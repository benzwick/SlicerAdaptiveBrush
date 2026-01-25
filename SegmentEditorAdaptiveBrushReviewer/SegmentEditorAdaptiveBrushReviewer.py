"""SegmentEditorAdaptiveBrushReviewer - Results Review Module.

A Slicer module for reviewing optimization results and managing gold standards.

See ADR-012 for architecture decisions.
"""

import logging
from pathlib import Path

import qt
import slicer
from SegmentEditorAdaptiveBrushReviewerLib import (
    ResultsLoader,
    VisualizationController,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)


class SegmentEditorAdaptiveBrushReviewer(ScriptedLoadableModule):
    """Module definition for Adaptive Brush Results Reviewer."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Adaptive Brush Reviewer"
        self.parent.categories = ["Testing"]
        self.parent.dependencies = ["SegmentEditor"]
        self.parent.contributors = ["SlicerAdaptiveBrush Team"]
        self.parent.helpText = """
        Review optimization results and compare segmentations against gold standards.

        Features:
        - Load optimization runs and browse trials
        - Dual segmentation display (gold vs test)
        - Screenshot thumbnail viewer
        - Save trial results as new gold standards

        See <a href="https://github.com/your-repo/SlicerAdaptiveBrush">documentation</a>.
        """
        self.parent.acknowledgementText = """
        Part of the SlicerAdaptiveBrush extension.
        """


class SegmentEditorAdaptiveBrushReviewerWidget(ScriptedLoadableModuleWidget):
    """Widget for the Reviewer module."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self.results_loader = None
        self.viz_controller = None
        self.current_run = None
        self.current_trial = None
        self.selected_screenshot_path = None

    def setup(self):
        """Set up the widget UI."""
        ScriptedLoadableModuleWidget.setup(self)

        # Initialize components
        self.logic = SegmentEditorAdaptiveBrushReviewerLogic()
        self.results_loader = ResultsLoader()
        self.viz_controller = VisualizationController()

        # Create UI
        self._create_run_selection_section()
        self._create_visualization_section()
        self._create_parameters_metrics_section()
        self._create_screenshots_section()
        self._create_actions_section()

        # Add vertical spacer
        self.layout.addStretch(1)

        # Initialize state
        self._refresh_run_list()

    def _create_run_selection_section(self):
        """Create the optimization run selection section."""
        collapsible = qt.QGroupBox("Optimization Run")
        layout = qt.QFormLayout(collapsible)

        # Run selector
        self.runComboBox = qt.QComboBox()
        self.runComboBox.currentTextChanged.connect(self._on_run_selected)

        run_row = qt.QHBoxLayout()
        run_row.addWidget(self.runComboBox)

        self.openFolderButton = qt.QPushButton("Open Folder")
        self.openFolderButton.clicked.connect(self._on_open_folder)
        run_row.addWidget(self.openFolderButton)

        self.refreshButton = qt.QPushButton("Refresh")
        self.refreshButton.clicked.connect(self._refresh_run_list)
        run_row.addWidget(self.refreshButton)

        layout.addRow("Run:", run_row)

        # Trial selector
        self.trialComboBox = qt.QComboBox()
        self.trialComboBox.currentIndexChanged.connect(self._on_trial_selected)
        layout.addRow("Trial:", self.trialComboBox)

        # Best trial indicator
        self.bestTrialLabel = qt.QLabel("Best: -")
        layout.addRow("", self.bestTrialLabel)

        self.layout.addWidget(collapsible)

    def _create_visualization_section(self):
        """Create the visualization controls section."""
        collapsible = qt.QGroupBox("Visualization")
        layout = qt.QVBoxLayout(collapsible)

        # View mode
        mode_row = qt.QHBoxLayout()
        mode_row.addWidget(qt.QLabel("View Mode:"))

        self.viewModeGroup = qt.QButtonGroup()
        for i, mode in enumerate(["Outline", "Transparent", "Fill"]):
            radio = qt.QRadioButton(mode)
            if i == 0:
                radio.setChecked(True)
            self.viewModeGroup.addButton(radio, i)
            mode_row.addWidget(radio)
        self.viewModeGroup.buttonClicked.connect(self._on_view_mode_changed)

        layout.addLayout(mode_row)

        # Legend
        legend_row = qt.QHBoxLayout()
        legend_row.addWidget(qt.QLabel("Legend:"))

        gold_label = qt.QLabel("Gold")
        gold_label.setStyleSheet("background-color: gold; padding: 2px 8px;")
        legend_row.addWidget(gold_label)

        test_label = qt.QLabel("Test")
        test_label.setStyleSheet("background-color: cyan; padding: 2px 8px;")
        legend_row.addWidget(test_label)

        overlap_label = qt.QLabel("Overlap")
        overlap_label.setStyleSheet("background-color: lightgreen; padding: 2px 8px;")
        legend_row.addWidget(overlap_label)

        legend_row.addStretch()
        layout.addLayout(legend_row)

        # Visibility toggles
        toggle_row = qt.QHBoxLayout()

        self.loadGoldButton = qt.QPushButton("Load Gold")
        self.loadGoldButton.clicked.connect(self._on_load_gold)
        toggle_row.addWidget(self.loadGoldButton)

        self.loadTestButton = qt.QPushButton("Load Test")
        self.loadTestButton.clicked.connect(self._on_load_test)
        toggle_row.addWidget(self.loadTestButton)

        self.toggleGoldCheck = qt.QCheckBox("Show Gold")
        self.toggleGoldCheck.setChecked(True)
        self.toggleGoldCheck.stateChanged.connect(self._on_toggle_gold)
        toggle_row.addWidget(self.toggleGoldCheck)

        self.toggleTestCheck = qt.QCheckBox("Show Test")
        self.toggleTestCheck.setChecked(True)
        self.toggleTestCheck.stateChanged.connect(self._on_toggle_test)
        toggle_row.addWidget(self.toggleTestCheck)

        layout.addLayout(toggle_row)

        self.layout.addWidget(collapsible)

    def _create_parameters_metrics_section(self):
        """Create the parameters and metrics display section."""
        collapsible = qt.QGroupBox("Parameters & Metrics")
        layout = qt.QHBoxLayout(collapsible)

        # Parameters
        params_group = qt.QGroupBox("Parameters")
        params_layout = qt.QVBoxLayout(params_group)
        self.paramsText = qt.QTextEdit()
        self.paramsText.setReadOnly(True)
        self.paramsText.setMaximumHeight(120)
        params_layout.addWidget(self.paramsText)
        layout.addWidget(params_group)

        # Metrics
        metrics_group = qt.QGroupBox("Metrics")
        metrics_layout = qt.QVBoxLayout(metrics_group)
        self.metricsText = qt.QTextEdit()
        self.metricsText.setReadOnly(True)
        self.metricsText.setMaximumHeight(120)
        metrics_layout.addWidget(self.metricsText)
        layout.addWidget(metrics_group)

        self.layout.addWidget(collapsible)

    def _create_screenshots_section(self):
        """Create the screenshots viewer section."""
        collapsible = qt.QGroupBox("Screenshots")
        layout = qt.QVBoxLayout(collapsible)

        # Thumbnail area (placeholder - actual viewer is more complex)
        self.screenshotList = qt.QListWidget()
        self.screenshotList.setViewMode(qt.QListWidget.IconMode)
        self.screenshotList.setIconSize(qt.QSize(80, 60))
        self.screenshotList.setMaximumHeight(100)
        self.screenshotList.itemClicked.connect(self._on_screenshot_selected)
        layout.addWidget(self.screenshotList)

        # Selected screenshot info
        info_row = qt.QHBoxLayout()
        self.screenshotPathLabel = qt.QLabel("Selected: -")
        info_row.addWidget(self.screenshotPathLabel)

        self.copyPathButton = qt.QPushButton("Copy Path")
        self.copyPathButton.clicked.connect(self._on_copy_path)
        info_row.addWidget(self.copyPathButton)

        self.viewFullButton = qt.QPushButton("View Full")
        self.viewFullButton.clicked.connect(self._on_view_full)
        info_row.addWidget(self.viewFullButton)

        layout.addLayout(info_row)

        self.layout.addWidget(collapsible)

    def _create_actions_section(self):
        """Create the actions section."""
        collapsible = qt.QGroupBox("Actions")
        layout = qt.QHBoxLayout(collapsible)

        self.saveGoldButton = qt.QPushButton("Save as Gold Standard")
        self.saveGoldButton.clicked.connect(self._on_save_as_gold)
        layout.addWidget(self.saveGoldButton)

        self.exportReportButton = qt.QPushButton("Export Report")
        self.exportReportButton.clicked.connect(self._on_export_report)
        layout.addWidget(self.exportReportButton)

        self.compareAlgosButton = qt.QPushButton("Compare Algorithms")
        self.compareAlgosButton.clicked.connect(self._on_compare_algorithms)
        layout.addWidget(self.compareAlgosButton)

        self.layout.addWidget(collapsible)

    def _refresh_run_list(self):
        """Refresh the list of optimization runs."""
        self.runComboBox.clear()

        runs = self.results_loader.list_runs()
        for run_path in runs:
            self.runComboBox.addItem(run_path.name, str(run_path))

        if runs:
            self._on_run_selected(runs[0].name)

    def _on_run_selected(self, run_name):
        """Handle run selection."""
        if not run_name:
            return

        run_path = Path(self.runComboBox.currentData)
        try:
            self.current_run = self.results_loader.load(run_path)
            self._populate_trials()
        except Exception as e:
            logging.error(f"Failed to load run: {e}")
            slicer.util.errorDisplay(f"Failed to load run: {e}")

    def _populate_trials(self):
        """Populate trial dropdown from current run."""
        self.trialComboBox.clear()

        if not self.current_run:
            return

        for trial in self.current_run.trials:
            label = f"#{trial.trial_number} - Dice: {trial.value:.3f}"
            if trial.pruned:
                label += " (pruned)"
            self.trialComboBox.addItem(label, trial.trial_number)

        # Update best trial label
        if self.current_run.best_trial:
            best = self.current_run.best_trial
            self.bestTrialLabel.setText(f"Best: #{best.trial_number} (Dice: {best.value:.3f})")

    def _on_trial_selected(self, index):
        """Handle trial selection."""
        if index < 0 or not self.current_run:
            return

        trial_number = self.trialComboBox.itemData(index)
        trial = next(
            (t for t in self.current_run.trials if t.trial_number == trial_number),
            None,
        )

        if trial:
            self.current_trial = trial
            self._display_trial(trial)

    def _display_trial(self, trial):
        """Display trial parameters and metrics."""
        # Parameters
        params_text = "\n".join(f"{k}: {v}" for k, v in trial.params.items())
        self.paramsText.setPlainText(params_text)

        # Metrics
        metrics_text = f"Dice: {trial.value:.3f}\n"
        metrics_text += f"Duration: {trial.duration_ms:.0f}ms\n"
        if trial.pruned:
            metrics_text += "Status: Pruned\n"
        self.metricsText.setPlainText(metrics_text)

        # Screenshots
        self._load_screenshots(trial)

    def _load_screenshots(self, trial):
        """Load screenshot thumbnails for trial."""
        self.screenshotList.clear()
        self.selected_screenshot_path = None
        self.screenshotPathLabel.setText("Selected: -")

        # Load from trial's screenshot list
        for screenshot_path in trial.screenshots:
            if not screenshot_path.exists():
                continue

            item = qt.QListWidgetItem()
            item.setData(qt.Qt.UserRole, str(screenshot_path))
            item.setToolTip(screenshot_path.name)

            # Load thumbnail
            pixmap = qt.QPixmap(str(screenshot_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(80, 60, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
                item.setIcon(qt.QIcon(scaled))

            self.screenshotList.addItem(item)

        # If no trial-specific screenshots, try loading from run's screenshots folder
        if self.screenshotList.count == 0 and self.current_run:
            ss_dir = self.current_run.path / "screenshots"
            if ss_dir.exists():
                for png_file in sorted(ss_dir.glob("*.png"))[:20]:  # Limit to 20
                    item = qt.QListWidgetItem()
                    item.setData(qt.Qt.UserRole, str(png_file))
                    item.setToolTip(png_file.name)

                    pixmap = qt.QPixmap(str(png_file))
                    if not pixmap.isNull():
                        scaled = pixmap.scaled(
                            80, 60, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation
                        )
                        item.setIcon(qt.QIcon(scaled))

                    self.screenshotList.addItem(item)

    def _on_view_mode_changed(self, button):
        """Handle view mode change."""
        modes = ["outline", "transparent", "fill"]
        mode_id = self.viewModeGroup.id(button)
        if 0 <= mode_id < len(modes):
            self.viz_controller.set_view_mode(modes[mode_id])

    def _on_load_gold(self):
        """Load gold standard segmentation."""
        if not self.current_run:
            slicer.util.warningDisplay("No run loaded")
            return

        # Try to get gold standard path from run config
        gold_path = self.results_loader.get_gold_standard_path(self.current_run)

        if gold_path and gold_path.exists():
            if self.viz_controller.load_gold_segmentation(gold_path):
                slicer.util.infoDisplay(f"Loaded gold standard: {gold_path.name}")
            else:
                slicer.util.errorDisplay(f"Failed to load: {gold_path}")
        else:
            # Try using GoldStandardManager if path not in config
            try:
                from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

                manager = GoldStandardManager()

                # Get gold name from config or trial
                gold_name = None
                if self.current_run.config.get("recipes"):
                    recipe = self.current_run.config["recipes"][0]
                    gold_name = recipe.get("gold_standard")

                if gold_name:
                    gold_node, metadata = manager.load_gold(gold_name)
                    if gold_node:
                        # Apply gold color and register with viz controller
                        self.viz_controller.gold_seg_node = gold_node
                        self.viz_controller._apply_color(gold_node, self.viz_controller.GOLD_COLOR)
                        self.viz_controller._set_display_mode(
                            gold_node, self.viz_controller.view_mode
                        )
                        gold_node.SetName("Gold Standard")
                        slicer.util.infoDisplay(f"Loaded gold standard: {gold_name}")
                        return

                slicer.util.warningDisplay("No gold standard found in run configuration")
            except Exception as e:
                logging.exception(f"Failed to load gold standard: {e}")
                slicer.util.errorDisplay(f"Failed to load gold standard: {e}")

    def _on_load_test(self):
        """Load test segmentation."""
        if not self.current_trial:
            slicer.util.warningDisplay("No trial selected")
            return

        seg_path = self.current_trial.segmentation_path
        if seg_path and seg_path.exists():
            if self.viz_controller.load_test_segmentation(seg_path):
                slicer.util.infoDisplay(f"Loaded test segmentation: {seg_path.name}")
            else:
                slicer.util.errorDisplay(f"Failed to load: {seg_path}")
        else:
            slicer.util.warningDisplay(
                f"No segmentation file found for trial #{self.current_trial.trial_number}"
            )

    def _on_toggle_gold(self, state):
        """Toggle gold standard visibility."""
        self.viz_controller.toggle_gold(state == qt.Qt.Checked)

    def _on_toggle_test(self, state):
        """Toggle test segmentation visibility."""
        self.viz_controller.toggle_test(state == qt.Qt.Checked)

    def _on_open_folder(self):
        """Open run folder in file manager."""
        if not self.current_run:
            return

        run_path = Path(self.runComboBox.currentData)
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(run_path)))

    def _on_screenshot_selected(self, item):
        """Handle screenshot selection."""
        path = item.data(qt.Qt.UserRole)
        if path:
            self.selected_screenshot_path = path
            self.screenshotPathLabel.setText(f"Selected: {Path(path).name}")

    def _on_copy_path(self):
        """Copy selected screenshot path to clipboard."""
        if not self.selected_screenshot_path:
            slicer.util.warningDisplay("No screenshot selected")
            return

        clipboard = qt.QApplication.clipboard()
        clipboard.setText(self.selected_screenshot_path)
        slicer.util.infoDisplay(f"Copied: {self.selected_screenshot_path}")

    def _on_view_full(self):
        """Open selected screenshot in system viewer."""
        if not self.selected_screenshot_path:
            slicer.util.warningDisplay("No screenshot selected")
            return

        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(self.selected_screenshot_path))

    def _on_save_as_gold(self):
        """Save current trial as gold standard."""
        if not self.current_trial:
            slicer.util.warningDisplay("No trial selected")
            return

        # Check if test segmentation is loaded
        test_node = self.viz_controller.get_test_node()
        if not test_node:
            slicer.util.warningDisplay("Load a test segmentation first")
            return

        # Ask for gold standard name
        default_name = f"trial_{self.current_trial.trial_number:03d}"
        if self.current_run:
            default_name = f"{self.current_run.name}_{default_name}"

        name, ok = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            "Save as Gold Standard",
            "Gold standard name:",
            qt.QLineEdit.Normal,
            default_name,
        )

        if not ok or not name:
            return

        try:
            from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

            manager = GoldStandardManager()

            # Get segment ID
            segmentation = test_node.GetSegmentation()
            segment_id = (
                segmentation.GetNthSegmentID(0) if segmentation.GetNumberOfSegments() > 0 else None
            )

            if not segment_id:
                slicer.util.errorDisplay("No segments found in test segmentation")
                return

            # Find source volume if possible
            volume_node = None
            # Try to find from current scene
            volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if volume_nodes:
                volume_node = volume_nodes[0]

            manager.save_gold(
                name=name,
                segmentation_node=test_node,
                segment_id=segment_id,
                volume_node=volume_node,
                metadata={
                    "source": "reviewer",
                    "run": self.current_run.name if self.current_run else "unknown",
                    "trial_number": self.current_trial.trial_number,
                    "dice": self.current_trial.value,
                    "params": self.current_trial.params,
                },
            )

            slicer.util.infoDisplay(f"Saved gold standard: {name}")

        except Exception as e:
            logging.exception(f"Failed to save gold standard: {e}")
            slicer.util.errorDisplay(f"Failed to save gold standard: {e}")

    def _on_export_report(self):
        """Export comparison report."""
        if not self.current_run:
            slicer.util.warningDisplay("No run loaded")
            return

        report_path = self.current_run.path / "review_report.md"

        lines = [
            "# Optimization Review Report",
            "",
            f"**Run:** {self.current_run.name}",
            f"**Total Trials:** {len(self.current_run.trials)}",
            "",
        ]

        # Best trial info
        if self.current_run.best_trial:
            best = self.current_run.best_trial
            lines.extend(
                [
                    "## Best Trial",
                    "",
                    f"- **Trial #:** {best.trial_number}",
                    f"- **Dice:** {best.value:.4f}",
                    f"- **Duration:** {best.duration_ms:.0f}ms",
                    "",
                    "### Parameters",
                    "",
                ]
            )
            for k, v in best.params.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Parameter importance
        if self.current_run.parameter_importance:
            lines.extend(
                [
                    "## Parameter Importance",
                    "",
                    "| Parameter | Importance |",
                    "|-----------|------------|",
                ]
            )
            sorted_importance = sorted(
                self.current_run.parameter_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for param, importance in sorted_importance:
                lines.append(f"| {param} | {importance:.4f} |")
            lines.append("")

        # All trials table
        lines.extend(
            [
                "## All Trials",
                "",
                "| # | Dice | Duration (ms) | Status |",
                "|---|------|---------------|--------|",
            ]
        )

        for trial in sorted(self.current_run.trials, key=lambda t: t.value, reverse=True):
            status = "Pruned" if trial.pruned else "Complete"
            lines.append(
                f"| {trial.trial_number} | {trial.value:.4f} | {trial.duration_ms:.0f} | {status} |"
            )

        lines.append("")

        # Write report
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        slicer.util.infoDisplay(f"Report saved: {report_path}")
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(report_path)))

    def _on_compare_algorithms(self):
        """Show algorithm comparison view."""
        if not self.current_run:
            slicer.util.warningDisplay("No run loaded")
            return

        # Group trials by algorithm
        algo_trials: dict[str, list] = {}
        for trial in self.current_run.trials:
            algo = trial.params.get("algorithm", "unknown")
            if algo not in algo_trials:
                algo_trials[algo] = []
            algo_trials[algo].append(trial)

        if not algo_trials:
            slicer.util.warningDisplay("No algorithm data found in trials")
            return

        # Build comparison text
        lines = ["Algorithm Comparison", "=" * 40, ""]

        for algo, trials in sorted(algo_trials.items()):
            values = [t.value for t in trials if not t.pruned]
            if not values:
                continue

            best_val = max(values)
            mean_val = sum(values) / len(values)
            times = [t.duration_ms for t in trials if not t.pruned]
            mean_time = sum(times) / len(times) if times else 0

            lines.extend(
                [
                    f"{algo}:",
                    f"  Trials: {len(trials)} ({len(values)} completed)",
                    f"  Best Dice: {best_val:.4f}",
                    f"  Mean Dice: {mean_val:.4f}",
                    f"  Mean Time: {mean_time:.0f}ms",
                    "",
                ]
            )

        # Show in a dialog
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("Algorithm Comparison")
        dialog.setMinimumSize(400, 300)

        layout = qt.QVBoxLayout(dialog)

        text_edit = qt.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText("\n".join(lines))
        text_edit.setFont(qt.QFont("Courier", 10))
        layout.addWidget(text_edit)

        close_button = qt.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec_()

    def cleanup(self):
        """Clean up when module is closed."""
        if self.viz_controller:
            self.viz_controller.cleanup()


class SegmentEditorAdaptiveBrushReviewerLogic(ScriptedLoadableModuleLogic):
    """Logic for the Reviewer module."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
