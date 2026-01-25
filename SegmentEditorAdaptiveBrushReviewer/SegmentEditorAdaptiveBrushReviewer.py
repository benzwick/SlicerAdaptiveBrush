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
    ScreenshotViewer,
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
        self.screenshot_viewer = None
        self.current_run = None
        self.current_trial = None

    def setup(self):
        """Set up the widget UI."""
        ScriptedLoadableModuleWidget.setup(self)

        # Initialize components
        self.logic = SegmentEditorAdaptiveBrushReviewerLogic()
        self.results_loader = ResultsLoader()
        self.viz_controller = VisualizationController()
        self.screenshot_viewer = ScreenshotViewer()

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
        # TODO: Load actual screenshots from trial data

    def _on_view_mode_changed(self, button):
        """Handle view mode change."""
        modes = ["outline", "transparent", "fill"]
        mode_id = self.viewModeGroup.id(button)
        if 0 <= mode_id < len(modes):
            self.viz_controller.set_view_mode(modes[mode_id])

    def _on_load_gold(self):
        """Load gold standard segmentation."""
        if not self.current_run:
            return
        # TODO: Get gold standard path from run config
        # self.viz_controller.load_gold_segmentation(gold_path)

    def _on_load_test(self):
        """Load test segmentation."""
        if not self.current_trial:
            return
        # TODO: Get segmentation path from trial
        # self.viz_controller.load_test_segmentation(test_path)

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
            self.screenshotPathLabel.setText(f"Selected: {Path(path).name}")

    def _on_copy_path(self):
        """Copy selected screenshot path to clipboard."""
        # TODO: Get selected screenshot path and copy to clipboard
        pass

    def _on_view_full(self):
        """Open selected screenshot in system viewer."""
        # TODO: Get selected screenshot path
        # qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(path))
        pass

    def _on_save_as_gold(self):
        """Save current trial as gold standard."""
        if not self.current_trial:
            slicer.util.warningDisplay("No trial selected")
            return

        # TODO: Implement save as gold standard
        slicer.util.infoDisplay("Save as gold standard - Not yet implemented")

    def _on_export_report(self):
        """Export comparison report."""
        if not self.current_run:
            slicer.util.warningDisplay("No run loaded")
            return

        # TODO: Implement report export
        slicer.util.infoDisplay("Export report - Not yet implemented")

    def _on_compare_algorithms(self):
        """Show algorithm comparison view."""
        if not self.current_run:
            slicer.util.warningDisplay("No run loaded")
            return

        # TODO: Implement algorithm comparison
        slicer.util.infoDisplay("Compare algorithms - Not yet implemented")

    def cleanup(self):
        """Clean up when module is closed."""
        if self.viz_controller:
            self.viz_controller.cleanup()


class SegmentEditorAdaptiveBrushReviewerLogic(ScriptedLoadableModuleLogic):
    """Logic for the Reviewer module."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
