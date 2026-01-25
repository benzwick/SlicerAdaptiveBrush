"""Embedded Wizard UI for the Quick Select Parameters Wizard.

This module provides a wizard panel that embeds directly in the
Segment Editor options frame, replacing the normal controls temporarily.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from WizardDataStructures import WizardRecommendation

logger = logging.getLogger(__name__)

# Import Qt - handle both standalone and Slicer environments
try:
    import qt

    HAS_QT = True
except ImportError:
    HAS_QT = False


class EmbeddedWizardPanel:
    """Embedded wizard panel that replaces normal controls in options frame.

    This panel provides a step-by-step wizard interface directly in the
    Segment Editor options area, avoiding dialog window management issues.
    """

    # Step indices
    STEP_FOREGROUND = 0
    STEP_BACKGROUND = 1
    STEP_BOUNDARY = 2
    STEP_QUESTIONS = 3
    STEP_RESULTS = 4

    def __init__(
        self,
        on_page_changed: Optional[Callable[[int], None]] = None,
        on_finished: Optional[Callable[[bool], None]] = None,
    ):
        """Initialize the embedded wizard.

        Args:
            on_page_changed: Callback when step changes, receives step index.
            on_finished: Callback when wizard completes, receives True if accepted.
        """
        self.on_page_changed = on_page_changed
        self.on_finished = on_finished
        self._current_step = 0
        self._sample_counts = {
            "foreground": 0,
            "background": 0,
            "boundary": 0,
        }
        self._recommendation: Optional[WizardRecommendation] = None
        self._answers: dict = {}

        # UI elements (created in build_ui)
        self.container: Any = None
        self.step_stack: Any = None
        self.step_label: Any = None
        self.status_label: Any = None
        self.back_button: Any = None
        self.next_button: Any = None
        self.cancel_button: Any = None

        # Step-specific widgets
        self.fg_count_label: Any = None
        self.bg_count_label: Any = None
        self.boundary_count_label: Any = None
        self.modality_combo: Any = None
        self.structure_combo: Any = None
        self.priority_combo: Any = None
        self.results_text: Any = None

    def build_ui(self) -> Any:
        """Build and return the wizard container widget.

        Returns:
            The container widget to add to the options frame.
        """
        if not HAS_QT:
            return None

        # Main container
        self.container = qt.QWidget()
        main_layout = qt.QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header with step indicator
        header = qt.QWidget()
        header_layout = qt.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 5)

        self.step_label = qt.QLabel("Step 1 of 5: Sample Foreground")
        self.step_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.step_label)
        header_layout.addStretch()

        main_layout.addWidget(header)

        # Step content area (stacked widget)
        self.step_stack = qt.QStackedWidget()
        self._build_step_pages()
        main_layout.addWidget(self.step_stack, 1)

        # Status label for sample counts
        self.status_label = qt.QLabel("")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addWidget(self.status_label)

        # Navigation buttons
        nav_widget = qt.QWidget()
        nav_layout = qt.QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 10, 0, 0)

        self.cancel_button = qt.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        nav_layout.addWidget(self.cancel_button)

        nav_layout.addStretch()

        self.back_button = qt.QPushButton("← Back")
        self.back_button.clicked.connect(self._on_back)
        self.back_button.setEnabled(False)
        nav_layout.addWidget(self.back_button)

        self.next_button = qt.QPushButton("Next →")
        self.next_button.clicked.connect(self._on_next)
        nav_layout.addWidget(self.next_button)

        main_layout.addWidget(nav_widget)

        self._update_step_ui()
        return self.container

    def _build_step_pages(self) -> None:
        """Build all step pages."""
        # Step 0: Foreground sampling
        fg_page = self._build_sampling_page(
            "Paint strokes <b>INSIDE</b> the structure you want to segment.",
            "foreground",
        )
        self.step_stack.addWidget(fg_page)

        # Step 1: Background sampling
        bg_page = self._build_sampling_page(
            "Paint strokes <b>OUTSIDE / AROUND</b> the target structure.",
            "background",
        )
        self.step_stack.addWidget(bg_page)

        # Step 2: Boundary tracing (optional)
        boundary_page = self._build_sampling_page(
            "Optionally, trace <b>ALONG THE EDGE</b> of the structure.\n"
            "This helps estimate boundary roughness. (Skip if not needed)",
            "boundary",
            optional=True,
        )
        self.step_stack.addWidget(boundary_page)

        # Step 3: Questions
        questions_page = self._build_questions_page()
        self.step_stack.addWidget(questions_page)

        # Step 4: Results
        results_page = self._build_results_page()
        self.step_stack.addWidget(results_page)

    def _build_sampling_page(
        self, instructions: str, sample_type: str, optional: bool = False
    ) -> Any:
        """Build a sampling step page."""
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 5, 0, 5)

        # Instructions
        instr_label = qt.QLabel(instructions)
        instr_label.setWordWrap(True)
        layout.addWidget(instr_label)

        # Sample count display
        count_label = qt.QLabel("Samples collected: 0")
        count_label.setStyleSheet("margin-top: 10px; font-weight: bold;")
        layout.addWidget(count_label)

        # Store reference to count label
        if sample_type == "foreground":
            self.fg_count_label = count_label
        elif sample_type == "background":
            self.bg_count_label = count_label
        else:
            self.boundary_count_label = count_label

        # Clear button
        clear_btn = qt.QPushButton("Clear Samples")
        clear_btn.setMaximumWidth(120)
        clear_btn.clicked.connect(lambda: self._on_clear_samples(sample_type))
        layout.addWidget(clear_btn)

        if optional:
            skip_note = qt.QLabel("(Optional - click Next to skip)")
            skip_note.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(skip_note)

        layout.addStretch()
        return page

    def _build_questions_page(self) -> Any:
        """Build the questions step page."""
        page = qt.QWidget()
        layout = qt.QFormLayout(page)
        layout.setContentsMargins(0, 5, 0, 5)

        # Modality selection
        self.modality_combo = qt.QComboBox()
        self.modality_combo.addItem("Unknown / Other", "")
        self.modality_combo.addItem("CT", "CT")
        self.modality_combo.addItem("MRI T1-weighted", "MRI_T1")
        self.modality_combo.addItem("MRI T2-weighted", "MRI_T2")
        self.modality_combo.addItem("Ultrasound", "Ultrasound")
        self.modality_combo.addItem("PET", "PET")
        layout.addRow("Imaging Modality:", self.modality_combo)

        # Structure type selection
        self.structure_combo = qt.QComboBox()
        self.structure_combo.addItem("Unknown / Other", "")
        self.structure_combo.addItem("Tumor / Lesion", "tumor")
        self.structure_combo.addItem("Blood Vessel", "vessel")
        self.structure_combo.addItem("Bone", "bone")
        self.structure_combo.addItem("Brain Tissue", "brain_tissue")
        self.structure_combo.addItem("Organ", "organ")
        layout.addRow("Target Structure:", self.structure_combo)

        # Priority selection
        self.priority_combo = qt.QComboBox()
        self.priority_combo.addItem("Balanced (Recommended)", "balanced")
        self.priority_combo.addItem("Faster Segmentation", "speed")
        self.priority_combo.addItem("Higher Precision", "precision")
        layout.addRow("Priority:", self.priority_combo)

        return page

    def _build_results_page(self) -> Any:
        """Build the results step page."""
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 5, 0, 5)

        # Results display
        self.results_text = qt.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("background-color: #f5f5f5;")
        layout.addWidget(self.results_text)

        return page

    def _update_step_ui(self) -> None:
        """Update UI based on current step."""
        step_titles = [
            "Step 1 of 5: Sample Foreground",
            "Step 2 of 5: Sample Background",
            "Step 3 of 5: Trace Boundary (Optional)",
            "Step 4 of 5: Questions",
            "Step 5 of 5: Recommendations",
        ]

        self.step_label.setText(step_titles[self._current_step])
        self.step_stack.setCurrentIndex(self._current_step)

        # Update navigation buttons
        self.back_button.setEnabled(self._current_step > 0)

        if self._current_step == self.STEP_RESULTS:
            self.next_button.setText("Apply && Finish")
        else:
            self.next_button.setText("Next →")

        # Update status based on step
        self._update_status()

        logger.debug(f"Wizard step changed to {self._current_step}")

    def _update_status(self) -> None:
        """Update status label based on current step."""
        if self._current_step == self.STEP_FOREGROUND:
            count = self._sample_counts["foreground"]
            self.status_label.setText(
                f"Click and drag in slice view to paint samples ({count} voxels)"
            )
        elif self._current_step == self.STEP_BACKGROUND:
            count = self._sample_counts["background"]
            self.status_label.setText(
                f"Click and drag in slice view to paint samples ({count} voxels)"
            )
        elif self._current_step == self.STEP_BOUNDARY:
            count = self._sample_counts["boundary"]
            self.status_label.setText(f"Optionally trace boundary ({count} points)")
        elif self._current_step == self.STEP_QUESTIONS:
            self.status_label.setText("Answer the questions above (optional)")
        elif self._current_step == self.STEP_RESULTS:
            self.status_label.setText("Review recommendations and click Apply")

    def _on_next(self) -> None:
        """Handle Next button click."""
        if self._current_step == self.STEP_RESULTS:
            # Finish and apply
            if self.on_finished:
                self.on_finished(True)
        else:
            self._current_step += 1
            self._update_step_ui()
            if self.on_page_changed:
                self.on_page_changed(self._current_step)

    def _on_back(self) -> None:
        """Handle Back button click."""
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step_ui()
            if self.on_page_changed:
                self.on_page_changed(self._current_step)

    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        if self.on_finished:
            self.on_finished(False)

    def _on_clear_samples(self, sample_type: str) -> None:
        """Handle clear samples button for a specific type."""
        self._sample_counts[sample_type] = 0
        self._update_sample_count_label(sample_type, 0)
        logger.debug(f"Cleared {sample_type} samples")

    def _update_sample_count_label(self, sample_type: str, count: int) -> None:
        """Update the sample count label for a specific type."""
        label = None
        if sample_type == "foreground":
            label = self.fg_count_label
        elif sample_type == "background":
            label = self.bg_count_label
        elif sample_type == "boundary":
            label = self.boundary_count_label

        if label:
            label.setText(f"Samples collected: {count}")

    def update_sample_count(self, sample_type: str, count: int) -> None:
        """Update sample count for display.

        Args:
            sample_type: 'foreground', 'background', or 'boundary'
            count: Number of samples collected
        """
        self._sample_counts[sample_type] = count
        self._update_sample_count_label(sample_type, count)
        self._update_status()

    def get_answers(self) -> dict:
        """Get user answers from questions page."""
        if not HAS_QT or not self.modality_combo:
            return {}

        return {
            "modality": self.modality_combo.currentData or None,
            "structure_type": self.structure_combo.currentData or None,
            "priority": self.priority_combo.currentData or "balanced",
        }

    def set_recommendation(self, recommendation: "WizardRecommendation") -> None:
        """Display the recommendation in results page."""
        self._recommendation = recommendation

        if not HAS_QT or not self.results_text:
            return

        # Format recommendation as HTML
        html = self._format_recommendation_html(recommendation)
        self.results_text.setHtml(html)

    def _format_recommendation_html(self, rec: "WizardRecommendation") -> str:
        """Format recommendation as HTML for display."""
        html = "<h3>Recommended Parameters</h3>"

        # Algorithm
        html += f"<p><b>Algorithm:</b> {rec.algorithm}</p>"
        html += f"<p style='color: #666; margin-left: 15px;'>{rec.algorithm_reason}</p>"

        # Brush radius
        html += f"<p><b>Brush Radius:</b> {rec.brush_radius_mm:.1f} mm</p>"
        html += f"<p style='color: #666; margin-left: 15px;'>{rec.radius_reason}</p>"

        # Edge sensitivity
        html += f"<p><b>Edge Sensitivity:</b> {rec.edge_sensitivity}%</p>"
        html += f"<p style='color: #666; margin-left: 15px;'>{rec.sensitivity_reason}</p>"

        # Thresholds
        if rec.has_threshold_suggestion():
            html += f"<p><b>Threshold Range:</b> {rec.threshold_lower:.1f} - {rec.threshold_upper:.1f}</p>"

        # Confidence
        confidence_pct = int(rec.confidence * 100)
        color = (
            "#4CAF50" if confidence_pct >= 70 else "#FF9800" if confidence_pct >= 50 else "#F44336"
        )
        html += f"<p><b>Confidence:</b> <span style='color: {color};'>{confidence_pct}%</span></p>"

        # Warnings
        if rec.warnings:
            html += "<h4 style='color: #FF9800;'>⚠️ Warnings</h4><ul>"
            for warning in rec.warnings:
                html += f"<li>{warning}</li>"
            html += "</ul>"

        # Alternatives
        if rec.alternative_algorithms:
            html += "<h4>Alternative Algorithms</h4><ul>"
            for algo, reason in rec.alternative_algorithms[:2]:
                html += f"<li><b>{algo}</b>: {reason}</li>"
            html += "</ul>"

        return html

    def get_recommendation(self) -> Optional["WizardRecommendation"]:
        """Return the current recommendation."""
        return self._recommendation

    @property
    def current_step(self) -> int:
        """Return the current step index."""
        return self._current_step

    def hide(self) -> None:
        """Hide the wizard panel."""
        if self.container:
            self.container.hide()

    def show(self) -> None:
        """Show the wizard panel."""
        if self.container:
            self.container.show()
