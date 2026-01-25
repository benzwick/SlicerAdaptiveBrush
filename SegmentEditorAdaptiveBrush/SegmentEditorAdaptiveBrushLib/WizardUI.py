"""Wizard UI components for the Quick Select Parameters Wizard.

This module provides Qt-based dialogs for the wizard workflow:
- WizardPanel: Multi-step wizard dialog
- WizardResultsDialog: Results display with explanations
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .WizardDataStructures import WizardRecommendation

logger = logging.getLogger(__name__)

# Import Qt - handle both standalone and Slicer environments
try:
    import qt
    from qt import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
        QWizard,
        QWizardPage,
    )

    HAS_QT = True
except ImportError:
    HAS_QT = False
    # Create stub classes for type checking when Qt not available
    QDialog = object
    QWizard = object
    QWizardPage = object


class WizardPage(QWizardPage if HAS_QT else object):  # type: ignore[misc]
    """Base class for wizard pages."""

    def __init__(self, title: str, parent: Any = None):
        """Initialize wizard page.

        Args:
            title: Page title displayed at top.
            parent: Parent widget.
        """
        if HAS_QT:
            super().__init__(parent)
            self.setTitle(title)
            self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the page UI. Override in subclasses."""
        pass


class ForegroundSamplingPage(WizardPage):
    """Wizard page for sampling foreground (target structure) intensities."""

    def __init__(self, on_sample_callback: Optional[Callable] = None, parent: Any = None):
        self.on_sample_callback = on_sample_callback
        self._sample_count = 0
        super().__init__("Step 1: Sample Foreground", parent)

    def _setup_ui(self) -> None:
        if not HAS_QT:
            return

        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "<b>Paint a few strokes on the INSIDE of the structure you want to segment.</b><br><br>"
            "The wizard will analyze the intensity values to determine optimal thresholds."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Status
        self._status_label = QLabel("Paint Mode Active - Click and drag on image")
        self._status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        layout.addWidget(self._status_label)

        # Sample count
        self._count_label = QLabel("Samples collected: 0 voxels")
        layout.addWidget(self._count_label)

        # Buttons
        button_layout = QHBoxLayout()
        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self._on_clear)
        button_layout.addWidget(self._clear_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        layout.addStretch()

    def update_sample_count(self, count: int) -> None:
        """Update the displayed sample count."""
        self._sample_count = count
        if HAS_QT and hasattr(self, "_count_label"):
            self._count_label.setText(f"Samples collected: {count:,} voxels")
            self.completeChanged.emit()

    def _on_clear(self) -> None:
        """Clear samples callback."""
        self._sample_count = 0
        if hasattr(self, "_count_label"):
            self._count_label.setText("Samples collected: 0 voxels")
        if self.on_sample_callback:
            self.on_sample_callback("clear_foreground")

    def isComplete(self) -> bool:
        """Return whether enough samples have been collected."""
        return self._sample_count >= 100  # Require at least 100 voxels


class BackgroundSamplingPage(WizardPage):
    """Wizard page for sampling background intensities."""

    def __init__(self, on_sample_callback: Optional[Callable] = None, parent: Any = None):
        self.on_sample_callback = on_sample_callback
        self._sample_count = 0
        super().__init__("Step 2: Sample Background", parent)

    def _setup_ui(self) -> None:
        if not HAS_QT:
            return

        layout = QVBoxLayout(self)

        instructions = QLabel(
            "<b>Paint a few strokes OUTSIDE or AROUND the structure.</b><br><br>"
            "Sample areas that should NOT be included in the segmentation."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self._status_label = QLabel("Paint Mode Active - Click and drag on image")
        self._status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        layout.addWidget(self._status_label)

        self._count_label = QLabel("Samples collected: 0 voxels")
        layout.addWidget(self._count_label)

        button_layout = QHBoxLayout()
        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self._on_clear)
        button_layout.addWidget(self._clear_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        layout.addStretch()

    def update_sample_count(self, count: int) -> None:
        """Update the displayed sample count."""
        self._sample_count = count
        if HAS_QT and hasattr(self, "_count_label"):
            self._count_label.setText(f"Samples collected: {count:,} voxels")
            self.completeChanged.emit()

    def _on_clear(self) -> None:
        """Clear samples callback."""
        self._sample_count = 0
        if hasattr(self, "_count_label"):
            self._count_label.setText("Samples collected: 0 voxels")
        if self.on_sample_callback:
            self.on_sample_callback("clear_background")

    def isComplete(self) -> bool:
        """Return whether enough samples have been collected."""
        return self._sample_count >= 100


class BoundaryTracingPage(WizardPage):
    """Wizard page for optional boundary tracing."""

    def __init__(self, on_sample_callback: Optional[Callable] = None, parent: Any = None):
        self.on_sample_callback = on_sample_callback
        self._sample_count = 0
        super().__init__("Step 3: Trace Boundary (Optional)", parent)

    def _setup_ui(self) -> None:
        if not HAS_QT:
            return

        layout = QVBoxLayout(self)

        instructions = QLabel(
            "<b>Optionally, trace along the EDGE of the structure.</b><br><br>"
            "This helps the wizard understand the boundary characteristics. "
            "You can skip this step if the boundary is clear."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self._status_label = QLabel("Paint Mode Active - Trace along the edge")
        self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self._status_label)

        self._count_label = QLabel("Boundary points: 0")
        layout.addWidget(self._count_label)

        button_layout = QHBoxLayout()
        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self._on_clear)
        button_layout.addWidget(self._clear_button)

        self._skip_button = QPushButton("Skip This Step")
        self._skip_button.clicked.connect(self._on_skip)
        button_layout.addWidget(self._skip_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        layout.addStretch()

    def update_sample_count(self, count: int) -> None:
        """Update the displayed boundary point count."""
        self._sample_count = count
        if HAS_QT and hasattr(self, "_count_label"):
            self._count_label.setText(f"Boundary points: {count:,}")

    def _on_clear(self) -> None:
        """Clear boundary samples."""
        self._sample_count = 0
        if hasattr(self, "_count_label"):
            self._count_label.setText("Boundary points: 0")
        if self.on_sample_callback:
            self.on_sample_callback("clear_boundary")

    def _on_skip(self) -> None:
        """Skip boundary tracing."""
        if self.wizard():
            self.wizard().next()

    def isComplete(self) -> bool:
        """Boundary tracing is always optional."""
        return True


class QuestionsPage(WizardPage):
    """Wizard page for modality and structure type questions."""

    def __init__(self, parent: Any = None):
        self._modality: Optional[str] = None
        self._structure_type: Optional[str] = None
        self._priority: str = "balanced"
        super().__init__("Step 4: Additional Information", parent)

    def _setup_ui(self) -> None:
        if not HAS_QT:
            return

        layout = QVBoxLayout(self)

        instructions = QLabel(
            "<b>Provide additional information to improve recommendations.</b><br><br>"
            "These questions help the wizard suggest optimal parameters."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Modality selection
        modality_label = QLabel("What type of imaging is this?")
        layout.addWidget(modality_label)

        self._modality_combo = qt.QComboBox()
        self._modality_combo.addItem("Unknown / Not sure", None)
        self._modality_combo.addItem("CT (Computed Tomography)", "CT")
        self._modality_combo.addItem("MRI T1-weighted", "MRI_T1")
        self._modality_combo.addItem("MRI T2-weighted", "MRI_T2")
        self._modality_combo.addItem("PET", "PET")
        self._modality_combo.addItem("Ultrasound", "Ultrasound")
        self._modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        layout.addWidget(self._modality_combo)

        # Structure type selection
        structure_label = QLabel("What are you trying to segment?")
        layout.addWidget(structure_label)

        self._structure_combo = qt.QComboBox()
        self._structure_combo.addItem("Unknown / Other", None)
        self._structure_combo.addItem("Tumor or lesion", "tumor")
        self._structure_combo.addItem("Bone", "bone")
        self._structure_combo.addItem("Vessel (blood vessel)", "vessel")
        self._structure_combo.addItem("Brain tissue", "brain_tissue")
        self._structure_combo.addItem("Organ (liver, kidney, etc.)", "organ")
        self._structure_combo.currentIndexChanged.connect(self._on_structure_changed)
        layout.addWidget(self._structure_combo)

        # Priority selection
        priority_label = QLabel("What is your priority?")
        layout.addWidget(priority_label)

        self._priority_combo = qt.QComboBox()
        self._priority_combo.addItem("Balanced (recommended)", "balanced")
        self._priority_combo.addItem("Speed - faster results", "speed")
        self._priority_combo.addItem("Precision - more accurate boundaries", "precision")
        self._priority_combo.currentIndexChanged.connect(self._on_priority_changed)
        layout.addWidget(self._priority_combo)

        layout.addStretch()

    def _on_modality_changed(self, index: int) -> None:
        self._modality = self._modality_combo.itemData(index)

    def _on_structure_changed(self, index: int) -> None:
        self._structure_type = self._structure_combo.itemData(index)

    def _on_priority_changed(self, index: int) -> None:
        self._priority = self._priority_combo.itemData(index)

    def get_answers(self) -> dict:
        """Return the user's answers."""
        return {
            "modality": self._modality,
            "structure_type": self._structure_type,
            "priority": self._priority,
        }

    def isComplete(self) -> bool:
        """Questions page is always complete (all optional)."""
        return True


class ResultsPage(WizardPage):
    """Wizard page displaying analysis results and recommendations."""

    def __init__(self, parent: Any = None):
        self._recommendation: Optional[WizardRecommendation] = None
        super().__init__("Step 5: Recommendations", parent)

    def _setup_ui(self) -> None:
        if not HAS_QT:
            return

        layout = QVBoxLayout(self)

        instructions = QLabel("<b>Based on your samples, here are the recommended parameters:</b>")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Results will be populated dynamically
        self._results_widget = QWidget()
        self._results_layout = QVBoxLayout(self._results_widget)
        layout.addWidget(self._results_widget)

        # Placeholder
        self._placeholder = QLabel("Analyzing samples...")
        self._results_layout.addWidget(self._placeholder)

        layout.addStretch()

    def set_recommendation(self, recommendation: "WizardRecommendation") -> None:
        """Set and display the recommendation."""
        self._recommendation = recommendation

        if not HAS_QT or not hasattr(self, "_results_layout"):
            return

        # Clear existing content
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Algorithm
        algo_label = QLabel(f"<b>Algorithm:</b> {recommendation.algorithm}")
        self._results_layout.addWidget(algo_label)

        algo_reason = QLabel(f"<i>{recommendation.algorithm_reason}</i>")
        algo_reason.setWordWrap(True)
        algo_reason.setStyleSheet("color: #666; margin-left: 20px;")
        self._results_layout.addWidget(algo_reason)

        # Brush Radius
        radius_label = QLabel(f"<b>Brush Radius:</b> {recommendation.brush_radius_mm:.1f} mm")
        self._results_layout.addWidget(radius_label)

        radius_reason = QLabel(f"<i>{recommendation.radius_reason}</i>")
        radius_reason.setWordWrap(True)
        radius_reason.setStyleSheet("color: #666; margin-left: 20px;")
        self._results_layout.addWidget(radius_reason)

        # Edge Sensitivity
        sens_label = QLabel(f"<b>Edge Sensitivity:</b> {recommendation.edge_sensitivity}%")
        self._results_layout.addWidget(sens_label)

        sens_reason = QLabel(f"<i>{recommendation.sensitivity_reason}</i>")
        sens_reason.setWordWrap(True)
        sens_reason.setStyleSheet("color: #666; margin-left: 20px;")
        self._results_layout.addWidget(sens_reason)

        # Thresholds
        if recommendation.has_threshold_suggestion():
            thresh_label = QLabel(
                f"<b>Thresholds:</b> {recommendation.threshold_lower:.1f} - "
                f"{recommendation.threshold_upper:.1f}"
            )
            self._results_layout.addWidget(thresh_label)

        # Confidence
        confidence_pct = int(recommendation.confidence * 100)
        conf_color = (
            "#4CAF50" if confidence_pct >= 75 else "#FF9800" if confidence_pct >= 50 else "#F44336"
        )
        conf_label = QLabel(
            f"<b>Confidence:</b> <span style='color: {conf_color};'>{confidence_pct}%</span>"
        )
        self._results_layout.addWidget(conf_label)

        # Warnings
        if recommendation.has_warnings():
            warnings_label = QLabel("<b>Warnings:</b>")
            self._results_layout.addWidget(warnings_label)
            for warning in recommendation.warnings:
                warn_text = QLabel(f"- {warning}")
                warn_text.setWordWrap(True)
                warn_text.setStyleSheet("color: #FF9800; margin-left: 20px;")
                self._results_layout.addWidget(warn_text)

        # Alternatives
        if recommendation.alternative_algorithms:
            alt_label = QLabel("<b>Alternative algorithms:</b>")
            self._results_layout.addWidget(alt_label)
            for algo, reason in recommendation.alternative_algorithms[:2]:
                alt_text = QLabel(f"- {algo}: {reason}")
                alt_text.setWordWrap(True)
                alt_text.setStyleSheet("color: #666; margin-left: 20px;")
                self._results_layout.addWidget(alt_text)

    def get_recommendation(self) -> Optional["WizardRecommendation"]:
        """Return the current recommendation."""
        return self._recommendation


class WizardPanel(QWizard if HAS_QT else object):  # type: ignore[misc]
    """Multi-step wizard dialog for parameter selection."""

    def __init__(self, parent: Any = None):
        """Initialize the wizard panel.

        Args:
            parent: Parent widget.
        """
        if HAS_QT:
            super().__init__(parent)
            self.setWindowTitle("Quick Select Parameters Wizard")
            self.setWizardStyle(QWizard.ModernStyle)
            self.setMinimumSize(500, 400)

            self._setup_pages()

    def _setup_pages(self) -> None:
        """Create and add wizard pages."""
        self.foreground_page = ForegroundSamplingPage(self._on_sample_action)
        self.background_page = BackgroundSamplingPage(self._on_sample_action)
        self.boundary_page = BoundaryTracingPage(self._on_sample_action)
        self.questions_page = QuestionsPage()
        self.results_page = ResultsPage()

        self.addPage(self.foreground_page)
        self.addPage(self.background_page)
        self.addPage(self.boundary_page)
        self.addPage(self.questions_page)
        self.addPage(self.results_page)

    def _on_sample_action(self, action: str) -> None:
        """Handle sampling actions from pages.

        Override in subclass to connect to actual sampling logic.
        """
        logger.debug(f"Sample action: {action}")

    def get_questions_answers(self) -> dict:
        """Get answers from questions page."""
        return self.questions_page.get_answers()

    def set_recommendation(self, recommendation: "WizardRecommendation") -> None:
        """Set the recommendation to display in results page."""
        self.results_page.set_recommendation(recommendation)


class WizardResultsDialog(QDialog if HAS_QT else object):  # type: ignore[misc]
    """Standalone dialog for displaying wizard results."""

    def __init__(self, recommendation: "WizardRecommendation", parent: Any = None):
        """Initialize the results dialog.

        Args:
            recommendation: The wizard recommendation to display.
            parent: Parent widget.
        """
        self.recommendation = recommendation
        self._applied = False

        if HAS_QT:
            super().__init__(parent)
            self.setWindowTitle("Parameter Recommendations")
            self.setMinimumSize(450, 350)
            self._setup_ui()

    def _setup_ui(self) -> None:
        if not HAS_QT:
            return

        layout = QVBoxLayout(self)

        # Results page content
        results_page = ResultsPage()
        results_page.set_recommendation(self.recommendation)
        layout.addWidget(results_page)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        apply_button = QPushButton("Apply Parameters")
        apply_button.setDefault(True)
        apply_button.clicked.connect(self._on_apply)
        button_layout.addWidget(apply_button)

        layout.addLayout(button_layout)

    def _on_apply(self) -> None:
        """Handle apply button click."""
        self._applied = True
        self.accept()

    def was_applied(self) -> bool:
        """Return whether the user clicked Apply."""
        return self._applied
