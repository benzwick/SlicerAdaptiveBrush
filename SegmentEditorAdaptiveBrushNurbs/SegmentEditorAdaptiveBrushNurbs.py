"""SegmentEditorAdaptiveBrushNurbs - Volumetric NURBS Export Module.

A Slicer module for converting painted segmentations into volumetric NURBS
elements for isogeometric analysis (IGA), triangulation, and CAD export.

Features:
- Automatic structure type detection (simple/tubular/branching)
- Hexahedral control mesh generation
- Volumetric NURBS construction using geomdl
- MFEM mesh export for IGA simulations
- STL/OBJ/VTK triangulated mesh export
- VTK visualization of control meshes and NURBS volumes

See the plan for architecture details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

if TYPE_CHECKING:
    from vtkMRMLSegmentationNode import vtkMRMLSegmentationNode

logger = logging.getLogger(__name__)


class SegmentEditorAdaptiveBrushNurbs(ScriptedLoadableModule):
    """Module definition for Adaptive Brush NURBS Export."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Adaptive Brush NURBS"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations", "SegmentEditor"]
        self.parent.contributors = ["SlicerAdaptiveBrush Team"]
        self.parent.helpText = """
        Convert painted segmentations into volumetric NURBS elements.

        <h3>Features:</h3>
        <ul>
        <li>Automatic structure type detection (simple/tubular/branching)</li>
        <li>Hexahedral NURBS volume generation</li>
        <li>MFEM mesh export for isogeometric analysis</li>
        <li>STL/OBJ triangulated mesh export</li>
        </ul>

        <h3>Usage:</h3>
        <ol>
        <li>Select a segmentation and segment</li>
        <li>Choose fitting parameters (degree, tolerance)</li>
        <li>Click "Generate NURBS" to create the volumetric NURBS</li>
        <li>Export to MFEM, STL, or other formats</li>
        </ol>

        See <a href="https://github.com/benzwick/SlicerAdaptiveBrush">documentation</a>.
        """
        self.parent.acknowledgementText = """
        Part of the SlicerAdaptiveBrush extension.

        Volumetric NURBS generation based on:
        - Patient-Specific Vascular NURBS Modeling for IGA (Zhang et al.)
        - geomdl library for NURBS construction
        """


class SegmentEditorAdaptiveBrushNurbsWidget(ScriptedLoadableModuleWidget):
    """Widget for the NURBS Export module."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic: SegmentEditorAdaptiveBrushNurbsLogic | None = None

        # Current state
        self._current_segmentation_node: vtkMRMLSegmentationNode | None = None
        self._current_segment_id: str | None = None
        self._generated_nurbs: object | None = None
        self._hex_mesh_actor: vtk.vtkActor | None = None

        # Progress tracking
        self._progress_bar: qt.QProgressBar | None = None
        self._status_label: qt.QLabel | None = None

    def setup(self):
        """Set up the widget UI."""
        ScriptedLoadableModuleWidget.setup(self)

        # Initialize logic
        self.logic = SegmentEditorAdaptiveBrushNurbsLogic()

        # Create UI sections
        self._create_progress_section()
        self._create_source_selection_section()
        self._create_structure_detection_section()
        self._create_fitting_parameters_section()
        self._create_quality_section()
        self._create_visualization_section()
        self._create_export_section()
        self._create_actions_section()

        # Add vertical spacer
        self.layout.addStretch(1)

        # Connect to scene events
        self._add_observers()

    def cleanup(self):
        """Clean up when module is closed."""
        self._remove_observers()
        self._clear_visualization()

    def _add_observers(self):
        """Add scene observers."""
        pass  # TODO: Add observers for segmentation changes

    def _remove_observers(self):
        """Remove scene observers."""
        pass  # TODO: Remove observers

    def _create_progress_section(self):
        """Create the progress feedback section."""
        progress_widget = qt.QWidget()
        layout = qt.QVBoxLayout(progress_widget)
        layout.setContentsMargins(0, 0, 0, 5)

        # Status label
        self._status_label = qt.QLabel("Ready")
        self._status_label.setStyleSheet("color: gray")
        layout.addWidget(self._status_label)

        # Progress bar
        self._progress_bar = qt.QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()  # Hidden by default
        layout.addWidget(self._progress_bar)

        self.layout.addWidget(progress_widget)

    def _show_progress(self, message: str, value: int = 0, max_value: int = 100):
        """Show progress bar with a status message.

        Args:
            message: Status message to display.
            value: Current progress value.
            max_value: Maximum progress value.
        """
        if self._status_label is not None:
            self._status_label.setText(message)
            self._status_label.setStyleSheet("color: #2196F3")  # Blue

        if self._progress_bar is not None:
            self._progress_bar.setMaximum(max_value)
            self._progress_bar.setValue(value)
            self._progress_bar.show()

        # Process events to update UI
        slicer.app.processEvents()

    def _update_progress(self, value: int, message: str | None = None):
        """Update progress bar value and optionally the message.

        Args:
            value: Current progress value.
            message: Optional new status message.
        """
        if self._progress_bar is not None:
            self._progress_bar.setValue(value)

        if message is not None and self._status_label is not None:
            self._status_label.setText(message)

        # Process events to update UI
        slicer.app.processEvents()

    def _hide_progress(self, message: str = "Ready"):
        """Hide progress bar and show completion message.

        Args:
            message: Final status message.
        """
        if self._progress_bar is not None:
            self._progress_bar.hide()
            self._progress_bar.setValue(0)

        if self._status_label is not None:
            self._status_label.setText(message)
            self._status_label.setStyleSheet("color: green")

        # Process events to update UI
        slicer.app.processEvents()

    def _show_error(self, message: str):
        """Show error status.

        Args:
            message: Error message to display.
        """
        if self._progress_bar is not None:
            self._progress_bar.hide()

        if self._status_label is not None:
            self._status_label.setText(message)
            self._status_label.setStyleSheet("color: red")

    def _create_source_selection_section(self):
        """Create the source segmentation selection section."""
        collapsible = qt.QGroupBox("Source")
        layout = qt.QFormLayout(collapsible)

        # Segmentation selector
        self.segmentationSelector = slicer.qMRMLNodeComboBox()
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.selectNodeUponCreation = True
        self.segmentationSelector.addEnabled = False
        self.segmentationSelector.removeEnabled = False
        self.segmentationSelector.noneEnabled = True
        self.segmentationSelector.showHidden = False
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationSelector.setToolTip(
            "Select segmentation containing the segment to convert"
        )
        self.segmentationSelector.currentNodeChanged.connect(self._on_segmentation_changed)
        layout.addRow("Segmentation:", self.segmentationSelector)

        # Segment selector
        self.segmentSelector = qt.QComboBox()
        self.segmentSelector.setToolTip("Select segment to convert to NURBS")
        self.segmentSelector.currentIndexChanged.connect(self._on_segment_changed)
        layout.addRow("Segment:", self.segmentSelector)

        self.layout.addWidget(collapsible)

    def _create_structure_detection_section(self):
        """Create the structure type detection section."""
        collapsible = qt.QGroupBox("Structure Type")
        layout = qt.QVBoxLayout(collapsible)

        # Auto-detect checkbox
        detect_row = qt.QHBoxLayout()
        self.autoDetectCheck = qt.QCheckBox("Auto-detect")
        self.autoDetectCheck.setChecked(True)
        self.autoDetectCheck.setToolTip("Automatically detect structure type from segment topology")
        self.autoDetectCheck.stateChanged.connect(self._on_auto_detect_changed)
        detect_row.addWidget(self.autoDetectCheck)

        self.detectButton = qt.QPushButton("Detect")
        self.detectButton.setToolTip("Run structure detection")
        self.detectButton.clicked.connect(self._on_detect_structure)
        detect_row.addWidget(self.detectButton)

        detect_row.addStretch()
        layout.addLayout(detect_row)

        # Structure type radio buttons
        type_row = qt.QHBoxLayout()

        self.structureTypeGroup = qt.QButtonGroup()
        self.simpleRadio = qt.QRadioButton("Simple")
        self.simpleRadio.setToolTip("Convex shape (tumors, nodules)")
        self.simpleRadio.setChecked(True)
        self.structureTypeGroup.addButton(self.simpleRadio, 0)
        type_row.addWidget(self.simpleRadio)

        self.tubularRadio = qt.QRadioButton("Tubular")
        self.tubularRadio.setToolTip("Single tube (vessel, airway)")
        self.structureTypeGroup.addButton(self.tubularRadio, 1)
        type_row.addWidget(self.tubularRadio)

        self.branchingRadio = qt.QRadioButton("Branching")
        self.branchingRadio.setToolTip("Branching tree (arterial, bronchial)")
        self.structureTypeGroup.addButton(self.branchingRadio, 2)
        type_row.addWidget(self.branchingRadio)

        type_row.addStretch()
        layout.addLayout(type_row)

        # Detection result label
        self.detectionResultLabel = qt.QLabel("Detection: Not run")
        self.detectionResultLabel.setStyleSheet("color: gray")
        layout.addWidget(self.detectionResultLabel)

        self.layout.addWidget(collapsible)

    def _create_fitting_parameters_section(self):
        """Create the NURBS fitting parameters section."""
        collapsible = qt.QGroupBox("Fitting Parameters")
        layout = qt.QFormLayout(collapsible)

        # Degree
        self.degreeSpinBox = qt.QSpinBox()
        self.degreeSpinBox.setMinimum(1)
        self.degreeSpinBox.setMaximum(5)
        self.degreeSpinBox.setValue(3)
        self.degreeSpinBox.setToolTip("NURBS polynomial degree (3 = cubic)")
        layout.addRow("Degree:", self.degreeSpinBox)

        # Control point resolution
        resolution_row = qt.QHBoxLayout()
        self.autoResolutionCheck = qt.QCheckBox("Auto")
        self.autoResolutionCheck.setChecked(True)
        self.autoResolutionCheck.setToolTip("Automatically determine control point density")
        self.autoResolutionCheck.stateChanged.connect(self._on_auto_resolution_changed)
        resolution_row.addWidget(self.autoResolutionCheck)

        self.resolutionSpinBox = qt.QSpinBox()
        self.resolutionSpinBox.setMinimum(2)
        self.resolutionSpinBox.setMaximum(20)
        self.resolutionSpinBox.setValue(4)
        self.resolutionSpinBox.setToolTip("Control points per direction (4x4x4 = 64 total)")
        self.resolutionSpinBox.setEnabled(False)
        resolution_row.addWidget(self.resolutionSpinBox)
        resolution_row.addStretch()
        layout.addRow("Resolution:", resolution_row)

        # Fitting tolerance
        self.toleranceSpinBox = qt.QDoubleSpinBox()
        self.toleranceSpinBox.setMinimum(0.1)
        self.toleranceSpinBox.setMaximum(10.0)
        self.toleranceSpinBox.setValue(0.5)
        self.toleranceSpinBox.setSuffix(" mm")
        self.toleranceSpinBox.setToolTip("Maximum allowed deviation from segment surface")
        layout.addRow("Tolerance:", self.toleranceSpinBox)

        self.layout.addWidget(collapsible)

    def _create_quality_section(self):
        """Create the quality metrics section."""
        collapsible = qt.QGroupBox("Quality")
        layout = qt.QFormLayout(collapsible)

        self.controlPointsLabel = qt.QLabel("-")
        layout.addRow("Control points:", self.controlPointsLabel)

        self.maxDeviationLabel = qt.QLabel("-")
        layout.addRow("Max deviation:", self.maxDeviationLabel)

        self.containmentLabel = qt.QLabel("-")
        layout.addRow("Containment:", self.containmentLabel)

        self.layout.addWidget(collapsible)

    def _create_visualization_section(self):
        """Create the visualization controls section."""
        collapsible = qt.QGroupBox("Visualization")
        layout = qt.QVBoxLayout(collapsible)

        # Checkboxes row
        check_row = qt.QHBoxLayout()

        self.showControlMeshCheck = qt.QCheckBox("Control mesh")
        self.showControlMeshCheck.setChecked(True)
        self.showControlMeshCheck.setToolTip("Show hexahedral control mesh")
        self.showControlMeshCheck.stateChanged.connect(self._on_visualization_changed)
        check_row.addWidget(self.showControlMeshCheck)

        self.showNurbsSurfaceCheck = qt.QCheckBox("NURBS surface")
        self.showNurbsSurfaceCheck.setChecked(True)
        self.showNurbsSurfaceCheck.setToolTip("Show evaluated NURBS surface")
        self.showNurbsSurfaceCheck.stateChanged.connect(self._on_visualization_changed)
        check_row.addWidget(self.showNurbsSurfaceCheck)

        self.showSegmentCheck = qt.QCheckBox("Segment overlay")
        self.showSegmentCheck.setChecked(False)
        self.showSegmentCheck.setToolTip("Show original segment for comparison")
        self.showSegmentCheck.stateChanged.connect(self._on_visualization_changed)
        check_row.addWidget(self.showSegmentCheck)

        check_row.addStretch()
        layout.addLayout(check_row)

        # Opacity slider
        opacity_row = qt.QHBoxLayout()
        opacity_row.addWidget(qt.QLabel("Opacity:"))
        self.opacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(80)
        self.opacitySlider.setToolTip("NURBS surface opacity")
        self.opacitySlider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self.opacitySlider)
        self.opacityLabel = qt.QLabel("80%")
        opacity_row.addWidget(self.opacityLabel)
        layout.addLayout(opacity_row)

        self.layout.addWidget(collapsible)

    def _create_export_section(self):
        """Create the export format selection section."""
        collapsible = qt.QGroupBox("Export")
        layout = qt.QVBoxLayout(collapsible)

        # Format checkboxes
        format_row = qt.QHBoxLayout()

        self.exportMfemCheck = qt.QCheckBox("MFEM mesh")
        self.exportMfemCheck.setChecked(True)
        self.exportMfemCheck.setToolTip("Export volumetric NURBS for IGA (MFEM format)")
        format_row.addWidget(self.exportMfemCheck)

        self.exportStlCheck = qt.QCheckBox("STL/OBJ")
        self.exportStlCheck.setChecked(False)
        self.exportStlCheck.setToolTip("Export triangulated surface mesh")
        format_row.addWidget(self.exportStlCheck)

        self.exportVtkCheck = qt.QCheckBox("VTK")
        self.exportVtkCheck.setChecked(False)
        self.exportVtkCheck.setToolTip("Export VTK unstructured grid")
        format_row.addWidget(self.exportVtkCheck)

        format_row.addStretch()
        layout.addLayout(format_row)

        # Output directory
        path_row = qt.QHBoxLayout()
        self.outputPathEdit = qt.QLineEdit()
        self.outputPathEdit.setPlaceholderText("Select output directory...")
        self.outputPathEdit.setToolTip("Directory for exported files")
        path_row.addWidget(self.outputPathEdit)

        self.browseButton = qt.QPushButton("Browse...")
        self.browseButton.clicked.connect(self._on_browse_output)
        path_row.addWidget(self.browseButton)

        layout.addLayout(path_row)

        self.layout.addWidget(collapsible)

    def _create_actions_section(self):
        """Create the main action buttons section."""
        collapsible = qt.QGroupBox("Actions")
        layout = qt.QHBoxLayout(collapsible)

        self.previewButton = qt.QPushButton("Preview")
        self.previewButton.setToolTip("Preview NURBS without full generation")
        self.previewButton.clicked.connect(self._on_preview)
        layout.addWidget(self.previewButton)

        self.generateButton = qt.QPushButton("Generate NURBS")
        self.generateButton.setToolTip("Generate volumetric NURBS from segment")
        self.generateButton.setStyleSheet("font-weight: bold")
        self.generateButton.clicked.connect(self._on_generate)
        layout.addWidget(self.generateButton)

        self.exportButton = qt.QPushButton("Export")
        self.exportButton.setToolTip("Export NURBS to selected formats")
        self.exportButton.setEnabled(False)
        self.exportButton.clicked.connect(self._on_export)
        layout.addWidget(self.exportButton)

        layout.addStretch()

        self.layout.addWidget(collapsible)

    # Event handlers

    def _on_segmentation_changed(self, node):
        """Handle segmentation selection change."""
        self._current_segmentation_node = node
        self._update_segment_list()
        self._clear_generated_nurbs()

    def _on_segment_changed(self, index: int):
        """Handle segment selection change."""
        if index < 0:
            self._current_segment_id = None
        else:
            self._current_segment_id = self.segmentSelector.itemData(index)
        self._clear_generated_nurbs()

    def _on_auto_detect_changed(self, state: int):
        """Handle auto-detect checkbox change."""
        enabled = state != qt.Qt.Checked
        self.simpleRadio.setEnabled(enabled)
        self.tubularRadio.setEnabled(enabled)
        self.branchingRadio.setEnabled(enabled)

    def _on_detect_structure(self):
        """Run structure type detection."""
        if not self._validate_input():
            return

        try:
            self._show_progress("Detecting structure type...", 0, 100)

            from SegmentEditorAdaptiveBrushNurbsLib import StructureDetector

            self._update_progress(30, "Analyzing topology...")

            detector = StructureDetector()
            structure_type = detector.detect(
                self._current_segmentation_node, self._current_segment_id
            )

            self._update_progress(100, "Detection complete")

            # Update radio buttons
            if structure_type == "simple":
                self.simpleRadio.setChecked(True)
            elif structure_type == "tubular":
                self.tubularRadio.setChecked(True)
            elif structure_type == "branching":
                self.branchingRadio.setChecked(True)

            self.detectionResultLabel.setText(f"Detection: {structure_type}")
            self.detectionResultLabel.setStyleSheet("color: green")
            logger.info(f"Structure detected as: {structure_type}")

            self._hide_progress(f"Detected: {structure_type}")

        except Exception as e:
            logger.exception("Structure detection failed")
            self.detectionResultLabel.setText("Detection: Failed")
            self.detectionResultLabel.setStyleSheet("color: red")
            self._show_error("Structure detection failed")
            slicer.util.errorDisplay(f"Structure detection failed:\n{e}")

    def _on_auto_resolution_changed(self, state: int):
        """Handle auto-resolution checkbox change."""
        self.resolutionSpinBox.setEnabled(state != qt.Qt.Checked)

    def _on_visualization_changed(self, state: int):
        """Handle visualization checkbox changes."""
        self._update_visualization()

    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change."""
        self.opacityLabel.setText(f"{value}%")
        self._update_visualization()

    def _on_browse_output(self):
        """Browse for output directory."""
        directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Output Directory")
        if directory:
            self.outputPathEdit.setText(directory)

    def _on_preview(self):
        """Generate quick preview of NURBS."""
        if not self._validate_input():
            return

        try:
            self._show_progress("Generating preview...", 0)
            self._generate_nurbs(preview=True)
            self._hide_progress("Preview complete")
        except Exception as e:
            logger.exception("Preview generation failed")
            self._show_error("Preview failed")
            slicer.util.errorDisplay(f"Preview failed:\n{e}")

    def _on_generate(self):
        """Generate full NURBS volume."""
        if not self._validate_input():
            return

        try:
            self._show_progress("Generating NURBS...", 0)
            self._generate_nurbs(preview=False)
            self.exportButton.setEnabled(True)
            self._hide_progress("NURBS generation complete")
        except Exception as e:
            logger.exception("NURBS generation failed")
            self._show_error("NURBS generation failed")
            slicer.util.errorDisplay(f"NURBS generation failed:\n{e}")

    def _on_export(self):
        """Export NURBS to selected formats."""
        if self._generated_nurbs is None:
            slicer.util.warningDisplay("No NURBS generated. Click 'Generate NURBS' first.")
            return

        output_dir = self.outputPathEdit.text.strip()
        if not output_dir:
            slicer.util.warningDisplay("Please select an output directory.")
            return

        try:
            self._show_progress("Exporting NURBS...", 0)
            self._export_nurbs(Path(output_dir))
            self._hide_progress("Export complete")
            slicer.util.infoDisplay(f"Export complete!\nFiles saved to: {output_dir}")
        except Exception as e:
            logger.exception("Export failed")
            self._show_error("Export failed")
            slicer.util.errorDisplay(f"Export failed:\n{e}")

    # Helper methods

    def _validate_input(self) -> bool:
        """Validate that a segmentation and segment are selected."""
        if self._current_segmentation_node is None:
            slicer.util.warningDisplay("Please select a segmentation.")
            return False

        if self._current_segment_id is None:
            slicer.util.warningDisplay("Please select a segment.")
            return False

        return True

    def _update_segment_list(self):
        """Update the segment selector with segments from current segmentation."""
        self.segmentSelector.clear()

        if self._current_segmentation_node is None:
            return

        segmentation = self._current_segmentation_node.GetSegmentation()
        if segmentation is None:
            return

        for i in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(segment_id)
            self.segmentSelector.addItem(segment.GetName(), segment_id)

    def _clear_generated_nurbs(self):
        """Clear any previously generated NURBS."""
        self._generated_nurbs = None
        self.exportButton.setEnabled(False)
        self._update_quality_display(None)
        self._clear_visualization()

    def _clear_visualization(self):
        """Clear visualization actors from 3D view."""
        if self._hex_mesh_actor is not None:
            # TODO: Remove from renderer
            self._hex_mesh_actor = None

    def _update_quality_display(self, quality_metrics: dict | None):
        """Update quality metrics display."""
        if quality_metrics is None:
            self.controlPointsLabel.setText("-")
            self.maxDeviationLabel.setText("-")
            self.containmentLabel.setText("-")
        else:
            self.controlPointsLabel.setText(str(quality_metrics.get("control_points", "-")))
            max_dev = quality_metrics.get("max_deviation")
            if max_dev is not None:
                self.maxDeviationLabel.setText(f"{max_dev:.2f} mm")
            else:
                self.maxDeviationLabel.setText("-")
            containment = quality_metrics.get("containment_percent")
            if containment is not None:
                self.containmentLabel.setText(f"{containment:.1f}%")
            else:
                self.containmentLabel.setText("-")

    def _update_visualization(self):
        """Update 3D visualization based on current settings."""
        # TODO: Implement visualization update
        pass

    def _generate_nurbs(self, preview: bool = False):
        """Generate NURBS from current segment.

        Args:
            preview: If True, use reduced resolution for quick preview.
        """
        from SegmentEditorAdaptiveBrushNurbsLib import (
            HexMeshGenerator,
            NurbsVolumeBuilder,
            StructureDetector,
        )

        from SegmentEditorAdaptiveBrushLib import dependency_manager  # type: ignore[attr-defined]

        # Ensure geomdl is available
        if not dependency_manager.ensure_available("geomdl"):
            raise RuntimeError("geomdl library is required for NURBS generation")

        # Get parameters
        degree = self.degreeSpinBox.value
        if self.autoResolutionCheck.isChecked():
            resolution = 4 if preview else None  # None = auto
        else:
            resolution = self.resolutionSpinBox.value

        tolerance = self.toleranceSpinBox.value

        # These are validated by _validate_input() before this method is called
        assert self._current_segmentation_node is not None
        assert self._current_segment_id is not None

        # Detect structure type if auto-detect is enabled
        self._update_progress(10, "Detecting structure type...")
        if self.autoDetectCheck.isChecked():
            detector = StructureDetector()
            structure_type = detector.detect(
                self._current_segmentation_node, self._current_segment_id
            )
        else:
            if self.simpleRadio.isChecked():
                structure_type = "simple"
            elif self.tubularRadio.isChecked():
                structure_type = "tubular"
            else:
                structure_type = "branching"

        logger.info(
            f"Generating NURBS: type={structure_type}, degree={degree}, "
            f"resolution={resolution}, tolerance={tolerance}"
        )

        # Generate hexahedral control mesh
        self._update_progress(20, f"Generating {structure_type} hex mesh...")
        hex_generator = HexMeshGenerator()

        # Build NURBS volume
        builder = NurbsVolumeBuilder()

        if structure_type == "simple":
            self._update_progress(30, "Creating bounding box mesh...")
            hex_mesh = hex_generator.generate_simple(
                self._current_segmentation_node,
                self._current_segment_id,
                resolution=resolution,
            )
            self._update_progress(50, "Building NURBS volume...")
            self._generated_nurbs = builder.build(hex_mesh, degree=degree)
            total_control_points = hex_mesh.num_control_points
        elif structure_type == "tubular":
            self._update_progress(30, "Extracting centerline...")
            hex_mesh = hex_generator.generate_tubular(
                self._current_segmentation_node,
                self._current_segment_id,
                resolution=resolution,
            )
            self._update_progress(50, "Building NURBS volume...")
            self._generated_nurbs = builder.build(hex_mesh, degree=degree)
            total_control_points = hex_mesh.num_control_points
        else:  # branching
            self._update_progress(30, "Extracting branch network...")
            hex_meshes = hex_generator.generate_branching(
                self._current_segmentation_node,
                self._current_segment_id,
                resolution=resolution,
            )
            self._update_progress(50, "Building multi-patch NURBS...")
            self._generated_nurbs = builder.build_multi_patch(hex_meshes, degree=degree)
            total_control_points = sum(hm.num_control_points for hm in hex_meshes)

        # Compute quality metrics
        self._update_progress(70, "Computing quality metrics...")

        # For multi-patch volumes, compute metrics for each patch and aggregate
        from SegmentEditorAdaptiveBrushNurbsLib.NurbsVolumeBuilder import (
            MultiPatchNurbsVolume,
            NurbsVolume,
        )

        if isinstance(self._generated_nurbs, MultiPatchNurbsVolume):
            # Aggregate metrics across all patches
            max_deviations = []
            containments = []
            for patch in self._generated_nurbs.patches:
                max_deviations.append(
                    builder.compute_max_deviation(
                        patch,
                        self._current_segmentation_node,
                        self._current_segment_id,
                    )
                )
                containments.append(
                    builder.compute_containment(
                        patch,
                        self._current_segmentation_node,
                        self._current_segment_id,
                    )
                )
            max_deviation = max(max_deviations) if max_deviations else 0.0
            containment_percent = min(containments) if containments else 100.0
        elif isinstance(self._generated_nurbs, NurbsVolume):
            # Single patch volume
            max_deviation = builder.compute_max_deviation(
                self._generated_nurbs,
                self._current_segmentation_node,
                self._current_segment_id,
            )
            containment_percent = builder.compute_containment(
                self._generated_nurbs,
                self._current_segmentation_node,
                self._current_segment_id,
            )
        else:
            # Unknown type, use defaults
            max_deviation = 0.0
            containment_percent = 100.0

        quality_metrics = {
            "control_points": total_control_points,
            "max_deviation": max_deviation,
            "containment_percent": containment_percent,
        }

        self._update_progress(90, "Updating display...")
        self._update_quality_display(quality_metrics)
        self._update_visualization()

        self._update_progress(100, "Complete")

        logger.info(
            f"NURBS generated: {quality_metrics['control_points']} control points, "
            f"max deviation {quality_metrics['max_deviation']:.2f} mm"
        )

    def _export_nurbs(self, output_dir: Path):
        """Export NURBS to selected formats.

        Args:
            output_dir: Directory to write output files.
        """
        from SegmentEditorAdaptiveBrushNurbsLib.Exporters import MeshExporter, MfemExporter
        from SegmentEditorAdaptiveBrushNurbsLib.NurbsVolumeBuilder import NurbsVolume

        output_dir.mkdir(parents=True, exist_ok=True)

        # Validated by caller
        assert self._current_segmentation_node is not None
        assert self._current_segment_id is not None
        assert self._generated_nurbs is not None

        # Get segment name for filename
        segmentation = self._current_segmentation_node.GetSegmentation()
        segment = segmentation.GetSegment(self._current_segment_id)
        base_name = segment.GetName().replace(" ", "_")

        # Cast to proper type for mypy
        nurbs_volume: NurbsVolume = self._generated_nurbs  # type: ignore[assignment]

        # Count export formats
        formats_to_export = sum(
            [
                self.exportMfemCheck.isChecked(),
                self.exportStlCheck.isChecked(),
                self.exportVtkCheck.isChecked(),
            ]
        )
        progress_per_format = 100 // max(formats_to_export, 1)
        current_progress = 0

        # Export MFEM
        if self.exportMfemCheck.isChecked():
            self._update_progress(current_progress, "Exporting MFEM mesh...")
            mfem_path = output_dir / f"{base_name}.mesh"
            mfem_exporter = MfemExporter()
            mfem_exporter.export(nurbs_volume, mfem_path)
            logger.info(f"Exported MFEM mesh: {mfem_path}")
            current_progress += progress_per_format

        # Export STL/OBJ
        if self.exportStlCheck.isChecked():
            self._update_progress(current_progress, "Exporting STL/OBJ...")
            stl_path = output_dir / f"{base_name}.stl"
            obj_path = output_dir / f"{base_name}.obj"
            mesh_exporter = MeshExporter()
            mesh_exporter.export_stl(nurbs_volume, stl_path)
            mesh_exporter.export_obj(nurbs_volume, obj_path)
            logger.info(f"Exported STL/OBJ: {stl_path}, {obj_path}")
            current_progress += progress_per_format

        # Export VTK
        if self.exportVtkCheck.isChecked():
            self._update_progress(current_progress, "Exporting VTK...")
            vtk_path = output_dir / f"{base_name}.vtk"
            vtk_exporter = MeshExporter()
            vtk_exporter.export_vtk(nurbs_volume, vtk_path)
            logger.info(f"Exported VTK: {vtk_path}")

        self._update_progress(100, "Export complete")


class SegmentEditorAdaptiveBrushNurbsLogic(ScriptedLoadableModuleLogic):
    """Logic class for NURBS Export module.

    Contains computational methods that can be called from other modules
    or from the widget.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def generate_nurbs_from_segment(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        structure_type: str = "auto",
        degree: int = 3,
        resolution: int | None = None,
    ) -> object:
        """Generate volumetric NURBS from a segment.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to convert.
            structure_type: "simple", "tubular", "branching", or "auto".
            degree: NURBS polynomial degree.
            resolution: Control points per direction (None for auto).

        Returns:
            Generated NURBS volume object.
        """
        from SegmentEditorAdaptiveBrushNurbsLib import (
            HexMeshGenerator,
            NurbsVolumeBuilder,
            StructureDetector,
        )

        from SegmentEditorAdaptiveBrushLib import dependency_manager  # type: ignore[attr-defined]

        # Ensure geomdl is available
        if not dependency_manager.ensure_available("geomdl"):
            raise RuntimeError("geomdl library is required for NURBS generation")

        # Detect structure type if auto
        if structure_type == "auto":
            detector = StructureDetector()
            structure_type = detector.detect(segmentation_node, segment_id)

        # Generate hexahedral control mesh and build NURBS volume
        hex_generator = HexMeshGenerator()
        builder = NurbsVolumeBuilder()

        if structure_type == "simple":
            hex_mesh = hex_generator.generate_simple(
                segmentation_node, segment_id, resolution=resolution
            )
            return builder.build(hex_mesh, degree=degree)
        elif structure_type == "tubular":
            hex_mesh = hex_generator.generate_tubular(
                segmentation_node, segment_id, resolution=resolution
            )
            return builder.build(hex_mesh, degree=degree)
        else:  # branching
            hex_meshes = hex_generator.generate_branching(
                segmentation_node, segment_id, resolution=resolution
            )
            return builder.build_multi_patch(hex_meshes, degree=degree)


class SegmentEditorAdaptiveBrushNurbsTest:
    """Test case for the NURBS Export module."""

    def setUp(self):
        """Set up test fixtures."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run the tests."""
        self.setUp()
        self.test_module_loads()
        self.test_simple_nurbs_generation()

    def test_module_loads(self):
        """Test that the module loads correctly."""
        logic = SegmentEditorAdaptiveBrushNurbsLogic()
        assert logic is not None

    def test_simple_nurbs_generation(self):
        """Test generating NURBS from a simple segment."""
        # TODO: Implement with test data
        pass
