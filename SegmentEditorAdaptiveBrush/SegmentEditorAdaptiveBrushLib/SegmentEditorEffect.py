"""Adaptive Brush Segment Editor Effect.

This effect provides an adaptive brush that automatically segments regions
based on image intensity similarity, adapting to image features (edges,
boundaries) rather than using a fixed geometric shape.
"""

import logging
import os
import sys
from typing import Dict

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.i18n import tr as _

# Add parent directory to path for imports when loaded by Slicer
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Import algorithm components (use non-relative imports for Slicer compatibility)
from IntensityAnalyzer import IntensityAnalyzer
from PerformanceCache import PerformanceCache

# Try to import SimpleITK (should be available in Slicer)
try:
    import SimpleITK as sitk
    import sitkUtils

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    logging.warning("SimpleITK not available - some features will be disabled")


class BrushOutlinePipeline:
    """VTK pipeline for brush outline visualization in a slice view.

    Shows a circle outline indicating the brush radius at the cursor position.
    The circle is created by cutting a sphere with the slice plane.
    """

    def __init__(self):
        """Initialize the brush outline pipeline."""
        # Create a circle using vtkRegularPolygonSource (simpler than sphere + cutter)
        self.circleSource = vtk.vtkRegularPolygonSource()
        self.circleSource.SetNumberOfSides(64)  # Smooth circle
        self.circleSource.SetRadius(1.0)  # Will be updated
        self.circleSource.GeneratePolygonOff()  # Just the outline
        self.circleSource.GeneratePolylineOn()

        # Transform to position in slice XY coordinates
        self.transform = vtk.vtkTransform()
        self.transformFilter = vtk.vtkTransformPolyDataFilter()
        self.transformFilter.SetTransform(self.transform)
        self.transformFilter.SetInputConnection(self.circleSource.GetOutputPort())

        # 2D mapper
        self.mapper = vtk.vtkPolyDataMapper2D()
        self.mapper.SetInputConnection(self.transformFilter.GetOutputPort())

        # 2D actor for the outline
        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(self.mapper)
        self.actor.VisibilityOff()
        self.actor.SetPickable(False)

        # Styling - yellow outline
        prop = self.actor.GetProperty()
        prop.SetColor(1.0, 0.9, 0.1)  # Yellow
        prop.SetLineWidth(2)
        prop.SetOpacity(0.8)

        # Store the renderer
        self.renderer = None
        self.sliceWidget = None

    def setSliceWidget(self, sliceWidget):
        """Attach the pipeline to a slice widget's renderer.

        Args:
            sliceWidget: The qMRMLSliceWidget to attach to.
        """
        if self.renderer is not None:
            self.renderer.RemoveActor2D(self.actor)

        self.sliceWidget = sliceWidget
        if sliceWidget is not None:
            self.renderer = sliceWidget.sliceView().renderWindow().GetRenderers().GetFirstRenderer()
            if self.renderer is not None:
                self.renderer.AddActor2D(self.actor)

    def updateOutline(self, xyPosition, radiusPixels):
        """Update the brush outline position and size.

        Args:
            xyPosition: Center position in slice XY coordinates (x, y).
            radiusPixels: Brush radius in pixels.
        """
        if self.renderer is None:
            return

        # Update circle radius
        self.circleSource.SetRadius(radiusPixels)

        # Position the circle at the cursor
        self.transform.Identity()
        self.transform.Translate(xyPosition[0], xyPosition[1], 0)

        self.actor.VisibilityOn()

        # Request render
        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def hide(self):
        """Hide the brush outline."""
        self.actor.VisibilityOff()
        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def cleanup(self):
        """Remove the actor from the renderer and clean up."""
        if self.renderer is not None:
            self.renderer.RemoveActor2D(self.actor)
            self.renderer = None
        self.sliceWidget = None


class SegmentEditorEffect:
    """Adaptive brush segment editor effect.

    This class implements the AbstractScriptedSegmentEditorEffect interface
    to provide an adaptive brush that segments based on image intensity.
    """

    def __init__(self, scriptedEffect):
        """Initialize the effect.

        Args:
            scriptedEffect: The scripted effect instance from Slicer.
        """
        scriptedEffect.name = "Adaptive Brush"
        scriptedEffect.perSegment = True
        self.scriptedEffect = scriptedEffect

        # Algorithm components
        self.intensityAnalyzer = IntensityAnalyzer()
        self.cache = PerformanceCache()

        # State
        self.isDrawing = False
        self.lastIjk = None

        # Default parameters
        self.radiusMm = 5.0
        self.edgeSensitivity = 50
        self.algorithm = "watershed"
        self.backend = "auto"
        self.sphereMode = False
        self.useThresholdCaching = False  # Disabled by default for accuracy

        # Brush outline visualization - one pipeline per slice view
        self.outlinePipelines: Dict[str, BrushOutlinePipeline] = {}
        self.activeViewWidget = None

    def clone(self):
        """Create a copy of this effect.

        Returns:
            New effect instance.
        """
        import qSlicerSegmentationsEditorEffectsPythonQt as effects

        clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
        clonedEffect.setPythonSource(__file__)
        return clonedEffect

    def icon(self):
        """Return the effect icon.

        Returns:
            QIcon for the effect toolbar button.
        """
        iconPath = os.path.join(
            os.path.dirname(__file__), "..", "Resources", "Icons", "SegmentEditorAdaptiveBrush.png"
        )
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def createCursor(self, widget):
        """Create a custom cursor for the effect.

        Returns a cursor with a small crosshair, similar to other segment editor effects.
        The base class implementation composites the effect icon with an arrow cursor.

        Args:
            widget: The view widget.

        Returns:
            QCursor for this effect.
        """
        # Use the default cursor creation from the base class
        # This composites a small effect icon with an arrow cursor
        return self.scriptedEffect.createCursor(widget)

    def helpText(self):
        """Return help text for the effect.

        Returns:
            HTML string with usage instructions.
        """
        return _(
            """<html>
<p><b>Adaptive Brush</b>: Paint with a brush that adapts to image intensity boundaries.</p>
<p>The brush automatically segments the region containing the cursor based on
intensity similarity, stopping at edges and boundaries.</p>
<h4>Usage:</h4>
<ul>
<li><b>Left-click and drag</b>: Paint adaptive regions</li>
<li><b>Radius</b>: Controls the maximum brush size</li>
<li><b>Edge Sensitivity</b>: How strictly to follow intensity boundaries</li>
<li><b>Algorithm</b>: Choose segmentation method (Watershed, Level Set, etc.)</li>
</ul>
<h4>Tips:</h4>
<ul>
<li>Start with Watershed algorithm for general use</li>
<li>Use Level Set (GPU) for faster processing if you have a GPU</li>
<li>Increase edge sensitivity for stricter boundary following</li>
<li>Enable 3D mode for volumetric painting</li>
</ul>
</html>"""
        )

    def setupOptionsFrame(self):
        """Create the effect options UI."""
        # ----- Brush Settings -----
        brushCollapsible = ctk.ctkCollapsibleButton()
        brushCollapsible.text = _("Brush Settings")
        self.scriptedEffect.addOptionsWidget(brushCollapsible)
        brushLayout = qt.QFormLayout(brushCollapsible)

        # Radius slider
        self.radiusSlider = ctk.ctkSliderWidget()
        self.radiusSlider.setToolTip(_("Brush radius in millimeters"))
        self.radiusSlider.minimum = 1
        self.radiusSlider.maximum = 100
        self.radiusSlider.value = self.radiusMm
        self.radiusSlider.singleStep = 0.5
        self.radiusSlider.decimals = 1
        self.radiusSlider.suffix = " mm"
        brushLayout.addRow(_("Radius:"), self.radiusSlider)

        # Edge sensitivity slider
        self.sensitivitySlider = ctk.ctkSliderWidget()
        self.sensitivitySlider.setToolTip(
            _("How strictly to follow intensity boundaries (0=permissive, 100=strict)")
        )
        self.sensitivitySlider.minimum = 0
        self.sensitivitySlider.maximum = 100
        self.sensitivitySlider.value = self.edgeSensitivity
        self.sensitivitySlider.singleStep = 5
        self.sensitivitySlider.suffix = "%"
        brushLayout.addRow(_("Edge Sensitivity:"), self.sensitivitySlider)

        # 3D sphere mode checkbox
        self.sphereModeCheckbox = qt.QCheckBox(_("3D Sphere Mode"))
        self.sphereModeCheckbox.setToolTip(_("Paint in 3D volume instead of 2D slice"))
        self.sphereModeCheckbox.checked = self.sphereMode
        brushLayout.addRow(self.sphereModeCheckbox)

        # ----- Algorithm Settings -----
        algorithmCollapsible = ctk.ctkCollapsibleButton()
        algorithmCollapsible.text = _("Algorithm")
        algorithmCollapsible.collapsed = True
        self.scriptedEffect.addOptionsWidget(algorithmCollapsible)
        algorithmLayout = qt.QFormLayout(algorithmCollapsible)

        # Algorithm dropdown
        self.algorithmCombo = qt.QComboBox()
        self.algorithmCombo.addItem(_("Watershed (Recommended)"), "watershed")
        self.algorithmCombo.addItem(_("Level Set (GPU)"), "level_set_gpu")
        self.algorithmCombo.addItem(_("Level Set (CPU)"), "level_set_cpu")
        self.algorithmCombo.addItem(_("Connected Threshold (Fast)"), "connected_threshold")
        self.algorithmCombo.addItem(_("Region Growing"), "region_growing")
        self.algorithmCombo.addItem(_("Threshold Brush (Simple)"), "threshold_brush")
        self.algorithmCombo.setToolTip(_("Segmentation algorithm to use"))
        algorithmLayout.addRow(_("Algorithm:"), self.algorithmCombo)

        # Backend dropdown
        self.backendCombo = qt.QComboBox()
        self.backendCombo.addItem(_("Auto"), "auto")
        self.backendCombo.addItem(_("CPU"), "cpu")
        self.backendCombo.addItem(_("GPU (OpenCL)"), "gpu_opencl")
        self.backendCombo.addItem(_("GPU (CUDA)"), "gpu_cuda")
        self.backendCombo.setToolTip(_("Computation backend"))
        algorithmLayout.addRow(_("Backend:"), self.backendCombo)

        # Threshold caching checkbox
        self.cachingCheckbox = qt.QCheckBox(_("Enable threshold caching"))
        self.cachingCheckbox.setToolTip(
            _(
                "Reuse threshold calculations when painting in similar intensity regions. "
                "Improves drag performance but may reduce accuracy at region boundaries."
            )
        )
        self.cachingCheckbox.checked = self.useThresholdCaching
        algorithmLayout.addRow(self.cachingCheckbox)

        # ----- Threshold Brush Settings -----
        self.thresholdGroup = qt.QWidget()
        thresholdLayout = qt.QFormLayout(self.thresholdGroup)
        thresholdLayout.setContentsMargins(0, 0, 0, 0)

        # Auto threshold checkbox
        self.autoThresholdCheckbox = qt.QCheckBox(_("Auto threshold"))
        self.autoThresholdCheckbox.setToolTip(
            _("Automatically compute thresholds using selected method")
        )
        self.autoThresholdCheckbox.checked = True
        thresholdLayout.addRow(self.autoThresholdCheckbox)

        # Threshold method dropdown (visible when auto is checked)
        self.thresholdMethodCombo = qt.QComboBox()
        self.thresholdMethodCombo.addItem(_("Otsu"), "otsu")
        self.thresholdMethodCombo.addItem(_("Huang"), "huang")
        self.thresholdMethodCombo.addItem(_("Triangle"), "triangle")
        self.thresholdMethodCombo.addItem(_("Maximum Entropy"), "max_entropy")
        self.thresholdMethodCombo.addItem(_("IsoData (Intermeans)"), "isodata")
        self.thresholdMethodCombo.addItem(_("Li"), "li")
        self.thresholdMethodCombo.setToolTip(_("Automatic threshold computation method"))
        thresholdLayout.addRow(_("Method:"), self.thresholdMethodCombo)

        # Manual threshold group (visible when auto is unchecked)
        self.manualThresholdGroup = qt.QWidget()
        manualLayout = qt.QFormLayout(self.manualThresholdGroup)
        manualLayout.setContentsMargins(0, 0, 0, 0)

        self.lowerThresholdSlider = ctk.ctkSliderWidget()
        self.lowerThresholdSlider.setToolTip(_("Lower intensity threshold"))
        self.lowerThresholdSlider.minimum = -2000
        self.lowerThresholdSlider.maximum = 5000
        self.lowerThresholdSlider.value = -100
        self.lowerThresholdSlider.singleStep = 10
        self.lowerThresholdSlider.decimals = 0
        manualLayout.addRow(_("Lower:"), self.lowerThresholdSlider)

        self.upperThresholdSlider = ctk.ctkSliderWidget()
        self.upperThresholdSlider.setToolTip(_("Upper intensity threshold"))
        self.upperThresholdSlider.minimum = -2000
        self.upperThresholdSlider.maximum = 5000
        self.upperThresholdSlider.value = 300
        self.upperThresholdSlider.singleStep = 10
        self.upperThresholdSlider.decimals = 0
        manualLayout.addRow(_("Upper:"), self.upperThresholdSlider)

        # Set from seed button
        self.setFromSeedButton = qt.QPushButton(_("Set from seed intensity"))
        self.setFromSeedButton.setToolTip(
            _("Set thresholds based on intensity at last click location")
        )
        manualLayout.addRow(self.setFromSeedButton)

        # Tolerance slider (for set from seed)
        self.toleranceSlider = ctk.ctkSliderWidget()
        self.toleranceSlider.setToolTip(_("Tolerance around seed intensity (% of local std dev)"))
        self.toleranceSlider.minimum = 1
        self.toleranceSlider.maximum = 100
        self.toleranceSlider.value = 20
        self.toleranceSlider.singleStep = 5
        self.toleranceSlider.decimals = 0
        self.toleranceSlider.suffix = "%"
        manualLayout.addRow(_("Tolerance:"), self.toleranceSlider)

        thresholdLayout.addRow(self.manualThresholdGroup)
        self.manualThresholdGroup.setVisible(False)  # Hidden when auto is checked

        algorithmLayout.addRow(self.thresholdGroup)
        self.thresholdGroup.setVisible(False)  # Hidden unless Threshold Brush selected

        # Connect signals
        self.radiusSlider.valueChanged.connect(self.onRadiusChanged)
        self.sensitivitySlider.valueChanged.connect(self.onSensitivityChanged)
        self.sphereModeCheckbox.toggled.connect(self.onSphereModeChanged)
        self.algorithmCombo.currentIndexChanged.connect(self.onAlgorithmChanged)
        self.backendCombo.currentIndexChanged.connect(self.onBackendChanged)
        self.cachingCheckbox.toggled.connect(self.onCachingChanged)
        self.lowerThresholdSlider.valueChanged.connect(self.onThresholdChanged)
        self.upperThresholdSlider.valueChanged.connect(self.onThresholdChanged)
        self.autoThresholdCheckbox.toggled.connect(self.onAutoThresholdChanged)
        self.thresholdMethodCombo.currentIndexChanged.connect(self.onThresholdMethodChanged)
        self.setFromSeedButton.clicked.connect(self.onSetFromSeedClicked)
        self.toleranceSlider.valueChanged.connect(self.onThresholdChanged)

    def onRadiusChanged(self, value):
        """Handle radius slider change."""
        self.radiusMm = value
        self.cache.invalidate()

    def onSensitivityChanged(self, value):
        """Handle edge sensitivity change."""
        self.edgeSensitivity = value
        self.cache.invalidate()

    def onSphereModeChanged(self, checked):
        """Handle 3D mode toggle."""
        self.sphereMode = checked
        self.cache.invalidate()

    def onAlgorithmChanged(self, index):
        """Handle algorithm selection change."""
        self.algorithm = self.algorithmCombo.currentData
        self.cache.invalidate()
        # Show/hide threshold controls based on algorithm
        isThresholdBrush = self.algorithm == "threshold_brush"
        self.thresholdGroup.setVisible(isThresholdBrush)

    def onThresholdChanged(self, value):
        """Handle manual threshold slider change."""
        self.cache.invalidate()

    def onAutoThresholdChanged(self, checked):
        """Toggle between auto and manual threshold modes."""
        self.thresholdMethodCombo.setVisible(checked)
        self.manualThresholdGroup.setVisible(not checked)
        self.cache.invalidate()

    def onThresholdMethodChanged(self, index):
        """Handle threshold method change."""
        self.cache.invalidate()

    def onSetFromSeedClicked(self):
        """Handle set from seed button click."""
        # Get source volume
        sourceVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        if sourceVolumeNode is None:
            logging.warning("No source volume selected")
            return

        # Use last clicked IJK position
        if self.lastIjk is None:
            logging.info("Click on the image first to set seed position")
            return

        volumeArray = slicer.util.arrayFromVolume(sourceVolumeNode)
        self._setThresholdsFromSeed(volumeArray, self.lastIjk, self.toleranceSlider.value)

    def onCachingChanged(self, checked):
        """Handle threshold caching toggle."""
        self.useThresholdCaching = checked
        self.cache.threshold_caching_enabled = checked
        if not checked:
            # Clear threshold cache when disabling
            self.cache.threshold_cache = None
            self.cache.threshold_seed_intensity = None

    def onBackendChanged(self, index):
        """Handle backend selection change."""
        self.backend = self.backendCombo.currentData
        self.cache.invalidate()

    def activate(self):
        """Called when the effect is selected."""
        self.cache.clear()
        self._createOutlinePipelines()

    def deactivate(self):
        """Called when the effect is deselected."""
        self.cache.clear()
        self.isDrawing = False
        self._cleanupOutlinePipelines()

    def _createOutlinePipelines(self):
        """Create brush outline pipelines for all slice views."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return

        for sliceViewName in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(sliceViewName)
            if sliceWidget is not None:
                pipeline = BrushOutlinePipeline()
                pipeline.setSliceWidget(sliceWidget)
                self.outlinePipelines[sliceViewName] = pipeline

    def _cleanupOutlinePipelines(self):
        """Clean up and remove all outline pipelines."""
        for pipeline in self.outlinePipelines.values():
            pipeline.cleanup()
        self.outlinePipelines.clear()
        self.activeViewWidget = None

    def _updateBrushPreview(self, xy, viewWidget):
        """Update the brush outline at the cursor position.

        Shows a circle indicating the brush radius.

        Args:
            xy: Screen coordinates (x, y).
            viewWidget: The slice view widget.
        """
        if viewWidget is None:
            return

        sliceLogic = viewWidget.sliceLogic()
        if sliceLogic is None:
            return

        sliceNode = sliceLogic.GetSliceNode()
        if sliceNode is None:
            return

        viewName = sliceNode.GetName()

        # Calculate radius in pixels
        # Get the spacing in the slice view to convert mm to pixels
        spacing = self._getSliceSpacing(viewWidget)
        radiusPixels = self.radiusMm / spacing if spacing > 0 else 50

        # Update the pipeline for this view
        if viewName in self.outlinePipelines:
            self.outlinePipelines[viewName].updateOutline(xy, radiusPixels)

        # Hide outlines in other views
        for name, pipeline in self.outlinePipelines.items():
            if name != viewName:
                pipeline.hide()

    def _getSliceSpacing(self, viewWidget):
        """Get the pixel spacing in mm for the slice view.

        Args:
            viewWidget: The slice view widget.

        Returns:
            Average spacing in mm per pixel.
        """
        sliceNode = viewWidget.sliceLogic().GetSliceNode()
        # Get field of view and dimensions
        fov = sliceNode.GetFieldOfView()
        dims = sliceNode.GetDimensions()

        if dims[0] > 0 and dims[1] > 0:
            # Average of X and Y spacing
            spacingX = fov[0] / dims[0]
            spacingY = fov[1] / dims[1]
            return (spacingX + spacingY) / 2
        return 1.0

    def _hideBrushPreview(self):
        """Hide brush outline in all views."""
        for pipeline in self.outlinePipelines.values():
            pipeline.hide()

    def processInteractionEvents(self, callerInteractor, eventId, viewWidget):
        """Handle mouse interaction events.

        Args:
            callerInteractor: VTK interactor that triggered the event.
            eventId: VTK event ID.
            viewWidget: The view widget where the event occurred.

        Returns:
            True if the event was handled, False otherwise.
        """
        if viewWidget.className() != "qMRMLSliceWidget":
            return False

        xy = callerInteractor.GetEventPosition()

        if eventId == vtk.vtkCommand.LeftButtonPressEvent:
            # Save undo state at the START of the stroke (once per stroke)
            self.scriptedEffect.saveStateForUndo()
            self.isDrawing = True
            self._updateBrushPreview(xy, viewWidget)
            self.processPoint(xy, viewWidget)
            return True

        elif eventId == vtk.vtkCommand.MouseMoveEvent:
            # Update preview on mouse move
            self._updateBrushPreview(xy, viewWidget)
            if self.isDrawing:
                self.processPoint(xy, viewWidget)
                return True  # Only consume when drawing
            return False  # Don't consume when just hovering

        elif eventId == vtk.vtkCommand.LeftButtonReleaseEvent:
            self.isDrawing = False
            self.lastIjk = None
            self.cache.onMouseRelease()
            return True

        elif eventId == vtk.vtkCommand.LeaveEvent:
            # Hide preview when mouse leaves the view
            self._hideBrushPreview()
            return False

        elif eventId == vtk.vtkCommand.EnterEvent:
            # Update preview when mouse enters
            self._updateBrushPreview(xy, viewWidget)
            return False

        return False

    def processPoint(self, xy, viewWidget):
        """Process a single point interaction.

        Args:
            xy: Screen coordinates (x, y).
            viewWidget: The slice view widget.
        """
        ijk = self.xyToIjk(xy, viewWidget)
        if ijk is None:
            return

        # Skip if same voxel as last time (optimization)
        if self.lastIjk is not None and ijk == self.lastIjk:
            return
        self.lastIjk = ijk

        # Get source volume
        sourceVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        if sourceVolumeNode is None:
            logging.warning("No source volume selected")
            return

        # Compute adaptive mask
        try:
            mask = self.computeAdaptiveMask(sourceVolumeNode, ijk, viewWidget)
            if mask is not None:
                self.applyMaskToSegment(mask)
        except Exception as e:
            logging.exception(f"Error computing adaptive mask: {e}")

    def xyToIjk(self, xy, viewWidget):
        """Convert screen XY coordinates to volume IJK.

        Args:
            xy: Screen coordinates (x, y).
            viewWidget: The slice view widget.

        Returns:
            Tuple (i, j, k) in volume coordinates, or None if invalid.
        """
        sliceLogic = viewWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        sourceVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()

        if sourceVolumeNode is None:
            return None

        # XY to RAS
        xyToRas = sliceNode.GetXYToRAS()
        rasPoint = [0, 0, 0, 1]
        xyPoint = [xy[0], xy[1], 0, 1]
        xyToRas.MultiplyPoint(xyPoint, rasPoint)

        # RAS to IJK
        rasToIjk = vtk.vtkMatrix4x4()
        sourceVolumeNode.GetRASToIJKMatrix(rasToIjk)
        ijkPoint = [0, 0, 0, 1]
        rasToIjk.MultiplyPoint(rasPoint, ijkPoint)

        # Get volume dimensions for bounds checking
        dims = sourceVolumeNode.GetImageData().GetDimensions()
        ijk = (
            int(round(ijkPoint[0])),
            int(round(ijkPoint[1])),
            int(round(ijkPoint[2])),
        )

        # Bounds check
        if not (0 <= ijk[0] < dims[0] and 0 <= ijk[1] < dims[1] and 0 <= ijk[2] < dims[2]):
            return None

        return ijk

    def computeAdaptiveMask(self, sourceVolumeNode, seedIjk, viewWidget):
        """Compute the adaptive segmentation mask.

        Args:
            sourceVolumeNode: The source volume MRML node.
            seedIjk: Seed point in IJK coordinates.
            viewWidget: The slice view widget.

        Returns:
            numpy array with binary mask, or None on failure.
        """
        if not HAS_SIMPLEITK:
            logging.error("SimpleITK required for adaptive brush")
            return None

        # Get volume as numpy array
        volumeArray = slicer.util.arrayFromVolume(sourceVolumeNode)
        spacing = sourceVolumeNode.GetSpacing()

        # Convert radius from mm to voxels
        radiusVoxels = [self.radiusMm / s for s in spacing]

        # Get slice index for 2D mode
        if not self.sphereMode:
            sliceIndex = seedIjk[2]
        else:
            sliceIndex = None

        # Build parameters
        params = {
            "radius_mm": self.radiusMm,
            "radius_voxels": radiusVoxels,
            "edge_sensitivity": self.edgeSensitivity / 100.0,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "sphere_mode": self.sphereMode,
            "slice_index": sliceIndex,
            "manual_lower": self.lowerThresholdSlider.value,
            "manual_upper": self.upperThresholdSlider.value,
            "auto_threshold": self.autoThresholdCheckbox.checked,
            "threshold_method": self.thresholdMethodCombo.currentData,
        }

        # Use cache for drag operations
        mask = self.cache.computeOrGetCached(
            volumeArray,
            seedIjk,
            params,
            self.intensityAnalyzer,
            self._runSegmentation,
        )

        return mask

    def _runSegmentation(self, volumeArray, seedIjk, params, thresholds):
        """Run the actual segmentation algorithm.

        Args:
            volumeArray: numpy array of the volume (z, y, x ordering).
            seedIjk: Seed point (i, j, k).
            params: Segmentation parameters dict.
            thresholds: Intensity thresholds from analyzer.

        Returns:
            Binary mask numpy array.
        """
        # Extract ROI around seed
        roi, roiStart = self._extractROI(volumeArray, seedIjk, params["radius_voxels"])

        # Local seed coordinates
        localSeed = tuple(seedIjk[i] - roiStart[i] for i in range(3))

        # Select algorithm
        algorithm = params["algorithm"]

        if algorithm == "connected_threshold":
            mask = self._connectedThreshold(roi, localSeed, thresholds)
        elif algorithm in ("level_set_gpu", "level_set_cpu"):
            mask = self._levelSet(roi, localSeed, thresholds, params)
        elif algorithm == "region_growing":
            mask = self._regionGrowing(roi, localSeed, thresholds, params)
        elif algorithm == "threshold_brush":
            mask = self._thresholdBrush(roi, localSeed, params)
        else:  # Default: watershed
            mask = self._watershed(roi, localSeed, thresholds, params)

        # Apply brush radius mask
        mask = self._applyBrushMask(mask, localSeed, params["radius_voxels"])

        # Expand back to full volume size
        fullMask = np.zeros_like(volumeArray, dtype=np.uint8)
        self._pasteROI(fullMask, mask, roiStart)

        return fullMask

    def _extractROI(self, volumeArray, seedIjk, radiusVoxels):
        """Extract region of interest around seed point.

        Args:
            volumeArray: Full volume array.
            seedIjk: Seed point (i, j, k).
            radiusVoxels: Radius in voxels per dimension.

        Returns:
            Tuple of (roi array, start indices).
        """
        shape = volumeArray.shape  # (z, y, x)
        margin = 1.2  # Extra margin for edge computation

        # Calculate bounds (note: numpy is z,y,x but seedIjk is i,j,k = x,y,z)
        startZ = max(0, int(seedIjk[2] - radiusVoxels[2] * margin))
        startY = max(0, int(seedIjk[1] - radiusVoxels[1] * margin))
        startX = max(0, int(seedIjk[0] - radiusVoxels[0] * margin))

        endZ = min(shape[0], int(seedIjk[2] + radiusVoxels[2] * margin) + 1)
        endY = min(shape[1], int(seedIjk[1] + radiusVoxels[1] * margin) + 1)
        endX = min(shape[2], int(seedIjk[0] + radiusVoxels[0] * margin) + 1)

        roi = volumeArray[startZ:endZ, startY:endY, startX:endX].copy()
        roiStart = (startX, startY, startZ)

        return roi, roiStart

    def _pasteROI(self, fullArray, roi, roiStart):
        """Paste ROI back into full array.

        Args:
            fullArray: Full volume array to modify.
            roi: ROI array to paste.
            roiStart: Start indices (x, y, z).
        """
        startX, startY, startZ = roiStart
        endZ = startZ + roi.shape[0]
        endY = startY + roi.shape[1]
        endX = startX + roi.shape[2]

        fullArray[startZ:endZ, startY:endY, startX:endX] |= roi

    def _connectedThreshold(self, roi, localSeed, thresholds):
        """Connected threshold segmentation (fast).

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates.
            thresholds: Dict with 'lower' and 'upper' thresholds.

        Returns:
            Binary mask array.
        """
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))

        # Note: SimpleITK uses (x, y, z) but our localSeed is (i, j, k) = (x, y, z)
        # and roi is (z, y, x) so we need to convert
        sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))

        try:
            result = sitk.ConnectedThreshold(
                sitkRoi,
                seedList=[sitkSeed],
                lower=float(thresholds["lower"]),
                upper=float(thresholds["upper"]),
                replaceValue=1,
            )
            return sitk.GetArrayFromImage(result).astype(np.uint8)
        except Exception as e:
            logging.error(f"Connected threshold failed: {e}")
            return np.zeros_like(roi, dtype=np.uint8)

    def _watershed(self, roi, localSeed, thresholds, params):
        """Watershed segmentation with marker refinement.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates.
            thresholds: Intensity thresholds.
            params: Algorithm parameters.

        Returns:
            Binary mask array.
        """
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))

        # First get initial region with connected threshold
        initialMask = self._connectedThreshold(roi, localSeed, thresholds)

        if np.sum(initialMask) == 0:
            return initialMask

        # Compute gradient magnitude for watershed
        gradient = sitk.GradientMagnitude(sitkRoi)

        # Scale gradient by edge sensitivity
        gradArray = sitk.GetArrayFromImage(gradient)
        sensitivity = params.get("edge_sensitivity", 0.5)
        gradMax = np.percentile(gradArray, 95)
        if gradMax > 0:
            gradArray = (gradArray / gradMax * 255 * sensitivity).astype(np.float32)

        gradient = sitk.GetImageFromArray(gradArray)

        # Create markers from initial mask
        # Foreground marker: eroded initial mask
        initialSitk = sitk.GetImageFromArray(initialMask)
        foreground = sitk.BinaryErode(initialSitk, [2, 2, 2])

        # Background marker: dilated then subtracted
        dilated = sitk.BinaryDilate(initialSitk, [3, 3, 3])
        background = sitk.Subtract(dilated, sitk.BinaryDilate(initialSitk, [1, 1, 1]))

        # Combine markers (foreground=1, background=2)
        markers = sitk.Add(foreground, sitk.Multiply(background, 2))
        markers = sitk.Cast(markers, sitk.sitkUInt8)

        # Run watershed
        try:
            watershed = sitk.MorphologicalWatershedFromMarkers(
                gradient, markers, markWatershedLine=False, fullyConnected=True
            )
            # Extract foreground (label 1)
            result = sitk.BinaryThreshold(watershed, 1, 1, 1, 0)
            return sitk.GetArrayFromImage(result).astype(np.uint8)
        except Exception as e:
            logging.error(f"Watershed failed: {e}")
            return initialMask

    def _levelSet(self, roi, localSeed, thresholds, params):
        """Level set segmentation.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates.
            thresholds: Intensity thresholds.
            params: Algorithm parameters.

        Returns:
            Binary mask array.
        """
        # Get initial segmentation
        initialMask = self._connectedThreshold(roi, localSeed, thresholds)

        if np.sum(initialMask) == 0:
            return initialMask

        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))
        initialSitk = sitk.GetImageFromArray(initialMask)

        # Create signed distance map for level set initialization
        distanceMap = sitk.SignedMaurerDistanceMap(
            initialSitk, insideIsPositive=True, squaredDistance=False, useImageSpacing=False
        )

        # Compute speed image from gradient (high speed where gradient is low)
        gradient = sitk.GradientMagnitude(sitkRoi)
        gradArray = sitk.GetArrayFromImage(gradient)
        gradMax = np.max(gradArray) + 1e-8
        speedArray = 1.0 - (gradArray / gradMax)
        speed = sitk.GetImageFromArray(speedArray.astype(np.float32))

        # Run geodesic active contour
        try:
            gac = sitk.GeodesicActiveContourLevelSetImageFilter()
            gac.SetPropagationScaling(1.0)
            gac.SetCurvatureScaling(0.5)
            gac.SetAdvectionScaling(1.0)
            gac.SetMaximumRMSError(0.01)
            gac.SetNumberOfIterations(100)

            levelSet = gac.Execute(
                sitk.Cast(distanceMap, sitk.sitkFloat32), sitk.Cast(speed, sitk.sitkFloat32)
            )

            # Threshold level set to get binary mask
            result = sitk.BinaryThreshold(levelSet, lowerThreshold=0)
            return sitk.GetArrayFromImage(result).astype(np.uint8)
        except Exception as e:
            logging.error(f"Level set failed: {e}")
            return initialMask

    def _regionGrowing(self, roi, localSeed, thresholds, params):
        """Region growing segmentation.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates.
            thresholds: Intensity thresholds.
            params: Algorithm parameters.

        Returns:
            Binary mask array.
        """
        # Use confidence connected for region growing behavior
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))
        sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))

        # Scale multiplier with edge sensitivity
        # sensitivity=0.0 -> multiplier=3.5 (permissive)
        # sensitivity=0.5 -> multiplier=2.5 (default)
        # sensitivity=1.0 -> multiplier=1.0 (strict)
        edge_sensitivity = params.get("edge_sensitivity", 0.5)
        multiplier = 3.5 - (2.5 * edge_sensitivity)

        try:
            result = sitk.ConfidenceConnected(
                sitkRoi,
                seedList=[sitkSeed],
                numberOfIterations=3,
                multiplier=multiplier,
                initialNeighborhoodRadius=2,
                replaceValue=1,
            )
            return sitk.GetArrayFromImage(result).astype(np.uint8)
        except Exception as e:
            logging.error(f"Region growing failed: {e}")
            return np.zeros_like(roi, dtype=np.uint8)

    def _thresholdBrush(self, roi, localSeed, params):
        """Threshold brush with auto or manual thresholds.

        Auto mode uses Otsu/Huang/etc to compute threshold, then auto-detects
        whether seed is in lighter or darker region to decide which side to segment.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates (i, j, k).
            params: Algorithm parameters including auto_threshold, threshold_method,
                    manual_lower, and manual_upper.

        Returns:
            Binary mask array.
        """
        if params.get("auto_threshold", True):
            method = params.get("threshold_method", "otsu")
            lower, upper = self._computeAutoThreshold(roi, localSeed, method)
        else:
            lower = params.get("manual_lower", -100)
            upper = params.get("manual_upper", 300)

        # Simple threshold - no connectivity, just intensity range
        mask = ((roi >= lower) & (roi <= upper)).astype(np.uint8)

        return mask

    def _applyBrushMask(self, mask, localSeed, radiusVoxels):
        """Apply circular/spherical brush constraint.

        Args:
            mask: Input mask array.
            localSeed: Seed point in local coordinates.
            radiusVoxels: Radius in voxels per dimension.

        Returns:
            Masked array.
        """
        shape = mask.shape  # (z, y, x)

        # Create coordinate grids
        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

        # Compute normalized distance
        dx = (x - localSeed[0]) / radiusVoxels[0]
        dy = (y - localSeed[1]) / radiusVoxels[1]
        dz = (z - localSeed[2]) / radiusVoxels[2]

        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        brushMask = distance <= 1.0

        return (mask & brushMask).astype(np.uint8)

    def _computeAutoThreshold(self, roi, localSeed, method="otsu"):
        """Compute automatic threshold using specified method.

        Auto-detects whether to segment above or below threshold based on
        whether the seed point is in the lighter or darker region.

        Args:
            roi: ROI array (will use histogram from this region).
            localSeed: Seed point in local ROI coordinates (i, j, k).
            method: One of "otsu", "huang", "triangle", "max_entropy", "isodata", "li".

        Returns:
            Tuple of (lower, upper) thresholds.
        """
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))

        if method == "otsu":
            filterObj = sitk.OtsuThresholdImageFilter()
        elif method == "huang":
            filterObj = sitk.HuangThresholdImageFilter()
        elif method == "triangle":
            filterObj = sitk.TriangleThresholdImageFilter()
        elif method == "max_entropy":
            filterObj = sitk.MaximumEntropyThresholdImageFilter()
        elif method == "isodata":
            filterObj = sitk.IsoDataThresholdImageFilter()
        elif method == "li":
            filterObj = sitk.LiThresholdImageFilter()
        else:
            filterObj = sitk.OtsuThresholdImageFilter()

        filterObj.SetInsideValue(1)
        filterObj.SetOutsideValue(0)
        filterObj.Execute(sitkRoi)

        threshold = filterObj.GetThreshold()
        data_min, data_max = float(roi.min()), float(roi.max())

        # Auto-detect: check if seed is above or below threshold
        # localSeed is (i, j, k) = (x, y, z), roi is (z, y, x)
        seed_intensity = roi[localSeed[2], localSeed[1], localSeed[0]]

        if seed_intensity >= threshold:
            # Seed is in brighter region - segment above threshold
            return (threshold, data_max)
        else:
            # Seed is in darker region - segment below threshold
            return (data_min, threshold)

    def _setThresholdsFromSeed(self, volumeArray, seedIjk, tolerancePercent):
        """Set threshold sliders based on intensity at seed point.

        Args:
            volumeArray: Full volume array (z, y, x ordering).
            seedIjk: Seed point (i, j, k).
            tolerancePercent: Tolerance as percentage of local std dev.
        """
        # Get seed intensity (array is z,y,x but seedIjk is i,j,k = x,y,z)
        seed_intensity = volumeArray[seedIjk[2], seedIjk[1], seedIjk[0]]

        # Compute local statistics in small ROI
        shape = volumeArray.shape
        radius = 10  # voxels
        startZ = max(0, seedIjk[2] - radius)
        startY = max(0, seedIjk[1] - radius)
        startX = max(0, seedIjk[0] - radius)
        endZ = min(shape[0], seedIjk[2] + radius + 1)
        endY = min(shape[1], seedIjk[1] + radius + 1)
        endX = min(shape[2], seedIjk[0] + radius + 1)

        roi = volumeArray[startZ:endZ, startY:endY, startX:endX]
        local_std = np.std(roi)

        # Compute tolerance
        # Scale factor 2.5 makes 20% tolerance ≈ 0.5 std dev, 100% ≈ 2.5 std dev
        tolerance = local_std * (tolerancePercent / 100.0) * 2.5

        # Update sliders
        self.lowerThresholdSlider.value = seed_intensity - tolerance
        self.upperThresholdSlider.value = seed_intensity + tolerance

        logging.info(
            f"Set thresholds from seed: intensity={seed_intensity:.1f}, "
            f"tolerance={tolerance:.1f}, range=[{seed_intensity - tolerance:.1f}, "
            f"{seed_intensity + tolerance:.1f}]"
        )

    def applyMaskToSegment(self, mask):
        """Apply the computed mask to the current segment.

        Args:
            mask: Binary mask numpy array (z, y, x ordering).
        """

        modifierLabelmap = self.scriptedEffect.defaultModifierLabelmap()

        # Get array from vtkOrientedImageData using vtk numpy support
        from vtk.util import numpy_support

        # Get the scalars from the image data
        imageData = modifierLabelmap
        dims = imageData.GetDimensions()

        # Get scalar array
        scalars = imageData.GetPointData().GetScalars()
        if scalars is None:
            # Initialize scalars if not present
            import vtk

            scalars = vtk.vtkUnsignedCharArray()
            scalars.SetNumberOfTuples(dims[0] * dims[1] * dims[2])
            scalars.Fill(0)
            imageData.GetPointData().SetScalars(scalars)

        # Convert to numpy array (this is a view, changes affect the original)
        modifierArray = numpy_support.vtk_to_numpy(scalars)
        modifierArray = modifierArray.reshape(dims[2], dims[1], dims[0])

        # Apply mask using OR operation
        # Make sure mask dimensions match
        if mask.shape == modifierArray.shape:
            np.logical_or(modifierArray, mask, out=modifierArray)
        else:
            logging.warning(
                f"Mask shape {mask.shape} doesn't match modifier shape {modifierArray.shape}"
            )
            # Try to apply anyway if shapes are compatible
            min_z = min(mask.shape[0], modifierArray.shape[0])
            min_y = min(mask.shape[1], modifierArray.shape[1])
            min_x = min(mask.shape[2], modifierArray.shape[2])
            modifierArray[:min_z, :min_y, :min_x] = np.logical_or(
                modifierArray[:min_z, :min_y, :min_x], mask[:min_z, :min_y, :min_x]
            )

        # Mark the image data as modified
        imageData.Modified()

        # Apply to segment using Add mode
        self.scriptedEffect.modifySelectedSegmentByLabelmap(
            modifierLabelmap, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
        )


# Required for Slicer to find the effect class
Effect = SegmentEditorEffect
