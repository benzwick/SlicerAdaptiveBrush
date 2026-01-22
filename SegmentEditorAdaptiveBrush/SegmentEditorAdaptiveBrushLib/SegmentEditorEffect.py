"""Adaptive Brush Segment Editor Effect.

This effect provides an adaptive brush that automatically segments regions
based on image intensity similarity, adapting to image features (edges,
boundaries) rather than using a fixed geometric shape.
"""

import logging
import os
import sys
import time
from typing import Dict

import ctk
import numpy as np
import qt
import slicer
import vtk
import vtk.util.numpy_support
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

# Try to import scikit-image for Random Walker algorithm
try:
    from skimage.segmentation import random_walker as skimage_random_walker

    HAS_SKIMAGE_RW = True
except ImportError:
    HAS_SKIMAGE_RW = False
    logging.debug("scikit-image not available - using fallback Random Walker")


class BrushOutlinePipeline:
    """VTK pipeline for brush outline visualization in a slice view.

    Shows two circle outlines:
    - Outer circle (yellow): Maximum brush extent
    - Inner circle (cyan): Threshold sampling zone
    - Preview overlay (green, semi-transparent): Segmentation preview
    """

    def __init__(self):
        """Initialize the brush outline pipeline."""
        # Preview overlay image actor
        self.previewImage = vtk.vtkImageData()
        self.previewMapper = vtk.vtkImageMapper()
        self.previewMapper.SetInputData(self.previewImage)
        self.previewMapper.SetColorWindow(255)
        self.previewMapper.SetColorLevel(127.5)

        self.previewActor = vtk.vtkActor2D()
        self.previewActor.SetMapper(self.previewMapper)
        self.previewActor.VisibilityOff()
        self.previewActor.SetPickable(False)
        self.previewActor.GetProperty().SetOpacity(0.4)

        # Lookup table for preview coloring (green with transparency)
        self.previewLUT = vtk.vtkLookupTable()
        self.previewLUT.SetNumberOfTableValues(2)
        self.previewLUT.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)  # Transparent for 0
        self.previewLUT.SetTableValue(1, 0.2, 0.8, 0.3, 0.5)  # Green semi-transparent for 1
        self.previewLUT.Build()

        # Use color mapping for preview
        self.previewColorMapper = vtk.vtkImageMapToColors()
        self.previewColorMapper.SetLookupTable(self.previewLUT)
        self.previewColorMapper.SetInputData(self.previewImage)

        # OUTER CIRCLE - Maximum extent (yellow)
        self.outerCircleSource = vtk.vtkRegularPolygonSource()
        self.outerCircleSource.SetNumberOfSides(64)
        self.outerCircleSource.SetRadius(1.0)
        self.outerCircleSource.GeneratePolygonOff()
        self.outerCircleSource.GeneratePolylineOn()

        self.outerTransform = vtk.vtkTransform()
        self.outerTransformFilter = vtk.vtkTransformPolyDataFilter()
        self.outerTransformFilter.SetTransform(self.outerTransform)
        self.outerTransformFilter.SetInputConnection(self.outerCircleSource.GetOutputPort())

        self.outerMapper = vtk.vtkPolyDataMapper2D()
        self.outerMapper.SetInputConnection(self.outerTransformFilter.GetOutputPort())

        self.outerActor = vtk.vtkActor2D()
        self.outerActor.SetMapper(self.outerMapper)
        self.outerActor.VisibilityOff()
        self.outerActor.SetPickable(False)

        # Outer circle styling - yellow, solid
        outerProp = self.outerActor.GetProperty()
        outerProp.SetColor(1.0, 0.9, 0.1)  # Yellow
        outerProp.SetLineWidth(2)
        outerProp.SetOpacity(0.8)

        # INNER CIRCLE - Threshold zone (cyan)
        self.innerCircleSource = vtk.vtkRegularPolygonSource()
        self.innerCircleSource.SetNumberOfSides(64)
        self.innerCircleSource.SetRadius(1.0)
        self.innerCircleSource.GeneratePolygonOff()
        self.innerCircleSource.GeneratePolylineOn()

        self.innerTransform = vtk.vtkTransform()
        self.innerTransformFilter = vtk.vtkTransformPolyDataFilter()
        self.innerTransformFilter.SetTransform(self.innerTransform)
        self.innerTransformFilter.SetInputConnection(self.innerCircleSource.GetOutputPort())

        self.innerMapper = vtk.vtkPolyDataMapper2D()
        self.innerMapper.SetInputConnection(self.innerTransformFilter.GetOutputPort())

        self.innerActor = vtk.vtkActor2D()
        self.innerActor.SetMapper(self.innerMapper)
        self.innerActor.VisibilityOff()
        self.innerActor.SetPickable(False)

        # Inner circle styling - cyan, dashed effect via stipple
        innerProp = self.innerActor.GetProperty()
        innerProp.SetColor(0.2, 0.9, 1.0)  # Cyan
        innerProp.SetLineWidth(1.5)
        innerProp.SetOpacity(0.7)

        # Store the renderer
        self.renderer = None
        self.sliceWidget = None

        # Keep reference to old actor for backwards compatibility
        self.actor = self.outerActor

    def setSliceWidget(self, sliceWidget):
        """Attach the pipeline to a slice widget's renderer.

        Args:
            sliceWidget: The qMRMLSliceWidget to attach to.
        """
        if self.renderer is not None:
            self.renderer.RemoveActor2D(self.outerActor)
            self.renderer.RemoveActor2D(self.innerActor)
            self.renderer.RemoveActor2D(self.previewActor)

        self.sliceWidget = sliceWidget
        if sliceWidget is not None:
            self.renderer = sliceWidget.sliceView().renderWindow().GetRenderers().GetFirstRenderer()
            if self.renderer is not None:
                # Add preview first so it's behind the outline circles
                self.renderer.AddActor2D(self.previewActor)
                self.renderer.AddActor2D(self.outerActor)
                self.renderer.AddActor2D(self.innerActor)

    def updateOutline(self, xyPosition, radiusPixels, innerRadiusRatio=0.5):
        """Update the brush outline position and size.

        Args:
            xyPosition: Center position in slice XY coordinates (x, y).
            radiusPixels: Outer brush radius in pixels.
            innerRadiusRatio: Inner circle as fraction of outer (0.0-1.0).
        """
        if self.renderer is None:
            return

        # Update outer circle
        self.outerCircleSource.SetRadius(radiusPixels)
        self.outerTransform.Identity()
        self.outerTransform.Translate(xyPosition[0], xyPosition[1], 0)
        self.outerActor.VisibilityOn()

        # Update inner circle
        innerRadius = radiusPixels * innerRadiusRatio
        self.innerCircleSource.SetRadius(innerRadius)
        self.innerTransform.Identity()
        self.innerTransform.Translate(xyPosition[0], xyPosition[1], 0)
        # Only show inner circle if it's meaningfully different from outer
        self.innerActor.SetVisibility(innerRadiusRatio < 0.95)

        # Request render
        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def updatePreview(self, mask2D, originXY, spacingXY):
        """Update the preview overlay with a 2D mask.

        Args:
            mask2D: 2D numpy array (height, width) with 0/1 values.
            originXY: Origin of the mask in slice XY coordinates (x, y).
            spacingXY: Pixel spacing in XY (sx, sy).
        """
        if self.renderer is None or mask2D is None:
            self.previewActor.VisibilityOff()
            return

        height, width = mask2D.shape

        # Create VTK image from numpy array
        self.previewImage.SetDimensions(width, height, 1)
        self.previewImage.SetSpacing(spacingXY[0], spacingXY[1], 1.0)
        self.previewImage.SetOrigin(originXY[0], originXY[1], 0)
        self.previewImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Copy mask data to VTK image
        # VTK expects row-major with Y going up, numpy has Y going down
        # Flip the mask vertically
        flippedMask = np.flipud(mask2D).astype(np.uint8) * 255

        # Get pointer to VTK image data and copy
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(
            flippedMask.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        self.previewImage.GetPointData().SetScalars(vtk_array)
        self.previewImage.Modified()

        # Update mapper
        self.previewMapper.SetInputData(self.previewImage)
        self.previewMapper.Modified()

        self.previewActor.VisibilityOn()

        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def hidePreview(self):
        """Hide only the preview overlay (keep circles visible)."""
        self.previewActor.VisibilityOff()
        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def hide(self):
        """Hide the brush outlines and preview."""
        self.outerActor.VisibilityOff()
        self.innerActor.VisibilityOff()
        self.previewActor.VisibilityOff()
        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def cleanup(self):
        """Remove the actors from the renderer and clean up."""
        if self.renderer is not None:
            self.renderer.RemoveActor2D(self.outerActor)
            self.renderer.RemoveActor2D(self.innerActor)
            self.renderer.RemoveActor2D(self.previewActor)
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

        # Default parameters - Basic
        self.radiusMm = 5.0
        self.edgeSensitivity = 50
        self.thresholdZone = 50  # Inner zone is 50% of brush radius
        self.samplingMethod = "mean_std"  # How to compute thresholds from zone
        self.algorithm = "geodesic_distance"
        self.backend = "auto"
        self.sphereMode = False
        self.previewMode = False  # Show segmentation preview on hover
        self.useThresholdCaching = False  # Disabled by default for accuracy

        # Preview throttling to avoid computing on every pixel
        self._lastPreviewTime = 0
        self._lastPreviewXY = None
        self._previewThrottleMs = 50  # Minimum ms between preview updates

        # Advanced parameters - Sampling
        # Gaussian sigma: controls center-weighting of intensity sampling
        # 0.5 gives moderate center bias - good balance of local accuracy vs noise robustness
        self.gaussianSigma = 0.5
        # Percentiles: 5-95 captures most of the tissue while excluding outliers
        self.percentileLow = 5
        self.percentileHigh = 95
        # Std multiplier: 2.0 captures ~95% of Gaussian distribution
        self.stdMultiplier = 2.0
        # Include zone: OFF by default - let algorithm determine boundaries naturally
        self.includeZoneInResult = False

        # Advanced parameters - Geodesic Distance
        # Edge weight 8.0: strong edge stopping for accurate boundaries
        self.geodesicEdgeWeight = 8.0
        # Distance scale 1.0: no scaling, use radius directly
        self.geodesicDistanceScale = 1.0
        # Smoothing 0.5: light smoothing reduces noise while preserving edges
        self.geodesicSmoothing = 0.5

        # Advanced parameters - Watershed
        # Gradient scale 1.5: moderate emphasis on edges
        self.watershedGradientScale = 1.5
        # Smoothing 0.5: light smoothing for noise reduction
        self.watershedSmoothing = 0.5

        # Advanced parameters - Level Set
        # Propagation 1.0: balanced expansion force
        self.levelSetPropagation = 1.0
        # Curvature 1.0: balanced smoothness constraint
        self.levelSetCurvature = 1.0
        # Iterations 50: sufficient for convergence in most cases
        self.levelSetIterations = 50

        # Advanced parameters - Region Growing
        # Multiplier 2.5: ~99% confidence interval for intensity matching
        self.regionGrowingMultiplier = 2.5
        # Iterations 4: enough passes for stable region
        self.regionGrowingIterations = 4

        # Advanced parameters - Random Walker
        # Beta 130: controls edge sensitivity (higher = stronger edges)
        self.randomWalkerBeta = 130.0
        # Mode: 'bf' (brute force) or 'cg' (conjugate gradient)
        self.randomWalkerMode = "cg"

        # Advanced parameters - Morphology
        # Fill holes: ON for cleaner results
        self.fillHoles = True
        # Closing radius 0: OFF by default to preserve fine detail
        self.closingRadius = 0

        # Brush outline visualization - one pipeline per slice view
        self.outlinePipelines: Dict[str, BrushOutlinePipeline] = {}
        self.activeViewWidget = None

        # Track source volume for detecting changes
        self._lastSourceVolumeId = None

        # Parameter presets for common tissue types
        # Each preset defines optimal parameters for that tissue
        self._presets = {
            "default": {
                "name": "Default",
                "description": "Balanced settings for general use",
                "algorithm": "geodesic_distance",
                "edge_sensitivity": 50,
                "threshold_zone": 50,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.0,
                "geodesic_edge_weight": 8.0,
                "geodesic_smoothing": 0.5,
                "random_walker_beta": 130.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "bone_ct": {
                "name": "Bone (CT)",
                "description": "High contrast bone segmentation in CT",
                "algorithm": "geodesic_distance",
                "edge_sensitivity": 70,
                "threshold_zone": 40,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.3,
                "std_multiplier": 1.5,
                "geodesic_edge_weight": 12.0,
                "geodesic_smoothing": 0.3,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "soft_tissue_ct": {
                "name": "Soft Tissue (CT)",
                "description": "Organs and soft tissue in CT (liver, muscle, etc.)",
                "algorithm": "watershed",
                "edge_sensitivity": 50,
                "threshold_zone": 60,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.5,
                "watershed_gradient_scale": 1.5,
                "watershed_smoothing": 0.7,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "lung_ct": {
                "name": "Lung (CT)",
                "description": "Lung parenchyma and airways in CT",
                "algorithm": "geodesic_distance",
                "edge_sensitivity": 60,
                "threshold_zone": 50,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.4,
                "std_multiplier": 2.0,
                "geodesic_edge_weight": 10.0,
                "geodesic_smoothing": 0.5,
                "fill_holes": False,  # Preserve airways
                "closing_radius": 0,
            },
            "brain_mri": {
                "name": "Brain (MRI)",
                "description": "Brain tissue segmentation in MRI",
                "algorithm": "level_set_cpu",
                "edge_sensitivity": 55,
                "threshold_zone": 50,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.0,
                "level_set_propagation": 1.0,
                "level_set_curvature": 1.2,
                "level_set_iterations": 60,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "tumor_lesion": {
                "name": "Tumor / Lesion",
                "description": "Tumors and lesions with irregular/ambiguous boundaries",
                "algorithm": "random_walker",
                "edge_sensitivity": 45,
                "threshold_zone": 60,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.6,
                "std_multiplier": 2.5,
                "random_walker_beta": 100.0,  # Lower beta for soft boundaries
                "fill_holes": True,
                "closing_radius": 1,
            },
            "vessel": {
                "name": "Vessels",
                "description": "Blood vessels and tubular structures",
                "algorithm": "geodesic_distance",
                "edge_sensitivity": 65,
                "threshold_zone": 35,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.3,
                "std_multiplier": 1.5,
                "geodesic_edge_weight": 10.0,
                "geodesic_distance_scale": 1.2,
                "geodesic_smoothing": 0.3,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "fat": {
                "name": "Fat",
                "description": "Adipose tissue (bright in T1 MRI, dark in CT)",
                "algorithm": "watershed",
                "edge_sensitivity": 40,
                "threshold_zone": 70,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.6,
                "std_multiplier": 2.5,
                "watershed_gradient_scale": 1.0,
                "watershed_smoothing": 0.8,
                "fill_holes": True,
                "closing_radius": 1,
            },
        }
        self._currentPreset = "default"

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
<li><b>Radius</b>: Maximum brush extent (circle shows limit)</li>
<li><b>Edge Sensitivity</b>: How strictly to follow intensity boundaries (0%=permissive, 100%=strict)</li>
<li><b>Algorithm</b>: Choose segmentation method</li>
</ul>
<h4>Algorithms:</h4>
<ul>
<li><b>Geodesic Distance</b>: Fast, follows edges naturally (recommended)</li>
<li><b>Watershed</b>: Good edge following, may be blocky on noisy images</li>
<li><b>Random Walker</b>: Probabilistic segmentation, excellent for ambiguous/blurry edges</li>
<li><b>Level Set</b>: Smooth contours, slower but precise</li>
<li><b>Region Growing</b>: Fast, good for homogeneous regions</li>
<li><b>Threshold Brush</b>: Simple intensity thresholding</li>
</ul>
<h4>Notes:</h4>
<ul>
<li>Circle outline shows <b>maximum extent</b> - actual painting may be smaller based on edges</li>
<li>Higher edge sensitivity = tighter boundary following = smaller regions</li>
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

        # Preset selector
        presetLayout = qt.QHBoxLayout()
        self.presetCombo = qt.QComboBox()
        for preset_id, preset_data in self._presets.items():
            self.presetCombo.addItem(preset_data["name"], preset_id)
        self.presetCombo.setToolTip(
            _(
                "Quick parameter presets optimized for different tissue types.\n\n"
                "Select a preset to automatically configure all parameters\n"
                "for that specific segmentation task."
            )
        )
        presetLayout.addWidget(self.presetCombo, 1)

        self.resetPresetButton = qt.QPushButton(_("Reset"))
        self.resetPresetButton.setToolTip(_("Reset all parameters to the selected preset"))
        self.resetPresetButton.setMaximumWidth(60)
        presetLayout.addWidget(self.resetPresetButton)

        brushLayout.addRow(_("Preset:"), presetLayout)

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

        # Threshold zone slider (inner circle for sampling)
        self.zoneSlider = ctk.ctkSliderWidget()
        self.zoneSlider.setToolTip(
            _(
                "Size of inner zone (cyan circle) used to sample intensities for threshold "
                "calculation. This zone is also guaranteed to be included in the result."
            )
        )
        self.zoneSlider.minimum = 10
        self.zoneSlider.maximum = 100
        self.zoneSlider.value = self.thresholdZone
        self.zoneSlider.singleStep = 5
        self.zoneSlider.suffix = "%"
        brushLayout.addRow(_("Threshold Zone:"), self.zoneSlider)

        # Intensity sampling method dropdown
        self.samplingMethodCombo = qt.QComboBox()
        self.samplingMethodCombo.addItem(_("Mean ± Std"), "mean_std")
        self.samplingMethodCombo.addItem(_("Percentile (5-95%)"), "percentile")
        self.samplingMethodCombo.addItem(_("Min / Max"), "minmax")
        self.samplingMethodCombo.setToolTip(
            _(
                "How to compute intensity thresholds from the sampling zone:\n"
                "• Mean ± Std: Use mean intensity ± (edge_sensitivity × std)\n"
                "• Percentile: Use 5th to 95th percentile range\n"
                "• Min / Max: Use actual min and max values"
            )
        )
        brushLayout.addRow(_("Sampling Method:"), self.samplingMethodCombo)

        # 3D sphere mode checkbox
        self.sphereModeCheckbox = qt.QCheckBox(_("3D Sphere Mode"))
        self.sphereModeCheckbox.setToolTip(_("Paint in 3D volume instead of 2D slice"))
        self.sphereModeCheckbox.checked = self.sphereMode
        brushLayout.addRow(self.sphereModeCheckbox)

        # Preview mode checkbox
        self.previewModeCheckbox = qt.QCheckBox(_("Preview Mode"))
        self.previewModeCheckbox.setToolTip(
            _(
                "Show semi-transparent preview of segmentation before clicking.\n\n"
                "When enabled, moving the mouse over the image will show what\n"
                "would be segmented if you clicked at that location.\n\n"
                "Note: May reduce responsiveness on slower systems."
            )
        )
        self.previewModeCheckbox.checked = self.previewMode
        brushLayout.addRow(self.previewModeCheckbox)

        # ----- Algorithm Settings -----
        algorithmCollapsible = ctk.ctkCollapsibleButton()
        algorithmCollapsible.text = _("Algorithm")
        algorithmCollapsible.collapsed = True
        self.scriptedEffect.addOptionsWidget(algorithmCollapsible)
        algorithmLayout = qt.QFormLayout(algorithmCollapsible)

        # Algorithm dropdown
        self.algorithmCombo = qt.QComboBox()
        self.algorithmCombo.addItem(_("Geodesic Distance (Recommended)"), "geodesic_distance")
        self.algorithmCombo.addItem(_("Watershed"), "watershed")
        self.algorithmCombo.addItem(_("Random Walker"), "random_walker")
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
        self.lowerThresholdSlider.setToolTip(
            _(
                "Lower intensity threshold for segmentation.\n\n"
                "Range automatically adjusts to image intensity (1st-99th percentile).\n"
                "Voxels with intensity below this value will not be included.\n\n"
                "Use 'Set from seed' button to set based on clicked location."
            )
        )
        self.lowerThresholdSlider.minimum = -2000
        self.lowerThresholdSlider.maximum = 5000
        self.lowerThresholdSlider.value = -100
        self.lowerThresholdSlider.singleStep = 1
        self.lowerThresholdSlider.decimals = 0
        manualLayout.addRow(_("Lower:"), self.lowerThresholdSlider)

        self.upperThresholdSlider = ctk.ctkSliderWidget()
        self.upperThresholdSlider.setToolTip(
            _(
                "Upper intensity threshold for segmentation.\n\n"
                "Range automatically adjusts to image intensity (1st-99th percentile).\n"
                "Voxels with intensity above this value will not be included.\n\n"
                "Use 'Set from seed' button to set based on clicked location."
            )
        )
        self.upperThresholdSlider.minimum = -2000
        self.upperThresholdSlider.maximum = 5000
        self.upperThresholdSlider.value = 300
        self.upperThresholdSlider.singleStep = 1
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

        # ----- Advanced Parameters -----
        advancedCollapsible = ctk.ctkCollapsibleButton()
        advancedCollapsible.text = _("Advanced Parameters")
        advancedCollapsible.collapsed = True
        self.scriptedEffect.addOptionsWidget(advancedCollapsible)
        advancedLayout = qt.QFormLayout(advancedCollapsible)

        # --- Sampling Parameters ---
        samplingLabel = qt.QLabel(_("<b>Sampling</b>"))
        advancedLayout.addRow(samplingLabel)

        # Gaussian sigma for distance weighting
        self.gaussianSigmaSlider = ctk.ctkSliderWidget()
        self.gaussianSigmaSlider.setToolTip(
            _(
                "Controls how much center pixels influence threshold computation.\n\n"
                "• 0.0 = Uniform weighting (all pixels in zone weighted equally)\n"
                "• 0.3-0.5 = Moderate center bias (recommended for most cases)\n"
                "• 1.0+ = Strong center bias (only pixels very close to cursor matter)\n\n"
                "Higher values make the brush more sensitive to the exact click location.\n"
                "Lower values average over a larger area, reducing noise sensitivity.\n\n"
                "Recommended: 0.5 (default)"
            )
        )
        self.gaussianSigmaSlider.minimum = 0.0
        self.gaussianSigmaSlider.maximum = 2.0
        self.gaussianSigmaSlider.value = self.gaussianSigma
        self.gaussianSigmaSlider.singleStep = 0.1
        self.gaussianSigmaSlider.decimals = 2
        advancedLayout.addRow(_("Gaussian Sigma:"), self.gaussianSigmaSlider)

        # Percentile low
        self.percentileLowSlider = ctk.ctkSliderWidget()
        self.percentileLowSlider.setToolTip(
            _(
                "Lower bound percentile for 'Percentile' sampling method.\n\n"
                "Intensities below this percentile are excluded from threshold range.\n"
                "• 0% = Include minimum intensity\n"
                "• 5% = Exclude darkest 5% (recommended - removes outliers)\n"
                "• 10-20% = More aggressive outlier removal\n\n"
                "Increase if dark noise/artifacts are being included.\n\n"
                "Recommended: 5%"
            )
        )
        self.percentileLowSlider.minimum = 0
        self.percentileLowSlider.maximum = 40
        self.percentileLowSlider.value = self.percentileLow
        self.percentileLowSlider.singleStep = 1
        self.percentileLowSlider.decimals = 0
        self.percentileLowSlider.suffix = "%"
        advancedLayout.addRow(_("Percentile Low:"), self.percentileLowSlider)

        # Percentile high
        self.percentileHighSlider = ctk.ctkSliderWidget()
        self.percentileHighSlider.setToolTip(
            _(
                "Upper bound percentile for 'Percentile' sampling method.\n\n"
                "Intensities above this percentile are excluded from threshold range.\n"
                "• 100% = Include maximum intensity\n"
                "• 95% = Exclude brightest 5% (recommended - removes outliers)\n"
                "• 80-90% = More aggressive outlier removal\n\n"
                "Decrease if bright noise/artifacts are being included.\n\n"
                "Recommended: 95%"
            )
        )
        self.percentileHighSlider.minimum = 60
        self.percentileHighSlider.maximum = 100
        self.percentileHighSlider.value = self.percentileHigh
        self.percentileHighSlider.singleStep = 1
        self.percentileHighSlider.decimals = 0
        self.percentileHighSlider.suffix = "%"
        advancedLayout.addRow(_("Percentile High:"), self.percentileHighSlider)

        # Std multiplier
        self.stdMultiplierSlider = ctk.ctkSliderWidget()
        self.stdMultiplierSlider.setToolTip(
            _(
                "Multiplier for standard deviation in 'Mean±Std' sampling method.\n\n"
                "Threshold range = mean ± (multiplier × std deviation)\n"
                "• 1.0 = ~68% of data (tight, may miss edges)\n"
                "• 2.0 = ~95% of data (recommended for most tissues)\n"
                "• 2.5 = ~99% of data (permissive)\n"
                "• 3.0+ = Very wide range (may leak into adjacent structures)\n\n"
                "Decrease for tighter boundaries, increase if segmentation has gaps.\n\n"
                "Recommended: 2.0"
            )
        )
        self.stdMultiplierSlider.minimum = 0.5
        self.stdMultiplierSlider.maximum = 4.0
        self.stdMultiplierSlider.value = self.stdMultiplier
        self.stdMultiplierSlider.singleStep = 0.1
        self.stdMultiplierSlider.decimals = 1
        advancedLayout.addRow(_("Std Multiplier:"), self.stdMultiplierSlider)

        # Include zone checkbox
        self.includeZoneCheckbox = qt.QCheckBox(_("Guarantee inner zone in result"))
        self.includeZoneCheckbox.setToolTip(
            _(
                "When enabled, the inner threshold zone (cyan circle) is always\n"
                "included in the segmentation result, regardless of algorithm output.\n\n"
                "• OFF (recommended): Algorithm determines all boundaries naturally\n"
                "• ON: Guarantees at least the inner zone is painted\n\n"
                "Enable if the brush sometimes produces empty results when clicking\n"
                "on valid tissue. Keep OFF for maximum boundary accuracy."
            )
        )
        self.includeZoneCheckbox.checked = self.includeZoneInResult
        advancedLayout.addRow(self.includeZoneCheckbox)

        # --- Geodesic Distance Parameters ---
        geodesicLabel = qt.QLabel(_("<b>Geodesic Distance</b>"))
        advancedLayout.addRow(geodesicLabel)

        self.geodesicEdgeWeightSlider = ctk.ctkSliderWidget()
        self.geodesicEdgeWeightSlider.setToolTip(
            _(
                "Controls how strongly edges/boundaries stop propagation.\n\n"
                "• 1-3 = Weak edge stopping (propagates through weak edges)\n"
                "• 5-8 = Moderate stopping (recommended for most tissues)\n"
                "• 10-15 = Strong stopping (stops at subtle intensity changes)\n"
                "• 15-20 = Very strong (may fragment into disconnected regions)\n\n"
                "Increase if brush leaks past boundaries.\n"
                "Decrease if brush stops too early or creates fragmented results.\n\n"
                "Recommended: 8.0"
            )
        )
        self.geodesicEdgeWeightSlider.minimum = 1.0
        self.geodesicEdgeWeightSlider.maximum = 20.0
        self.geodesicEdgeWeightSlider.value = self.geodesicEdgeWeight
        self.geodesicEdgeWeightSlider.singleStep = 0.5
        self.geodesicEdgeWeightSlider.decimals = 1
        advancedLayout.addRow(_("Edge Weight:"), self.geodesicEdgeWeightSlider)

        self.geodesicDistanceScaleSlider = ctk.ctkSliderWidget()
        self.geodesicDistanceScaleSlider.setToolTip(
            _(
                "Scales the distance threshold that determines region size.\n\n"
                "• 0.5 = Half the brush radius (tighter segmentation)\n"
                "• 1.0 = Match brush radius (recommended)\n"
                "• 1.5-2.0 = Expand beyond brush radius\n\n"
                "Increase to capture more of the structure.\n"
                "Decrease for tighter, more conservative segmentation.\n\n"
                "Recommended: 1.0"
            )
        )
        self.geodesicDistanceScaleSlider.minimum = 0.3
        self.geodesicDistanceScaleSlider.maximum = 2.5
        self.geodesicDistanceScaleSlider.value = self.geodesicDistanceScale
        self.geodesicDistanceScaleSlider.singleStep = 0.1
        self.geodesicDistanceScaleSlider.decimals = 1
        advancedLayout.addRow(_("Distance Scale:"), self.geodesicDistanceScaleSlider)

        self.geodesicSmoothingSlider = ctk.ctkSliderWidget()
        self.geodesicSmoothingSlider.setToolTip(
            _(
                "Gaussian smoothing applied before edge detection.\n\n"
                "• 0.0 = No smoothing (sensitive to noise, sharp edges)\n"
                "• 0.3-0.5 = Light smoothing (recommended - reduces noise)\n"
                "• 1.0+ = Heavy smoothing (blurs fine edges)\n\n"
                "Increase for noisy images or to ignore fine texture.\n"
                "Decrease to preserve sharp, fine boundaries.\n\n"
                "Recommended: 0.5"
            )
        )
        self.geodesicSmoothingSlider.minimum = 0.0
        self.geodesicSmoothingSlider.maximum = 2.0
        self.geodesicSmoothingSlider.value = self.geodesicSmoothing
        self.geodesicSmoothingSlider.singleStep = 0.1
        self.geodesicSmoothingSlider.decimals = 1
        advancedLayout.addRow(_("Smoothing:"), self.geodesicSmoothingSlider)

        # --- Watershed Parameters ---
        watershedLabel = qt.QLabel(_("<b>Watershed</b>"))
        advancedLayout.addRow(watershedLabel)

        self.watershedGradientScaleSlider = ctk.ctkSliderWidget()
        self.watershedGradientScaleSlider.setToolTip(
            _(
                "Amplification factor for image gradients in watershed.\n\n"
                "• 0.5-1.0 = Weak gradient emphasis (may leak through edges)\n"
                "• 1.5 = Moderate emphasis (recommended)\n"
                "• 2.0-3.0 = Strong emphasis (sensitive to all edges)\n\n"
                "Increase to make watershed respect weaker boundaries.\n"
                "Decrease if getting over-segmented/blocky results.\n\n"
                "Recommended: 1.5"
            )
        )
        self.watershedGradientScaleSlider.minimum = 0.3
        self.watershedGradientScaleSlider.maximum = 4.0
        self.watershedGradientScaleSlider.value = self.watershedGradientScale
        self.watershedGradientScaleSlider.singleStep = 0.1
        self.watershedGradientScaleSlider.decimals = 1
        advancedLayout.addRow(_("Gradient Scale:"), self.watershedGradientScaleSlider)

        self.watershedSmoothingSlider = ctk.ctkSliderWidget()
        self.watershedSmoothingSlider.setToolTip(
            _(
                "Gaussian smoothing before computing watershed gradients.\n\n"
                "• 0.0 = No smoothing (noisy gradients, over-segmentation)\n"
                "• 0.5 = Light smoothing (recommended)\n"
                "• 1.0+ = Heavy smoothing (merges small regions)\n\n"
                "Increase if results are too blocky or fragmented.\n"
                "Decrease to preserve fine boundary detail.\n\n"
                "Recommended: 0.5"
            )
        )
        self.watershedSmoothingSlider.minimum = 0.0
        self.watershedSmoothingSlider.maximum = 2.0
        self.watershedSmoothingSlider.value = self.watershedSmoothing
        self.watershedSmoothingSlider.singleStep = 0.1
        self.watershedSmoothingSlider.decimals = 1
        advancedLayout.addRow(_("Smoothing:"), self.watershedSmoothingSlider)

        # --- Level Set Parameters ---
        levelSetLabel = qt.QLabel(_("<b>Level Set</b>"))
        advancedLayout.addRow(levelSetLabel)

        self.levelSetPropagationSlider = ctk.ctkSliderWidget()
        self.levelSetPropagationSlider.setToolTip(
            _(
                "Controls the expansion/contraction force of the level set.\n\n"
                "• 0.5 = Weak expansion (conservative, may underestimate)\n"
                "• 1.0 = Balanced expansion (recommended)\n"
                "• 1.5-2.0 = Strong expansion (may overshoot boundaries)\n\n"
                "Increase if segmentation doesn't fill the target region.\n"
                "Decrease if segmentation leaks past boundaries.\n\n"
                "Recommended: 1.0"
            )
        )
        self.levelSetPropagationSlider.minimum = 0.2
        self.levelSetPropagationSlider.maximum = 3.0
        self.levelSetPropagationSlider.value = self.levelSetPropagation
        self.levelSetPropagationSlider.singleStep = 0.1
        self.levelSetPropagationSlider.decimals = 1
        advancedLayout.addRow(_("Propagation:"), self.levelSetPropagationSlider)

        self.levelSetCurvatureSlider = ctk.ctkSliderWidget()
        self.levelSetCurvatureSlider.setToolTip(
            _(
                "Controls boundary smoothness constraint.\n\n"
                "• 0.0-0.5 = Low smoothing (jagged boundaries, follows edges closely)\n"
                "• 1.0 = Balanced smoothing (recommended)\n"
                "• 2.0+ = Heavy smoothing (very smooth, may miss fine detail)\n\n"
                "Increase for smoother boundaries and to reduce noise.\n"
                "Decrease to preserve fine boundary detail.\n\n"
                "Recommended: 1.0"
            )
        )
        self.levelSetCurvatureSlider.minimum = 0.0
        self.levelSetCurvatureSlider.maximum = 3.0
        self.levelSetCurvatureSlider.value = self.levelSetCurvature
        self.levelSetCurvatureSlider.singleStep = 0.1
        self.levelSetCurvatureSlider.decimals = 1
        advancedLayout.addRow(_("Curvature:"), self.levelSetCurvatureSlider)

        self.levelSetIterationsSlider = ctk.ctkSliderWidget()
        self.levelSetIterationsSlider.setToolTip(
            _(
                "Maximum number of level set evolution iterations.\n\n"
                "• 20-30 = Fast but may not converge fully\n"
                "• 50 = Usually sufficient (recommended)\n"
                "• 100+ = More accurate but slower\n\n"
                "Increase if segmentation seems incomplete.\n"
                "Decrease for faster (but less accurate) results.\n\n"
                "Recommended: 50"
            )
        )
        self.levelSetIterationsSlider.minimum = 10
        self.levelSetIterationsSlider.maximum = 200
        self.levelSetIterationsSlider.value = self.levelSetIterations
        self.levelSetIterationsSlider.singleStep = 10
        self.levelSetIterationsSlider.decimals = 0
        advancedLayout.addRow(_("Iterations:"), self.levelSetIterationsSlider)

        # --- Region Growing Parameters ---
        regionGrowingLabel = qt.QLabel(_("<b>Region Growing</b>"))
        advancedLayout.addRow(regionGrowingLabel)

        self.regionGrowingMultiplierSlider = ctk.ctkSliderWidget()
        self.regionGrowingMultiplierSlider.setToolTip(
            _(
                "Confidence interval multiplier for intensity matching.\n\n"
                "Region grows to include voxels within mean ± (multiplier × std).\n"
                "• 1.0 = Tight (~68% confidence, may miss parts of region)\n"
                "• 2.0-2.5 = Moderate (~95-99% confidence, recommended)\n"
                "• 3.0+ = Permissive (may leak into adjacent structures)\n\n"
                "Increase if segmentation has holes or gaps.\n"
                "Decrease if segmentation leaks into surrounding tissue.\n\n"
                "Recommended: 2.5"
            )
        )
        self.regionGrowingMultiplierSlider.minimum = 0.5
        self.regionGrowingMultiplierSlider.maximum = 5.0
        self.regionGrowingMultiplierSlider.value = self.regionGrowingMultiplier
        self.regionGrowingMultiplierSlider.singleStep = 0.1
        self.regionGrowingMultiplierSlider.decimals = 1
        advancedLayout.addRow(_("Multiplier:"), self.regionGrowingMultiplierSlider)

        self.regionGrowingIterationsSlider = ctk.ctkSliderWidget()
        self.regionGrowingIterationsSlider.setToolTip(
            _(
                "Number of region growing iterations.\n\n"
                "Each iteration recomputes statistics from the current region.\n"
                "• 1-2 = Fast but may not reach full extent\n"
                "• 3-4 = Usually sufficient (recommended)\n"
                "• 5+ = More iterations, slower but more complete\n\n"
                "Increase if segmentation seems to stop too early.\n"
                "Decrease for faster results.\n\n"
                "Recommended: 4"
            )
        )
        self.regionGrowingIterationsSlider.minimum = 1
        self.regionGrowingIterationsSlider.maximum = 10
        self.regionGrowingIterationsSlider.value = self.regionGrowingIterations
        self.regionGrowingIterationsSlider.singleStep = 1
        self.regionGrowingIterationsSlider.decimals = 0
        advancedLayout.addRow(_("Iterations:"), self.regionGrowingIterationsSlider)

        # --- Random Walker Parameters ---
        rwLabel = qt.QLabel(_("<b>Random Walker</b>"))
        advancedLayout.addRow(rwLabel)

        self.randomWalkerBetaSlider = ctk.ctkSliderWidget()
        self.randomWalkerBetaSlider.setToolTip(
            _(
                "Edge sensitivity for Random Walker algorithm.\n\n"
                "Controls how strongly edges affect the random walk.\n"
                "Higher values = stronger edge boundaries.\n\n"
                "• 50-80: Low edge sensitivity, smoother boundaries\n"
                "• 100-150: Moderate sensitivity (recommended)\n"
                "• 200-500: High sensitivity, tighter boundaries\n\n"
                "Increase for sharp edges (bone/air).\n"
                "Decrease for soft tissue or noisy images.\n\n"
                "Recommended: 130"
            )
        )
        self.randomWalkerBetaSlider.minimum = 10
        self.randomWalkerBetaSlider.maximum = 500
        self.randomWalkerBetaSlider.value = self.randomWalkerBeta
        self.randomWalkerBetaSlider.singleStep = 10
        self.randomWalkerBetaSlider.decimals = 0
        advancedLayout.addRow(_("Beta (edge weight):"), self.randomWalkerBetaSlider)

        # --- Morphology Parameters ---
        morphLabel = qt.QLabel(_("<b>Post-processing</b>"))
        advancedLayout.addRow(morphLabel)

        self.fillHolesCheckbox = qt.QCheckBox(_("Fill holes in result"))
        self.fillHolesCheckbox.setToolTip(
            _(
                "Fill enclosed holes inside the segmentation.\n\n"
                "• ON (recommended): Removes internal holes/gaps\n"
                "• OFF: Preserve holes (e.g., for hollow structures)\n\n"
                "Enable for solid structures like organs or tumors.\n"
                "Disable if segmenting structures with intentional holes."
            )
        )
        self.fillHolesCheckbox.checked = self.fillHoles
        advancedLayout.addRow(self.fillHolesCheckbox)

        self.closingRadiusSlider = ctk.ctkSliderWidget()
        self.closingRadiusSlider.setToolTip(
            _(
                "Morphological closing to bridge small gaps in boundaries.\n\n"
                "Closing = dilation followed by erosion.\n"
                "• 0 = Disabled (recommended - preserves fine detail)\n"
                "• 1 = Small closing (fills 1-voxel gaps)\n"
                "• 2-3 = Moderate closing (fills small holes, smooths boundary)\n"
                "• 4+ = Large closing (significantly smooths, may lose detail)\n\n"
                "Increase if result has small gaps or rough boundaries.\n"
                "Keep at 0 to preserve accurate boundary detail.\n\n"
                "Recommended: 0"
            )
        )
        self.closingRadiusSlider.minimum = 0
        self.closingRadiusSlider.maximum = 5
        self.closingRadiusSlider.value = self.closingRadius
        self.closingRadiusSlider.singleStep = 1
        self.closingRadiusSlider.decimals = 0
        advancedLayout.addRow(_("Closing Radius:"), self.closingRadiusSlider)

        # Connect signals
        self.presetCombo.currentIndexChanged.connect(self.onPresetChanged)
        self.resetPresetButton.clicked.connect(self.onResetPreset)
        self.radiusSlider.valueChanged.connect(self.onRadiusChanged)
        self.sensitivitySlider.valueChanged.connect(self.onSensitivityChanged)
        self.zoneSlider.valueChanged.connect(self.onZoneChanged)
        self.samplingMethodCombo.currentIndexChanged.connect(self.onSamplingMethodChanged)
        self.sphereModeCheckbox.toggled.connect(self.onSphereModeChanged)
        self.previewModeCheckbox.toggled.connect(self.onPreviewModeChanged)
        self.algorithmCombo.currentIndexChanged.connect(self.onAlgorithmChanged)
        self.backendCombo.currentIndexChanged.connect(self.onBackendChanged)
        self.cachingCheckbox.toggled.connect(self.onCachingChanged)
        self.lowerThresholdSlider.valueChanged.connect(self.onThresholdChanged)
        self.upperThresholdSlider.valueChanged.connect(self.onThresholdChanged)
        self.autoThresholdCheckbox.toggled.connect(self.onAutoThresholdChanged)
        self.thresholdMethodCombo.currentIndexChanged.connect(self.onThresholdMethodChanged)
        self.setFromSeedButton.clicked.connect(self.onSetFromSeedClicked)
        self.toleranceSlider.valueChanged.connect(self.onThresholdChanged)

        # Advanced parameter signals
        self.gaussianSigmaSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.percentileLowSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.percentileHighSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.stdMultiplierSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.includeZoneCheckbox.toggled.connect(self.onAdvancedParamChanged)
        self.geodesicEdgeWeightSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.geodesicDistanceScaleSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.geodesicSmoothingSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.watershedGradientScaleSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.watershedSmoothingSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.levelSetPropagationSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.levelSetCurvatureSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.levelSetIterationsSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.regionGrowingMultiplierSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.regionGrowingIterationsSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.randomWalkerBetaSlider.valueChanged.connect(self.onAdvancedParamChanged)
        self.fillHolesCheckbox.toggled.connect(self.onAdvancedParamChanged)
        self.closingRadiusSlider.valueChanged.connect(self.onAdvancedParamChanged)

    def onPresetChanged(self, index):
        """Handle preset selection change."""
        preset_id = self.presetCombo.currentData
        if preset_id and preset_id in self._presets:
            self._applyPreset(preset_id)

    def onResetPreset(self):
        """Reset all parameters to the currently selected preset."""
        preset_id = self.presetCombo.currentData
        if preset_id and preset_id in self._presets:
            self._applyPreset(preset_id)

    def _applyPreset(self, preset_id):
        """Apply a parameter preset.

        Args:
            preset_id: The preset identifier (key in self._presets).
        """
        if preset_id not in self._presets:
            return

        preset = self._presets[preset_id]
        self._currentPreset = preset_id

        # Block signals to prevent multiple cache invalidations
        widgets_to_block = [
            self.sensitivitySlider,
            self.zoneSlider,
            self.samplingMethodCombo,
            self.algorithmCombo,
            self.gaussianSigmaSlider,
            self.stdMultiplierSlider,
            self.geodesicEdgeWeightSlider,
            self.geodesicDistanceScaleSlider,
            self.geodesicSmoothingSlider,
            self.watershedGradientScaleSlider,
            self.watershedSmoothingSlider,
            self.levelSetPropagationSlider,
            self.levelSetCurvatureSlider,
            self.levelSetIterationsSlider,
            self.regionGrowingMultiplierSlider,
            self.regionGrowingIterationsSlider,
            self.randomWalkerBetaSlider,
            self.fillHolesCheckbox,
            self.closingRadiusSlider,
        ]

        # Block signals
        for widget in widgets_to_block:
            widget.blockSignals(True)

        try:
            # Apply basic parameters
            if "edge_sensitivity" in preset:
                self.edgeSensitivity = preset["edge_sensitivity"]
                self.sensitivitySlider.value = preset["edge_sensitivity"]

            if "threshold_zone" in preset:
                self.thresholdZone = preset["threshold_zone"]
                self.zoneSlider.value = preset["threshold_zone"]

            if "sampling_method" in preset:
                self.samplingMethod = preset["sampling_method"]
                idx = self.samplingMethodCombo.findData(preset["sampling_method"])
                if idx >= 0:
                    self.samplingMethodCombo.setCurrentIndex(idx)

            if "algorithm" in preset:
                self.algorithm = preset["algorithm"]
                idx = self.algorithmCombo.findData(preset["algorithm"])
                if idx >= 0:
                    self.algorithmCombo.setCurrentIndex(idx)

            # Apply sampling parameters
            if "gaussian_sigma" in preset:
                self.gaussianSigma = preset["gaussian_sigma"]
                self.gaussianSigmaSlider.value = preset["gaussian_sigma"]

            if "std_multiplier" in preset:
                self.stdMultiplier = preset["std_multiplier"]
                self.stdMultiplierSlider.value = preset["std_multiplier"]

            # Apply geodesic parameters
            if "geodesic_edge_weight" in preset:
                self.geodesicEdgeWeight = preset["geodesic_edge_weight"]
                self.geodesicEdgeWeightSlider.value = preset["geodesic_edge_weight"]

            if "geodesic_distance_scale" in preset:
                self.geodesicDistanceScale = preset["geodesic_distance_scale"]
                self.geodesicDistanceScaleSlider.value = preset["geodesic_distance_scale"]

            if "geodesic_smoothing" in preset:
                self.geodesicSmoothing = preset["geodesic_smoothing"]
                self.geodesicSmoothingSlider.value = preset["geodesic_smoothing"]

            # Apply watershed parameters
            if "watershed_gradient_scale" in preset:
                self.watershedGradientScale = preset["watershed_gradient_scale"]
                self.watershedGradientScaleSlider.value = preset["watershed_gradient_scale"]

            if "watershed_smoothing" in preset:
                self.watershedSmoothing = preset["watershed_smoothing"]
                self.watershedSmoothingSlider.value = preset["watershed_smoothing"]

            # Apply level set parameters
            if "level_set_propagation" in preset:
                self.levelSetPropagation = preset["level_set_propagation"]
                self.levelSetPropagationSlider.value = preset["level_set_propagation"]

            if "level_set_curvature" in preset:
                self.levelSetCurvature = preset["level_set_curvature"]
                self.levelSetCurvatureSlider.value = preset["level_set_curvature"]

            if "level_set_iterations" in preset:
                self.levelSetIterations = preset["level_set_iterations"]
                self.levelSetIterationsSlider.value = preset["level_set_iterations"]

            # Apply region growing parameters
            if "region_growing_multiplier" in preset:
                self.regionGrowingMultiplier = preset["region_growing_multiplier"]
                self.regionGrowingMultiplierSlider.value = preset["region_growing_multiplier"]

            if "region_growing_iterations" in preset:
                self.regionGrowingIterations = preset["region_growing_iterations"]
                self.regionGrowingIterationsSlider.value = preset["region_growing_iterations"]

            # Apply random walker parameters
            if "random_walker_beta" in preset:
                self.randomWalkerBeta = preset["random_walker_beta"]
                self.randomWalkerBetaSlider.value = preset["random_walker_beta"]

            # Apply morphology parameters
            if "fill_holes" in preset:
                self.fillHoles = preset["fill_holes"]
                self.fillHolesCheckbox.checked = preset["fill_holes"]

            if "closing_radius" in preset:
                self.closingRadius = preset["closing_radius"]
                self.closingRadiusSlider.value = preset["closing_radius"]

        finally:
            # Unblock signals
            for widget in widgets_to_block:
                widget.blockSignals(False)

        # Invalidate cache once after all changes
        self.cache.invalidate()

        logging.debug(f"Applied preset: {preset['name']}")

    def onRadiusChanged(self, value):
        """Handle radius slider change."""
        self.radiusMm = value
        self.cache.invalidate()

    def onSensitivityChanged(self, value):
        """Handle edge sensitivity change."""
        self.edgeSensitivity = value
        self.cache.invalidate()

    def onZoneChanged(self, value):
        """Handle threshold zone size change."""
        self.thresholdZone = value
        self.cache.invalidate()

    def onSamplingMethodChanged(self, index):
        """Handle sampling method change."""
        self.samplingMethod = self.samplingMethodCombo.currentData
        self.cache.invalidate()

    def onAdvancedParamChanged(self, value=None):
        """Handle any advanced parameter change."""
        # Update all advanced parameters from UI
        self.gaussianSigma = self.gaussianSigmaSlider.value
        self.percentileLow = self.percentileLowSlider.value
        self.percentileHigh = self.percentileHighSlider.value
        self.stdMultiplier = self.stdMultiplierSlider.value
        self.includeZoneInResult = self.includeZoneCheckbox.checked
        self.geodesicEdgeWeight = self.geodesicEdgeWeightSlider.value
        self.geodesicDistanceScale = self.geodesicDistanceScaleSlider.value
        self.geodesicSmoothing = self.geodesicSmoothingSlider.value
        self.watershedGradientScale = self.watershedGradientScaleSlider.value
        self.watershedSmoothing = self.watershedSmoothingSlider.value
        self.levelSetPropagation = self.levelSetPropagationSlider.value
        self.levelSetCurvature = self.levelSetCurvatureSlider.value
        self.levelSetIterations = int(self.levelSetIterationsSlider.value)
        self.regionGrowingMultiplier = self.regionGrowingMultiplierSlider.value
        self.regionGrowingIterations = int(self.regionGrowingIterationsSlider.value)
        self.randomWalkerBeta = self.randomWalkerBetaSlider.value
        self.fillHoles = self.fillHolesCheckbox.checked
        self.closingRadius = int(self.closingRadiusSlider.value)
        self.cache.invalidate()

    def onSphereModeChanged(self, checked):
        """Handle 3D mode toggle."""
        self.sphereMode = checked
        self.cache.invalidate()

    def onPreviewModeChanged(self, checked):
        """Handle preview mode toggle."""
        self.previewMode = checked
        if not checked:
            # Hide any existing preview
            self._hideSegmentationPreview()

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
        self._updateThresholdRanges()
        # Track current source volume for change detection
        self._lastSourceVolumeId = self._getCurrentSourceVolumeId()

    def deactivate(self):
        """Called when the effect is deselected."""
        self.cache.clear()
        self.isDrawing = False
        self._cleanupOutlinePipelines()
        self._lastSourceVolumeId = None

    def sourceVolumeNodeChanged(self):
        """Called when the source volume node changes.

        Updates threshold slider ranges to match the new volume's intensity range.
        """
        self._updateThresholdRanges()
        self._lastSourceVolumeId = self._getCurrentSourceVolumeId()
        self.cache.invalidate()

    def _getCurrentSourceVolumeId(self):
        """Get the ID of the current source volume, or None."""
        try:
            parameterSetNode = self.scriptedEffect.parameterSetNode()
            if parameterSetNode is None:
                return None
            sourceVolumeNode = parameterSetNode.GetSourceVolumeNode()
            if sourceVolumeNode is None:
                return None
            return sourceVolumeNode.GetID()
        except Exception:
            return None

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

        Shows two circles: outer (max extent) and inner (threshold zone).

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

        # Inner zone ratio (0.0 to 1.0)
        innerRatio = self.thresholdZone / 100.0

        # Update the pipeline for this view
        if viewName in self.outlinePipelines:
            self.outlinePipelines[viewName].updateOutline(xy, radiusPixels, innerRatio)

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

    def _updateSegmentationPreview(self, xy, viewWidget):
        """Compute and display segmentation preview at cursor position.

        Args:
            xy: Screen coordinates (x, y).
            viewWidget: The slice view widget.
        """
        # Throttle preview updates to avoid performance issues
        current_time = time.time() * 1000  # Convert to ms
        if current_time - self._lastPreviewTime < self._previewThrottleMs:
            return

        # Skip if mouse hasn't moved significantly (more than 3 pixels)
        if self._lastPreviewXY is not None:
            dx = abs(xy[0] - self._lastPreviewXY[0])
            dy = abs(xy[1] - self._lastPreviewXY[1])
            if dx < 3 and dy < 3:
                return

        self._lastPreviewTime = current_time
        self._lastPreviewXY = xy

        try:
            # Get the IJK coordinates
            ijk = self.xyToIjk(xy, viewWidget)
            if ijk is None:
                self._hideSegmentationPreview()
                return

            # Get source volume
            sourceVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
            if sourceVolumeNode is None:
                return

            # Compute the mask (same as when painting)
            mask = self.computeAdaptiveMask(sourceVolumeNode, ijk, viewWidget)
            if mask is None or not np.any(mask):
                self._hideSegmentationPreview()
                return

            # Get the slice view info
            sliceLogic = viewWidget.sliceLogic()
            sliceNode = sliceLogic.GetSliceNode()
            viewName = sliceNode.GetName()

            if viewName not in self.outlinePipelines:
                return

            # Extract 2D slice from the 3D mask
            # Get the current slice index
            sliceIndex = self._getSliceIndex(viewWidget, sourceVolumeNode)
            if sliceIndex is None:
                return

            # Get orientation to extract correct slice
            orientation = sliceNode.GetOrientationString()

            # Extract 2D slice based on orientation
            if orientation == "Axial":
                mask2D = mask[sliceIndex, :, :]
            elif orientation == "Sagittal":
                mask2D = mask[:, :, sliceIndex]
            elif orientation == "Coronal":
                mask2D = mask[:, sliceIndex, :]
            else:
                # For oblique, just use axial
                mask2D = mask[sliceIndex, :, :] if sliceIndex < mask.shape[0] else None

            if mask2D is None or not np.any(mask2D):
                self._hideSegmentationPreview()
                return

            # Convert mask coordinates to XY screen coordinates
            # Get the transform from IJK to XY
            rasToXY = vtk.vtkMatrix4x4()
            sliceNode.GetXYToRAS().GetInverse(rasToXY)

            ijkToRAS = vtk.vtkMatrix4x4()
            sourceVolumeNode.GetIJKToRASMatrix(ijkToRAS)

            spacing = sourceVolumeNode.GetSpacing()

            # For now, use a simple approach: position the preview at the brush location
            # and scale based on spacing
            sliceSpacing = self._getSliceSpacing(viewWidget)

            # Calculate the origin of the mask in XY coordinates
            # The mask is centered around the seed point
            radiusVoxels = [self.radiusMm / s for s in spacing]
            margin = 1.2

            # Origin in XY: center minus radius
            originXY = (
                xy[0] - radiusVoxels[0] * margin / sliceSpacing,
                xy[1] - radiusVoxels[1] * margin / sliceSpacing,
            )

            # Spacing in XY (pixels per voxel)
            spacingXY = (1.0 / sliceSpacing, 1.0 / sliceSpacing)

            # Update the preview overlay
            self.outlinePipelines[viewName].updatePreview(mask2D, originXY, spacingXY)

        except Exception as e:
            logging.debug(f"Preview computation failed: {e}")
            self._hideSegmentationPreview()

    def _hideSegmentationPreview(self):
        """Hide segmentation preview overlay in all views."""
        for pipeline in self.outlinePipelines.values():
            pipeline.hidePreview()

    def _getSliceIndex(self, viewWidget, volumeNode):
        """Get the current slice index for the given view and volume.

        Args:
            viewWidget: The slice view widget.
            volumeNode: The volume node.

        Returns:
            Slice index (int) or None.
        """
        try:
            sliceLogic = viewWidget.sliceLogic()
            sliceNode = sliceLogic.GetSliceNode()

            # Get the slice offset (position along the normal)
            sliceOffset = sliceLogic.GetSliceOffset()

            # Get volume origin and spacing
            origin = volumeNode.GetOrigin()
            spacing = volumeNode.GetSpacing()
            dims = volumeNode.GetImageData().GetDimensions()

            # Determine orientation
            orientation = sliceNode.GetOrientationString()

            if orientation == "Axial":
                # Z axis
                sliceIndex = int(round((sliceOffset - origin[2]) / spacing[2]))
                sliceIndex = max(0, min(sliceIndex, dims[2] - 1))
            elif orientation == "Sagittal":
                # X axis
                sliceIndex = int(round((sliceOffset - origin[0]) / spacing[0]))
                sliceIndex = max(0, min(sliceIndex, dims[0] - 1))
            elif orientation == "Coronal":
                # Y axis
                sliceIndex = int(round((sliceOffset - origin[1]) / spacing[1]))
                sliceIndex = max(0, min(sliceIndex, dims[1] - 1))
            else:
                # Default to axial for oblique
                sliceIndex = int(round((sliceOffset - origin[2]) / spacing[2]))
                sliceIndex = max(0, min(sliceIndex, dims[2] - 1))

            return sliceIndex

        except Exception:
            return None

    def _updateThresholdRanges(self):
        """Update threshold slider ranges based on source volume intensity.

        Sets slider min/max to the 1st-99th percentile of image intensities
        for a more meaningful range than hardcoded values. Also sets
        appropriate step size and sensible default values.
        """
        try:
            parameterSetNode = self.scriptedEffect.parameterSetNode()
            if parameterSetNode is None:
                return

            sourceVolumeNode = parameterSetNode.GetSourceVolumeNode()
            if sourceVolumeNode is None:
                return

            volumeArray = slicer.util.arrayFromVolume(sourceVolumeNode)
            if volumeArray is None or volumeArray.size == 0:
                return

            # Compute percentiles for robust range estimation
            # p1/p99 for slider range, p25/p75 for default values
            p1, p25, p50, p75, p99 = np.percentile(volumeArray, [1, 25, 50, 75, 99])

            # Add small margin for slider usability
            margin = (p99 - p1) * 0.05
            range_min = p1 - margin
            range_max = p99 + margin
            total_range = range_max - range_min

            # Update slider ranges
            self.lowerThresholdSlider.minimum = range_min
            self.lowerThresholdSlider.maximum = range_max
            self.upperThresholdSlider.minimum = range_min
            self.upperThresholdSlider.maximum = range_max

            # Set appropriate step size based on range
            # Aim for ~200-500 steps across the range
            step = max(1, total_range / 300)
            # Round to nice values
            if step >= 10:
                step = round(step / 10) * 10
            elif step >= 1:
                step = round(step)
            else:
                step = round(step, 1)

            self.lowerThresholdSlider.singleStep = step
            self.upperThresholdSlider.singleStep = step

            # Set default values to interquartile range (25th to 75th percentile)
            # This captures the "middle 50%" of intensities - a sensible starting point
            self.lowerThresholdSlider.value = p25
            self.upperThresholdSlider.value = p75

            logging.debug(
                f"Updated threshold ranges: [{range_min:.1f}, {range_max:.1f}], "
                f"step: {step}, defaults (IQR): [{p25:.1f}, {p75:.1f}]"
            )
        except Exception as e:
            logging.warning(f"Could not update threshold ranges: {e}")

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
            # Hide segmentation preview when starting to draw
            self._hideSegmentationPreview()
            self._updateBrushPreview(xy, viewWidget)
            self.processPoint(xy, viewWidget)
            return True

        elif eventId == vtk.vtkCommand.MouseMoveEvent:
            # Update brush outline preview on mouse move
            self._updateBrushPreview(xy, viewWidget)
            if self.isDrawing:
                self.processPoint(xy, viewWidget)
                return True  # Only consume when drawing
            else:
                # Show segmentation preview when hovering (if preview mode enabled)
                if self.previewMode:
                    self._updateSegmentationPreview(xy, viewWidget)
            return False  # Don't consume when just hovering

        elif eventId == vtk.vtkCommand.LeftButtonReleaseEvent:
            self.isDrawing = False
            self.lastIjk = None
            self.cache.onMouseRelease()
            # Show preview again after drawing
            if self.previewMode:
                self._updateSegmentationPreview(xy, viewWidget)
            return True

        elif eventId == vtk.vtkCommand.LeaveEvent:
            # Hide preview when mouse leaves the view
            self._hideBrushPreview()
            self._hideSegmentationPreview()
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
            # Basic
            "radius_mm": self.radiusMm,
            "radius_voxels": radiusVoxels,
            "edge_sensitivity": self.edgeSensitivity / 100.0,
            "threshold_zone": self.thresholdZone / 100.0,
            "sampling_method": self.samplingMethod,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "sphere_mode": self.sphereMode,
            "slice_index": sliceIndex,
            "manual_lower": self.lowerThresholdSlider.value,
            "manual_upper": self.upperThresholdSlider.value,
            "auto_threshold": self.autoThresholdCheckbox.checked,
            "threshold_method": self.thresholdMethodCombo.currentData,
            # Advanced - Sampling
            "gaussian_sigma": self.gaussianSigma,
            "percentile_low": self.percentileLow,
            "percentile_high": self.percentileHigh,
            "std_multiplier": self.stdMultiplier,
            "include_zone_in_result": self.includeZoneInResult,
            # Advanced - Geodesic
            "geodesic_edge_weight": self.geodesicEdgeWeight,
            "geodesic_distance_scale": self.geodesicDistanceScale,
            "geodesic_smoothing": self.geodesicSmoothing,
            # Advanced - Watershed
            "watershed_gradient_scale": self.watershedGradientScale,
            "watershed_smoothing": self.watershedSmoothing,
            # Advanced - Level Set
            "level_set_propagation": self.levelSetPropagation,
            "level_set_curvature": self.levelSetCurvature,
            "level_set_iterations": self.levelSetIterations,
            # Advanced - Region Growing
            "region_growing_multiplier": self.regionGrowingMultiplier,
            "region_growing_iterations": self.regionGrowingIterations,
            # Advanced - Random Walker
            "random_walker_beta": self.randomWalkerBeta,
            # Advanced - Morphology
            "fill_holes": self.fillHoles,
            "closing_radius": self.closingRadius,
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

        # Compute zone-based thresholds (overrides analyzer thresholds if zone is set)
        zone_thresholds = self._computeZoneThresholds(roi, localSeed, params)
        if zone_thresholds is not None:
            thresholds = zone_thresholds

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
        elif algorithm == "geodesic_distance":
            mask = self._geodesicDistance(roi, localSeed, thresholds, params)
        elif algorithm == "random_walker":
            mask = self._randomWalker(roi, localSeed, thresholds, params)
        else:  # Default: watershed
            mask = self._watershed(roi, localSeed, thresholds, params)

        # Apply circular brush mask as MAXIMUM extent for ALL algorithms
        # Adaptive algorithms use edges to stop earlier, but should never exceed brush radius
        mask = self._applyBrushMask(mask, localSeed, params["radius_voxels"])

        # Apply morphological operations based on advanced parameters
        fill_holes = params.get("fill_holes", True)
        closing_radius = params.get("closing_radius", 1)

        if np.any(mask) and (fill_holes or closing_radius > 0):
            maskSitk = sitk.GetImageFromArray(mask)

            # Fill holes inside the segmentation
            if fill_holes:
                try:
                    maskSitk = sitk.BinaryFillhole(maskSitk)
                except Exception:
                    pass  # Skip if fillhole fails

            # Close small gaps (morphological closing)
            if closing_radius > 0:
                try:
                    kernel_size = [int(closing_radius)] * 3
                    maskSitk = sitk.BinaryMorphologicalClosing(maskSitk, kernel_size)
                except Exception:
                    pass  # Skip if closing fails

            mask = sitk.GetArrayFromImage(maskSitk).astype(np.uint8)

        # Ensure inner zone (threshold zone) is included in result if enabled
        include_zone = params.get("include_zone_in_result", True)
        if include_zone:
            inner_zone_mask = self._createInnerZoneMask(roi.shape, localSeed, params)
            mask = mask | inner_zone_mask

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
            params: Algorithm parameters including watershed_gradient_scale
                    and watershed_smoothing.

        Returns:
            Binary mask array.
        """
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))

        # Advanced parameters
        watershed_gradient_scale = params.get("watershed_gradient_scale", 1.0)
        watershed_smoothing = params.get("watershed_smoothing", 0.5)
        edge_sensitivity = params.get("edge_sensitivity", 0.5)

        # Smooth the image to reduce noise - user-controllable
        if watershed_smoothing > 0.01:
            smoothed = sitk.SmoothingRecursiveGaussian(sitkRoi, sigma=watershed_smoothing)
        else:
            smoothed = sitkRoi

        # Compute gradient magnitude for watershed
        gradient = sitk.GradientMagnitude(smoothed)
        gradArray = sitk.GetArrayFromImage(gradient)

        # Normalize gradient to 0-255 range
        gradMax = np.percentile(gradArray, 98) + 1e-8

        # Combine user-controllable gradient_scale with edge_sensitivity
        # gradient_scale is a base multiplier, edge_sensitivity fine-tunes it
        effective_scale = watershed_gradient_scale * (0.5 + 1.5 * edge_sensitivity)
        gradArray = np.clip(gradArray / gradMax * 255 * effective_scale, 0, 255).astype(
            np.float32
        )

        gradient = sitk.GetImageFromArray(gradArray)

        # Create foreground marker at seed point (small sphere)
        fgMarker = np.zeros_like(roi, dtype=np.uint8)
        fgMarker[localSeed[2], localSeed[1], localSeed[0]] = 1
        # Dilate slightly to make it more robust
        fgSitk = sitk.GetImageFromArray(fgMarker)
        fgSitk = sitk.BinaryDilate(fgSitk, [1, 1, 1])

        # Create background marker at ROI boundary
        bgMarker = np.zeros_like(roi, dtype=np.uint8)
        bgMarker[0, :, :] = 1
        bgMarker[-1, :, :] = 1
        bgMarker[:, 0, :] = 1
        bgMarker[:, -1, :] = 1
        bgMarker[:, :, 0] = 1
        bgMarker[:, :, -1] = 1
        bgSitk = sitk.GetImageFromArray(bgMarker)

        # Combine markers (foreground=1, background=2)
        markers = sitk.Add(fgSitk, sitk.Multiply(bgSitk, 2))
        markers = sitk.Cast(markers, sitk.sitkUInt8)

        # Run watershed
        try:
            watershed = sitk.MorphologicalWatershedFromMarkers(
                gradient, markers, markWatershedLine=False, fullyConnected=True
            )
            # Extract foreground (label 1)
            result = sitk.BinaryThreshold(watershed, 1, 1, 1, 0)
            mask = sitk.GetArrayFromImage(result).astype(np.uint8)

            # Also constrain by intensity thresholds for safety
            intensity_mask = (roi >= thresholds["lower"]) & (roi <= thresholds["upper"])
            mask = mask & intensity_mask.astype(np.uint8)

            return mask
        except Exception as e:
            logging.error(f"Watershed failed: {e}")
            return self._connectedThreshold(roi, localSeed, thresholds)

    def _levelSet(self, roi, localSeed, thresholds, params):
        """Level set segmentation using threshold-based approach.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates.
            thresholds: Intensity thresholds.
            params: Algorithm parameters including level_set_propagation,
                    level_set_curvature, and level_set_iterations.

        Returns:
            Binary mask array.
        """
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))

        edge_sensitivity = params.get("edge_sensitivity", 0.5)

        # Advanced parameters
        level_set_propagation = params.get("level_set_propagation", 1.5)
        level_set_curvature = params.get("level_set_curvature", 1.0)
        level_set_iterations = params.get("level_set_iterations", 50)

        # Use ThresholdSegmentationLevelSetImageFilter which is more reliable
        # It grows/shrinks based on whether pixels are in threshold range
        try:
            # Smooth input slightly
            smoothed = sitk.SmoothingRecursiveGaussian(sitkRoi, sigma=0.5)

            # Create seed image with signed distance from seed point
            radius_voxels = params.get("radius_voxels", [5, 5, 5])
            init_radius = max(2, min(radius_voxels) / 3)

            # Create distance from seed
            shape = roi.shape
            z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
            dist_from_seed = np.sqrt(
                (x - localSeed[0]) ** 2 + (y - localSeed[1]) ** 2 + (z - localSeed[2]) ** 2
            )
            # Signed distance: negative inside, positive outside
            init_levelset = dist_from_seed - init_radius
            seedImage = sitk.GetImageFromArray(init_levelset.astype(np.float32))

            # Configure level set filter
            lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
            lsFilter.SetLowerThreshold(float(thresholds["lower"]))
            lsFilter.SetUpperThreshold(float(thresholds["upper"]))

            # Propagation: positive = grow into threshold region
            # Curvature: smooths the boundary
            # Combine user params with edge_sensitivity for fine control
            # Higher edge sensitivity = more curvature, less propagation
            effective_prop = level_set_propagation * (1.0 - 0.5 * edge_sensitivity)
            effective_curv = level_set_curvature * (0.5 + edge_sensitivity)

            lsFilter.SetPropagationScaling(effective_prop)
            lsFilter.SetCurvatureScaling(effective_curv)
            lsFilter.SetMaximumRMSError(0.02)
            lsFilter.SetNumberOfIterations(int(level_set_iterations))

            levelSet = lsFilter.Execute(seedImage, smoothed)

            # Threshold: inside is negative in level set convention
            result = sitk.BinaryThreshold(levelSet, upperThreshold=0)
            return sitk.GetArrayFromImage(result).astype(np.uint8)

        except Exception as e:
            logging.error(f"Level set failed: {e}")
            # Fall back to connected threshold
            return self._connectedThreshold(roi, localSeed, thresholds)

    def _regionGrowing(self, roi, localSeed, thresholds, params):
        """Region growing segmentation.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates.
            thresholds: Intensity thresholds.
            params: Algorithm parameters including region_growing_multiplier
                    and region_growing_iterations.

        Returns:
            Binary mask array.
        """
        # Use confidence connected for region growing behavior
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))
        sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))

        # Advanced parameters
        region_growing_multiplier = params.get("region_growing_multiplier", 2.0)
        region_growing_iterations = params.get("region_growing_iterations", 4)
        edge_sensitivity = params.get("edge_sensitivity", 0.5)

        # Combine user multiplier with edge sensitivity
        # Higher sensitivity = tighter bounds (smaller effective multiplier)
        effective_multiplier = region_growing_multiplier * (1.0 - 0.5 * edge_sensitivity)
        effective_multiplier = max(0.5, effective_multiplier)  # Keep minimum multiplier

        try:
            result = sitk.ConfidenceConnected(
                sitkRoi,
                seedList=[sitkSeed],
                numberOfIterations=int(region_growing_iterations),
                multiplier=effective_multiplier,
                initialNeighborhoodRadius=2,
                replaceValue=1,
            )
            mask = sitk.GetArrayFromImage(result).astype(np.uint8)

            # Also constrain by intensity thresholds
            intensity_mask = (roi >= thresholds["lower"]) & (roi <= thresholds["upper"])
            mask = mask & intensity_mask.astype(np.uint8)

            return mask
        except Exception as e:
            logging.error(f"Region growing failed: {e}")
            return np.zeros_like(roi, dtype=np.uint8)

    def _thresholdBrush(self, roi, localSeed, params):
        """Threshold brush with auto or manual thresholds.

        Auto mode uses Otsu/Huang/etc to compute threshold, then auto-detects
        whether seed is in lighter or darker region to decide which side to segment.

        Edge sensitivity controls connectivity: at high sensitivity, only the
        connected component containing the seed is kept.

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

        # At high edge sensitivity, use connected components to keep only
        # the region connected to the seed
        edge_sensitivity = params.get("edge_sensitivity", 0.5)
        if edge_sensitivity > 0.3:
            sitkMask = sitk.GetImageFromArray(mask)
            sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))

            # Use connected threshold on the binary mask to get only connected region
            try:
                connected = sitk.ConnectedThreshold(
                    sitkMask, seedList=[sitkSeed], lower=1, upper=1, replaceValue=1
                )
                mask = sitk.GetArrayFromImage(connected).astype(np.uint8)
            except Exception:
                pass  # Keep original mask if connectivity fails

        return mask

    def _geodesicDistance(self, roi, localSeed, thresholds, params):
        """Geodesic distance-based segmentation.

        Combines spatial distance from seed with intensity gradient penalty
        and intensity similarity weighting. Works better for narrow segments
        by incorporating intensity constraints into the speed function.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates (i, j, k).
            thresholds: Intensity thresholds (used for initial mask).
            params: Algorithm parameters including geodesic_edge_weight,
                    geodesic_distance_scale, and geodesic_smoothing.

        Returns:
            Binary mask array.
        """
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))
        sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))

        # Get parameters
        edge_sensitivity = params.get("edge_sensitivity", 0.5)
        radius_voxels = params.get("radius_voxels", [10, 10, 10])
        avg_radius = np.mean(radius_voxels)

        # Advanced parameters
        geodesic_edge_weight = params.get("geodesic_edge_weight", 5.0)
        geodesic_distance_scale = params.get("geodesic_distance_scale", 1.0)
        geodesic_smoothing = params.get("geodesic_smoothing", 0.5)

        # Smooth to reduce noise - user-controllable smoothing
        if geodesic_smoothing > 0.01:
            smoothed = sitk.SmoothingRecursiveGaussian(sitkRoi, sigma=geodesic_smoothing)
        else:
            smoothed = sitkRoi

        smoothedArray = sitk.GetArrayFromImage(smoothed)

        # Get seed intensity for intensity similarity computation
        seed_intensity = roi[localSeed[2], localSeed[1], localSeed[0]]

        # Compute gradient magnitude
        gradient = sitk.GradientMagnitude(smoothed)
        gradArray = sitk.GetArrayFromImage(gradient)

        # Normalize gradient
        gradMax = np.percentile(gradArray, 95) + 1e-8
        gradNorm = gradArray / gradMax

        # Create gradient-based speed: high speed in homogeneous regions
        effective_edge_weight = geodesic_edge_weight * (0.5 + edge_sensitivity)
        gradientSpeed = 1.0 / (1.0 + gradNorm * effective_edge_weight)

        # Create intensity similarity speed: high speed when similar to seed
        # This helps narrow segments stay within intensity bounds
        intensity_std = max(np.std(smoothedArray), 1.0)
        intensity_diff = np.abs(smoothedArray - seed_intensity) / intensity_std
        # Sigmoid-like falloff: high speed (1.0) when similar, low when different
        intensity_weight = 2.0 + 3.0 * edge_sensitivity  # 2-5
        intensitySpeed = 1.0 / (1.0 + intensity_diff * intensity_weight)

        # Combine speeds: use geometric mean for balanced effect
        # This ensures narrow segments don't leak into dissimilar intensities
        speedArray = np.sqrt(gradientSpeed * intensitySpeed)

        # Pre-mask by intensity threshold to prevent any propagation outside bounds
        intensity_mask = (roi >= thresholds["lower"]) & (roi <= thresholds["upper"])
        speedArray = speedArray * intensity_mask

        # Ensure minimum speed in valid regions to prevent infinite distances
        speedArray = np.where(intensity_mask, np.maximum(speedArray, 0.01), 0.0)
        speed = sitk.GetImageFromArray(speedArray.astype(np.float32))

        # Run Fast Marching to compute geodesic distance from seed
        try:
            # FastMarching computes arrival times (geodesic distances)
            fastMarching = sitk.FastMarchingImageFilter()
            fastMarching.SetStoppingValue(avg_radius * 3)  # Stop at 3x radius

            # Set trial points (seeds)
            fastMarching.AddTrialPoint(sitkSeed)

            distance = fastMarching.Execute(speed)
            distArray = sitk.GetArrayFromImage(distance)

            # Threshold at radius to get mask
            # distance_scale allows user to expand/contract the result
            # Higher sensitivity = tighter boundary
            base_threshold = avg_radius * (1.5 - 0.5 * edge_sensitivity)
            dist_threshold = base_threshold * geodesic_distance_scale
            mask = (distArray < dist_threshold).astype(np.uint8)

            # Final intensity threshold (should already be satisfied but for safety)
            mask = mask & intensity_mask.astype(np.uint8)

            # Ensure result is connected to seed using connected components
            # This is crucial for narrow segments to avoid disconnected blobs
            if np.any(mask):
                sitkMask = sitk.GetImageFromArray(mask)
                try:
                    connected = sitk.ConnectedThreshold(
                        sitkMask, seedList=[sitkSeed], lower=1, upper=1, replaceValue=1
                    )
                    mask = sitk.GetArrayFromImage(connected).astype(np.uint8)
                except Exception:
                    pass  # Keep original mask if connectivity check fails

            return mask

        except Exception as e:
            logging.error(f"Geodesic distance failed: {e}")
            # Fall back to connected threshold
            return self._connectedThreshold(roi, localSeed, thresholds)

    def _randomWalker(self, roi, localSeed, thresholds, params):
        """Random Walker segmentation using scikit-image.

        Uses the Random Walker algorithm which treats segmentation as an
        electrical potential problem. Excellent at handling ambiguous boundaries
        by computing probability of random walk reaching foreground vs background.

        Falls back to gradient-weighted region growing if scikit-image unavailable.

        Args:
            roi: ROI array (z, y, x).
            localSeed: Seed point in local coordinates (i, j, k).
            thresholds: Intensity thresholds.
            params: Algorithm parameters including random_walker_beta.

        Returns:
            Binary mask array.
        """
        edge_sensitivity = params.get("edge_sensitivity", 0.5)
        radius_voxels = params.get("radius_voxels", [10, 10, 10])

        # Get user-configurable beta parameter
        base_beta = params.get("random_walker_beta", 130.0)
        # Scale beta by edge sensitivity (higher sensitivity = higher beta)
        beta = base_beta * (0.5 + edge_sensitivity)  # 65-195 range at default

        if HAS_SKIMAGE_RW:
            try:
                return self._randomWalkerSkimage(
                    roi, localSeed, thresholds, beta, radius_voxels
                )
            except Exception as e:
                logging.warning(f"scikit-image Random Walker failed: {e}, using fallback")

        # Fallback: gradient-weighted region growing
        return self._randomWalkerFallback(roi, localSeed, thresholds, params)

    def _randomWalkerSkimage(self, roi, localSeed, thresholds, beta, radius_voxels):
        """Scikit-image Random Walker implementation.

        Args:
            roi: ROI array (z, y, x).
            localSeed: Seed point in local coordinates (i, j, k).
            thresholds: Intensity thresholds.
            beta: Edge sensitivity parameter for random walker.
            radius_voxels: Radius in voxels per dimension.

        Returns:
            Binary mask array.
        """
        # Normalize ROI for better numerical stability
        roi_min = np.min(roi)
        roi_max = np.max(roi)
        roi_range = roi_max - roi_min
        if roi_range < 1e-6:
            return np.zeros_like(roi, dtype=np.uint8)
        roi_norm = (roi - roi_min) / roi_range

        # Create markers array: 0=unknown, 1=foreground, 2=background
        markers = np.zeros(roi.shape, dtype=np.int8)

        # Foreground marker: small sphere around seed
        shape = roi.shape  # (z, y, x)
        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
        dx = (x - localSeed[0]) / max(radius_voxels[0] * 0.15, 1)
        dy = (y - localSeed[1]) / max(radius_voxels[1] * 0.15, 1)
        dz = (z - localSeed[2]) / max(radius_voxels[2] * 0.15, 1)
        seed_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        markers[seed_dist <= 1.0] = 1  # Foreground seed region

        # Background marker: ring at outer edge of ROI
        # Create ring at ~90% of the ROI extent
        outer_ring_factor = 0.9
        dx_outer = (x - localSeed[0]) / (radius_voxels[0] * outer_ring_factor)
        dy_outer = (y - localSeed[1]) / (radius_voxels[1] * outer_ring_factor)
        dz_outer = (z - localSeed[2]) / (radius_voxels[2] * outer_ring_factor)
        outer_dist = np.sqrt(dx_outer**2 + dy_outer**2 + dz_outer**2)
        markers[(outer_dist >= 1.0) & (outer_dist <= 1.2)] = 2  # Background ring

        # Also mark regions outside intensity range as background
        intensity_mask = (roi >= thresholds["lower"]) & (roi <= thresholds["upper"])
        # Mark definitely-out-of-range voxels as background
        markers[~intensity_mask & (markers == 0)] = 2

        # Ensure we have both foreground and background markers
        if not np.any(markers == 1) or not np.any(markers == 2):
            logging.debug("Random Walker: insufficient markers, using fallback")
            return self._connectedThreshold(roi, localSeed, thresholds)

        # Run Random Walker
        # mode='cg' (conjugate gradient) is faster for larger images
        labels = skimage_random_walker(
            roi_norm,
            markers,
            beta=beta,
            mode="cg",
            return_full_prob=False,
        )

        # Extract foreground (label 1)
        mask = (labels == 1).astype(np.uint8)

        # Constrain to intensity range for safety
        mask = mask & intensity_mask.astype(np.uint8)

        # Ensure connectivity to seed
        if np.any(mask):
            sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))
            sitkMask = sitk.GetImageFromArray(mask)
            try:
                connected = sitk.ConnectedThreshold(
                    sitkMask, seedList=[sitkSeed], lower=1, upper=1, replaceValue=1
                )
                mask = sitk.GetArrayFromImage(connected).astype(np.uint8)
            except Exception:
                pass  # Keep original mask if connectivity check fails

        return mask

    def _randomWalkerFallback(self, roi, localSeed, thresholds, params):
        """Fallback Random Walker using gradient-weighted region growing.

        Used when scikit-image is not available.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates (i, j, k).
            thresholds: Intensity thresholds.
            params: Algorithm parameters.

        Returns:
            Binary mask array.
        """
        edge_sensitivity = params.get("edge_sensitivity", 0.5)

        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))
        sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))

        # Get seed intensity
        seed_intensity = roi[localSeed[2], localSeed[1], localSeed[0]]

        # Smooth the image based on edge sensitivity
        sigma = 0.5 + 1.5 * (1.0 - edge_sensitivity)  # 0.5-2.0
        smoothed = sitk.SmoothingRecursiveGaussian(sitkRoi, sigma=sigma)

        # Compute gradient on smoothed image
        gradient = sitk.GradientMagnitude(smoothed)
        gradArray = sitk.GetArrayFromImage(gradient)

        # Normalize gradient
        gradMax = np.percentile(gradArray, 95) + 1e-8
        gradNorm = gradArray / gradMax

        # Create edge-aware weight using beta parameter
        beta = 5.0 + 15.0 * edge_sensitivity  # 5-20
        edgeWeight = np.exp(-beta * gradNorm)

        # Create intensity similarity weight
        smoothedArray = sitk.GetArrayFromImage(smoothed)
        intensity_diff = np.abs(smoothedArray - seed_intensity)
        intensity_std = max(np.std(smoothedArray), 1.0)
        intensity_weight = np.exp(-intensity_diff / (intensity_std * 2))

        # Combined weight
        weight = edgeWeight * intensity_weight

        # Use connected threshold on weighted image
        weighted_roi = (roi * weight).astype(np.float32)
        sitkWeighted = sitk.GetImageFromArray(weighted_roi)

        weight_at_seed = weight[localSeed[2], localSeed[1], localSeed[0]]

        if weight_at_seed > 0.1:
            weighted_lower = thresholds["lower"] * weight_at_seed * 0.5
            weighted_upper = thresholds["upper"] * weight_at_seed * 1.5

            try:
                result = sitk.ConnectedThreshold(
                    sitkWeighted,
                    seedList=[sitkSeed],
                    lower=float(weighted_lower),
                    upper=float(weighted_upper),
                    replaceValue=1,
                )
                mask = sitk.GetArrayFromImage(result).astype(np.uint8)

                # Morphological cleanup
                maskSitk = sitk.GetImageFromArray(mask)
                maskSitk = sitk.BinaryMorphologicalClosing(maskSitk, [1, 1, 1])
                mask = sitk.GetArrayFromImage(maskSitk).astype(np.uint8)

                # Constrain by original intensity thresholds
                intensity_mask = (roi >= thresholds["lower"]) & (roi <= thresholds["upper"])
                mask = mask & intensity_mask.astype(np.uint8)

                return mask
            except Exception as e:
                logging.error(f"Random walker fallback failed: {e}")

        return self._connectedThreshold(roi, localSeed, thresholds)

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

    def _computeZoneThresholds(self, roi, localSeed, params):
        """Compute intensity thresholds from the inner threshold zone.

        Uses Gaussian distance weighting so points closer to the cursor have
        more influence on the threshold computation than points further away.

        Args:
            roi: ROI array (z, y, x).
            localSeed: Seed point in local coordinates (i, j, k).
            params: Parameters including threshold_zone, sampling_method, and
                    advanced parameters like gaussian_sigma, percentile_low, etc.

        Returns:
            Dict with 'lower' and 'upper' thresholds, or None to use default.
        """
        threshold_zone = params.get("threshold_zone", 0.5)
        sampling_method = params.get("sampling_method", "mean_std")
        radius_voxels = params.get("radius_voxels", [10, 10, 10])

        # Advanced parameters with defaults
        gaussian_sigma = params.get("gaussian_sigma", 0.5)
        percentile_low = params.get("percentile_low", 5)
        percentile_high = params.get("percentile_high", 95)
        std_multiplier = params.get("std_multiplier", 2.0)

        # Create coordinate grids for the ROI
        shape = roi.shape
        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

        # Zone radius is threshold_zone fraction of brush radius
        zone_radius = [r * threshold_zone for r in radius_voxels]

        # Compute normalized distance from seed (0 at center, 1 at zone boundary)
        dx = (x - localSeed[0]) / max(zone_radius[0], 0.1)
        dy = (y - localSeed[1]) / max(zone_radius[1], 0.1)
        dz = (z - localSeed[2]) / max(zone_radius[2], 0.1)
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        zone_mask = distance <= 1.0

        # Compute Gaussian weights based on distance from seed
        # sigma controls how quickly weight falls off with distance
        # weight = exp(-distance^2 / (2 * sigma^2))
        # At distance=0: weight=1.0, at distance=sigma: weight~0.61
        weights = np.exp(-(distance**2) / (2 * gaussian_sigma**2))
        weights = weights * zone_mask  # Zero outside zone

        # Flatten for weighted computations
        zone_intensities = roi[zone_mask].astype(np.float64)
        zone_weights = weights[zone_mask]

        if len(zone_intensities) < 5:
            return None  # Not enough samples, use default

        # Normalize weights to sum to 1
        weight_sum = np.sum(zone_weights)
        if weight_sum < 1e-10:
            return None
        normalized_weights = zone_weights / weight_sum

        # Compute thresholds based on sampling method
        if sampling_method == "mean_std":
            # Weighted mean and std
            weighted_mean = np.sum(zone_intensities * normalized_weights)
            weighted_variance = np.sum(
                normalized_weights * (zone_intensities - weighted_mean) ** 2
            )
            weighted_std = np.sqrt(weighted_variance)

            lower = weighted_mean - std_multiplier * weighted_std
            upper = weighted_mean + std_multiplier * weighted_std

        elif sampling_method == "percentile":
            # For weighted percentile, we use the weighted quantile algorithm
            # Sort by intensity
            sorted_indices = np.argsort(zone_intensities)
            sorted_intensities = zone_intensities[sorted_indices]
            sorted_weights = normalized_weights[sorted_indices]

            # Cumulative weights
            cumulative_weights = np.cumsum(sorted_weights)

            # Find intensities at specified percentiles
            low_target = percentile_low / 100.0
            high_target = percentile_high / 100.0

            # Find where cumulative weight crosses the target
            low_idx = np.searchsorted(cumulative_weights, low_target)
            high_idx = np.searchsorted(cumulative_weights, high_target)

            # Clamp indices
            low_idx = min(low_idx, len(sorted_intensities) - 1)
            high_idx = min(high_idx, len(sorted_intensities) - 1)

            lower = sorted_intensities[low_idx]
            upper = sorted_intensities[high_idx]

        elif sampling_method == "minmax":
            # For min/max, we use weighted extremes (values with significant weight)
            # Filter to significant weights (>1% of max weight)
            significant_mask = zone_weights > 0.01 * np.max(zone_weights)
            significant_intensities = roi[zone_mask][significant_mask]

            if len(significant_intensities) < 2:
                lower = np.min(zone_intensities)
                upper = np.max(zone_intensities)
            else:
                lower = np.min(significant_intensities)
                upper = np.max(significant_intensities)

        else:
            return None

        return {"lower": float(lower), "upper": float(upper)}

    def _createInnerZoneMask(self, shape, localSeed, params):
        """Create a mask for the inner threshold zone.

        This mask is OR'd with the algorithm result to guarantee the
        inner zone is always included in the segmentation.

        Args:
            shape: Shape of the ROI (z, y, x).
            localSeed: Seed point in local coordinates (i, j, k).
            params: Parameters including threshold_zone and radius_voxels.

        Returns:
            Binary mask array for inner zone.
        """
        threshold_zone = params.get("threshold_zone", 0.5)
        radius_voxels = params.get("radius_voxels", [10, 10, 10])

        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

        # Zone radius is threshold_zone fraction of brush radius
        zone_radius = [r * threshold_zone for r in radius_voxels]

        dx = (x - localSeed[0]) / max(zone_radius[0], 0.1)
        dy = (y - localSeed[1]) / max(zone_radius[1], 0.1)
        dz = (z - localSeed[2]) / max(zone_radius[2], 0.1)
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        return (distance <= 1.0).astype(np.uint8)

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
