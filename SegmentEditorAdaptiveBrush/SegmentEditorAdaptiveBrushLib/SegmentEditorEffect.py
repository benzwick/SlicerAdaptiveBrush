"""Adaptive Brush Segment Editor Effect.

This effect provides an adaptive brush that automatically segments regions
based on image intensity similarity, adapting to image features (edges,
boundaries) rather than using a fixed geometric shape.
"""

import logging
import os
import sys
import time
from typing import Any, Optional

import ctk
import numpy as np  # noqa: E402
import qt  # noqa: E402
import slicer  # noqa: E402
import vtk  # noqa: E402
import vtk.util.numpy_support  # noqa: E402
from SegmentEditorEffects import AbstractScriptedSegmentEditorEffect  # noqa: E402
from slicer.i18n import tr as _  # noqa: E402

# Add parent directory to path for imports when loaded by Slicer
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Import algorithm components (use non-relative imports for Slicer compatibility)
from DependencyManager import dependency_manager  # noqa: E402
from IntensityAnalyzer import IntensityAnalyzer  # noqa: E402
from PerformanceCache import PerformanceCache  # noqa: E402

# Try to import SimpleITK (should be available in Slicer)
try:
    import SimpleITK as sitk  # noqa: E402
    import sitkUtils  # noqa: E402, F401 - for future use with volume push/pull

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    logging.warning("SimpleITK not available - some features will be disabled")

# Check initial scikit-image availability for Random Walker (without prompting)
HAS_SKIMAGE_RW = dependency_manager.is_available("skimage")
skimage_random_walker = None  # type: ignore[no-redef]

if HAS_SKIMAGE_RW:
    from skimage.segmentation import (  # type: ignore[no-redef]
        random_walker as skimage_random_walker,
    )
else:
    logging.info(
        "scikit-image not available - Random Walker will use fallback algorithm. "
        "User will be prompted to install when selecting the algorithm."
    )


def _ensure_random_walker_available() -> bool:
    """Prompt to install scikit-image if not available for Random Walker.

    Returns:
        True if skimage is now available, False otherwise
    """
    global HAS_SKIMAGE_RW, skimage_random_walker

    if HAS_SKIMAGE_RW:
        return True

    if dependency_manager.ensure_available("skimage"):
        from skimage.segmentation import random_walker as rw

        skimage_random_walker = rw
        HAS_SKIMAGE_RW = True
        return True

    return False


class BrushOutlinePipeline:
    """VTK pipeline for brush outline visualization in a slice view.

    Shows two circle outlines:
    - Outer circle (yellow): Maximum brush extent
    - Inner circle (cyan): Threshold sampling zone
    - Preview overlay (green, semi-transparent): Segmentation preview
    - Crosshair (optional): Center crosshair lines
    """

    # Crosshair style constants
    CROSSHAIR_STYLE_CROSS = "cross"  # Simple + shape
    CROSSHAIR_STYLE_BULLSEYE = "bullseye"  # + with gap at center
    CROSSHAIR_STYLE_DOT = "dot"  # Small dot at center
    CROSSHAIR_STYLE_CROSSHAIR = "crosshair"  # Full lines extending to brush edge

    def __init__(self):
        """Initialize the brush outline pipeline."""
        # Crosshair settings
        self.crosshairEnabled = True
        self.crosshairSize = 10  # Size in pixels (for cross/bullseye styles)
        self.crosshairThickness = 1  # Line width
        self.crosshairStyle = self.CROSSHAIR_STYLE_CROSS
        self.crosshairColor = (1.0, 1.0, 1.0)  # White by default
        self.crosshairOpacity = 0.9

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

        # CROSSHAIR - Center indicator lines
        self.crosshairPoints = vtk.vtkPoints()
        self.crosshairLines = vtk.vtkCellArray()
        self.crosshairPolyData = vtk.vtkPolyData()
        self.crosshairPolyData.SetPoints(self.crosshairPoints)
        self.crosshairPolyData.SetLines(self.crosshairLines)

        self.crosshairMapper = vtk.vtkPolyDataMapper2D()
        self.crosshairMapper.SetInputData(self.crosshairPolyData)

        self.crosshairActor = vtk.vtkActor2D()
        self.crosshairActor.SetMapper(self.crosshairMapper)
        self.crosshairActor.VisibilityOff()
        self.crosshairActor.SetPickable(False)

        # Crosshair styling - white by default
        crosshairProp = self.crosshairActor.GetProperty()
        crosshairProp.SetColor(1.0, 1.0, 1.0)
        crosshairProp.SetLineWidth(1)
        crosshairProp.SetOpacity(0.9)

        # CENTER DOT - for dot style
        self.dotSource = vtk.vtkRegularPolygonSource()
        self.dotSource.SetNumberOfSides(16)
        self.dotSource.SetRadius(2.0)
        self.dotSource.GeneratePolygonOn()
        self.dotSource.GeneratePolylineOff()

        self.dotTransform = vtk.vtkTransform()
        self.dotTransformFilter = vtk.vtkTransformPolyDataFilter()
        self.dotTransformFilter.SetTransform(self.dotTransform)
        self.dotTransformFilter.SetInputConnection(self.dotSource.GetOutputPort())

        self.dotMapper = vtk.vtkPolyDataMapper2D()
        self.dotMapper.SetInputConnection(self.dotTransformFilter.GetOutputPort())

        self.dotActor = vtk.vtkActor2D()
        self.dotActor.SetMapper(self.dotMapper)
        self.dotActor.VisibilityOff()
        self.dotActor.SetPickable(False)

        dotProp = self.dotActor.GetProperty()
        dotProp.SetColor(1.0, 1.0, 1.0)
        dotProp.SetOpacity(0.9)

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
            self.renderer.RemoveActor2D(self.crosshairActor)
            self.renderer.RemoveActor2D(self.dotActor)

        self.sliceWidget = sliceWidget
        if sliceWidget is not None:
            self.renderer = sliceWidget.sliceView().renderWindow().GetRenderers().GetFirstRenderer()
            if self.renderer is not None:
                # Add preview first so it's behind the outline circles
                self.renderer.AddActor2D(self.previewActor)
                self.renderer.AddActor2D(self.outerActor)
                self.renderer.AddActor2D(self.innerActor)
                self.renderer.AddActor2D(self.crosshairActor)
                self.renderer.AddActor2D(self.dotActor)

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

        # Update crosshair
        self._updateCrosshair(xyPosition, radiusPixels)

        # Request render
        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def _updateCrosshair(self, xyPosition, radiusPixels):
        """Update the crosshair visualization.

        Args:
            xyPosition: Center position in slice XY coordinates (x, y).
            radiusPixels: Brush radius in pixels (used for crosshair style).
        """
        if not self.crosshairEnabled:
            self.crosshairActor.VisibilityOff()
            self.dotActor.VisibilityOff()
            return

        cx, cy = xyPosition[0], xyPosition[1]

        # Update crosshair color and style
        crosshairProp = self.crosshairActor.GetProperty()
        crosshairProp.SetColor(*self.crosshairColor)
        crosshairProp.SetLineWidth(self.crosshairThickness)
        crosshairProp.SetOpacity(self.crosshairOpacity)

        dotProp = self.dotActor.GetProperty()
        dotProp.SetColor(*self.crosshairColor)
        dotProp.SetOpacity(self.crosshairOpacity)

        # Build crosshair geometry based on style
        self.crosshairPoints.Reset()
        self.crosshairLines.Reset()

        if self.crosshairStyle == self.CROSSHAIR_STYLE_DOT:
            # Just show a dot at center
            self.crosshairActor.VisibilityOff()
            self.dotSource.SetRadius(max(2, self.crosshairSize / 5))
            self.dotTransform.Identity()
            self.dotTransform.Translate(cx, cy, 0)
            self.dotActor.VisibilityOn()

        elif self.crosshairStyle == self.CROSSHAIR_STYLE_CROSS:
            # Simple + shape
            size = self.crosshairSize
            self._addCrossLines(cx, cy, size, gap=0)
            self.crosshairActor.VisibilityOn()
            self.dotActor.VisibilityOff()

        elif self.crosshairStyle == self.CROSSHAIR_STYLE_BULLSEYE:
            # + with gap at center
            size = self.crosshairSize
            gap = max(3, size / 4)
            self._addCrossLines(cx, cy, size, gap=gap)
            self.crosshairActor.VisibilityOn()
            # Also show small dot at center
            self.dotSource.SetRadius(1.5)
            self.dotTransform.Identity()
            self.dotTransform.Translate(cx, cy, 0)
            self.dotActor.VisibilityOn()

        elif self.crosshairStyle == self.CROSSHAIR_STYLE_CROSSHAIR:
            # Full lines extending to brush edge
            size = radiusPixels * 0.9  # Slightly inside the brush circle
            gap = max(3, radiusPixels * 0.1)
            self._addCrossLines(cx, cy, size, gap=gap)
            self.crosshairActor.VisibilityOn()
            self.dotActor.VisibilityOff()

        self.crosshairPolyData.Modified()

    def _addCrossLines(self, cx, cy, size, gap=0):
        """Add cross lines to the crosshair polydata.

        Args:
            cx, cy: Center position.
            size: Half-length of each arm.
            gap: Gap size at center (0 for solid cross).
        """
        # Horizontal line
        if gap > 0:
            # Left segment
            p0 = self.crosshairPoints.InsertNextPoint(cx - size, cy, 0)
            p1 = self.crosshairPoints.InsertNextPoint(cx - gap, cy, 0)
            self.crosshairLines.InsertNextCell(2)
            self.crosshairLines.InsertCellPoint(p0)
            self.crosshairLines.InsertCellPoint(p1)
            # Right segment
            p2 = self.crosshairPoints.InsertNextPoint(cx + gap, cy, 0)
            p3 = self.crosshairPoints.InsertNextPoint(cx + size, cy, 0)
            self.crosshairLines.InsertNextCell(2)
            self.crosshairLines.InsertCellPoint(p2)
            self.crosshairLines.InsertCellPoint(p3)
        else:
            p0 = self.crosshairPoints.InsertNextPoint(cx - size, cy, 0)
            p1 = self.crosshairPoints.InsertNextPoint(cx + size, cy, 0)
            self.crosshairLines.InsertNextCell(2)
            self.crosshairLines.InsertCellPoint(p0)
            self.crosshairLines.InsertCellPoint(p1)

        # Vertical line
        if gap > 0:
            # Bottom segment
            p4 = self.crosshairPoints.InsertNextPoint(cx, cy - size, 0)
            p5 = self.crosshairPoints.InsertNextPoint(cx, cy - gap, 0)
            self.crosshairLines.InsertNextCell(2)
            self.crosshairLines.InsertCellPoint(p4)
            self.crosshairLines.InsertCellPoint(p5)
            # Top segment
            p6 = self.crosshairPoints.InsertNextPoint(cx, cy + gap, 0)
            p7 = self.crosshairPoints.InsertNextPoint(cx, cy + size, 0)
            self.crosshairLines.InsertNextCell(2)
            self.crosshairLines.InsertCellPoint(p6)
            self.crosshairLines.InsertCellPoint(p7)
        else:
            p4 = self.crosshairPoints.InsertNextPoint(cx, cy - size, 0)
            p5 = self.crosshairPoints.InsertNextPoint(cx, cy + size, 0)
            self.crosshairLines.InsertNextCell(2)
            self.crosshairLines.InsertCellPoint(p4)
            self.crosshairLines.InsertCellPoint(p5)

    def setCrosshairSettings(self, enabled=None, size=None, thickness=None, style=None, color=None):
        """Update crosshair settings.

        Args:
            enabled: Whether crosshair is shown.
            size: Size in pixels.
            thickness: Line width.
            style: One of CROSSHAIR_STYLE_* constants.
            color: RGB tuple (0-1 range).
        """
        if enabled is not None:
            self.crosshairEnabled = enabled
        if size is not None:
            self.crosshairSize = size
        if thickness is not None:
            self.crosshairThickness = thickness
        if style is not None:
            self.crosshairStyle = style
        if color is not None:
            self.crosshairColor = color

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
        self.crosshairActor.VisibilityOff()
        self.dotActor.VisibilityOff()
        if self.sliceWidget is not None:
            self.sliceWidget.sliceView().scheduleRender()

    def cleanup(self):
        """Remove the actors from the renderer and clean up."""
        if self.renderer is not None:
            self.renderer.RemoveActor2D(self.outerActor)
            self.renderer.RemoveActor2D(self.innerActor)
            self.renderer.RemoveActor2D(self.previewActor)
            self.renderer.RemoveActor2D(self.crosshairActor)
            self.renderer.RemoveActor2D(self.dotActor)
            self.renderer = None
        self.sliceWidget = None


class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
    """Adaptive brush segment editor effect.

    This effect provides an adaptive brush that segments based on image intensity.
    """

    # Brush outline colors for add mode (yellow outer, cyan inner)
    BRUSH_COLOR_ADD = (1.0, 0.9, 0.1)
    BRUSH_COLOR_ADD_INNER = (0.2, 0.9, 1.0)

    # Brush outline colors for erase mode (red/orange outer, lighter inner)
    BRUSH_COLOR_ERASE = (1.0, 0.3, 0.1)
    BRUSH_COLOR_ERASE_INNER = (1.0, 0.5, 0.3)

    def __init__(self, scriptedEffect):
        """Initialize the effect.

        Args:
            scriptedEffect: The scripted effect instance from Slicer.
        """
        scriptedEffect.name = "Adaptive Brush"
        scriptedEffect.perSegment = True
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

        # Algorithm components
        self.intensityAnalyzer = IntensityAnalyzer()
        self.cache = PerformanceCache()

        # State
        self.isDrawing = False
        self.lastIjk = None
        self.eraseMode = False  # True = erase, False = add
        self._currentStrokeEraseMode = False  # Locked mode for current stroke
        self._isMiddleButtonHeld = False  # Track middle button for erase modifier
        self._updatingThresholdRanges = False  # Reentrancy guard for threshold updates

        # Default parameters - Basic
        # Note: Some defaults optimized for brain MRI tumor (MRBrainTumor1).
        # May need adjustment for other anatomy/modalities.
        self.radiusMm = 5.0
        self.edgeSensitivity = 40  # Optimized: 35, kept conservative
        self.thresholdZone = 50  # Inner zone is 50% of brush radius
        self.samplingMethod = "mean_std"  # How to compute thresholds from zone
        self.algorithm = "watershed"  # Best performer in optimization
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
        # Optimized on brain MRI tumor (Dice 0.9998)
        # Gradient scale 1.8: stronger edge emphasis
        self.watershedGradientScale = 1.8
        # Smoothing 0.6: moderate smoothing for noise reduction
        self.watershedSmoothing = 0.6

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
        self.outlinePipelines: dict[str, BrushOutlinePipeline] = {}
        self.activeViewWidget = None

        # Track source volume for detecting changes
        self._lastSourceVolumeId = None

        # Parameter presets by modality and tissue type
        # These set common parameters (edge sensitivity, threshold zone, etc.)
        # Algorithm selection is separate - user picks algorithm, then picks preset
        # For algorithm recommendations, see user documentation or use the wizard
        #
        # Preset parameters:
        #   edge_sensitivity: 0-100, higher = more edge-following
        #   threshold_zone: 0-100, higher = wider intensity acceptance
        #   sampling_method: "mean_std", "percentile", or "histogram"
        #   gaussian_sigma: 0.1-2.0, smoothing before segmentation
        #   std_multiplier: 1.0-4.0, threshold width multiplier
        #   fill_holes: True/False, fill holes in result
        #   closing_radius: 0-3, morphological closing radius
        self._presets = {
            # ================================================================
            # DEFAULT / BALANCED
            # ================================================================
            "default": {
                "name": "Default",
                "description": "Balanced settings for general use",
                "edge_sensitivity": 50,
                "threshold_zone": 50,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # CT PRESETS
            # ================================================================
            "ct_bone": {
                "name": "CT - Bone",
                "description": "High contrast bone (cortical and trabecular)",
                "algorithm": "watershed",
                "edge_sensitivity": 70,
                "threshold_zone": 35,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.25,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "ct_soft_tissue": {
                "name": "CT - Soft Tissue/Organ",
                "description": "Organs (liver, kidney, spleen, pancreas)",
                "algorithm": "geodesic_distance",
                "edge_sensitivity": 50,
                "threshold_zone": 55,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "ct_lung": {
                "name": "CT - Lung Parenchyma",
                "description": "Lung tissue (preserves airways)",
                "algorithm": "region_growing",
                "edge_sensitivity": 55,
                "threshold_zone": 45,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.35,
                "std_multiplier": 2.0,
                "fill_holes": False,
                "closing_radius": 0,
            },
            "ct_airway": {
                "name": "CT - Airway",
                "description": "Trachea and bronchi (tubular low-density)",
                "edge_sensitivity": 60,
                "threshold_zone": 40,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.3,
                "std_multiplier": 2.0,
                "fill_holes": False,
                "closing_radius": 0,
            },
            "ct_vessel_contrast": {
                "name": "CT - Vessels (CTA)",
                "description": "Contrast-enhanced vessels",
                "algorithm": "connected_threshold",
                "edge_sensitivity": 65,
                "threshold_zone": 35,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.25,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "ct_muscle": {
                "name": "CT - Muscle",
                "description": "Skeletal muscle tissue",
                "edge_sensitivity": 45,
                "threshold_zone": 60,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "ct_fat": {
                "name": "CT - Fat/Adipose",
                "description": "Subcutaneous and visceral fat",
                "edge_sensitivity": 55,
                "threshold_zone": 50,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.4,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "ct_lymph_node": {
                "name": "CT - Lymph Node",
                "description": "Lymph nodes (with or without contrast)",
                "edge_sensitivity": 50,
                "threshold_zone": 55,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.4,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # MRI T1 PRESETS
            # ================================================================
            "mri_t1_brain_gm": {
                "name": "MRI T1 - Gray Matter",
                "description": "Cortical gray matter in T1",
                "edge_sensitivity": 55,
                "threshold_zone": 50,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "mri_t1_brain_wm": {
                "name": "MRI T1 - White Matter",
                "description": "Deep white matter in T1",
                "edge_sensitivity": 50,
                "threshold_zone": 55,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "mri_t1_fat": {
                "name": "MRI T1 - Fat/Adipose",
                "description": "Adipose tissue (bright in T1)",
                "edge_sensitivity": 40,
                "threshold_zone": 65,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.55,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "mri_t1_muscle": {
                "name": "MRI T1 - Muscle",
                "description": "Skeletal muscle in T1",
                "edge_sensitivity": 45,
                "threshold_zone": 60,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            # ================================================================
            # MRI T1+Gd (CONTRAST) PRESETS
            # ================================================================
            "mri_t1gd_tumor": {
                "name": "MRI T1+Gd - Enhancing Tumor",
                "description": "Contrast-enhancing tumors",
                "algorithm": "watershed",
                "edge_sensitivity": 45,
                "threshold_zone": 60,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.55,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "mri_t1gd_vessel": {
                "name": "MRI T1+Gd - Vessel (MRA)",
                "description": "Contrast-enhanced vessels in MRA",
                "edge_sensitivity": 60,
                "threshold_zone": 40,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.35,
                "std_multiplier": 1.8,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # MRI T2/FLAIR PRESETS
            # ================================================================
            "mri_t2_lesion": {
                "name": "MRI T2 - Lesion/Edema",
                "description": "Hyperintense lesions in T2",
                "algorithm": "threshold_brush",
                "edge_sensitivity": 50,
                "threshold_zone": 55,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "mri_flair_lesion": {
                "name": "MRI FLAIR - WM Lesion",
                "description": "White matter lesions (MS, ischemia)",
                "edge_sensitivity": 55,
                "threshold_zone": 50,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.4,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "mri_t2_csf": {
                "name": "MRI T2 - CSF/Fluid",
                "description": "Cerebrospinal fluid (very bright in T2)",
                "edge_sensitivity": 60,
                "threshold_zone": 40,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.35,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # MRI DWI PRESETS
            # ================================================================
            "mri_dwi_stroke": {
                "name": "MRI DWI - Acute Stroke",
                "description": "Restricted diffusion in acute stroke",
                "edge_sensitivity": 55,
                "threshold_zone": 55,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.7,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 1,
            },
            # ================================================================
            # PET PRESETS
            # ================================================================
            "pet_tumor": {
                "name": "PET - Hypermetabolic Tumor",
                "description": "FDG-avid tumors and metastases",
                "algorithm": "connected_threshold",
                "edge_sensitivity": 50,
                "threshold_zone": 55,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.6,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "pet_suv_threshold": {
                "name": "PET - SUV-based",
                "description": "SUV threshold-based segmentation",
                "edge_sensitivity": 45,
                "threshold_zone": 45,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.5,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # ULTRASOUND PRESETS
            # ================================================================
            "us_general": {
                "name": "Ultrasound - General",
                "description": "General ultrasound (noisy, speckle)",
                "edge_sensitivity": 55,
                "threshold_zone": 65,
                "sampling_method": "percentile",
                "gaussian_sigma": 1.0,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "us_cyst": {
                "name": "Ultrasound - Cyst/Fluid",
                "description": "Anechoic cysts and fluid collections",
                "edge_sensitivity": 50,
                "threshold_zone": 60,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.8,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # CBCT PRESETS
            # ================================================================
            "cbct_bone": {
                "name": "CBCT - Bone/Dental",
                "description": "Bone and teeth in cone-beam CT",
                "edge_sensitivity": 65,
                "threshold_zone": 40,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.4,
                "std_multiplier": 1.8,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # MICROSCOPY PRESETS
            # ================================================================
            "micro_cell": {
                "name": "Microscopy - Cell",
                "description": "Individual cells in microscopy",
                "edge_sensitivity": 60,
                "threshold_zone": 45,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.3,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "micro_nucleus": {
                "name": "Microscopy - Nucleus",
                "description": "Cell nuclei (often DAPI stained)",
                "edge_sensitivity": 65,
                "threshold_zone": 40,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.25,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
            # ================================================================
            # GENERIC / MULTI-MODALITY PRESETS
            # ================================================================
            "generic_tumor": {
                "name": "Generic - Tumor/Mass",
                "description": "Tumors with irregular boundaries",
                "edge_sensitivity": 45,
                "threshold_zone": 60,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.55,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "generic_vessel": {
                "name": "Generic - Vessels",
                "description": "Blood vessels and tubular structures",
                "edge_sensitivity": 65,
                "threshold_zone": 35,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.3,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "generic_organ": {
                "name": "Generic - Organ",
                "description": "Solid organs with smooth boundaries",
                "edge_sensitivity": 50,
                "threshold_zone": 55,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.5,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 1,
            },
            "generic_lesion": {
                "name": "Generic - Small Lesion",
                "description": "Small focal lesions",
                "edge_sensitivity": 55,
                "threshold_zone": 50,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.4,
                "std_multiplier": 2.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "generic_high_contrast": {
                "name": "Generic - High Contrast",
                "description": "Structures with strong edges",
                "edge_sensitivity": 70,
                "threshold_zone": 35,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.3,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "generic_low_contrast": {
                "name": "Generic - Low Contrast",
                "description": "Subtle intensity differences",
                "edge_sensitivity": 40,
                "threshold_zone": 65,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.6,
                "std_multiplier": 2.5,
                "fill_holes": True,
                "closing_radius": 1,
            },
            # ================================================================
            # SPEED / PRECISION PRESETS
            # ================================================================
            "speed_fast": {
                "name": "Speed - Fast Rough",
                "description": "Quick approximate segmentation",
                "edge_sensitivity": 40,
                "threshold_zone": 70,
                "sampling_method": "mean_std",
                "gaussian_sigma": 0.3,
                "std_multiplier": 3.0,
                "fill_holes": True,
                "closing_radius": 0,
            },
            "precision_high": {
                "name": "Precision - Careful Edges",
                "description": "Maximum boundary accuracy",
                "edge_sensitivity": 75,
                "threshold_zone": 35,
                "sampling_method": "percentile",
                "gaussian_sigma": 0.25,
                "std_multiplier": 1.5,
                "fill_holes": True,
                "closing_radius": 0,
            },
        }
        self._currentPreset = "default"

    def __repr__(self):
        """Return string representation to prevent default recursion."""
        return "<SegmentEditorEffect 'Adaptive Brush'>"

    def __str__(self):
        """Return string representation to prevent default recursion."""
        return "SegmentEditorEffect(Adaptive Brush)"

    # -------------------------------------------------------------------------
    # Action Logging for Test Replay
    # -------------------------------------------------------------------------
    # These methods enable structured logging of user actions for:
    # - Generating test code from manual sessions
    # - Debugging algorithm behavior
    # - Reproducing issues reported by users

    def _log_action(self, action_type: str, **kwargs) -> None:
        """Log an action in structured format for test generation.

        Args:
            action_type: Type of action (e.g., "algorithm_changed", "paint_stroke").
            **kwargs: Action-specific parameters.
        """
        import json
        from datetime import datetime

        action = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "params": kwargs,
            "state": self._get_state_snapshot(),
        }

        # Log to dedicated action logger for easy filtering
        action_logger = logging.getLogger("AdaptiveBrush.Actions")
        action_logger.info(f"ACTION: {json.dumps(action)}")

    def _get_state_snapshot(self) -> dict:
        """Capture current effect state for action logging.

        Returns:
            Dictionary with all effect parameters.
        """
        return {
            # Basic parameters
            "algorithm": self.algorithm,
            "radius_mm": self.radiusMm,
            "edge_sensitivity": self.edgeSensitivity,
            "threshold_zone": self.thresholdZone,
            "sphere_mode": self.sphereMode,
            "erase_mode": self.eraseMode,
            "sampling_method": self.samplingMethod,
            "backend": self.backend,
            # Sampling parameters
            "gaussian_sigma": self.gaussianSigma,
            "percentile_low": self.percentileLow,
            "percentile_high": self.percentileHigh,
            "std_multiplier": self.stdMultiplier,
            "include_zone_in_result": self.includeZoneInResult,
            # Geodesic distance parameters
            "geodesic_edge_weight": self.geodesicEdgeWeight,
            "geodesic_distance_scale": self.geodesicDistanceScale,
            "geodesic_smoothing": self.geodesicSmoothing,
            # Watershed parameters
            "watershed_gradient_scale": self.watershedGradientScale,
            "watershed_smoothing": self.watershedSmoothing,
            # Level set parameters
            "level_set_propagation": self.levelSetPropagation,
            "level_set_curvature": self.levelSetCurvature,
            "level_set_iterations": self.levelSetIterations,
            # Region growing parameters
            "region_growing_multiplier": self.regionGrowingMultiplier,
            "region_growing_iterations": self.regionGrowingIterations,
            # Random walker parameters
            "random_walker_beta": self.randomWalkerBeta,
            # Morphology parameters
            "fill_holes": self.fillHoles,
            "closing_radius": self.closingRadius,
        }

    def _get_full_state(self) -> dict:
        """Capture complete effect state including UI state.

        Returns:
            Dictionary with all effect parameters and UI state.
        """
        state = self._get_state_snapshot()
        state.update(
            {
                "is_drawing": self.isDrawing,
                "last_ijk": self.lastIjk,
                "current_stroke_erase_mode": self._currentStrokeEraseMode,
                "preview_mode": self.previewMode,
                "use_threshold_caching": self.useThresholdCaching,
            }
        )
        return state

    def register(self) -> None:
        """Register the effect with the segment editor effect factory.

        This method is copied from AbstractScriptedSegmentEditorEffect since
        we don't inherit from it (to avoid recursion issues).
        """
        import slicer

        effectFactorySingleton = slicer.qSlicerSegmentEditorEffectFactory.instance()
        effectFactorySingleton.registerEffect(self.scriptedEffect)

    def clone(self) -> Any:
        """Create a copy of this effect.

        Returns:
            New effect instance.
        """
        import qSlicerSegmentationsEditorEffectsPythonQt as effects

        clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
        clonedEffect.setPythonSource(__file__)
        return clonedEffect

    def icon(self) -> Any:
        """Return the effect icon.

        Returns:
            QIcon for the effect toolbar button.
        """
        # Icon is in the same directory as the effect file (extension pattern)
        iconPath = os.path.join(os.path.dirname(__file__), "SegmentEditorEffect.png")
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    # Note: createCursor is NOT overridden here - the default C++ implementation
    # in qSlicerSegmentEditorAbstractEffect::createCursor is used, which creates
    # a cursor combining NullEffect.png with our icon() result

    def helpText(self) -> str:
        """Return help text for the effect.

        Returns:
            HTML string with usage instructions.
            First line before <br>. shown as collapsed summary.
        """
        return "<html>" + str(
            _(
                """Paint with a brush that adapts to image intensity boundaries<br>.
Left-click and drag to paint. Ctrl+click or Middle+click to invert mode. Shift+scroll to adjust brush size.<p>
<b>Brush Circles:</b>
<ul style="margin: 0">
<li><b>Yellow (outer)</b>: Maximum brush extent
<li><b>Cyan (inner)</b>: Threshold sampling zone
</ul><p>
<b>Algorithms:</b>
<ul style="margin: 0">
<li><b>Geodesic Distance</b>: Fast, follows edges naturally (recommended)
<li><b>Watershed</b>: Good edge following, may be blocky on noisy images
<li><b>Random Walker</b>: Excellent for ambiguous/blurry edges
<li><b>Level Set</b>: Smooth contours, slower but precise
<li><b>Region Growing</b>: Fast, good for homogeneous regions
<li><b>Threshold Brush</b>: Simple intensity thresholding
</ul>"""
            )
        )

    def setupOptionsFrame(self):
        """Create the effect options UI."""
        # ----- Brush Settings -----
        self.brushCollapsible = ctk.ctkCollapsibleButton()
        self.brushCollapsible.text = _("Brush Settings")
        brushCollapsible = self.brushCollapsible  # Local alias for compatibility
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

        # Algorithm dropdown (most important choice - visible immediately)
        self.algorithmCombo = qt.QComboBox()
        self.algorithmCombo.addItem(_("Geodesic Distance (Recommended)"), "geodesic_distance")
        self.algorithmCombo.addItem(_("Watershed"), "watershed")
        self.algorithmCombo.addItem(_("Random Walker"), "random_walker")
        self.algorithmCombo.addItem(_("Level Set"), "level_set")
        self.algorithmCombo.addItem(_("Connected Threshold (Fast)"), "connected_threshold")
        self.algorithmCombo.addItem(_("Region Growing"), "region_growing")
        self.algorithmCombo.addItem(_("Threshold Brush (Simple)"), "threshold_brush")
        self.algorithmCombo.setToolTip(_("Segmentation algorithm to use"))
        brushLayout.addRow(_("Algorithm:"), self.algorithmCombo)

        # Quick Select Parameters Wizard button
        self.wizardButton = qt.QPushButton(_("Quick Select Parameters..."))
        self.wizardButton.setToolTip(
            _(
                "Launch an interactive wizard to help determine optimal\n"
                "algorithm and parameters for your specific image and target.\n\n"
                "The wizard will guide you through:\n"
                "1. Sampling the target structure (foreground)\n"
                "2. Sampling the surrounding area (background)\n"
                "3. Optionally tracing the boundary\n"
                "4. Answering questions about the imaging type\n"
                "5. Reviewing recommended parameters"
            )
        )
        self.wizardButton.clicked.connect(self.onWizardClicked)
        brushLayout.addRow(self.wizardButton)

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

        # Threshold zone slider (inner circle for sampling) - before edge sensitivity
        self.zoneSlider = ctk.ctkSliderWidget()
        self.zoneSlider.setToolTip(
            _(
                "Size of inner zone (cyan circle) as percentage of brush radius.\n"
                "Intensities are sampled from this zone to compute thresholds.\n\n"
                "Larger zone = more samples = more robust thresholds.\n"
                "Smaller zone = samples closer to cursor = more precise."
            )
        )
        self.zoneSlider.decimals = 0
        self.zoneSlider.minimum = 10
        self.zoneSlider.maximum = 100
        self.zoneSlider.value = self.thresholdZone
        self.zoneSlider.singleStep = 5
        self.zoneSlider.suffix = "%"
        brushLayout.addRow(_("Threshold Zone:"), self.zoneSlider)

        # Edge sensitivity slider
        self.sensitivitySlider = ctk.ctkSliderWidget()
        self.sensitivitySlider.setToolTip(
            _("How strictly to follow intensity boundaries (0=permissive, 100=strict)")
        )
        self.sensitivitySlider.decimals = 0
        self.sensitivitySlider.minimum = 0
        self.sensitivitySlider.maximum = 100
        self.sensitivitySlider.value = self.edgeSensitivity
        self.sensitivitySlider.singleStep = 5
        self.sensitivitySlider.suffix = "%"
        brushLayout.addRow(_("Edge Sensitivity:"), self.sensitivitySlider)

        # Include zone checkbox (grouped with threshold zone)
        self.includeZoneCheckbox = qt.QCheckBox(_("Guarantee inner zone in result"))
        self.includeZoneCheckbox.setToolTip(
            _(
                "Guarantees that clicking always paints at least the inner zone (cyan circle).\n\n"
                "• OFF (default): Only paint what the algorithm detects as similar tissue.\n"
                "  Best for accurate boundaries.\n\n"
                "• ON: Force-paint the inner zone even if the algorithm finds nothing.\n"
                "  Use if clicks sometimes produce no result on valid tissue."
            )
        )
        self.includeZoneCheckbox.checked = self.includeZoneInResult
        brushLayout.addRow(self.includeZoneCheckbox)

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

        # Mode selection (Add/Erase)
        modeLayout = qt.QHBoxLayout()
        modeLabel = qt.QLabel(_("Mode:"))
        self.addModeRadio = qt.QRadioButton(_("Add"))
        self.eraseModeRadio = qt.QRadioButton(_("Erase"))
        self.addModeRadio.setChecked(True)
        self.addModeRadio.setToolTip(
            _("Paint to add to segment (Ctrl+click or Middle+click to temporarily erase)")
        )
        self.eraseModeRadio.setToolTip(
            _("Paint to erase from segment (Ctrl+click or Middle+click to temporarily add)")
        )
        modeLayout.addWidget(modeLabel)
        modeLayout.addWidget(self.addModeRadio)
        modeLayout.addWidget(self.eraseModeRadio)
        modeLayout.addStretch()
        brushLayout.addRow(modeLayout)

        # Connect mode signals
        self.addModeRadio.toggled.connect(self.onModeChanged)

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

        # ----- Algorithm Parameters (dynamic section) -----
        # Title updates based on selected algorithm
        self.algorithmParamsCollapsible = ctk.ctkCollapsibleButton()
        self.algorithmParamsCollapsible.text = _("Geodesic Distance Parameters")
        self.algorithmParamsCollapsible.collapsed = True
        self.scriptedEffect.addOptionsWidget(self.algorithmParamsCollapsible)
        self.algorithmParamsLayout = qt.QFormLayout(self.algorithmParamsCollapsible)

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

        self.algorithmParamsLayout.addRow(self.thresholdGroup)
        self.thresholdGroup.setVisible(False)  # Hidden unless Threshold Brush selected

        # --- Geodesic Distance Parameters (in Algorithm Parameters section) ---
        self.geodesicParamsGroup = qt.QWidget()
        geodesicLayout = qt.QFormLayout(self.geodesicParamsGroup)
        geodesicLayout.setContentsMargins(0, 0, 0, 0)

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
        geodesicLayout.addRow(_("Edge Weight:"), self.geodesicEdgeWeightSlider)

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
        geodesicLayout.addRow(_("Distance Scale:"), self.geodesicDistanceScaleSlider)

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
        geodesicLayout.addRow(_("Smoothing:"), self.geodesicSmoothingSlider)

        self.algorithmParamsLayout.addRow(self.geodesicParamsGroup)

        # --- Watershed Parameters (in Algorithm Parameters section) ---
        self.watershedParamsGroup = qt.QWidget()
        watershedLayout = qt.QFormLayout(self.watershedParamsGroup)
        watershedLayout.setContentsMargins(0, 0, 0, 0)

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
        watershedLayout.addRow(_("Gradient Scale:"), self.watershedGradientScaleSlider)

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
        watershedLayout.addRow(_("Smoothing:"), self.watershedSmoothingSlider)

        self.algorithmParamsLayout.addRow(self.watershedParamsGroup)
        self.watershedParamsGroup.setVisible(False)

        # --- Level Set Parameters (in Algorithm Parameters section) ---
        self.levelSetParamsGroup = qt.QWidget()
        levelSetLayout = qt.QFormLayout(self.levelSetParamsGroup)
        levelSetLayout.setContentsMargins(0, 0, 0, 0)

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
        levelSetLayout.addRow(_("Propagation:"), self.levelSetPropagationSlider)

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
        levelSetLayout.addRow(_("Curvature:"), self.levelSetCurvatureSlider)

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
        levelSetLayout.addRow(_("Iterations:"), self.levelSetIterationsSlider)

        self.algorithmParamsLayout.addRow(self.levelSetParamsGroup)
        self.levelSetParamsGroup.setVisible(False)

        # --- Region Growing Parameters (in Algorithm Parameters section) ---
        self.regionGrowingParamsGroup = qt.QWidget()
        regionGrowingLayout = qt.QFormLayout(self.regionGrowingParamsGroup)
        regionGrowingLayout.setContentsMargins(0, 0, 0, 0)

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
        regionGrowingLayout.addRow(_("Multiplier:"), self.regionGrowingMultiplierSlider)

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
        regionGrowingLayout.addRow(_("Iterations:"), self.regionGrowingIterationsSlider)

        self.algorithmParamsLayout.addRow(self.regionGrowingParamsGroup)
        self.regionGrowingParamsGroup.setVisible(False)

        # --- Random Walker Parameters (in Algorithm Parameters section) ---
        self.randomWalkerParamsGroup = qt.QWidget()
        rwLayout = qt.QFormLayout(self.randomWalkerParamsGroup)
        rwLayout.setContentsMargins(0, 0, 0, 0)

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
        rwLayout.addRow(_("Beta (edge weight):"), self.randomWalkerBetaSlider)

        self.algorithmParamsLayout.addRow(self.randomWalkerParamsGroup)
        self.randomWalkerParamsGroup.setVisible(False)

        # ----- Sampling & Post-processing -----
        self.samplingCollapsible = ctk.ctkCollapsibleButton()
        self.samplingCollapsible.text = _("Sampling & Post-processing")
        self.samplingCollapsible.collapsed = True
        self.scriptedEffect.addOptionsWidget(self.samplingCollapsible)
        samplingCollapsible = self.samplingCollapsible  # Local alias for compatibility
        samplingLayout = qt.QFormLayout(samplingCollapsible)

        # --- Sampling Parameters ---
        samplingLabel = qt.QLabel(_("<b>Sampling</b>"))
        samplingLayout.addRow(samplingLabel)

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
        samplingLayout.addRow(_("Gaussian Sigma:"), self.gaussianSigmaSlider)

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
        samplingLayout.addRow(_("Percentile Low:"), self.percentileLowSlider)

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
        samplingLayout.addRow(_("Percentile High:"), self.percentileHighSlider)

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
        samplingLayout.addRow(_("Std Multiplier:"), self.stdMultiplierSlider)

        # --- Post-processing Parameters ---
        postProcLabel = qt.QLabel(_("<b>Post-processing</b>"))
        samplingLayout.addRow(postProcLabel)

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
        samplingLayout.addRow(self.fillHolesCheckbox)

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
        samplingLayout.addRow(_("Closing Radius:"), self.closingRadiusSlider)

        # --- Backend & Performance ---
        backendLabel = qt.QLabel(_("<b>Backend & Performance</b>"))
        samplingLayout.addRow(backendLabel)

        # Backend dropdown
        self.backendCombo = qt.QComboBox()
        self.backendCombo.addItem(_("Auto"), "auto")
        self.backendCombo.addItem(_("CPU"), "cpu")
        self.backendCombo.addItem(_("GPU (OpenCL)"), "gpu_opencl")
        self.backendCombo.addItem(_("GPU (CUDA)"), "gpu_cuda")
        self.backendCombo.setToolTip(_("Computation backend"))
        samplingLayout.addRow(_("Backend:"), self.backendCombo)

        # Threshold caching checkbox
        self.cachingCheckbox = qt.QCheckBox(_("Enable threshold caching"))
        self.cachingCheckbox.setToolTip(
            _(
                "Reuse threshold calculations when painting in similar intensity regions. "
                "Improves drag performance but may reduce accuracy at region boundaries."
            )
        )
        self.cachingCheckbox.checked = self.useThresholdCaching
        samplingLayout.addRow(self.cachingCheckbox)

        # ----- Display Settings -----
        self.displayCollapsible = ctk.ctkCollapsibleButton()
        self.displayCollapsible.text = _("Display Settings")
        self.displayCollapsible.collapsed = True
        self.scriptedEffect.addOptionsWidget(self.displayCollapsible)
        displayCollapsible = self.displayCollapsible  # Local alias for compatibility
        displayLayout = qt.QFormLayout(displayCollapsible)

        # Crosshair enable checkbox
        self.crosshairCheckbox = qt.QCheckBox(_("Show crosshair"))
        self.crosshairCheckbox.setToolTip(
            _(
                "Display a crosshair at the brush center for precise positioning.\n\n"
                "The crosshair helps identify the exact center point where\n"
                "the segmentation algorithm will be seeded."
            )
        )
        self.crosshairCheckbox.checked = True  # Default on
        displayLayout.addRow(self.crosshairCheckbox)

        # Crosshair style dropdown
        self.crosshairStyleCombo = qt.QComboBox()
        self.crosshairStyleCombo.addItem(_("Cross (+)"), BrushOutlinePipeline.CROSSHAIR_STYLE_CROSS)
        self.crosshairStyleCombo.addItem(
            _("Bullseye (+ with gap)"), BrushOutlinePipeline.CROSSHAIR_STYLE_BULLSEYE
        )
        self.crosshairStyleCombo.addItem(_("Dot"), BrushOutlinePipeline.CROSSHAIR_STYLE_DOT)
        self.crosshairStyleCombo.addItem(
            _("Full crosshair"), BrushOutlinePipeline.CROSSHAIR_STYLE_CROSSHAIR
        )
        self.crosshairStyleCombo.setToolTip(
            _(
                "Crosshair style:\n"
                "• Cross: Simple + shape at center\n"
                "• Bullseye: Cross with gap and center dot\n"
                "• Dot: Small dot at center only\n"
                "• Full crosshair: Lines extending to brush edge"
            )
        )
        displayLayout.addRow(_("Crosshair Style:"), self.crosshairStyleCombo)

        # Crosshair size slider
        self.crosshairSizeSlider = ctk.ctkSliderWidget()
        self.crosshairSizeSlider.setToolTip(_("Size of crosshair in pixels"))
        self.crosshairSizeSlider.minimum = 5
        self.crosshairSizeSlider.maximum = 50
        self.crosshairSizeSlider.value = 10
        self.crosshairSizeSlider.singleStep = 1
        self.crosshairSizeSlider.decimals = 0
        self.crosshairSizeSlider.suffix = " px"
        displayLayout.addRow(_("Crosshair Size:"), self.crosshairSizeSlider)

        # Crosshair thickness slider
        self.crosshairThicknessSlider = ctk.ctkSliderWidget()
        self.crosshairThicknessSlider.setToolTip(_("Line thickness of crosshair"))
        self.crosshairThicknessSlider.minimum = 1
        self.crosshairThicknessSlider.maximum = 5
        self.crosshairThicknessSlider.value = 1
        self.crosshairThicknessSlider.singleStep = 1
        self.crosshairThicknessSlider.decimals = 0
        displayLayout.addRow(_("Crosshair Thickness:"), self.crosshairThicknessSlider)

        # Crosshair color picker
        crosshairColorLayout = qt.QHBoxLayout()
        self.crosshairColorButton = qt.QPushButton()
        self.crosshairColorButton.setToolTip(_("Click to change crosshair color"))
        self.crosshairColorButton.setFixedSize(30, 30)
        self._crosshairColor = qt.QColor(255, 255, 255)  # White default
        self._updateColorButton()
        crosshairColorLayout.addWidget(self.crosshairColorButton)

        # Preset color buttons
        self.crosshairWhiteButton = qt.QPushButton(_("White"))
        self.crosshairYellowButton = qt.QPushButton(_("Yellow"))
        self.crosshairCyanButton = qt.QPushButton(_("Cyan"))
        crosshairColorLayout.addWidget(self.crosshairWhiteButton)
        crosshairColorLayout.addWidget(self.crosshairYellowButton)
        crosshairColorLayout.addWidget(self.crosshairCyanButton)
        crosshairColorLayout.addStretch()
        displayLayout.addRow(_("Crosshair Color:"), crosshairColorLayout)

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

        # Display settings signals
        self.crosshairCheckbox.toggled.connect(self.onCrosshairSettingsChanged)
        self.crosshairStyleCombo.currentIndexChanged.connect(self.onCrosshairSettingsChanged)
        self.crosshairSizeSlider.valueChanged.connect(self.onCrosshairSettingsChanged)
        self.crosshairThicknessSlider.valueChanged.connect(self.onCrosshairSettingsChanged)
        self.crosshairColorButton.clicked.connect(self.onCrosshairColorPicker)
        self.crosshairWhiteButton.clicked.connect(lambda: self._setCrosshairColor(255, 255, 255))
        self.crosshairYellowButton.clicked.connect(lambda: self._setCrosshairColor(255, 230, 50))
        self.crosshairCyanButton.clicked.connect(lambda: self._setCrosshairColor(50, 230, 255))

    def _updateColorButton(self):
        """Update the crosshair color button appearance."""
        self.crosshairColorButton.setStyleSheet(
            f"background-color: rgb({self._crosshairColor.red()}, "
            f"{self._crosshairColor.green()}, {self._crosshairColor.blue()}); "
            f"border: 1px solid gray;"
        )

    def _setCrosshairColor(self, r, g, b):
        """Set crosshair color from RGB values."""
        self._crosshairColor = qt.QColor(r, g, b)
        self._updateColorButton()
        self.onCrosshairSettingsChanged()

    def onCrosshairColorPicker(self):
        """Open color picker dialog for crosshair color."""
        color = qt.QColorDialog.getColor(self._crosshairColor, None, _("Select Crosshair Color"))
        if color.isValid():
            self._crosshairColor = color
            self._updateColorButton()
            self.onCrosshairSettingsChanged()

    def onCrosshairSettingsChanged(self):
        """Handle crosshair settings changes."""
        enabled = self.crosshairCheckbox.checked
        style = self.crosshairStyleCombo.currentData
        size = int(self.crosshairSizeSlider.value)
        thickness = int(self.crosshairThicknessSlider.value)
        color = (
            self._crosshairColor.red() / 255.0,
            self._crosshairColor.green() / 255.0,
            self._crosshairColor.blue() / 255.0,
        )

        # Update all brush outline pipelines
        for pipeline in self.outlinePipelines.values():
            pipeline.setCrosshairSettings(
                enabled=enabled,
                style=style,
                size=size,
                thickness=thickness,
                color=color,
            )

        # Trigger a redraw if brush is visible
        self._forceOutlineUpdate()

    def _forceOutlineUpdate(self):
        """Force redraw of brush outline in all views."""
        # Schedule render in all slice views with pipelines
        for pipeline in self.outlinePipelines.values():
            if pipeline.sliceWidget is not None:
                # Note: Widget may be deleted during Slicer shutdown, causing
                # RuntimeError. This is expected and non-critical for refresh.
                try:
                    pipeline.sliceWidget.sliceView().scheduleRender()
                except RuntimeError as e:
                    # Widget deleted - expected during cleanup, non-critical
                    logging.debug(f"Render skipped (widget deleted): {e}")

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
            self.algorithmComboBox,
            self.sensitivitySlider,
            self.zoneSlider,
            self.samplingMethodCombo,
            self.gaussianSigmaSlider,
            self.stdMultiplierSlider,
            self.fillHolesCheckbox,
            self.closingRadiusSlider,
        ]

        # Block signals
        for widget in widgets_to_block:
            widget.blockSignals(True)

        try:
            # Apply algorithm if specified in preset
            if "algorithm" in preset:
                algorithm = preset["algorithm"]
                self.setAlgorithm(algorithm)
                idx = self.algorithmComboBox.findData(algorithm)
                if idx >= 0:
                    self.algorithmComboBox.setCurrentIndex(idx)

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

            # Apply sampling parameters
            if "gaussian_sigma" in preset:
                self.gaussianSigma = preset["gaussian_sigma"]
                self.gaussianSigmaSlider.value = preset["gaussian_sigma"]

            if "std_multiplier" in preset:
                self.stdMultiplier = preset["std_multiplier"]
                self.stdMultiplierSlider.value = preset["std_multiplier"]

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

        # Update algorithm parameters visibility (since signals were blocked)
        self._updateAlgorithmParamsVisibility()

        logging.info(f"Applied preset: {preset['name']}")

    # =========================================================================
    # Public API for scripting and recipes
    # =========================================================================

    def applyPreset(self, preset_id: str) -> None:
        """Apply a parameter preset.

        Args:
            preset_id: Preset name (e.g., "tumor_lesion", "bone_ct", "default").

        Available presets:
            - "default": Balanced settings for general use
            - "bone_ct": High contrast bone in CT
            - "soft_tissue_ct": Organs/soft tissue in CT
            - "lung_ct": Lung parenchyma in CT
            - "brain_mri": Brain tissue in MRI
            - "tumor_lesion": Tumors with irregular boundaries
            - "vessel": Blood vessels
            - "smooth_edge": Smooth-edged structures
            - "sharp_edge": Sharp-edged structures
            - "fast_rough": Quick rough segmentation
        """
        self._applyPreset(preset_id)

    def paintAt(self, r: float, a: float, s: float, erase: bool = False) -> None:
        """Apply a brush stroke at the given RAS coordinates.

        This is the main method for programmatic painting. It navigates to
        the location, applies the brush with current parameters, and updates
        the segmentation.

        Args:
            r: Right coordinate (mm).
            a: Anterior coordinate (mm).
            s: Superior coordinate (mm).
            erase: If True, erase instead of add.

        Example:
            effect.applyPreset("tumor_lesion")
            effect.brushRadiusMm = 20.0
            effect.paintAt(-5.31, 34.77, 20.83)
        """
        import slicer

        ras = (r, a, s)

        # Get the red slice widget (primary 2D view)
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget("Red")
        sliceLogic = sliceWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()

        # Navigate to the RAS location
        sliceNode.JumpSliceByCentering(r, a, s)
        slicer.app.processEvents()

        # Convert RAS to XY in slice view
        xy = self._rasToXy(ras, sliceWidget)

        if xy is None:
            logging.warning(f"Could not convert RAS {ras} to XY coordinates")
            return

        logging.debug(f"paintAt: RAS={ras} -> XY={xy}")

        # Apply the brush stroke
        self.scriptedEffect.saveStateForUndo()
        self.isDrawing = True
        self._currentStrokeEraseMode = erase
        self.processPoint(xy, sliceWidget)
        self.isDrawing = False
        slicer.app.processEvents()

        logging.debug(f"paintAt RAS=({r}, {a}, {s}), erase={erase}")

    def _rasToXy(self, ras, sliceWidget):
        """Convert RAS coordinates to screen XY in slice widget.

        Args:
            ras: (R, A, S) coordinates.
            sliceWidget: The slice widget.

        Returns:
            (x, y) screen coordinates or None if conversion fails.
        """
        import vtk

        sliceLogic = sliceWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()

        # Get XY to RAS matrix and invert it to get RAS to XY
        xyToRas = sliceNode.GetXYToRAS()
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xyToRas, rasToXy)

        # Transform RAS to screen XY
        rasPoint = [ras[0], ras[1], ras[2], 1.0]
        xyPoint = [0, 0, 0, 1]
        rasToXy.MultiplyPoint(rasPoint, xyPoint)

        return (int(xyPoint[0]), int(xyPoint[1]))

    @property
    def brushRadiusMm(self) -> float:
        """Get/set the brush radius in millimeters."""
        return float(self.radiusMm)

    @brushRadiusMm.setter
    def brushRadiusMm(self, value: float) -> None:
        self.radiusMm = value
        self.radiusSlider.value = value

    @property
    def edgeSensitivityValue(self) -> int:
        """Get/set edge sensitivity (0-100)."""
        return int(self.edgeSensitivity)

    @edgeSensitivityValue.setter
    def edgeSensitivityValue(self, value: int) -> None:
        self.edgeSensitivity = value
        self.sensitivitySlider.value = value

    def onRadiusChanged(self, value):
        """Handle radius slider change."""
        old_radius = self.radiusMm
        self.radiusMm = value
        self._log_action("radius_changed", old=old_radius, new=value)
        self.cache.invalidate()

    def onSensitivityChanged(self, value):
        """Handle edge sensitivity change."""
        old_sensitivity = self.edgeSensitivity
        self.edgeSensitivity = value
        self._log_action("sensitivity_changed", old=old_sensitivity, new=value)
        self.cache.invalidate()

    def onZoneChanged(self, value):
        """Handle threshold zone size change."""
        old_zone = self.thresholdZone
        self.thresholdZone = value
        self._log_action("zone_changed", old=old_zone, new=value)
        self.cache.invalidate()

    def onSamplingMethodChanged(self, index):
        """Handle sampling method change."""
        old_method = self.samplingMethod
        self.samplingMethod = self.samplingMethodCombo.currentData
        self._log_action("sampling_method_changed", old=old_method, new=self.samplingMethod)
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
        old_mode = self.sphereMode
        self.sphereMode = checked
        self._log_action("sphere_mode_changed", old=old_mode, new=checked)
        self.cache.invalidate()

    def onPreviewModeChanged(self, checked):
        """Handle preview mode toggle."""
        self.previewMode = checked
        if not checked:
            # Hide any existing preview
            self._hideSegmentationPreview()

    def onModeChanged(self, checked):
        """Handle add/erase mode toggle.

        Args:
            checked: True if the "Add" radio button is checked.
        """
        # Only respond to the radio that became checked
        if not checked:
            return
        self.eraseMode = self.eraseModeRadio.isChecked()
        # Update any visible brush preview to show new color
        self._updateBrushColors()

    def onWizardClicked(self):
        """Launch the Quick Select Parameters wizard."""
        try:
            from ParameterWizard import ParameterWizard

            wizard = ParameterWizard(self)
            wizard.start()
        except Exception as e:
            logging.exception("Failed to start parameter wizard")
            slicer.util.errorDisplay(
                f"Failed to start wizard: {e}",
                windowTitle="Wizard Error",
            )

    def setAlgorithm(self, algorithm: str) -> None:
        """Set the current algorithm.

        Args:
            algorithm: Algorithm identifier string.
        """
        # Find the algorithm in the combo box
        index = self.algorithmCombo.findData(algorithm)
        if index >= 0:
            self.algorithmCombo.setCurrentIndex(index)
        self.algorithm = algorithm
        self._updateAlgorithmParamsVisibility()

    def setRadiusMm(self, radius_mm: float) -> None:
        """Set the brush radius.

        Args:
            radius_mm: Brush radius in millimeters (clamped to 1.0-100.0).
        """
        clamped = max(1.0, min(100.0, radius_mm))
        if clamped != radius_mm:
            logging.info(f"Brush radius clamped: requested={radius_mm}, applied={clamped}")
        self.radiusMm = clamped
        self.radiusSlider.value = self.radiusMm

    def setEdgeSensitivity(self, sensitivity: int) -> None:
        """Set the edge sensitivity.

        Args:
            sensitivity: Edge sensitivity value (clamped to 0-100).
        """
        clamped = max(0, min(100, sensitivity))
        if clamped != sensitivity:
            logging.info(f"Edge sensitivity clamped: requested={sensitivity}, applied={clamped}")
        self.edgeSensitivity = clamped
        self.sensitivitySlider.value = self.edgeSensitivity

    def setThresholdRange(self, lower: float, upper: float) -> None:
        """Set the threshold range for threshold-based algorithms.

        Args:
            lower: Lower threshold value.
            upper: Upper threshold value.
        """
        self.lowerThresholdSlider.value = lower
        self.upperThresholdSlider.value = upper

    def _updateBrushColors(self):
        """Update brush outline colors based on current erase mode."""
        for pipeline in self.outlinePipelines.values():
            if self.eraseMode:
                # Erase mode: red/orange colors
                pipeline.outerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ERASE)
                pipeline.innerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ERASE_INNER)
            else:
                # Add mode: yellow/cyan colors
                pipeline.outerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ADD)
                pipeline.innerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ADD_INNER)
        # Request render for visible views
        for pipeline in self.outlinePipelines.values():
            if pipeline.sliceWidget is not None:
                pipeline.sliceWidget.sliceView().scheduleRender()

    def _getAlgorithmDisplayName(self, algorithm_id: str) -> str:
        """Get human-readable display name for an algorithm."""
        names = {
            "geodesic_distance": _("Geodesic Distance"),
            "watershed": _("Watershed"),
            "random_walker": _("Random Walker"),
            "level_set": _("Level Set"),
            "connected_threshold": _("Connected Threshold"),
            "region_growing": _("Region Growing"),
            "threshold_brush": _("Threshold Brush"),
        }
        return str(names.get(algorithm_id, algorithm_id))

    def _updateAlgorithmParamsVisibility(self):
        """Update the algorithm parameters section title and visibility.

        Called when the algorithm changes, either directly or via preset.
        """
        # Update dynamic section title
        displayName = self._getAlgorithmDisplayName(self.algorithm)
        self.algorithmParamsCollapsible.text = _("{name} Parameters").format(name=displayName)

        # Hide all algorithm parameter groups first
        self.thresholdGroup.setVisible(False)
        self.geodesicParamsGroup.setVisible(False)
        self.watershedParamsGroup.setVisible(False)
        self.levelSetParamsGroup.setVisible(False)
        self.regionGrowingParamsGroup.setVisible(False)
        self.randomWalkerParamsGroup.setVisible(False)

        # Show the appropriate parameter group and section
        hasParams = True
        if self.algorithm == "geodesic_distance":
            self.geodesicParamsGroup.setVisible(True)
        elif self.algorithm == "watershed":
            self.watershedParamsGroup.setVisible(True)
        elif self.algorithm == "level_set":
            self.levelSetParamsGroup.setVisible(True)
        elif self.algorithm == "region_growing":
            self.regionGrowingParamsGroup.setVisible(True)
        elif self.algorithm == "random_walker":
            self.randomWalkerParamsGroup.setVisible(True)
        elif self.algorithm == "threshold_brush":
            self.thresholdGroup.setVisible(True)
        elif self.algorithm == "connected_threshold":
            # No custom parameters for Connected Threshold
            hasParams = False

        # Hide the entire section if algorithm has no parameters
        self.algorithmParamsCollapsible.setVisible(hasParams)

    def onAlgorithmChanged(self, index):
        """Handle algorithm selection change."""
        old_algorithm = self.algorithm
        self.algorithm = self.algorithmCombo.currentData
        logging.info(f"Algorithm changed to: {self.algorithm}")
        self._log_action("algorithm_changed", old=old_algorithm, new=self.algorithm)
        self.cache.invalidate()

        # Update visibility
        self._updateAlgorithmParamsVisibility()

        # Prompt to install scikit-image if Random Walker selected without it
        if self.algorithm == "random_walker" and not HAS_SKIMAGE_RW:
            # Try to install (will prompt user)
            if not _ensure_random_walker_available():
                # User declined or installation failed - warn about fallback
                slicer.util.warningDisplay(
                    "scikit-image is not available. Random Walker will use a fallback "
                    "algorithm with reduced accuracy.\n\n"
                    "You can install it manually and restart Slicer for best results.",
                    windowTitle="Random Walker - Using Fallback",
                )

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

    def activate(self) -> None:
        """Called when the effect is selected."""
        logging.debug("AdaptiveBrush activate() starting")
        self.cache.clear()
        logging.debug("activate: cache cleared")
        self._createOutlinePipelines()
        logging.debug("activate: outline pipelines created")
        self._updateThresholdRanges()
        logging.debug("activate: threshold ranges updated")
        # Track current source volume for change detection
        self._lastSourceVolumeId = self._getCurrentSourceVolumeId()
        logging.debug("activate: source volume ID tracked")

        # Prompt to install sklearn for GMM if not available
        # Do this at activation time, not during painting
        from IntensityAnalyzer import HAS_SKLEARN, _ensure_sklearn

        logging.debug(f"activate: HAS_SKLEARN={HAS_SKLEARN}")
        if not HAS_SKLEARN:
            logging.debug("activate: calling _ensure_sklearn()")
            if _ensure_sklearn():
                logging.debug("activate: sklearn now available, recreating analyzer")
                # Recreate analyzer with GMM now available
                from IntensityAnalyzer import IntensityAnalyzer

                self.intensityAnalyzer = IntensityAnalyzer()
            else:
                logging.debug("activate: sklearn not available, using fallback")
        logging.debug("AdaptiveBrush activate() complete")

    def deactivate(self) -> None:
        """Called when the effect is deselected."""
        self.cache.clear()
        self.isDrawing = False
        self._cleanupOutlinePipelines()
        self._lastSourceVolumeId = None

    # -------------------------------------------------------------------------
    # Methods that Slicer may call on effects - provide stubs to prevent
    # AttributeError which could trigger recursive exception handlers
    # -------------------------------------------------------------------------

    def setMRMLDefaults(self) -> None:
        """Called to set default MRML parameters. No-op for this effect."""
        pass

    def updateGUIFromMRML(self) -> None:
        """Called to sync GUI from MRML parameters. No-op for this effect."""
        pass

    def updateMRMLFromGUI(self) -> None:
        """Called to sync MRML from GUI. No-op for this effect."""
        pass

    def interactionNodeModified(self, interactionNode: Any) -> None:
        """Called when the interaction node changes. No-op for this effect."""
        pass

    def layoutChanged(self):
        """Called when the application layout changes. No-op for this effect."""
        pass

    def processViewNodeEvents(self, callerViewNode, eventId, viewWidget):
        """Called to process view node events. No-op for this effect."""
        pass

    def cleanup(self):
        """Clean up resources to prevent memory leaks.

        Disconnects all signal/slot connections and cleans up VTK pipelines.
        This method is called by Slicer before the effect is deleted.

        See: https://github.com/Slicer/Slicer/issues/7392
        """
        # Disconnect all signal/slot connections from UI widgets
        # Using try/except to handle cases where widgets may not exist
        widgets_to_disconnect = [
            "presetCombo",
            "resetPresetButton",
            "radiusSlider",
            "sensitivitySlider",
            "zoneSlider",
            "samplingMethodCombo",
            "sphereModeCheckbox",
            "previewModeCheckbox",
            "algorithmCombo",
            "backendCombo",
            "cachingCheckbox",
            "lowerThresholdSlider",
            "upperThresholdSlider",
            "autoThresholdCheckbox",
            "thresholdMethodCombo",
            "setFromSeedButton",
            "toleranceSlider",
            "gaussianSigmaSlider",
            "percentileLowSlider",
            "percentileHighSlider",
            "stdMultiplierSlider",
            "includeZoneCheckbox",
            "geodesicEdgeWeightSlider",
            "geodesicDistanceScaleSlider",
            "geodesicSmoothingSlider",
            "watershedGradientScaleSlider",
            "watershedSmoothingSlider",
            "levelSetPropagationSlider",
            "levelSetCurvatureSlider",
            "levelSetIterationsSlider",
            "regionGrowingMultiplierSlider",
            "regionGrowingIterationsSlider",
            "randomWalkerBetaSlider",
            "fillHolesCheckbox",
            "closingRadiusSlider",
            "addModeRadio",
            "eraseModeRadio",
        ]

        for widget_name in widgets_to_disconnect:
            try:
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.disconnect()
            except (RuntimeError, ValueError) as e:
                # Widget already deleted or no matching overload during cleanup - expected
                logging.debug(f"Widget {widget_name} disconnect skipped: {e}")

        # Clean up brush outline pipelines
        self._cleanupOutlinePipelines()

        # Clear cache
        if hasattr(self, "cache") and self.cache is not None:
            self.cache.clear()

        # Call parent cleanup
        AbstractScriptedSegmentEditorEffect.cleanup(self)

    def sourceVolumeNodeChanged(self):
        """Called when the source volume node changes.

        Updates threshold slider ranges to match the new volume's intensity range.
        """
        self._updateThresholdRanges()
        self._lastSourceVolumeId = self._getCurrentSourceVolumeId()
        self.cache.invalidate()

    def masterVolumeNodeChanged(self):
        """Called when the master volume node changes (deprecated name).

        Delegates to sourceVolumeNodeChanged for backward compatibility.
        """
        self.sourceVolumeNodeChanged()

    def referenceGeometryChanged(self):
        """Called when the reference geometry changes.

        No-op for this effect - we handle volume changes in sourceVolumeNodeChanged.
        """
        pass

    def _getCurrentSourceVolumeId(self):
        """Get the ID of the current source volume, or None if not set.

        Returns:
            Volume ID string, or None if no source volume is configured.

        Note:
            Returns None during Slicer state transitions when effect not fully initialized.
        """
        try:
            parameterSetNode = self.scriptedEffect.parameterSetNode()
            if parameterSetNode is None:
                return None
            sourceVolumeNode = parameterSetNode.GetSourceVolumeNode()
            if sourceVolumeNode is None:
                return None
            return sourceVolumeNode.GetID()
        except (AttributeError, RuntimeError) as e:
            # Effect not fully initialized or Slicer shutting down
            logging.debug(f"Source volume query during transition: {e}")
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

    def _updateBrushPreview(self, xy, viewWidget, eraseMode=None):
        """Update the brush outline at the cursor position.

        Shows two circles: outer (max extent) and inner (threshold zone).

        Args:
            xy: Screen coordinates (x, y).
            viewWidget: The slice view widget.
            eraseMode: If True, show erase colors. If None, use self.eraseMode.
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
            pipeline = self.outlinePipelines[viewName]
            pipeline.updateOutline(xy, radiusPixels, innerRatio)

            # Update colors based on erase mode
            effectiveEraseMode = eraseMode if eraseMode is not None else self.eraseMode
            if effectiveEraseMode:
                # Erase mode: red/orange colors
                pipeline.outerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ERASE)
                pipeline.innerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ERASE_INNER)
            else:
                # Add mode: yellow/cyan colors
                pipeline.outerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ADD)
                pipeline.innerActor.GetProperty().SetColor(*self.BRUSH_COLOR_ADD_INNER)

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
            ijk = self._xyToIjk(xy, viewWidget)
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
            Slice index (int) or None if index cannot be determined.

        Note:
            Returns None for oblique views or during widget state transitions.
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

        except (AttributeError, RuntimeError) as e:
            # Widget not fully initialized or volume without image data
            logging.debug(f"Slice index query during transition: {e}")
            return None

    def _updateThresholdRanges(self):
        """Update threshold slider ranges based on source volume intensity.

        Sets slider min/max to the 1st-99th percentile of image intensities
        for a more meaningful range than hardcoded values. Also sets
        appropriate step size and sensible default values.
        """
        # Reentrancy guard to prevent recursive calls
        if self._updatingThresholdRanges:
            return
        self._updatingThresholdRanges = True

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

            # Block signals while updating slider ranges to prevent recursive updates
            # This follows the pattern used in Slicer's ThresholdEffect
            wasLowerBlocked = self.lowerThresholdSlider.blockSignals(True)
            wasUpperBlocked = self.upperThresholdSlider.blockSignals(True)

            try:
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
            finally:
                # Restore signal state
                self.lowerThresholdSlider.blockSignals(wasLowerBlocked)
                self.upperThresholdSlider.blockSignals(wasUpperBlocked)

            logging.debug(
                f"Updated threshold ranges: [{range_min:.1f}, {range_max:.1f}], "
                f"step: {step}, defaults (IQR): [{p25:.1f}, {p75:.1f}]"
            )
        except Exception as e:
            logging.warning(f"Could not update threshold ranges: {e}")
        finally:
            self._updatingThresholdRanges = False

    def processInteractionEvents(
        self, callerInteractor: Any, eventId: int, viewWidget: Any
    ) -> bool:
        """Handle mouse interaction events.

        Args:
            callerInteractor: VTK interactor that triggered the event.
            eventId: VTK event ID.
            viewWidget: The view widget where the event occurred.

        Returns:
            True if the event was handled, False otherwise.
        """
        # When wizard is active, forward events to the wizard's sampler
        if getattr(self, "_wizardActive", False):
            if hasattr(self, "_activeWizard") and self._activeWizard:
                return bool(
                    self._activeWizard.handle_interaction_event(
                        callerInteractor, eventId, viewWidget
                    )
                )
            return False

        if viewWidget.className() != "qMRMLSliceWidget":
            return False

        xy = callerInteractor.GetEventPosition()

        # Track middle button state for erase modifier
        if eventId == vtk.vtkCommand.MiddleButtonPressEvent:
            self._isMiddleButtonHeld = True
            # Don't consume - let pan work
            return False

        elif eventId == vtk.vtkCommand.MiddleButtonReleaseEvent:
            self._isMiddleButtonHeld = False
            return False

        # Handle scroll wheel events for brush radius and threshold zone adjustment
        if eventId in (
            vtk.vtkCommand.MouseWheelForwardEvent,
            vtk.vtkCommand.MouseWheelBackwardEvent,
        ):
            isCtrlPressed = callerInteractor.GetControlKey()
            isShiftPressed = callerInteractor.GetShiftKey()
            forward = eventId == vtk.vtkCommand.MouseWheelForwardEvent

            if isShiftPressed and isCtrlPressed:
                # Ctrl+Shift+scroll: adjust threshold zone
                delta = 5 if forward else -5
                newValue = max(10, min(100, self.thresholdZone + delta))
                self.thresholdZone = newValue
                self.zoneSlider.value = newValue
                self._updateBrushPreview(xy, viewWidget)
                return True  # Consume event

            elif isShiftPressed and not isCtrlPressed:
                # Shift+scroll: adjust brush radius by ~20%
                factor = 1.2 if forward else 0.8
                newRadius = max(1.0, min(100.0, self.radiusMm * factor))
                self.radiusMm = newRadius
                self.radiusSlider.value = newRadius
                self._updateBrushPreview(xy, viewWidget)
                return True  # Consume event

            # If neither modifier combo, don't consume (let Slicer handle Ctrl+scroll for zoom)
            return False

        # Detect Ctrl key for temporary mode inversion
        isCtrlPressed = callerInteractor.GetControlKey()

        # Preview mode (hovering): only Ctrl inverts mode (middle button doesn't affect preview)
        # This is because middle button alone should allow pan without changing brush color
        previewEraseMode = self.eraseMode != bool(isCtrlPressed)

        # Paint mode: both Ctrl and middle button can invert mode
        # Middle button only takes effect when combined with left-click
        isModifierActive = bool(isCtrlPressed) or self._isMiddleButtonHeld
        paintEraseMode = self.eraseMode != isModifierActive

        if eventId == vtk.vtkCommand.LeftButtonPressEvent:
            # Save undo state at the START of the stroke (once per stroke)
            self.scriptedEffect.saveStateForUndo()
            self.isDrawing = True
            # Lock the erase mode for this stroke (can't change mid-stroke)
            self._currentStrokeEraseMode = paintEraseMode
            # Hide segmentation preview when starting to draw
            self._hideSegmentationPreview()
            self._updateBrushPreview(xy, viewWidget, paintEraseMode)
            self.processPoint(xy, viewWidget)
            return True

        elif eventId == vtk.vtkCommand.MouseMoveEvent:
            if self.isDrawing:
                # While drawing, use the locked stroke mode for preview
                self._updateBrushPreview(xy, viewWidget, self._currentStrokeEraseMode)
                self.processPoint(xy, viewWidget)
                return True  # Only consume when drawing
            else:
                # While hovering, only Ctrl affects preview (not middle button)
                self._updateBrushPreview(xy, viewWidget, previewEraseMode)
                # Show segmentation preview when hovering (if preview mode enabled)
                if self.previewMode:
                    self._updateSegmentationPreview(xy, viewWidget)
            return False  # Don't consume when just hovering

        elif eventId == vtk.vtkCommand.LeftButtonReleaseEvent:
            self.isDrawing = False
            self.lastIjk = None
            self.cache.onMouseRelease()
            # Show preview again after drawing (revert to hover preview mode)
            self._updateBrushPreview(xy, viewWidget, previewEraseMode)
            if self.previewMode:
                self._updateSegmentationPreview(xy, viewWidget)
            return True

        elif eventId == vtk.vtkCommand.LeaveEvent:
            # Hide preview when mouse leaves the view
            self._hideBrushPreview()
            self._hideSegmentationPreview()
            return False

        elif eventId == vtk.vtkCommand.EnterEvent:
            # Update preview when mouse enters (hover mode)
            self._updateBrushPreview(xy, viewWidget, previewEraseMode)
            return False

        return False

    def processPoint(self, xy, viewWidget):
        """Process a single point interaction.

        Args:
            xy: Screen coordinates (x, y).
            viewWidget: The slice view widget.
        """
        ijk = self._xyToIjk(xy, viewWidget)
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

        # Get RAS coordinates for logging
        sliceLogic = viewWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        viewName = sliceNode.GetName() if sliceNode else "Unknown"

        # Compute RAS from IJK for logging
        ras = None
        ijkToRas = vtk.vtkMatrix4x4()
        sourceVolumeNode.GetIJKToRASMatrix(ijkToRas)
        ijkPoint = [ijk[0], ijk[1], ijk[2], 1]
        rasPoint = [0, 0, 0, 1]
        ijkToRas.MultiplyPoint(ijkPoint, rasPoint)
        ras = (rasPoint[0], rasPoint[1], rasPoint[2])

        # Log paint stroke action
        self._log_action(
            "paint_stroke",
            xy=xy,
            ijk=ijk,
            ras=ras,
            view=viewName,
            erase_mode=self._currentStrokeEraseMode,
        )

        # Compute adaptive mask
        start_time = time.time()
        try:
            mask = self.computeAdaptiveMask(sourceVolumeNode, ijk, viewWidget)
            if mask is not None:
                voxels_modified = int(np.sum(mask > 0))
                self.applyMaskToSegment(mask, erase=self._currentStrokeEraseMode)
                elapsed_ms = (time.time() - start_time) * 1000
                logging.debug(
                    f"Paint stroke: ijk={ijk}, voxels={voxels_modified}, "
                    f"time={elapsed_ms:.1f}ms, algorithm={self.algorithm}"
                )
        except RuntimeError as e:
            # SimpleITK filter failure - show user-facing error
            logging.exception(f"Algorithm '{self.algorithm}' failed")
            slicer.util.errorDisplay(
                f"Segmentation algorithm '{self.algorithm}' failed:\n\n{e}\n\n"
                "Try a different algorithm or adjust parameters.",
                windowTitle="Adaptive Brush Error",
            )

    def _xyToIjk(self, xy, viewWidget):
        """Convert screen XY coordinates to volume IJK (internal helper).

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

    def computeAdaptiveMask(
        self, sourceVolumeNode: Any, seedIjk: tuple[int, int, int], viewWidget: Any
    ) -> Optional[np.ndarray]:
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
        mask: Optional[np.ndarray] = self.cache.computeOrGetCached(
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
        logging.debug(
            f"_runSegmentation: algorithm={params['algorithm']}, seedIjk={seedIjk}, "
            f"thresholds=[{thresholds.get('lower', 'N/A')}, {thresholds.get('upper', 'N/A')}]"
        )
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
        elif algorithm == "level_set":
            mask = self._levelSet(roi, localSeed, thresholds, params)
        elif algorithm == "region_growing":
            mask = self._regionGrowing(roi, localSeed, thresholds, params)
        elif algorithm == "threshold_brush":
            mask = self._thresholdBrush(roi, localSeed, params)
        elif algorithm == "geodesic_distance":
            mask = self._geodesicDistance(roi, localSeed, thresholds, params)
        elif algorithm == "random_walker":
            mask = self._randomWalker(roi, localSeed, thresholds, params)
        elif algorithm == "watershed":
            mask = self._watershed(roi, localSeed, thresholds, params)
        else:
            raise ValueError(
                f"Unknown algorithm: '{algorithm}'. "
                f"Valid algorithms: geodesic_distance, watershed, random_walker, level_set, "
                f"connected_threshold, region_growing, threshold_brush"
            )

        # Apply circular brush mask as MAXIMUM extent for ALL algorithms
        # Adaptive algorithms use edges to stop earlier, but should never exceed brush radius
        mask = self._applyBrushMask(mask, localSeed, params["radius_voxels"])

        # Apply morphological operations based on advanced parameters
        fill_holes = params.get("fill_holes", True)
        closing_radius = params.get("closing_radius", 1)

        if np.any(mask) and (fill_holes or closing_radius > 0):
            maskSitk = sitk.GetImageFromArray(mask)

            # Fill holes inside the segmentation (optional post-processing)
            # RuntimeError: SimpleITK filter failure (e.g., degenerate mask)
            # Result: segmentation still valid, just without hole filling
            if fill_holes:
                try:
                    maskSitk = sitk.BinaryFillhole(maskSitk)
                except RuntimeError as e:
                    logging.warning(f"Binary fillhole failed (result may have holes): {e}")

            # Close small gaps (optional morphological post-processing)
            # RuntimeError: SimpleITK filter failure (e.g., invalid kernel size)
            # Result: segmentation still valid, just without gap closing
            if closing_radius > 0:
                try:
                    kernel_size = [int(closing_radius)] * 3
                    maskSitk = sitk.BinaryMorphologicalClosing(maskSitk, kernel_size)
                except RuntimeError as e:
                    logging.warning(f"Morphological closing failed (result may have gaps): {e}")

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
        gradArray = np.clip(gradArray / gradMax * 255 * effective_scale, 0, 255).astype(np.float32)

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
        # BinaryThreshold with only upperThreshold=0 has lowerThreshold default of 0,
        # creating range [0,0] which only keeps exactly 0. We need <= 0.
        levelSetArray = sitk.GetArrayFromImage(levelSet)
        return (levelSetArray <= 0).astype(np.uint8)

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

            # Filter to only regions connected to seed (optional post-processing)
            # RuntimeError: seed outside mask or SimpleITK failure
            # Result: may include disconnected blobs if this fails
            try:
                connected = sitk.ConnectedThreshold(
                    sitkMask, seedList=[sitkSeed], lower=1, upper=1, replaceValue=1
                )
                mask = sitk.GetArrayFromImage(connected).astype(np.uint8)
            except RuntimeError as e:
                logging.warning(f"Connectivity filter failed (may have disconnected regions): {e}")

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

        # Filter to only regions connected to seed (important for narrow segments)
        if np.any(mask):
            sitkMask = sitk.GetImageFromArray(mask)
            try:
                connected = sitk.ConnectedThreshold(
                    sitkMask, seedList=[sitkSeed], lower=1, upper=1, replaceValue=1
                )
                mask = sitk.GetArrayFromImage(connected).astype(np.uint8)
            except RuntimeError as e:
                # Seed outside mask after thresholding - result may have disconnected blobs
                # This is a known edge case that doesn't warrant failing the whole operation
                logging.warning(
                    f"Connectivity filter failed (result may have disconnected regions): {e}"
                )

        return mask

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
            return self._randomWalkerSkimage(roi, localSeed, thresholds, beta, radius_voxels)

        # scikit-image not available - use gradient-weighted fallback
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
            raise RuntimeError(
                "Random Walker: insufficient markers. "
                "Try increasing brush size or adjusting thresholds."
            )

        # Run Random Walker
        # mode='cg_j' (conjugate gradient with Jacobi preconditioning) avoids
        # UMFPACK dependency warning and is typically faster
        # prob_tol=0.1 accepts slight numerical imprecision in probabilities
        # (we only use argmax for labels, not actual probability values)
        labels = skimage_random_walker(
            roi_norm,
            markers,
            beta=beta,
            mode="cg_j",
            prob_tol=0.1,
            return_full_prob=False,
        )

        # Extract foreground (label 1)
        mask = (labels == 1).astype(np.uint8)

        # Constrain to intensity range for safety
        mask = mask & intensity_mask.astype(np.uint8)

        # Filter to only regions connected to seed
        # RuntimeError: seed outside mask or SimpleITK failure
        # Result: may include disconnected blobs if this fails
        if np.any(mask):
            sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))
            sitkMask = sitk.GetImageFromArray(mask)
            try:
                connected = sitk.ConnectedThreshold(
                    sitkMask, seedList=[sitkSeed], lower=1, upper=1, replaceValue=1
                )
                mask = sitk.GetArrayFromImage(connected).astype(np.uint8)
            except RuntimeError as e:
                logging.warning(f"Connectivity filter failed (may have disconnected regions): {e}")

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

        if weight_at_seed <= 0.1:
            raise RuntimeError(
                f"Random Walker: seed weight too low ({weight_at_seed:.3f}). "
                "Seed may be on an edge. Try clicking in a more homogeneous region."
            )

        # weight_at_seed > 0.1 - proceed with weighted connected threshold
        weighted_lower = thresholds["lower"] * weight_at_seed * 0.5
        weighted_upper = thresholds["upper"] * weight_at_seed * 1.5

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
            weighted_variance = np.sum(normalized_weights * (zone_intensities - weighted_mean) ** 2)
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

    def applyMaskToSegment(self, mask: np.ndarray, erase: bool = False) -> None:
        """Apply the computed mask to the current segment.

        Args:
            mask: Binary mask numpy array (z, y, x ordering).
            erase: If True, remove mask from segment. If False, add to segment.
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

        # Apply to segment using appropriate mode
        if erase:
            mode = slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeRemove
        else:
            mode = slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
        self.scriptedEffect.modifySelectedSegmentByLabelmap(modifierLabelmap, mode)


# Required for Slicer to find the effect class
Effect = SegmentEditorEffect
