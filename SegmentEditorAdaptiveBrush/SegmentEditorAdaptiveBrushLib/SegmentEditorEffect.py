"""Adaptive Brush Segment Editor Effect.

This effect provides an adaptive brush that automatically segments regions
based on image intensity similarity, adapting to image features (edges,
boundaries) rather than using a fixed geometric shape.
"""

import logging
import os
import sys

import ctk
import numpy as np
import qt
import vtk

import slicer
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

        # Connect signals
        self.radiusSlider.valueChanged.connect(self.onRadiusChanged)
        self.sensitivitySlider.valueChanged.connect(self.onSensitivityChanged)
        self.sphereModeCheckbox.toggled.connect(self.onSphereModeChanged)
        self.algorithmCombo.currentIndexChanged.connect(self.onAlgorithmChanged)
        self.backendCombo.currentIndexChanged.connect(self.onBackendChanged)

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

    def onBackendChanged(self, index):
        """Handle backend selection change."""
        self.backend = self.backendCombo.currentData
        self.cache.invalidate()

    def activate(self):
        """Called when the effect is selected."""
        self.cache.clear()

    def deactivate(self):
        """Called when the effect is deselected."""
        self.cache.clear()
        self.isDrawing = False

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

        if eventId == vtk.vtkCommand.LeftButtonPressEvent:
            # Save undo state at the START of the stroke (once per stroke)
            self.scriptedEffect.saveStateForUndo()
            self.isDrawing = True
            xy = callerInteractor.GetEventPosition()
            self.processPoint(xy, viewWidget)
            return True

        elif eventId == vtk.vtkCommand.MouseMoveEvent:
            if self.isDrawing:
                xy = callerInteractor.GetEventPosition()
                self.processPoint(xy, viewWidget)
                return True

        elif eventId == vtk.vtkCommand.LeftButtonReleaseEvent:
            self.isDrawing = False
            self.lastIjk = None
            self.cache.onMouseRelease()
            return True

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
            mask = self._regionGrowing(roi, localSeed, thresholds)
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

    def _regionGrowing(self, roi, localSeed, thresholds):
        """Region growing segmentation.

        Args:
            roi: ROI array.
            localSeed: Seed point in local coordinates.
            thresholds: Intensity thresholds.

        Returns:
            Binary mask array.
        """
        # Use confidence connected for region growing behavior
        sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))
        sitkSeed = (int(localSeed[0]), int(localSeed[1]), int(localSeed[2]))

        try:
            result = sitk.ConfidenceConnected(
                sitkRoi,
                seedList=[sitkSeed],
                numberOfIterations=3,
                multiplier=2.5,
                initialNeighborhoodRadius=2,
                replaceValue=1,
            )
            return sitk.GetArrayFromImage(result).astype(np.uint8)
        except Exception as e:
            logging.error(f"Region growing failed: {e}")
            return np.zeros_like(roi, dtype=np.uint8)

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

    def applyMaskToSegment(self, mask):
        """Apply the computed mask to the current segment.

        Args:
            mask: Binary mask numpy array (z, y, x ordering).
        """
        import vtkSegmentationCorePython as vtkSegmentationCore

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
