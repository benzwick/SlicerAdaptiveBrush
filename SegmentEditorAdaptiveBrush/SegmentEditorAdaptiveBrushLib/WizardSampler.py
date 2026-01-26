"""Wizard Sampler for interactive sample collection in slice views.

This module handles mouse events for paint-style sampling during the
Quick Select Parameters wizard workflow.
"""

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class WizardSampler:
    """Handles interactive sampling in slice views.

    This class captures mouse events in Slicer slice views to collect
    intensity samples at painted locations. Visual feedback is provided
    via labelmap overlay for foreground/background (paint strokes) and
    markup curves for boundary tracing.
    """

    # Brush radius in mm for paint-style visual feedback
    PAINT_BRUSH_RADIUS_MM = 2.0

    def __init__(
        self,
        volume_node: Any,
        callback: Optional[Callable[[list, np.ndarray], None]] = None,
    ):
        """Initialize the sampler.

        Args:
            volume_node: The source volume node to sample from.
            callback: Optional callback function called with (points, intensities)
                     when sampling completes.
        """
        self.volume_node = volume_node
        self.callback = callback
        self._points: list[tuple[int, int, int]] = []
        self._intensities: list[float] = []
        self._active = False
        self._view_widget: Any = None

        # Visual feedback - labelmap for paint, curves for boundary
        self._labelmap_node: Any = None
        self._labelmap_display_node: Any = None
        self._curve_nodes: list[Any] = []
        self._current_curve: Any = None
        self._sample_type: str = "foreground"

        # Sampling parameters
        self._sample_spacing = 1  # Sample every N pixels during drag
        self._last_sample_pos: Optional[tuple[int, int]] = None

    @property
    def is_active(self) -> bool:
        """Return whether sampling mode is currently active."""
        return self._active

    @property
    def _is_boundary_mode(self) -> bool:
        """Return whether we're in boundary tracing mode."""
        return self._sample_type == "boundary"

    def activate(self, view_widget: Any, sample_type: str = "foreground") -> None:
        """Activate sampling mode in a slice view."""
        if self._active:
            self.deactivate()

        self._view_widget = view_widget
        self._active = True
        self._points = []
        self._intensities = []
        self._last_sample_pos = None
        self._sample_type = sample_type

        if self._is_boundary_mode:
            # Boundary uses curves - will create on first stroke
            self._curve_nodes = []
            self._current_curve = None
        else:
            # Foreground/background use labelmap paint
            self._create_labelmap_overlay()

        logger.debug(f"WizardSampler activated for {sample_type} sampling")

    def deactivate(self) -> None:
        """Deactivate sampling mode."""
        self._active = False
        self._view_widget = None
        self._last_sample_pos = None

        # Clean up both visualization types
        self._remove_labelmap_overlay()
        self._remove_all_curves()

        logger.debug("WizardSampler deactivated")

    # ========== Labelmap (Paint) Visualization ==========

    def _create_labelmap_overlay(self) -> None:
        """Create a labelmap volume for paint-style visual feedback."""
        try:
            import slicer
            import vtk

            self._remove_labelmap_overlay()

            if not self.volume_node:
                return

            # Create labelmap with same geometry as source volume
            self._labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode",
                f"WizardSampling_{self._sample_type}",
            )

            # Copy geometry from source volume
            imageData = vtk.vtkImageData()
            imageData.SetDimensions(self.volume_node.GetImageData().GetDimensions())
            imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
            imageData.GetPointData().GetScalars().Fill(0)

            self._labelmap_node.SetAndObserveImageData(imageData)
            self._labelmap_node.CopyOrientation(self.volume_node)

            # Set up display with appropriate color
            self._labelmap_display_node = self._labelmap_node.GetDisplayNode()
            if not self._labelmap_display_node:
                self._labelmap_node.CreateDefaultDisplayNodes()
                self._labelmap_display_node = self._labelmap_node.GetDisplayNode()

            if self._labelmap_display_node:
                # Create color table with sample type color
                colors = {
                    "foreground": (0, 200, 0),  # Green
                    "background": (200, 0, 0),  # Red
                }
                rgb = colors.get(self._sample_type, (128, 128, 128))

                # Create a simple color node
                colorNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLColorTableNode",
                    f"WizardColor_{self._sample_type}",
                )
                colorNode.SetTypeToUser()
                colorNode.SetNumberOfColors(2)
                colorNode.SetColor(0, "Background", 0, 0, 0, 0)  # Transparent
                colorNode.SetColor(1, "Paint", rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 0.6)

                self._labelmap_display_node.SetAndObserveColorNodeID(colorNode.GetID())
                self._labelmap_display_node.SetVisibility(True)

            # Set labelmap as label layer in all slice views so it's visible
            self._set_labelmap_in_slice_views(self._labelmap_node)

            logger.debug(f"Created labelmap overlay for {self._sample_type}")

        except ImportError:
            self._labelmap_node = None

    def _set_labelmap_in_slice_views(self, labelmap_node: Any) -> None:
        """Set the labelmap as the label layer in all slice views."""
        try:
            import slicer

            # Store previous label volumes to restore later
            self._previous_label_volumes: dict[str, str] = {}

            layoutManager = slicer.app.layoutManager()
            for sliceViewName in layoutManager.sliceViewNames():
                sliceWidget = layoutManager.sliceWidget(sliceViewName)
                compositeNode = sliceWidget.sliceLogic().GetSliceCompositeNode()

                # Store previous label volume
                prevLabelID = compositeNode.GetLabelVolumeID()
                self._previous_label_volumes[sliceViewName] = prevLabelID

                # Set our labelmap as the label layer
                compositeNode.SetLabelVolumeID(labelmap_node.GetID())
                compositeNode.SetLabelOpacity(0.6)

        except Exception as e:
            logger.warning(f"Failed to set labelmap in slice views: {e}")

    def _restore_previous_label_volumes(self) -> None:
        """Restore previous label volumes in slice views."""
        try:
            import slicer

            if not hasattr(self, "_previous_label_volumes"):
                return

            layoutManager = slicer.app.layoutManager()
            for sliceViewName, prevLabelID in self._previous_label_volumes.items():
                sliceWidget = layoutManager.sliceWidget(sliceViewName)
                if sliceWidget:
                    compositeNode = sliceWidget.sliceLogic().GetSliceCompositeNode()
                    compositeNode.SetLabelVolumeID(prevLabelID if prevLabelID else "")

            self._previous_label_volumes = {}

        except Exception as e:
            logger.warning(f"Failed to restore label volumes: {e}")

    def _remove_labelmap_overlay(self) -> None:
        """Remove the labelmap overlay."""
        if self._labelmap_node is None:
            return
        try:
            import slicer

            # Restore previous label volumes first
            self._restore_previous_label_volumes()

            # Remove associated color node
            if self._labelmap_display_node:
                colorNodeID = self._labelmap_display_node.GetColorNodeID()
                if colorNodeID:
                    colorNode = slicer.mrmlScene.GetNodeByID(colorNodeID)
                    if colorNode and "WizardColor" in colorNode.GetName():
                        slicer.mrmlScene.RemoveNode(colorNode)

            slicer.mrmlScene.RemoveNode(self._labelmap_node)
        except ImportError:
            pass
        self._labelmap_node = None
        self._labelmap_display_node = None

    def _paint_at_ijk(self, ijk: tuple[int, int, int]) -> None:
        """Paint a brush stroke at the given IJK coordinates."""
        if not self._labelmap_node:
            return

        imageData = self._labelmap_node.GetImageData()
        if not imageData:
            return

        dims = imageData.GetDimensions()
        spacing = self.volume_node.GetSpacing()

        # Calculate brush radius in voxels
        radius_voxels = [max(1, int(self.PAINT_BRUSH_RADIUS_MM / spacing[i])) for i in range(3)]

        # Paint a small sphere/circle at the location
        for di in range(-radius_voxels[0], radius_voxels[0] + 1):
            for dj in range(-radius_voxels[1], radius_voxels[1] + 1):
                # Check if within circular brush (2D painting)
                if di * di + dj * dj > radius_voxels[0] * radius_voxels[1]:
                    continue

                ni, nj, nk = ijk[0] + di, ijk[1] + dj, ijk[2]

                # Bounds check
                if 0 <= ni < dims[0] and 0 <= nj < dims[1] and 0 <= nk < dims[2]:
                    imageData.SetScalarComponentFromFloat(ni, nj, nk, 0, 1)

        imageData.Modified()
        self._labelmap_node.Modified()

    # ========== Curve (Boundary) Visualization ==========

    def _start_new_curve(self) -> None:
        """Start a new curve for boundary tracing."""
        try:
            import slicer

            stroke_num = len(self._curve_nodes) + 1
            curve = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode",
                f"WizardBoundary_{stroke_num}",
            )

            # Use linear interpolation (not spline) so curve follows exactly what user draws
            curve.SetCurveTypeToLinear()

            # Just set color to yellow, keep other defaults like Markups module
            display_node = curve.GetDisplayNode()
            if display_node:
                display_node.SetSelectedColor(1.0, 1.0, 0.0)
                display_node.SetColor(1.0, 1.0, 0.0)
                display_node.SetTextScale(0)  # No labels

            self._current_curve = curve
            self._curve_nodes.append(curve)
            logger.debug(f"Started new boundary curve {stroke_num}")
        except ImportError:
            self._current_curve = None

    def _remove_all_curves(self) -> None:
        """Remove all boundary curve nodes."""
        if not self._curve_nodes:
            return
        try:
            import slicer

            for curve in self._curve_nodes:
                if curve:
                    slicer.mrmlScene.RemoveNode(curve)
        except ImportError:
            pass
        self._curve_nodes = []
        self._current_curve = None

    def _add_curve_point(self, ras: tuple[float, float, float]) -> None:
        """Add a point to the current boundary curve."""
        if self._current_curve:
            n = self._current_curve.AddControlPoint(ras[0], ras[1], ras[2])
            # Update curve line representation
            self._current_curve.UpdateCurvePolyFromControlPoints()
            if n < 3:  # Log first few points for debugging
                logger.debug(
                    f"Added curve point {n}: RAS=({ras[0]:.1f}, {ras[1]:.1f}, {ras[2]:.1f})"
                )

    # ========== Event Handling ==========

    def process_event(self, caller: Any, event_id: str) -> bool:
        """Process mouse events for sampling."""
        if not self._active:
            return False

        if event_id == "LeftButtonPressEvent":
            self._on_left_button_press(caller)
            return True
        elif event_id == "LeftButtonReleaseEvent":
            self._on_left_button_release()
            return True
        elif event_id == "MouseMoveEvent":
            self._on_mouse_move(caller)
            return True

        return False

    def _on_left_button_press(self, caller: Any) -> None:
        """Handle left mouse button press - start sampling stroke."""
        if not self._active:
            return

        # Start new curve for boundary tracing
        if self._is_boundary_mode:
            self._start_new_curve()

        xy = self._get_event_position(caller)
        if xy:
            self._sample_at_xy(xy)
            self._last_sample_pos = xy
            logger.debug(f"WizardSampler: started stroke at {xy}, count={len(self._points)}")

    def _on_left_button_release(self) -> None:
        """Handle left mouse button release - finalize sampling stroke."""
        if not self._active:
            return

        self._last_sample_pos = None

        # Notify callback if registered
        if self.callback and len(self._points) > 0:
            logger.info(f"WizardSampler: stroke complete with {len(self._points)} samples")
            # Wrap callback in try/except - it's external code we don't control
            try:
                self.callback(self._points.copy(), np.array(self._intensities))
            except Exception as e:
                logger.exception(f"Error in sampling callback: {e}")

    def _on_mouse_move(self, caller: Any) -> None:
        """Handle mouse movement - sample during drag.

        Note: This is only called when the left button is down (checked by
        ParameterWizard.handle_interaction_event before forwarding).
        """
        if not self._active:
            return

        xy = self._get_event_position(caller)
        if xy and self._should_sample(xy):
            self._sample_at_xy(xy)
            self._last_sample_pos = xy

    def _get_event_position(self, caller: Any) -> Optional[tuple[int, int]]:
        """Get the current mouse position from the interactor."""
        pos = caller.GetEventPosition()
        return (pos[0], pos[1])

    def _should_sample(self, xy: tuple[int, int]) -> bool:
        """Check if we should sample at this position (spacing check)."""
        if self._last_sample_pos is None:
            return True

        dx = abs(xy[0] - self._last_sample_pos[0])
        dy = abs(xy[1] - self._last_sample_pos[1])
        distance = (dx**2 + dy**2) ** 0.5

        return bool(distance >= self._sample_spacing)

    def _sample_at_xy(self, xy: tuple[int, int]) -> None:
        """Sample intensity at screen coordinates."""
        if not self._view_widget or not self.volume_node:
            return

        ras, ijk = self._xy_to_coords(xy)
        if ijk is None:
            return

        intensity = self._get_intensity_at_ijk(ijk)
        if intensity is not None:
            self._points.append(ijk)
            self._intensities.append(intensity)

            # Visual feedback based on mode
            if self._is_boundary_mode:
                if ras:
                    self._add_curve_point(ras)
            else:
                self._paint_at_ijk(ijk)

    def _xy_to_coords(
        self, xy: tuple[int, int]
    ) -> tuple[Optional[tuple[float, float, float]], Optional[tuple[int, int, int]]]:
        """Convert screen XY to RAS and volume IJK coordinates."""
        import vtk

        sliceLogic = self._view_widget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()

        # Get XY to RAS transform
        xyToRas = sliceNode.GetXYToRAS()
        ras_result = xyToRas.MultiplyPoint([xy[0], xy[1], 0, 1])
        ras = (float(ras_result[0]), float(ras_result[1]), float(ras_result[2]))

        # Get RAS to IJK transform
        rasToIjk = vtk.vtkMatrix4x4()
        self.volume_node.GetRASToIJKMatrix(rasToIjk)

        # Convert RAS to IJK
        ijk_float = rasToIjk.MultiplyPoint([ras[0], ras[1], ras[2], 1])
        ijk = (
            int(round(ijk_float[0])),
            int(round(ijk_float[1])),
            int(round(ijk_float[2])),
        )

        # Bounds check
        dims = self.volume_node.GetImageData().GetDimensions()
        if not (0 <= ijk[0] < dims[0] and 0 <= ijk[1] < dims[1] and 0 <= ijk[2] < dims[2]):
            return ras, None

        return ras, ijk

    def _get_intensity_at_ijk(self, ijk: tuple[int, int, int]) -> Optional[float]:
        """Get intensity value at IJK coordinates."""
        imageData = self.volume_node.GetImageData()
        if imageData is None:
            return None

        scalar = imageData.GetScalarComponentAsFloat(ijk[0], ijk[1], ijk[2], 0)
        return float(scalar)

    def get_samples(self) -> tuple[list[tuple[int, int, int]], np.ndarray]:
        """Return collected points and intensities."""
        if len(self._intensities) == 0:
            return [], np.array([])
        return self._points.copy(), np.array(self._intensities)

    def clear(self) -> None:
        """Clear collected samples and visual feedback."""
        self._points = []
        self._intensities = []
        self._last_sample_pos = None

        # Clear visual feedback based on mode
        if self._is_boundary_mode:
            self._remove_all_curves()
        else:
            # Clear the labelmap
            if self._labelmap_node:
                imageData = self._labelmap_node.GetImageData()
                if imageData:
                    imageData.GetPointData().GetScalars().Fill(0)
                    imageData.Modified()
                    self._labelmap_node.Modified()

    @property
    def sample_count(self) -> int:
        """Return the number of samples collected."""
        return len(self._points)
