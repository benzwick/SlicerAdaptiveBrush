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
    intensity samples at painted locations.
    """

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

        # Visual feedback - use curve nodes for drawing strokes
        self._curve_nodes: list[Any] = []  # All curves (one per stroke)
        self._current_curve: Any = None  # Current stroke's curve
        self._sample_type: str = "foreground"

        # Sampling parameters
        self._sample_spacing = 1  # Sample every N pixels during drag
        self._last_sample_pos: Optional[tuple[int, int]] = None

    @property
    def is_active(self) -> bool:
        """Return whether sampling mode is currently active."""
        return self._active

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
        self._curve_nodes = []
        self._current_curve = None

        logger.debug(f"WizardSampler activated for {sample_type} sampling")

    def deactivate(self) -> None:
        """Deactivate sampling mode."""
        self._active = False
        self._view_widget = None
        self._last_sample_pos = None
        self._current_curve = None
        self._remove_all_curves()
        logger.debug("WizardSampler deactivated")

    def _start_new_curve(self) -> None:
        """Start a new curve for a new stroke.

        Creates a new curve node for each stroke so strokes don't connect.
        """
        try:
            import slicer

            stroke_num = len(self._curve_nodes) + 1
            curve = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode",
                f"WizardSampling_{self._sample_type}_{stroke_num}",
            )

            display_node = curve.GetDisplayNode()
            if display_node:
                colors = {
                    "foreground": (0.0, 0.8, 0.0),  # Green
                    "background": (0.8, 0.0, 0.0),  # Red
                    "boundary": (0.8, 0.8, 0.0),  # Yellow
                }
                color = colors.get(self._sample_type, (0.5, 0.5, 0.5))
                display_node.SetSelectedColor(*color)
                display_node.SetColor(*color)
                display_node.SetLineThickness(0.5)  # Thicker line for visibility
                display_node.SetTextScale(0)  # No labels
                display_node.SetVisibility(True)
                display_node.SetPointLabelsVisibility(False)
                display_node.SetGlyphScale(0.3)  # Small dots at control points

            self._current_curve = curve
            self._curve_nodes.append(curve)
            logger.debug(f"Started new curve stroke {stroke_num}")
        except ImportError:
            self._current_curve = None

    def _remove_all_curves(self) -> None:
        """Remove all visual feedback curve nodes."""
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

    def _add_visual_point(self, ras: tuple[float, float, float]) -> None:
        """Add a visual point at the given RAS coordinates."""
        if self._current_curve:
            self._current_curve.AddControlPoint(ras[0], ras[1], ras[2])

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

        # Start a new curve for this stroke
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

            if ras:
                self._add_visual_point(ras)

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
        ijk = (int(round(ijk_float[0])), int(round(ijk_float[1])), int(round(ijk_float[2])))

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
        self._remove_all_curves()

    @property
    def sample_count(self) -> int:
        """Return the number of samples collected."""
        return len(self._points)
