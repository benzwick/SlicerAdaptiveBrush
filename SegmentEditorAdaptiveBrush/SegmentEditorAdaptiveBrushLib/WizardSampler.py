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
        self._observations: list[tuple[Any, int]] = []

        # Sampling parameters
        self._sample_spacing = 1  # Sample every N pixels during drag
        self._last_sample_pos: Optional[tuple[int, int]] = None

    @property
    def is_active(self) -> bool:
        """Return whether sampling mode is currently active."""
        return self._active

    def activate(self, view_widget: Any) -> None:
        """Activate sampling mode in a slice view.

        Args:
            view_widget: The Slicer slice view widget to capture events from.
        """
        if self._active:
            self.deactivate()

        self._view_widget = view_widget
        self._active = True
        self._points = []
        self._intensities = []
        self._last_sample_pos = None

        # Add event observers
        try:
            interactor = view_widget.sliceView().interactor()
            if interactor:
                # Observe mouse events
                events_to_observe = [
                    ("LeftButtonPressEvent", self._on_left_button_press),
                    ("LeftButtonReleaseEvent", self._on_left_button_release),
                    ("MouseMoveEvent", self._on_mouse_move),
                ]
                for event_name, handler in events_to_observe:
                    tag = interactor.AddObserver(event_name, handler)
                    self._observations.append((interactor, tag))

                logger.debug("WizardSampler activated on view widget")
        except Exception as e:
            logger.warning(f"Failed to activate WizardSampler: {e}")
            self._active = False

    def deactivate(self) -> None:
        """Deactivate sampling mode and clean up observers."""
        # Remove event observers
        for interactor, tag in self._observations:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._observations = []

        self._active = False
        self._view_widget = None
        self._last_sample_pos = None

        logger.debug("WizardSampler deactivated")

    def process_event(self, caller: Any, event_id: str) -> bool:
        """Process mouse events for sampling.

        Args:
            caller: The interactor that generated the event.
            event_id: The event identifier string.

        Returns:
            True if the event was handled, False otherwise.
        """
        if not self._active:
            return False

        if event_id == "LeftButtonPressEvent":
            self._on_left_button_press(caller, event_id)
            return True
        elif event_id == "LeftButtonReleaseEvent":
            self._on_left_button_release(caller, event_id)
            return True
        elif event_id == "MouseMoveEvent":
            self._on_mouse_move(caller, event_id)
            return True

        return False

    def _on_left_button_press(self, caller: Any, event: str) -> None:
        """Handle left mouse button press - start sampling."""
        if not self._active:
            return

        try:
            xy = self._get_event_position(caller)
            if xy:
                self._sample_at_xy(xy)
                self._last_sample_pos = xy
        except Exception as e:
            logger.debug(f"Error in left button press: {e}")

    def _on_left_button_release(self, caller: Any, event: str) -> None:
        """Handle left mouse button release - finalize sampling stroke."""
        if not self._active:
            return

        self._last_sample_pos = None

        # Notify callback if registered
        if self.callback and len(self._points) > 0:
            try:
                self.callback(self._points.copy(), np.array(self._intensities))
            except Exception as e:
                logger.warning(f"Error in sampling callback: {e}")

    def _on_mouse_move(self, caller: Any, event: str) -> None:
        """Handle mouse movement - sample during drag."""
        if not self._active:
            return

        # Only sample during drag (left button held)
        try:
            interactor = caller
            if not interactor.GetLeftButtonDown():
                return

            xy = self._get_event_position(caller)
            if xy and self._should_sample(xy):
                self._sample_at_xy(xy)
                self._last_sample_pos = xy
        except Exception as e:
            logger.debug(f"Error in mouse move: {e}")

    def _get_event_position(self, caller: Any) -> Optional[tuple[int, int]]:
        """Get the current mouse position from the interactor.

        Returns:
            (x, y) screen coordinates or None if unavailable.
        """
        try:
            pos = caller.GetEventPosition()
            return (pos[0], pos[1])
        except Exception:
            return None

    def _should_sample(self, xy: tuple[int, int]) -> bool:
        """Check if we should sample at this position (spacing check)."""
        if self._last_sample_pos is None:
            return True

        dx = abs(xy[0] - self._last_sample_pos[0])
        dy = abs(xy[1] - self._last_sample_pos[1])
        distance = (dx**2 + dy**2) ** 0.5

        return bool(distance >= self._sample_spacing)

    def _sample_at_xy(self, xy: tuple[int, int]) -> None:
        """Sample intensity at screen coordinates.

        Args:
            xy: Screen (x, y) coordinates.
        """
        if not self._view_widget or not self.volume_node:
            return

        try:
            # Convert screen XY to volume IJK coordinates
            ijk = self._xy_to_ijk(xy)
            if ijk is None:
                return

            # Get intensity at IJK
            intensity = self._get_intensity_at_ijk(ijk)
            if intensity is not None:
                self._points.append(ijk)
                self._intensities.append(intensity)

        except Exception as e:
            logger.debug(f"Error sampling at {xy}: {e}")

    def _xy_to_ijk(self, xy: tuple[int, int]) -> Optional[tuple[int, int, int]]:
        """Convert screen XY to volume IJK coordinates.

        Args:
            xy: Screen coordinates (x, y).

        Returns:
            Volume IJK coordinates (i, j, k) or None if conversion fails.
        """
        try:
            import vtk

            sliceLogic = self._view_widget.sliceLogic()
            sliceNode = sliceLogic.GetSliceNode()

            # Get XY to RAS transform
            xyToRas = sliceNode.GetXYToRAS()
            ras = xyToRas.MultiplyPoint([xy[0], xy[1], 0, 1])

            # Get RAS to IJK transform
            rasToIjk = vtk.vtkMatrix4x4()
            self.volume_node.GetRASToIJKMatrix(rasToIjk)

            # Convert RAS to IJK
            ijk_float = rasToIjk.MultiplyPoint([ras[0], ras[1], ras[2], 1])
            ijk = (int(round(ijk_float[0])), int(round(ijk_float[1])), int(round(ijk_float[2])))

            # Bounds check
            dims = self.volume_node.GetImageData().GetDimensions()
            if not (0 <= ijk[0] < dims[0] and 0 <= ijk[1] < dims[1] and 0 <= ijk[2] < dims[2]):
                return None

            return ijk

        except Exception as e:
            logger.debug(f"XY to IJK conversion failed: {e}")
            return None

    def _get_intensity_at_ijk(self, ijk: tuple[int, int, int]) -> Optional[float]:
        """Get intensity value at IJK coordinates.

        Args:
            ijk: Volume coordinates (i, j, k).

        Returns:
            Intensity value or None if unavailable.
        """
        try:
            imageData = self.volume_node.GetImageData()
            if imageData is None:
                return None

            scalar = imageData.GetScalarComponentAsFloat(ijk[0], ijk[1], ijk[2], 0)
            return float(scalar)

        except Exception as e:
            logger.debug(f"Failed to get intensity at {ijk}: {e}")
            return None

    def get_samples(self) -> tuple[list[tuple[int, int, int]], np.ndarray]:
        """Return collected points and intensities.

        Returns:
            Tuple of (points list, intensities array).
        """
        if len(self._intensities) == 0:
            return [], np.array([])
        return self._points.copy(), np.array(self._intensities)

    def clear(self) -> None:
        """Clear collected samples."""
        self._points = []
        self._intensities = []
        self._last_sample_pos = None

    @property
    def sample_count(self) -> int:
        """Return the number of samples collected."""
        return len(self._points)
