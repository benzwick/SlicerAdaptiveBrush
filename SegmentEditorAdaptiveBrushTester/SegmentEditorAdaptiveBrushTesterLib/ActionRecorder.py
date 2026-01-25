"""Action recorder for manual testing.

Records user actions during manual testing sessions for:
1. Reproducibility - understand what was tested
2. Test generation - convert actions to automated tests
3. Documentation - capture testing workflows

Uses Slicer observers to automatically capture:
- Segmentation modifications (detected via node observers)
- Effect parameter state at time of modification
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .TestRunFolder import TestRunFolder

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of recordable actions."""

    PAINT = "paint"
    PARAMETER_CHANGE = "parameter_change"
    ALGORITHM_CHANGE = "algorithm_change"
    SCREENSHOT = "screenshot"
    NOTE = "note"
    MARK_PASS = "mark_pass"
    MARK_FAIL = "mark_fail"
    SESSION_START = "session_start"
    SESSION_STOP = "session_stop"


class ActionRecorder:
    """Records user actions during manual testing.

    Automatically captures segmentation modifications via Slicer observers.
    Records effect parameters at time of each paint action for reproducibility.

    Usage:
        recorder = ActionRecorder(test_run_folder)

        # Start recording with automatic capture
        recorder.start()

        # Manual records (screenshots, notes) still work
        recorder.record_screenshot("001_test", "After painting")
        recorder.record_note("Watershed stopped at expected boundary")
        recorder.record_pass("Segmentation looks correct")

        # Stop recording
        recorder.stop()

    Callback mode (for branch recording):
        def my_callback(action_dict):
            print(f"Action recorded: {action_dict['type']}")

        recorder = ActionRecorder(test_run_folder, action_callback=my_callback)
        recorder.start()
        # ... actions are recorded and callback is called ...
    """

    def __init__(
        self,
        test_run_folder: TestRunFolder | None = None,
        action_callback: Any | None = None,
    ) -> None:
        """Initialize action recorder.

        Args:
            test_run_folder: Folder to save actions to. Optional if using callback mode.
            action_callback: Optional callback function that receives each recorded action.
                The callback receives a dict with action details (type, timestamp, etc.).
                Useful for branch recording integration.
        """
        self._test_run_folder = test_run_folder
        self._action_callback = action_callback
        self._recording = False
        self._action_count = 0
        self._observers: list[tuple[Any, int]] = []  # (node, observerId) pairs
        self._last_effect_state: dict | None = None
        self._segmentation_node = None

    @property
    def is_recording(self) -> bool:
        """True if currently recording."""
        return self._recording

    @property
    def action_count(self) -> int:
        """Number of actions recorded in current session."""
        return self._action_count

    @property
    def has_callback(self) -> bool:
        """True if an action callback is set."""
        return self._action_callback is not None

    def set_action_callback(self, callback: Any | None) -> None:
        """Set or clear the action callback.

        Args:
            callback: Callback function that receives action dicts, or None to clear.
        """
        self._action_callback = callback

    def start(self) -> None:
        """Start recording actions.

        Sets up observers on the current segmentation node to automatically
        capture paint events and parameter changes.
        """
        if self._recording:
            logger.warning("Already recording")
            return

        self._recording = True
        self._action_count = 0

        # Set up automatic capture via observers
        self._setup_observers()

        # Record initial effect state
        self._last_effect_state = self._get_effect_state()

        self._record_action(ActionType.SESSION_START, {"initial_state": self._last_effect_state})
        logger.info("Started recording manual actions with automatic capture")

    def stop(self) -> None:
        """Stop recording actions.

        Removes all observers and saves final state.
        """
        if not self._recording:
            logger.warning("Not recording")
            return

        # Clean up observers
        self._remove_observers()

        self._record_action(ActionType.SESSION_STOP, {"action_count": self._action_count})
        self._recording = False
        self._last_effect_state = None

        logger.info(f"Stopped recording. Total actions: {self._action_count}")

    def _setup_observers(self) -> None:
        """Set up Slicer observers for automatic action capture."""
        try:
            import slicer

            # Find the active segmentation node
            segmentEditorNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentEditorNode")
            if segmentEditorNode:
                self._segmentation_node = segmentEditorNode.GetSegmentationNode()

            if self._segmentation_node:
                # Observe segmentation modifications (paint events)
                observerId = self._segmentation_node.AddObserver(
                    slicer.vtkMRMLSegmentationNode.SegmentModified,
                    self._on_segment_modified,
                )
                self._observers.append((self._segmentation_node, observerId))
                logger.info(f"Observing segmentation: {self._segmentation_node.GetName()}")
            else:
                logger.warning("No segmentation node found for observation")

        except Exception as e:
            logger.warning(f"Could not set up observers: {e}")

    def _remove_observers(self) -> None:
        """Remove all Slicer observers."""
        for node, observerId in self._observers:
            try:
                node.RemoveObserver(observerId)
            except Exception as e:
                logger.warning(f"Error removing observer: {e}")
        self._observers.clear()
        self._segmentation_node = None

    def _on_segment_modified(self, caller, event) -> None:
        """Callback when a segment is modified (paint event detected)."""
        if not self._recording:
            return

        # Get current effect state
        current_state = self._get_effect_state()
        if current_state is None:
            return

        # Detect parameter changes since last action
        if self._last_effect_state:
            self._detect_and_record_changes(self._last_effect_state, current_state)

        # Get cursor position for reproducibility
        cursor_ras, cursor_ijk = self._get_cursor_position()

        # Record the paint action with full state
        self._record_action(
            ActionType.PAINT,
            {
                "algorithm": current_state.get("algorithm"),
                "mode": "erase" if current_state.get("erase_mode") else "add",
                "radius_mm": current_state.get("radius_mm"),
                "is_3d": current_state.get("sphere_mode"),
                "edge_sensitivity": current_state.get("edge_sensitivity"),
                "threshold_zone": current_state.get("threshold_zone"),
                "cursor_ras": cursor_ras,
                "cursor_ijk": cursor_ijk,
                "full_state": current_state,
            },
        )

        self._last_effect_state = current_state
        logger.debug(f"Recorded paint at {cursor_ijk} with {current_state.get('algorithm')}")

    def _get_effect_state(self) -> dict | None:
        """Get current Adaptive Brush effect state.

        Returns:
            Dictionary of effect parameters, or None if effect not active.
        """
        try:
            import slicer

            # Get the segment editor widget
            segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            if segmentEditorWidget is None:
                return None

            effect = segmentEditorWidget.activeEffect()
            if effect is None or effect.name != "Adaptive Brush":
                return None

            # Get the scripted effect instance
            scripted = effect.self()
            if scripted is None:
                return None

            # Capture all relevant parameters
            return {
                "algorithm": getattr(scripted, "algorithm", None),
                "radius_mm": getattr(scripted, "radiusMm", None),
                "sphere_mode": getattr(scripted, "sphereMode", None),
                "erase_mode": getattr(scripted, "eraseMode", None),
                "edge_sensitivity": getattr(scripted, "edgeSensitivity", None),
                "threshold_zone": getattr(scripted, "thresholdZone", None),
                "threshold_min": getattr(scripted, "thresholdMin", None),
                "threshold_max": getattr(scripted, "thresholdMax", None),
                "preview_mode": getattr(scripted, "previewMode", None),
                "level_set_iterations": getattr(scripted, "levelSetIterations", None),
                "threshold_method": (
                    scripted.thresholdMethodCombo.currentData
                    if hasattr(scripted, "thresholdMethodCombo")
                    else None
                ),
            }

        except Exception as e:
            logger.debug(f"Could not get effect state: {e}")
            return None

    def _detect_and_record_changes(self, old_state: dict, new_state: dict) -> None:
        """Detect and record parameter changes between states."""
        for key in new_state:
            old_val = old_state.get(key)
            new_val = new_state.get(key)
            if old_val != new_val and new_val is not None:
                if key == "algorithm":
                    self.record_algorithm_change(new_val)
                else:
                    self.record_parameter_change(key, new_val)

    def _get_cursor_position(self) -> tuple[list[float] | None, list[int] | None]:
        """Get current crosshair cursor position.

        Returns:
            Tuple of (RAS coordinates, IJK coordinates) or (None, None).
        """
        try:
            import slicer

            crosshairNode = slicer.util.getNode("Crosshair")
            if crosshairNode is None:
                return None, None

            ras = [0.0, 0.0, 0.0]
            crosshairNode.GetCursorPositionRAS(ras)

            # Convert RAS to IJK if we have a source volume
            ijk = None
            segmentEditorNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentEditorNode")
            if segmentEditorNode:
                sourceVolume = segmentEditorNode.GetSourceVolumeNode()
                if sourceVolume:
                    import vtk

                    rasToIjk = vtk.vtkMatrix4x4()
                    sourceVolume.GetRASToIJKMatrix(rasToIjk)
                    ijkFloat = rasToIjk.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])
                    ijk = [
                        int(round(ijkFloat[0])),
                        int(round(ijkFloat[1])),
                        int(round(ijkFloat[2])),
                    ]

            return ras, ijk

        except Exception as e:
            logger.debug(f"Could not get cursor position: {e}")
            return None, None

    def _record_action(self, action_type: ActionType, details: dict) -> None:
        """Record an action to the log file and/or callback.

        Args:
            action_type: Type of action.
            details: Action-specific details.
        """
        action = {
            "type": action_type.value,
            "timestamp": datetime.now().isoformat(),
            "sequence": self._action_count,
            **details,
        }

        # Save to test run folder if available
        if self._test_run_folder is not None:
            self._test_run_folder.append_manual_action(action)

        # Call callback if set (for branch recording integration)
        if self._action_callback is not None:
            try:
                self._action_callback(action)
            except Exception as e:
                logger.warning(f"Action callback failed: {e}")

        self._action_count += 1

    def record_paint(
        self,
        ijk: tuple[int, int, int],
        algorithm: str,
        mode: str = "add",
        radius_mm: float | None = None,
        is_3d: bool = False,
    ) -> None:
        """Record a paint action.

        Args:
            ijk: Voxel coordinates (i, j, k).
            algorithm: Algorithm name used for painting.
            mode: "add" or "erase".
            radius_mm: Brush radius in mm.
            is_3d: Whether 3D brush mode is enabled.
        """
        if not self._recording:
            return

        self._record_action(
            ActionType.PAINT,
            {
                "ijk": list(ijk),
                "algorithm": algorithm,
                "mode": mode,
                "radius_mm": radius_mm,
                "is_3d": is_3d,
            },
        )

        logger.debug(f"Recorded paint at {ijk} with {algorithm}")

    def record_parameter_change(self, parameter: str, value) -> None:
        """Record a parameter change.

        Args:
            parameter: Parameter name.
            value: New value.
        """
        if not self._recording:
            return

        self._record_action(
            ActionType.PARAMETER_CHANGE,
            {
                "parameter": parameter,
                "value": value,
            },
        )

        logger.debug(f"Recorded parameter change: {parameter} = {value}")

    def record_algorithm_change(self, algorithm: str) -> None:
        """Record an algorithm change.

        Args:
            algorithm: New algorithm name.
        """
        if not self._recording:
            return

        self._record_action(
            ActionType.ALGORITHM_CHANGE,
            {
                "algorithm": algorithm,
            },
        )

        logger.debug(f"Recorded algorithm change: {algorithm}")

    def record_screenshot(self, screenshot_id: str, description: str) -> None:
        """Record a screenshot capture.

        Args:
            screenshot_id: Screenshot identifier.
            description: Screenshot description.
        """
        if not self._recording:
            return

        self._record_action(
            ActionType.SCREENSHOT,
            {
                "screenshot_id": screenshot_id,
                "description": description,
            },
        )

        logger.debug(f"Recorded screenshot: {screenshot_id}")

    def record_note(self, note: str) -> None:
        """Record a user note.

        Args:
            note: User's observation or note.
        """
        if not self._recording:
            return

        self._record_action(
            ActionType.NOTE,
            {
                "note": note,
            },
        )

        logger.debug(f"Recorded note: {note[:50]}...")

    def record_pass(self, reason: str = "") -> None:
        """Record a manual pass mark.

        Args:
            reason: Reason for passing.
        """
        if not self._recording:
            return

        self._record_action(
            ActionType.MARK_PASS,
            {
                "reason": reason,
            },
        )

        logger.info(f"Marked PASS: {reason}")

    def record_fail(self, reason: str = "") -> None:
        """Record a manual fail mark.

        Args:
            reason: Reason for failing.
        """
        if not self._recording:
            return

        self._record_action(
            ActionType.MARK_FAIL,
            {
                "reason": reason,
            },
        )

        logger.warning(f"Marked FAIL: {reason}")
