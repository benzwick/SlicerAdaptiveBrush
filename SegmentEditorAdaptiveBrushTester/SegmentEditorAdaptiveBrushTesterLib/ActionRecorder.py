"""Action recorder for manual testing.

Records user actions during manual testing sessions for:
1. Reproducibility - understand what was tested
2. Test generation - convert actions to automated tests
3. Documentation - capture testing workflows
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

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

    Usage:
        recorder = ActionRecorder(test_run_folder)

        # Start recording
        recorder.start()

        # Record actions
        recorder.record_paint(ijk=(128, 100, 90), algorithm="watershed")
        recorder.record_parameter_change("edge_sensitivity", 0.7)
        recorder.record_screenshot("001_test", "After painting")
        recorder.record_note("Watershed stopped at expected boundary")
        recorder.record_pass("Segmentation looks correct")

        # Stop recording
        recorder.stop()
    """

    def __init__(self, test_run_folder: TestRunFolder) -> None:
        """Initialize action recorder.

        Args:
            test_run_folder: Folder to save actions to.
        """
        self._test_run_folder = test_run_folder
        self._recording = False
        self._action_count = 0

    @property
    def is_recording(self) -> bool:
        """True if currently recording."""
        return self._recording

    @property
    def action_count(self) -> int:
        """Number of actions recorded in current session."""
        return self._action_count

    def start(self) -> None:
        """Start recording actions."""
        if self._recording:
            logger.warning("Already recording")
            return

        self._recording = True
        self._action_count = 0

        self._record_action(ActionType.SESSION_START, {})
        logger.info("Started recording manual actions")

    def stop(self) -> None:
        """Stop recording actions."""
        if not self._recording:
            logger.warning("Not recording")
            return

        self._record_action(ActionType.SESSION_STOP, {"action_count": self._action_count})
        self._recording = False

        logger.info(f"Stopped recording. Total actions: {self._action_count}")

    def _record_action(self, action_type: ActionType, details: dict) -> None:
        """Record an action to the log file.

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

        self._test_run_folder.append_manual_action(action)
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
