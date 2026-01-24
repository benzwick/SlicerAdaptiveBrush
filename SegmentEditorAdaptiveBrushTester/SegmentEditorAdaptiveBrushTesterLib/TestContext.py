"""Test context providing test-specific utilities.

TestContext provides utilities for:
- Screenshots: Capture views during tests
- Timing: Measure operation performance
- Assertions: Record pass/fail conditions
- Logging: Test-specific notes

Tests call Slicer API directly - TestContext does NOT wrap Slicer functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .TestCase import Assertion

if TYPE_CHECKING:
    from .MetricsCollector import MetricsCollector, TimingContext
    from .ScreenshotCapture import ScreenshotCapture, ScreenshotInfo
    from .TestRunFolder import TestRunFolder

logger = logging.getLogger(__name__)


class TestContext:
    """Test execution context with test-specific utilities.

    Provides:
    - output_folder: Path to current test run folder
    - screenshot(): Capture and save screenshots
    - timing(): Measure operation duration
    - assert_*(): Record test assertions

    Does NOT provide wrappers for Slicer API. Tests should call Slicer
    functions directly (SampleData.downloadSample, slicer.mrmlScene, etc.).
    """

    def __init__(
        self,
        test_run_folder: TestRunFolder,
        screenshot_capture: ScreenshotCapture,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Initialize test context.

        Args:
            test_run_folder: Folder for test output.
            screenshot_capture: Screenshot capture utility.
            metrics_collector: Metrics collection utility.
        """
        self._test_run_folder = test_run_folder
        self._screenshot_capture = screenshot_capture
        self._metrics_collector = metrics_collector
        self._assertions: list[Assertion] = []
        self._screenshots: list[str] = []

    @property
    def output_folder(self) -> Path:
        """Path to current test run output folder."""
        return self._test_run_folder.path

    @property
    def assertions(self) -> list[Assertion]:
        """List of assertions made during the test."""
        return self._assertions.copy()

    @property
    def screenshots(self) -> list[str]:
        """List of screenshot filenames captured during the test."""
        return self._screenshots.copy()

    @property
    def metrics(self) -> dict:
        """Collected metrics for this test."""
        return self._metrics_collector.get_metrics()

    def screenshot(self, screenshot_id: str, description: str) -> ScreenshotInfo:
        """Capture a screenshot of the current Slicer state.

        Args:
            screenshot_id: Unique identifier for this screenshot (e.g., "001_before").
            description: Human-readable description of what the screenshot shows.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        logger.info(f"Screenshot: {screenshot_id} - {description}")
        info = self._screenshot_capture.capture_layout(
            screenshot_id=screenshot_id,
            description=description,
            output_folder=self._test_run_folder.screenshots_folder,
        )
        self._screenshots.append(info.filename)
        return info

    def screenshot_slice_view(
        self, view: str, screenshot_id: str, description: str
    ) -> ScreenshotInfo:
        """Capture a screenshot of a specific slice view.

        Args:
            view: View name ("Red", "Yellow", "Green").
            screenshot_id: Unique identifier for this screenshot.
            description: Human-readable description.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        logger.info(f"Screenshot slice {view}: {screenshot_id} - {description}")
        info = self._screenshot_capture.capture_slice_view(
            view=view,
            screenshot_id=screenshot_id,
            description=description,
            output_folder=self._test_run_folder.screenshots_folder,
        )
        self._screenshots.append(info.filename)
        return info

    def screenshot_3d_view(self, screenshot_id: str, description: str) -> ScreenshotInfo:
        """Capture a screenshot of the 3D view.

        Args:
            screenshot_id: Unique identifier for this screenshot.
            description: Human-readable description.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        logger.info(f"Screenshot 3D: {screenshot_id} - {description}")
        info = self._screenshot_capture.capture_3d_view(
            screenshot_id=screenshot_id,
            description=description,
            output_folder=self._test_run_folder.screenshots_folder,
        )
        self._screenshots.append(info.filename)
        return info

    def timing(self, operation: str) -> TimingContext:
        """Create a timing context for measuring operation duration.

        Usage:
            with ctx.timing("watershed_stroke"):
                effect.apply(...)

        Args:
            operation: Name of the operation being timed.

        Returns:
            Context manager that records duration on exit.
        """
        logger.debug(f"Starting timing: {operation}")
        return self._metrics_collector.timing(operation)

    def record_metric(self, name: str, value: float, unit: str = "") -> None:
        """Record a custom metric.

        Args:
            name: Metric name (e.g., "voxel_count", "dice_coefficient").
            value: Metric value.
            unit: Optional unit string (e.g., "voxels", "ms").
        """
        logger.debug(f"Metric: {name} = {value} {unit}")
        self._metrics_collector.record_metric(name, value, unit)

    def log(self, message: str) -> None:
        """Log a test-specific note.

        Use for observations during test execution that may help
        with debugging or understanding test behavior.

        Args:
            message: Note to log.
        """
        logger.info(f"[TEST NOTE] {message}")

    def assert_true(self, condition: bool, message: str) -> Assertion:
        """Assert that a condition is true.

        Args:
            condition: Condition to check.
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        assertion = Assertion(passed=condition, message=message)
        self._assertions.append(assertion)
        if condition:
            logger.info(f"PASS: {message}")
        else:
            logger.error(f"FAIL: {message}")
        return assertion

    def assert_false(self, condition: bool, message: str) -> Assertion:
        """Assert that a condition is false.

        Args:
            condition: Condition to check.
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        return self.assert_true(not condition, message)

    def assert_equal(self, actual: object, expected: object, message: str) -> Assertion:
        """Assert that two values are equal.

        Args:
            actual: Actual value.
            expected: Expected value.
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = actual == expected
        assertion = Assertion(
            passed=passed,
            message=message,
            expected=expected,
            actual=actual,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} (actual={actual})")
        else:
            logger.error(f"FAIL: {message} (expected={expected}, actual={actual})")
        return assertion

    def assert_not_equal(self, actual: object, expected: object, message: str) -> Assertion:
        """Assert that two values are not equal.

        Args:
            actual: Actual value.
            expected: Value that should not match.
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = actual != expected
        assertion = Assertion(
            passed=passed,
            message=f"{message} (should not equal {expected})",
            expected=f"not {expected}",
            actual=actual,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} (actual={actual} != {expected})")
        else:
            logger.error(f"FAIL: {message} (actual={actual} equals {expected})")
        return assertion

    def assert_greater(self, actual: float, expected: float, message: str) -> Assertion:
        """Assert that actual > expected.

        Args:
            actual: Actual value.
            expected: Minimum threshold (exclusive).
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = actual > expected
        assertion = Assertion(
            passed=passed,
            message=message,
            expected=f"> {expected}",
            actual=actual,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} ({actual} > {expected})")
        else:
            logger.error(f"FAIL: {message} ({actual} not > {expected})")
        return assertion

    def assert_greater_equal(self, actual: float, expected: float, message: str) -> Assertion:
        """Assert that actual >= expected.

        Args:
            actual: Actual value.
            expected: Minimum threshold (inclusive).
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = actual >= expected
        assertion = Assertion(
            passed=passed,
            message=message,
            expected=f">= {expected}",
            actual=actual,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} ({actual} >= {expected})")
        else:
            logger.error(f"FAIL: {message} ({actual} not >= {expected})")
        return assertion

    def assert_less(self, actual: float, expected: float, message: str) -> Assertion:
        """Assert that actual < expected.

        Args:
            actual: Actual value.
            expected: Maximum threshold (exclusive).
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = actual < expected
        assertion = Assertion(
            passed=passed,
            message=message,
            expected=f"< {expected}",
            actual=actual,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} ({actual} < {expected})")
        else:
            logger.error(f"FAIL: {message} ({actual} not < {expected})")
        return assertion

    def assert_less_equal(self, actual: float, expected: float, message: str) -> Assertion:
        """Assert that actual <= expected.

        Args:
            actual: Actual value.
            expected: Maximum threshold (inclusive).
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = actual <= expected
        assertion = Assertion(
            passed=passed,
            message=message,
            expected=f"<= {expected}",
            actual=actual,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} ({actual} <= {expected})")
        else:
            logger.error(f"FAIL: {message} ({actual} not <= {expected})")
        return assertion

    def assert_is_not_none(self, value: object, message: str) -> Assertion:
        """Assert that a value is not None.

        Args:
            value: Value to check.
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = value is not None
        assertion = Assertion(
            passed=passed,
            message=message,
            expected="not None",
            actual=value,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} (value is not None)")
        else:
            logger.error(f"FAIL: {message} (value is None)")
        return assertion

    def assert_is_none(self, value: object, message: str) -> Assertion:
        """Assert that a value is None.

        Args:
            value: Value to check.
            message: Description of what is being checked.

        Returns:
            Assertion result.
        """
        passed = value is None
        assertion = Assertion(
            passed=passed,
            message=message,
            expected="None",
            actual=value,
        )
        self._assertions.append(assertion)
        if passed:
            logger.info(f"PASS: {message} (value is None)")
        else:
            logger.error(f"FAIL: {message} (value is {value}, not None)")
        return assertion
