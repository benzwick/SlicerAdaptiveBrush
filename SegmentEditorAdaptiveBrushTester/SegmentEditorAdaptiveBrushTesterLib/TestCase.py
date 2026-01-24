"""Base test case class for Slicer testing.

Test cases inherit from TestCase and implement setup, run, verify, and teardown methods.
Tests call Slicer API directly - no wrapper classes are provided.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .TestContext import TestContext

logger = logging.getLogger(__name__)


@dataclass
class Assertion:
    """Result of a test assertion."""

    passed: bool
    message: str
    expected: object = None
    actual: object = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.message}"


@dataclass
class TestResult:
    """Result of running a single test case."""

    name: str
    passed: bool
    assertions: list[Assertion] = field(default_factory=list)
    error: str | None = None
    error_traceback: str | None = None
    duration_seconds: float = 0.0
    screenshots: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def failed_assertions(self) -> list[Assertion]:
        """Return only failed assertions."""
        return [a for a in self.assertions if not a.passed]


@dataclass
class TestCaseInfo:
    """Information about a registered test case."""

    name: str
    description: str
    category: str
    cls: type[TestCase]


class TestCase(ABC):
    """Base class for all test cases.

    Test cases implement:
    - setup(): Prepare test environment (load data, create nodes)
    - run(): Execute the test (paint, change parameters, etc.)
    - verify(): Check results with assertions
    - teardown(): Clean up (optional)

    Tests call Slicer functions directly. The TestContext provides only
    test-specific utilities (screenshots, timing, assertions) - not wrappers
    for Slicer API.

    Example:
        class TestAlgorithmWatershed(TestCase):
            name = "algorithm_watershed"
            description = "Test watershed algorithm on brain tissue"
            category = "algorithm"

            def setup(self, ctx: TestContext):
                import SampleData
                self.volume = SampleData.downloadSample("MRHead")

            def run(self, ctx: TestContext):
                ctx.screenshot("001_before", "Before painting")
                # Call Slicer/effect methods directly
                self._apply_paint()
                ctx.screenshot("002_after", "After painting")

            def verify(self, ctx: TestContext):
                voxel_count = self._count_voxels()
                ctx.assert_greater(voxel_count, 100, "Should segment tissue")
    """

    # Subclasses must set these
    name: str = ""
    description: str = ""
    category: str = "general"

    def __init__(self) -> None:
        """Initialize test case."""
        if not self.name:
            self.name = self.__class__.__name__
        logger.debug(f"Initializing test case: {self.name}")

    @abstractmethod
    def setup(self, ctx: TestContext) -> None:
        """Set up test environment.

        Called before run(). Load sample data, create segmentation nodes,
        configure the effect. Call Slicer API directly.

        Args:
            ctx: Test context for screenshots, timing, assertions.
        """
        pass

    @abstractmethod
    def run(self, ctx: TestContext) -> None:
        """Execute the test actions.

        Perform the main test operations: paint with the brush, change
        parameters, interact with UI. Call Slicer API directly.

        Args:
            ctx: Test context for screenshots, timing, assertions.
        """
        pass

    @abstractmethod
    def verify(self, ctx: TestContext) -> None:
        """Verify test results with assertions.

        Check that the test produced expected results. Use ctx.assert_*
        methods to record pass/fail conditions.

        Args:
            ctx: Test context for screenshots, timing, assertions.
        """
        pass

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test (optional).

        Override to perform cleanup. Called after verify(), even if
        verify() raised an exception.

        Args:
            ctx: Test context for screenshots, timing, assertions.
        """
        logger.debug(f"Teardown for test: {self.name}")
