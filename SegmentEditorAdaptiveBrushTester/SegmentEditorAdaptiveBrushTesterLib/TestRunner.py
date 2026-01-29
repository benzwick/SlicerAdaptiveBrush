"""Test runner for executing registered test cases.

The TestRunner coordinates test execution:
1. Creates test run folder for output
2. Sets up TestContext with utilities
3. Runs test setup, run, verify, teardown phases
4. Collects results and generates reports
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .MetricsCollector import MetricsCollector
from .ScreenshotCapture import ScreenshotCapture
from .TestCase import TestCaseInfo, TestResult
from .TestContext import TestContext
from .TestRegistry import TestRegistry
from .TestRunFolder import TestRunFolder

logger = logging.getLogger(__name__)


@dataclass
class TestSuiteResult:
    """Result of running a test suite."""

    suite_name: str
    output_folder: Path
    results: list[TestResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        """True if all tests passed."""
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of tests that failed."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def total_count(self) -> int:
        """Total number of tests."""
        return len(self.results)


class TestRunner:
    """Executes registered test cases and collects results.

    Usage:
        runner = TestRunner()

        # List available tests
        for info in runner.list_tests():
            print(f"{info.name}: {info.description}")

        # Run a single test
        result = runner.run_test("algorithm_watershed")

        # Run all tests in a category
        suite_result = runner.run_suite("algorithms")

        # Run all tests
        suite_result = runner.run_suite("all")
    """

    def __init__(self, output_base: Path | None = None) -> None:
        """Initialize test runner.

        Args:
            output_base: Base directory for test output. Defaults to
                         ./test_runs in the extension directory.
        """
        if output_base is None:
            # Default to test_runs/ in the project root (where .git is)
            output_base = Path(__file__).parent.parent.parent / "test_runs"
        self._output_base = output_base
        self._screenshot_capture = ScreenshotCapture()
        logger.info(f"TestRunner initialized with output base: {output_base}")

    def list_tests(self, category: str | None = None) -> list[TestCaseInfo]:
        """List available test cases.

        Args:
            category: Optional category filter.

        Returns:
            List of TestCaseInfo objects.
        """
        return TestRegistry.list_tests(category=category)

    def list_categories(self) -> list[str]:
        """List available test categories.

        Returns:
            List of category names.
        """
        return TestRegistry.list_categories()

    def run_test(
        self,
        name: str,
        test_run_folder: TestRunFolder | None = None,
    ) -> TestResult:
        """Run a single test case by name.

        Args:
            name: Name of test to run.
            test_run_folder: Optional existing test run folder.

        Returns:
            TestResult with pass/fail status and details.

        Raises:
            ValueError: If test name not found.
        """
        info = TestRegistry.get(name)
        if info is None:
            raise ValueError(f"Test not found: {name}")

        logger.info(f"Running test: {name}")

        # Create test run folder if not provided
        if test_run_folder is None:
            test_run_folder = TestRunFolder.create(
                base_path=self._output_base,
                run_name=name,
            )

        # Set up screenshot capture for this test run
        self._screenshot_capture.set_base_folder(test_run_folder.screenshots_folder)
        self._screenshot_capture.set_group(name)

        # Create test context
        metrics_collector = MetricsCollector()
        ctx = TestContext(
            test_run_folder=test_run_folder,
            screenshot_capture=self._screenshot_capture,
            metrics_collector=metrics_collector,
        )

        # Create test instance
        test_instance = info.cls()

        # Run test phases
        start_time = time.time()
        error = None
        error_traceback = None

        try:
            logger.info(f"[{name}] Setup")
            test_instance.setup(ctx)

            logger.info(f"[{name}] Run")
            test_instance.run(ctx)

            logger.info(f"[{name}] Verify")
            test_instance.verify(ctx)

        except Exception as e:
            error = str(e)
            error_traceback = traceback.format_exc()
            logger.exception(f"[{name}] Error: {e}")

        finally:
            try:
                logger.info(f"[{name}] Teardown")
                test_instance.teardown(ctx)
            except Exception as e:
                logger.exception(f"[{name}] Teardown error: {e}")
                if error is None:
                    error = f"Teardown error: {e}"
                    error_traceback = traceback.format_exc()

        duration = time.time() - start_time

        # Determine pass/fail
        failed_assertions = [a for a in ctx.assertions if not a.passed]
        passed = error is None and len(failed_assertions) == 0

        result = TestResult(
            name=name,
            passed=passed,
            assertions=ctx.assertions,
            error=error,
            error_traceback=error_traceback,
            duration_seconds=duration,
            screenshots=ctx.screenshots,
            metrics=ctx.metrics,
        )

        # Log summary
        status = "PASSED" if passed else "FAILED"
        logger.info(
            f"[{name}] {status} in {duration:.2f}s "
            f"({len(ctx.assertions)} assertions, {len(failed_assertions)} failed)"
        )

        return result

    def run_suite(self, suite: str = "all") -> TestSuiteResult:
        """Run a suite of tests.

        Args:
            suite: Suite name. Can be "all" or a category name
                   (e.g., "algorithm", "ui", "workflow").

        Returns:
            TestSuiteResult with all test results.
        """
        # Determine which tests to run
        if suite == "all":
            tests = self.list_tests()
        else:
            tests = self.list_tests(category=suite)

        if not tests:
            logger.warning(f"No tests found for suite: {suite}")
            return TestSuiteResult(
                suite_name=suite,
                output_folder=self._output_base,
                results=[],
                duration_seconds=0.0,
            )

        logger.info(f"Running suite '{suite}' with {len(tests)} tests")

        # Create test run folder for the suite
        test_run_folder = TestRunFolder.create(
            base_path=self._output_base,
            run_name=suite,
        )

        # Run all tests
        start_time = time.time()
        results = []

        for i, info in enumerate(tests):
            print(f"[{i + 1}/{len(tests)}] Running: {info.name}...", flush=True)
            result = self.run_test(info.name, test_run_folder=test_run_folder)
            status = "PASS" if result.passed else "FAIL"
            print(f"[{i + 1}/{len(tests)}] {info.name}: {status}", flush=True)
            results.append(result)

        duration = time.time() - start_time

        suite_result = TestSuiteResult(
            suite_name=suite,
            output_folder=test_run_folder.path,
            results=results,
            duration_seconds=duration,
        )

        # Save results to files
        test_run_folder.save_results(results)
        test_run_folder.save_metadata(
            {
                "suite": suite,
                "total_tests": suite_result.total_count,
                "passed": suite_result.passed_count,
                "failed": suite_result.failed_count,
                "duration_seconds": duration,
            }
        )

        # Save screenshot manifest
        self._screenshot_capture.save_manifest()

        # Log summary
        logger.info(
            f"Suite '{suite}' completed: "
            f"{suite_result.passed_count}/{suite_result.total_count} passed "
            f"in {duration:.2f}s"
        )

        return suite_result

    def discover_tests(self) -> None:
        """Discover and register test cases from TestCases package.

        This imports all test modules to trigger @register_test decorators.
        """
        import importlib
        import pkgutil

        # Import TestCases package
        try:
            test_cases_path = Path(__file__).parent.parent / "TestCases"
            if not test_cases_path.exists():
                logger.warning(f"TestCases directory not found: {test_cases_path}")
                return

            # Import each test module
            for module_info in pkgutil.iter_modules([str(test_cases_path)]):
                if module_info.name.startswith("test_"):
                    module_name = f"TestCases.{module_info.name}"
                    try:
                        importlib.import_module(module_name)
                        logger.debug(f"Discovered test module: {module_name}")
                    except Exception as e:
                        logger.exception(f"Error loading test module {module_name}: {e}")

            logger.info(f"Discovered {len(TestRegistry.list_tests())} tests")

        except Exception as e:
            logger.exception(f"Error discovering tests: {e}")
