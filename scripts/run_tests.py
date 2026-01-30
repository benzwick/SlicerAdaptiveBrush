#!/usr/bin/env python
"""Run AdaptiveBrush tests in Slicer.

This script is executed by Slicer via:
    Slicer --python-script scripts/run_tests.py [options] [suite]

Arguments:
    suite: Test suite to run. Options: "all", "ci", "algorithm", "ui", "workflow".
           Default: "all"
           Special suites:
           - "ci": Runs all tests except gold_standard and reviewer_integration
                   (these require special setup not available in CI)

Options:
    --exit: Exit Slicer after tests complete (for automated runs).
            Without this flag, Slicer stays open for manual testing.

Examples:
    # Run all tests and stay open for manual testing:
    Slicer --python-script scripts/run_tests.py all

    # Run CI-compatible tests and exit:
    Slicer --python-script scripts/run_tests.py --exit ci

    # Run algorithm tests and exit when done:
    Slicer --python-script scripts/run_tests.py --exit algorithm

    # Run all tests and exit:
    Slicer --python-script scripts/run_tests.py --exit
"""

from __future__ import annotations

import logging
import sys

# Configure logging with immediate flush
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None

logger = logging.getLogger(__name__)

print("=" * 60, flush=True)
print("run_tests.py starting", flush=True)
print(f"Arguments: {sys.argv}", flush=True)
print("=" * 60, flush=True)


def main() -> None:
    """Run tests and optionally show interactive panel or exit."""
    import slicer

    # Parse arguments
    args = sys.argv[1:]
    exit_after_tests = "--exit" in args
    if exit_after_tests:
        args.remove("--exit")

    suite = args[0] if args else "all"

    logger.info(f"Running test suite: {suite}")
    logger.info(f"Exit after tests: {exit_after_tests}")

    # Import the testing framework - FAIL FAST if not available
    print("Importing SegmentEditorAdaptiveBrushTesterLib...", flush=True)
    try:
        from SegmentEditorAdaptiveBrushTesterLib import TestRunner

        print("Import successful!", flush=True)
    except ImportError as e:
        error_msg = (
            f"FATAL: SegmentEditorAdaptiveBrushTesterLib import failed: {e}\n"
            "Make sure --additional-module-paths is set BEFORE --python-script"
        )
        print(error_msg, flush=True)
        logger.error(error_msg)
        # Always exit on import failure - don't hang
        slicer.app.exit(1)
        return

    # Create and configure test runner
    print("Creating TestRunner...", flush=True)
    runner = TestRunner()
    print("TestRunner created", flush=True)

    # Discover tests
    print("Discovering tests...", flush=True)
    runner.discover_tests()
    print(f"Tests discovered: {len(runner.list_tests())}", flush=True)

    # Run tests
    # "ci" suite excludes tests that require special setup not available in CI
    CI_EXCLUDED_CATEGORIES = {"gold_standard", "reviewer_integration"}

    if suite == "ci":
        # Get all tests except excluded categories
        all_tests = runner.list_tests()
        ci_tests = [t for t in all_tests if t.category not in CI_EXCLUDED_CATEGORIES]
        excluded_count = len(all_tests) - len(ci_tests)
        print(
            f"Running CI suite: {len(ci_tests)} tests (excluding {excluded_count} from {CI_EXCLUDED_CATEGORIES})",
            flush=True,
        )

        # Run each test individually
        import time

        from SegmentEditorAdaptiveBrushTesterLib import TestRunFolder, TestSuiteResult

        test_run_folder = TestRunFolder.create(
            base_path=runner._output_base,
            run_name="ci",
        )

        start_time = time.time()
        results = []
        for i, info in enumerate(ci_tests):
            print(f"[{i + 1}/{len(ci_tests)}] Running: {info.name}...", flush=True)
            test_result = runner.run_test(info.name, test_run_folder=test_run_folder)
            status = "PASS" if test_result.passed else "FAIL"
            print(f"[{i + 1}/{len(ci_tests)}] {info.name}: {status}", flush=True)
            results.append(test_result)

        duration = time.time() - start_time
        result = TestSuiteResult(
            suite_name="ci",
            output_folder=test_run_folder.path,
            results=results,
            duration_seconds=duration,
        )
        print("CI suite complete", flush=True)
    else:
        print(f"Running suite: {suite}", flush=True)
        result = runner.run_suite(suite)
        print("Suite complete", flush=True)

    # Print summary
    status = "PASSED" if result.passed else "FAILED"
    print(f"\n{'=' * 60}")
    print(f"Test Suite: {suite}")
    print(f"Status: {status}")
    print(f"Tests: {result.passed_count}/{result.total_count} passed")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Output: {result.output_folder}")
    print(f"{'=' * 60}\n")

    # Copy Slicer log
    from SegmentEditorAdaptiveBrushTesterLib import TestRunFolder

    test_run_folder = TestRunFolder(result.output_folder)
    test_run_folder.copy_slicer_log()

    if exit_after_tests:
        # Exit Slicer with appropriate return code
        exit_code = 0 if result.passed else 1
        logger.info(f"Exiting Slicer with code {exit_code}")
        slicer.app.exit(exit_code)
    else:
        # Show the tester module for interactive follow-up
        slicer.util.selectModule("SegmentEditorAdaptiveBrushTester")
        logger.info(
            "Tests complete. Slicer is open for manual testing. "
            "Use the Adaptive Brush Tester panel to continue."
        )


if __name__ == "__main__":
    main()
