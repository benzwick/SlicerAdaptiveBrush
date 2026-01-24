#!/usr/bin/env python
"""Run AdaptiveBrush tests in Slicer.

This script is executed by Slicer via:
    Slicer --python-script scripts/run_tests.py [options] [suite]

Arguments:
    suite: Test suite to run. Options: "all", "algorithm", "ui", "workflow".
           Default: "all"

Options:
    --exit: Exit Slicer after tests complete (for automated runs).
            Without this flag, Slicer stays open for manual testing.

Examples:
    # Run all tests and stay open for manual testing:
    Slicer --python-script scripts/run_tests.py all

    # Run algorithm tests and exit when done:
    Slicer --python-script scripts/run_tests.py --exit algorithm

    # Run all tests and exit:
    Slicer --python-script scripts/run_tests.py --exit
"""

from __future__ import annotations

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger(__name__)


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

    # Import the testing framework
    try:
        from SegmentEditorAdaptiveBrushTesterLib import TestRunner
    except ImportError:
        logger.error(
            "SegmentEditorAdaptiveBrushTester module not found. "
            "Make sure the extension is installed."
        )
        if exit_after_tests:
            slicer.app.exit(1)
        return

    # Create and configure test runner
    runner = TestRunner()

    # Discover tests
    runner.discover_tests()

    # Run tests
    result = runner.run_suite(suite)

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
