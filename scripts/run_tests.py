#!/usr/bin/env python
"""Run AdaptiveBrush tests in Slicer.

This script is executed by Slicer via:
    Slicer --python-script scripts/run_tests.py [suite]

Arguments:
    suite: Test suite to run. Options: "all", "algorithm", "ui", "workflow".
           Default: "all"

After tests complete, Slicer stays open with the interactive testing panel
visible for manual follow-up testing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run tests and show interactive panel."""
    import slicer

    # Parse arguments
    suite = "all"
    if len(sys.argv) > 1:
        suite = sys.argv[1]

    logger.info(f"Running test suite: {suite}")

    # Import the testing framework
    try:
        from SegmentEditorAdaptiveBrushTesterLib import TestRunner
    except ImportError:
        logger.error(
            "SegmentEditorAdaptiveBrushTester module not found. "
            "Make sure the extension is installed."
        )
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

    # Show the tester module for interactive follow-up
    slicer.util.selectModule("SegmentEditorAdaptiveBrushTester")

    logger.info(
        "Tests complete. Slicer is open for manual testing. "
        "Use the Adaptive Brush Tester panel to continue."
    )


if __name__ == "__main__":
    main()
