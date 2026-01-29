#!/usr/bin/env python
"""Run Reviewer module tests inside Slicer.

This script runs pytest for the SegmentEditorAdaptiveBrushReviewer tests
inside the Slicer Python environment, which provides access to DICOMLib.

Usage:
    Slicer --python-script scripts/run_reviewer_tests.py [--exit]

Options:
    --exit: Exit Slicer after tests complete (for automated runs).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run pytest for Reviewer module tests."""
    import slicer

    # Parse arguments
    exit_after_tests = "--exit" in sys.argv

    # Get the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Add paths for imports
    reviewer_lib = (
        project_root
        / "SegmentEditorAdaptiveBrushReviewer"
        / "SegmentEditorAdaptiveBrushReviewerLib"
    )
    test_dir = project_root / "SegmentEditorAdaptiveBrushReviewer" / "Testing" / "Python"

    if str(reviewer_lib) not in sys.path:
        sys.path.insert(0, str(reviewer_lib))

    logger.info(f"Project root: {project_root}")
    logger.info(f"Test directory: {test_dir}")

    # Install pytest if needed
    try:
        import pytest
    except ImportError:
        logger.info("Installing pytest...")
        slicer.util.pip_install("pytest")
        import pytest

    # Run pytest
    logger.info("Running Reviewer module tests with pytest...")
    print("\n" + "=" * 60)
    print("Running SegmentEditorAdaptiveBrushReviewer Tests in Slicer")
    print("=" * 60 + "\n")

    # Run pytest programmatically
    exit_code = pytest.main(
        [
            str(test_dir),
            "-v",
            "--tb=short",
            "-p",
            "no:cacheprovider",  # Disable cache to avoid permission issues
        ]
    )

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("All tests PASSED!")
    else:
        print(f"Tests FAILED (exit code: {exit_code})")
    print("=" * 60 + "\n")

    if exit_after_tests:
        logger.info(f"Exiting Slicer with code {exit_code}")
        slicer.app.exit(exit_code)
    else:
        logger.info("Slicer staying open for manual inspection")


if __name__ == "__main__":
    main()
