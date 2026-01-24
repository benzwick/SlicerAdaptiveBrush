"""Test run folder organization.

Creates and manages the output folder structure for test runs.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .TestCase import TestResult

logger = logging.getLogger(__name__)


class TestRunFolder:
    """Manages the folder structure for a test run.

    Structure:
        test_runs/2026-01-24_143025_algorithms/
        ├── metadata.json          # Run config, summary
        ├── results.json           # Test results
        ├── metrics.json           # Performance metrics
        ├── manual_actions.jsonl   # Recorded manual testing
        ├── screenshots/
        │   ├── manifest.json      # Screenshot descriptions
        │   └── *.png              # Captured screenshots
        └── logs/
            ├── test_run.log       # Test execution log
            └── slicer_session.log # Copy of Slicer log
    """

    def __init__(self, path: Path) -> None:
        """Initialize test run folder.

        Args:
            path: Path to the test run folder.
        """
        self._path = path
        self._screenshots_folder = path / "screenshots"
        self._logs_folder = path / "logs"

        # Create subdirectories
        self._screenshots_folder.mkdir(parents=True, exist_ok=True)
        self._logs_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Test run folder: {path}")

    @classmethod
    def create(cls, base_path: Path, run_name: str) -> TestRunFolder:
        """Create a new test run folder with timestamp.

        Args:
            base_path: Base directory for test runs.
            run_name: Name of the test run (e.g., "algorithms", "ui").

        Returns:
            New TestRunFolder instance.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder_name = f"{timestamp}_{run_name}"
        path = base_path / folder_name

        path.mkdir(parents=True, exist_ok=True)

        folder = cls(path)
        folder._initialize_log_file()

        return folder

    @property
    def path(self) -> Path:
        """Path to the test run folder."""
        return self._path

    @property
    def screenshots_folder(self) -> Path:
        """Path to screenshots subfolder."""
        return self._screenshots_folder

    @property
    def logs_folder(self) -> Path:
        """Path to logs subfolder."""
        return self._logs_folder

    def _initialize_log_file(self) -> None:
        """Set up logging to file."""
        log_file = self._logs_folder / "test_run.log"

        # Create file handler for this test run
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s")
        )

        # Add to root logger
        logging.getLogger().addHandler(handler)

        logger.info(f"Logging to: {log_file}")

    def save_metadata(self, metadata: dict) -> None:
        """Save test run metadata.

        Args:
            metadata: Metadata dictionary to save.
        """
        filepath = self._path / "metadata.json"

        # Add standard fields
        full_metadata = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            **metadata,
        }

        with open(filepath, "w") as f:
            json.dump(full_metadata, f, indent=2)

        logger.info(f"Saved metadata to: {filepath}")

    def save_results(self, results: list[TestResult]) -> None:
        """Save test results to JSON.

        Args:
            results: List of TestResult objects.
        """
        filepath = self._path / "results.json"

        # Convert to serializable format
        results_data = []
        for r in results:
            result_dict = {
                "name": r.name,
                "passed": r.passed,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
                "error_traceback": r.error_traceback,
                "screenshots": r.screenshots,
                "assertions": [
                    {
                        "passed": a.passed,
                        "message": a.message,
                        "expected": str(a.expected) if a.expected is not None else None,
                        "actual": str(a.actual) if a.actual is not None else None,
                    }
                    for a in r.assertions
                ],
                "metrics": r.metrics,
            }
            results_data.append(result_dict)

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved results to: {filepath}")

        # Also save metrics separately for easier analysis
        self._save_metrics(results)

    def _save_metrics(self, results: list[TestResult]) -> None:
        """Save aggregated metrics to JSON.

        Args:
            results: List of TestResult objects.
        """
        filepath = self._path / "metrics.json"

        all_metrics = {}
        for r in results:
            if r.metrics:
                all_metrics[r.name] = r.metrics

        with open(filepath, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.debug(f"Saved metrics to: {filepath}")

    def save_screenshot_manifest(self, screenshots: list[dict]) -> None:
        """Save screenshot manifest.

        Args:
            screenshots: List of screenshot info dictionaries.
        """
        filepath = self._screenshots_folder / "manifest.json"

        with open(filepath, "w") as f:
            json.dump(screenshots, f, indent=2)

        logger.debug(f"Saved screenshot manifest to: {filepath}")

    def copy_slicer_log(self) -> Path | None:
        """Copy Slicer session log to test run folder.

        Returns:
            Path to copied log file, or None if not found.
        """
        import slicer

        slicer_log = Path(slicer.app.errorLogModel().filePath())

        if not slicer_log.exists():
            logger.warning(f"Slicer log not found: {slicer_log}")
            return None

        dest = self._logs_folder / "slicer_session.log"
        shutil.copy2(slicer_log, dest)

        logger.info(f"Copied Slicer log to: {dest}")
        return dest

    def append_manual_action(self, action: dict) -> None:
        """Append a manual action to the actions log.

        Args:
            action: Action dictionary with type, timestamp, and details.
        """
        filepath = self._path / "manual_actions.jsonl"

        with open(filepath, "a") as f:
            f.write(json.dumps(action) + "\n")

        logger.debug(f"Recorded action: {action.get('type', 'unknown')}")
