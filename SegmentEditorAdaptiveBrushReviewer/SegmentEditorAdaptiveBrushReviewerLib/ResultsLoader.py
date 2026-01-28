"""Load and parse optimization run results.

Loads results from the optimization_results directory structure.
All segmentations are stored as DICOM SEG (no .seg.nrrd support).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DicomInfo:
    """DICOM study/volume information for an optimization run."""

    patient_id: str
    study_description: str
    volume_series_uid: str
    volume_name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DicomInfo:
        """Create from dictionary."""
        return cls(
            patient_id=data.get("patient_id", ""),
            study_description=data.get("study_description", ""),
            volume_series_uid=data.get("volume_series_uid", ""),
            volume_name=data.get("volume_name", ""),
        )


@dataclass
class TrialData:
    """Data for a single optimization trial."""

    trial_number: int
    params: dict[str, Any]
    value: float
    duration_ms: float
    pruned: bool = False
    user_attrs: dict[str, Any] = field(default_factory=dict)
    intermediate_values: dict[int, float] = field(default_factory=dict)
    screenshots: list[Path] = field(default_factory=list)
    # DICOM SEG information
    dicom_series_uid: str | None = None
    dicom_seg_path: Path | None = None


@dataclass
class OptimizationRun:
    """Data for a complete optimization run."""

    path: Path
    name: str
    config: dict[str, Any]
    trials: list[TrialData]
    best_trial: TrialData | None
    parameter_importance: dict[str, float]
    dicom_info: DicomInfo | None = None

    def get_trial(self, trial_number: int) -> TrialData | None:
        """Get trial by number."""
        return next(
            (t for t in self.trials if t.trial_number == trial_number),
            None,
        )

    def get_algorithm_trials(self, algorithm: str) -> list[TrialData]:
        """Get all trials for a specific algorithm."""
        return [t for t in self.trials if t.params.get("algorithm") == algorithm]


class ResultsLoader:
    """Load optimization run results.

    Scans the optimization_results directory and loads run data
    including trials, metrics, and segmentations.
    """

    def __init__(self, results_dir: Path | str | None = None):
        """Initialize loader.

        Args:
            results_dir: Directory containing optimization results.
                        Defaults to "optimization_results" relative to extension root.
        """
        if results_dir is None:
            # Find optimization_results relative to this module
            module_dir = Path(__file__).parent.parent.parent
            results_dir = module_dir / "optimization_results"
        self.results_dir = Path(results_dir)

    def list_runs(self) -> list[Path]:
        """List available optimization runs.

        Returns:
            List of paths to run directories, sorted by date descending.
        """
        if not self.results_dir.exists():
            return []

        runs = [
            p for p in self.results_dir.iterdir() if p.is_dir() and (p / "results.json").exists()
        ]

        # Sort by name (which includes timestamp) descending
        runs.sort(reverse=True)
        return runs

    def load(self, run_path: Path | str) -> OptimizationRun:
        """Load complete optimization run.

        Args:
            run_path: Path to run directory.

        Returns:
            OptimizationRun with all data.

        Raises:
            FileNotFoundError: If results file doesn't exist.
        """
        run_path = Path(run_path)

        # Load results
        results_path = run_path / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path) as f:
            results = json.load(f)

        # Load config if available
        config_path = run_path / "config.yaml"
        config = {}
        if config_path.exists():
            try:
                import yaml  # type: ignore[import-untyped]

                with open(config_path) as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        # Load parameter importance
        importance_path = run_path / "parameter_importance.json"
        importance = {}
        if importance_path.exists():
            with open(importance_path) as f:
                importance = json.load(f)

        # Parse trials - handle both formats:
        # 1. Optuna format: {"trials": [...], "best_trial": {...}}
        # 2. Test format: [{...}, {...}, ...] - array of test results
        trials = []
        best_trial = None
        run_name = run_path.name  # Default name from directory

        dicom_info = None
        if isinstance(results, list):
            # Test runner format - array of test results
            for i, test_data in enumerate(results):
                trial = self._parse_test_result(test_data, i, run_path)
                trials.append(trial)
            # Find best (highest value/passed tests)
            passed_trials = [t for t in trials if t.value > 0]
            if passed_trials:
                best_trial = max(passed_trials, key=lambda t: t.value)
        else:
            # Optuna format
            run_name = results.get("config_name", run_path.name)

            # Parse DICOM info (required for optimization runs)
            if "dicom" in results:
                dicom_info = DicomInfo.from_dict(results["dicom"])
            else:
                logger.warning(f"No DICOM info in results - legacy run? {run_path}")

            for trial_data in results.get("trials", []):
                trial = self._parse_trial(trial_data, run_path)
                trials.append(trial)
            if results.get("best_trial"):
                best_trial = self._parse_trial(results["best_trial"], run_path)

        run = OptimizationRun(
            path=run_path,
            name=run_name,
            config=config,
            trials=trials,
            best_trial=best_trial,
            parameter_importance=importance,
            dicom_info=dicom_info,
        )

        logger.info(f"Loaded run '{run.name}' with {len(trials)} trials")
        return run

    def _parse_trial(self, data: dict, run_path: Path) -> TrialData:
        """Parse trial data from Optuna format dictionary."""
        trial_num = data["trial_number"]
        user_attrs = data.get("user_attrs", {})

        # Get DICOM SEG info from user_attrs
        dicom_series_uid = user_attrs.get("dicom_series_uid")
        dicom_seg_path_str = user_attrs.get("dicom_seg_path")
        dicom_seg_path = run_path / dicom_seg_path_str if dicom_seg_path_str else None

        if not dicom_series_uid:
            logger.warning(f"Trial {trial_num} has no DICOM series UID - legacy format?")

        # Find screenshots
        screenshots = []
        ss_dir = run_path / "screenshots" / f"trial_{trial_num:03d}"
        if ss_dir.exists():
            screenshots = sorted(ss_dir.glob("*.png"))

        return TrialData(
            trial_number=trial_num,
            params=data.get("params", {}),
            value=data.get("value", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            pruned=data.get("pruned", False),
            user_attrs=user_attrs,
            intermediate_values=data.get("intermediate_values", {}),
            screenshots=screenshots,
            dicom_series_uid=dicom_series_uid,
            dicom_seg_path=dicom_seg_path,
        )

    def _parse_test_result(self, data: dict, index: int, run_path: Path) -> TrialData:
        """Parse trial data from test runner format dictionary."""
        # Test runner format has: name, passed, duration_seconds, screenshots, assertions, metrics

        # Find screenshots in the screenshots folder
        screenshots = []
        ss_dir = run_path / "screenshots"
        if ss_dir.exists():
            # Screenshots are listed by filename in the data
            for ss_name in data.get("screenshots", []):
                ss_path = ss_dir / ss_name
                if ss_path.exists():
                    screenshots.append(ss_path)

        # Extract metrics as params
        params = {"test_name": data.get("name", f"test_{index}")}
        metrics_data = data.get("metrics", {})
        for metric in metrics_data.get("metrics", []):
            params[metric["name"]] = metric["value"]

        # Value is 1.0 if passed, 0.0 if failed
        # Or use dice score if available in metrics
        value = 1.0 if data.get("passed", False) else 0.0
        for metric in metrics_data.get("metrics", []):
            if "dice" in metric["name"].lower():
                value = metric["value"]
                break

        return TrialData(
            trial_number=index + 1,
            params=params,
            value=value,
            duration_ms=data.get("duration_seconds", 0.0) * 1000,
            pruned=False,
            user_attrs={
                "passed": data.get("passed", False),
                "error": data.get("error"),
                "assertions": data.get("assertions", []),
            },
            intermediate_values={},
            screenshots=screenshots,
            # Test runner format doesn't have DICOM info
            dicom_series_uid=None,
            dicom_seg_path=None,
        )

    def get_gold_standard_path(self, run: OptimizationRun) -> Path | None:
        """Get gold standard DICOM path from run config.

        Args:
            run: Optimization run.

        Returns:
            Path to gold standard DICOM directory, or None if not found.
        """
        recipes = run.config.get("recipes", [])
        if recipes and "gold_standard" in recipes[0]:
            gold_path = Path(recipes[0]["gold_standard"])

            # Extract the gold standard directory from various path formats:
            # - "GoldStandards/X/gold.seg.nrrd" -> "GoldStandards/X"
            # - "GoldStandards/X/dicom" -> "GoldStandards/X"
            # - "GoldStandards/X" -> "GoldStandards/X"
            if gold_path.name == "gold.seg.nrrd":
                gold_dir = gold_path.parent
            elif gold_path.name == "dicom":
                gold_dir = gold_path.parent
            else:
                gold_dir = gold_path

            # Try to resolve relative path from extension root (Tester module)
            tester_dir = Path(__file__).parent.parent.parent / "SegmentEditorAdaptiveBrushTester"
            resolved_dir = tester_dir / gold_dir
            if resolved_dir.exists():
                dicom_path = resolved_dir / "dicom"
                if dicom_path.exists():
                    return dicom_path
                # Fall back to directory for legacy check
                return resolved_dir

            # Try relative to run directory
            run_resolved = run.path / gold_dir
            if run_resolved.exists():
                dicom_path = run_resolved / "dicom"
                if dicom_path.exists():
                    return dicom_path
                return run_resolved

            # Try absolute path
            if gold_dir.is_absolute() and gold_dir.exists():
                dicom_path = gold_dir / "dicom"
                if dicom_path.exists():
                    return dicom_path
                return gold_dir

        return None
