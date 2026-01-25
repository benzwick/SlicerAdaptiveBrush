"""Load and parse optimization run results.

Loads results from the optimization_results directory structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    segmentation_path: Path | None = None
    screenshots: list[Path] = field(default_factory=list)


@dataclass
class OptimizationRun:
    """Data for a complete optimization run."""

    path: Path
    name: str
    config: dict[str, Any]
    trials: list[TrialData]
    best_trial: TrialData | None
    parameter_importance: dict[str, float]

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
                        Defaults to "optimization_results".
        """
        if results_dir is None:
            results_dir = Path("optimization_results")
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

        # Parse trials
        trials = []
        for trial_data in results.get("trials", []):
            trial = self._parse_trial(trial_data, run_path)
            trials.append(trial)

        # Parse best trial
        best_trial = None
        if results.get("best_trial"):
            best_trial = self._parse_trial(results["best_trial"], run_path)

        run = OptimizationRun(
            path=run_path,
            name=results.get("config_name", run_path.name),
            config=config,
            trials=trials,
            best_trial=best_trial,
            parameter_importance=importance,
        )

        logger.info(f"Loaded run '{run.name}' with {len(trials)} trials")
        return run

    def _parse_trial(self, data: dict, run_path: Path) -> TrialData:
        """Parse trial data from dictionary."""
        trial_num = data["trial_number"]

        # Find segmentation file
        seg_path = None
        seg_dir = run_path / "segmentations"
        if seg_dir.exists():
            candidates = list(seg_dir.glob(f"trial_{trial_num:03d}*.seg.nrrd"))
            if candidates:
                seg_path = candidates[0]

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
            user_attrs=data.get("user_attrs", {}),
            intermediate_values=data.get("intermediate_values", {}),
            segmentation_path=seg_path,
            screenshots=screenshots,
        )

    def get_gold_standard_path(self, run: OptimizationRun) -> Path | None:
        """Get gold standard path from run config.

        Args:
            run: Optimization run.

        Returns:
            Path to gold standard or None.
        """
        recipes = run.config.get("recipes", [])
        if recipes and "gold_standard" in recipes[0]:
            return Path(recipes[0]["gold_standard"])
        return None
