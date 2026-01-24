"""Parameter optimization for segmentation algorithms.

Provides Optuna-style parameter optimization framework for finding
optimal algorithm parameters through systematic trial-and-error.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Define search space for a parameter.

    Supports continuous, discrete step, and categorical parameters.
    """

    name: str
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None  # None = continuous
    choices: list | None = None  # For categorical parameters

    def sample(self) -> float | str:
        """Sample a random value from this parameter space."""
        if self.choices is not None:
            result: str = random.choice(self.choices)
            return result

        if self.min_val is None or self.max_val is None:
            raise ValueError(f"Parameter {self.name} has no range defined")

        min_v = self.min_val
        max_v = self.max_val

        if self.step is not None:
            steps = int((max_v - min_v) / self.step)
            return min_v + random.randint(0, steps) * self.step

        return random.uniform(min_v, max_v)


@dataclass
class OptimizationTrial:
    """Result of one optimization trial."""

    trial_id: int
    params: dict
    dice: float
    hausdorff_95: float
    strokes: int
    duration_ms: float
    voxels: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "dice": self.dice,
            "hausdorff_95": self.hausdorff_95,
            "strokes": self.strokes,
            "duration_ms": self.duration_ms,
            "voxels": self.voxels,
            "timestamp": self.timestamp,
        }


class ParameterOptimizer:
    """Optuna-style parameter optimization for segmentation.

    Supports random search and grid search over algorithm-specific parameter spaces.

    Usage:
        optimizer = ParameterOptimizer("watershed", "MRBrainTumor1_tumor")

        for trial_num in range(20):
            params = optimizer.suggest_params()

            # Run segmentation with params...
            dice, hd95, strokes, duration, voxels = run_trial(params)

            optimizer.record_trial(params, dice, hd95, strokes, duration, voxels)

        best = optimizer.get_best_params()
        optimizer.save_results(output_path)
    """

    # Default search spaces per algorithm
    SEARCH_SPACES: dict[str, list[ParameterSpace]] = {
        "watershed": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("threshold_zone", 30, 70, 10),
            ParameterSpace("watershed_gradient_scale", 0.5, 2.0, 0.25),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
        "level_set_cpu": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("level_set_iterations", 50, 500, 50),
            ParameterSpace("level_set_propagation", 0.5, 2.0, 0.25),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
        "level_set_gpu": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("level_set_iterations", 50, 500, 50),
            ParameterSpace("level_set_propagation", 0.5, 2.0, 0.25),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
        "connected_threshold": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("threshold_zone", 30, 70, 10),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
        "region_growing": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("threshold_zone", 30, 70, 10),
            ParameterSpace("region_growing_multiplier", 1.5, 3.5, 0.5),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
        "threshold_brush": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("threshold_zone", 30, 70, 10),
            ParameterSpace(
                "threshold_method",
                choices=["otsu", "huang", "triangle", "maximum_entropy", "isodata", "li"],
            ),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
        "geodesic_distance": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("threshold_zone", 30, 70, 10),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
        "random_walker": [
            ParameterSpace("edge_sensitivity", 10, 90, 10),
            ParameterSpace("threshold_zone", 30, 70, 10),
            ParameterSpace("random_walker_beta", 50, 500, 50),
            ParameterSpace("brush_radius_mm", 10, 40, 5),
        ],
    }

    def __init__(
        self,
        algorithm: str,
        gold_standard_name: str,
        custom_spaces: list[ParameterSpace] | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            algorithm: Algorithm name to optimize.
            gold_standard_name: Name of gold standard to compare against.
            custom_spaces: Override default search spaces.
        """
        self.algorithm = algorithm
        self.gold_standard_name = gold_standard_name
        self.trials: list[OptimizationTrial] = []

        if custom_spaces is not None:
            self.search_space = custom_spaces
        else:
            self.search_space = self.SEARCH_SPACES.get(algorithm, [])

        if not self.search_space:
            logger.warning(f"No search space defined for algorithm: {algorithm}")

    def suggest_params(self) -> dict[str, str | float]:
        """Suggest next parameters to try (random search).

        Returns:
            Dictionary of parameter values.
        """
        params: dict[str, str | float] = {"algorithm": self.algorithm}
        for space in self.search_space:
            params[space.name] = space.sample()

        logger.debug(f"Suggested params: {params}")
        return params

    def suggest_grid_params(self) -> list[dict]:
        """Generate all parameter combinations for grid search.

        Returns:
            List of parameter dictionaries covering all combinations.
        """
        import itertools

        # Generate all values for each parameter
        param_values: dict[str, list] = {}
        for space in self.search_space:
            if space.choices is not None:
                param_values[space.name] = space.choices
            elif space.step is not None and space.min_val is not None and space.max_val is not None:
                min_v, max_v, step_v = space.min_val, space.max_val, space.step
                steps = int((max_v - min_v) / step_v) + 1
                param_values[space.name] = [min_v + i * step_v for i in range(steps)]
            elif space.min_val is not None and space.max_val is not None:
                # For continuous, sample 5 evenly spaced values
                min_v, max_v = space.min_val, space.max_val
                param_values[space.name] = [min_v + i * (max_v - min_v) / 4 for i in range(5)]

        # Generate all combinations
        combinations: list[dict] = []
        keys = list(param_values.keys())
        for values in itertools.product(*[param_values[k] for k in keys]):
            params: dict = {"algorithm": self.algorithm}
            params.update(dict(zip(keys, values)))
            combinations.append(params)

        logger.info(f"Grid search: {len(combinations)} combinations")
        return combinations

    def record_trial(
        self,
        params: dict,
        dice: float,
        hausdorff_95: float,
        strokes: int,
        duration_ms: float,
        voxels: int = 0,
    ) -> OptimizationTrial:
        """Record trial result.

        Args:
            params: Parameters used for this trial.
            dice: Dice coefficient achieved.
            hausdorff_95: 95th percentile Hausdorff distance.
            strokes: Number of strokes used.
            duration_ms: Total duration in milliseconds.
            voxels: Number of voxels segmented.

        Returns:
            The recorded trial.
        """
        trial = OptimizationTrial(
            trial_id=len(self.trials) + 1,
            params=params,
            dice=dice,
            hausdorff_95=hausdorff_95,
            strokes=strokes,
            duration_ms=duration_ms,
            voxels=voxels,
        )
        self.trials.append(trial)

        logger.info(
            f"Trial {trial.trial_id}: Dice={dice:.3f}, HD95={hausdorff_95:.1f}mm, "
            f"{strokes} strokes, {duration_ms:.0f}ms"
        )

        return trial

    def get_best_params(self, metric: str = "dice") -> dict:
        """Get parameters with best metric value.

        Args:
            metric: Metric to optimize ("dice" = maximize, "hausdorff_95" = minimize).

        Returns:
            Best parameter dictionary.
        """
        if not self.trials:
            return {}

        if metric == "dice":
            best = max(self.trials, key=lambda t: t.dice)
        elif metric == "hausdorff_95":
            best = min(self.trials, key=lambda t: t.hausdorff_95)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best.params

    def get_best_trial(self, metric: str = "dice") -> OptimizationTrial | None:
        """Get the best trial by metric.

        Args:
            metric: Metric to optimize.

        Returns:
            Best trial or None if no trials.
        """
        if not self.trials:
            return None

        if metric == "dice":
            return max(self.trials, key=lambda t: t.dice)
        elif metric == "hausdorff_95":
            return min(self.trials, key=lambda t: t.hausdorff_95)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_summary(self) -> dict:
        """Get optimization summary statistics.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.trials:
            return {
                "algorithm": self.algorithm,
                "gold_standard": self.gold_standard_name,
                "total_trials": 0,
                "best_dice": 0.0,
                "best_hausdorff_95": float("inf"),
            }

        best_dice_trial = max(self.trials, key=lambda t: t.dice)
        best_hd_trial = min(self.trials, key=lambda t: t.hausdorff_95)

        dice_values = [t.dice for t in self.trials]
        hd_values = [t.hausdorff_95 for t in self.trials if t.hausdorff_95 < float("inf")]

        return {
            "algorithm": self.algorithm,
            "gold_standard": self.gold_standard_name,
            "total_trials": len(self.trials),
            "best_dice": best_dice_trial.dice,
            "best_dice_params": best_dice_trial.params,
            "best_hausdorff_95": best_hd_trial.hausdorff_95,
            "best_hausdorff_params": best_hd_trial.params,
            "mean_dice": sum(dice_values) / len(dice_values),
            "mean_hausdorff_95": sum(hd_values) / len(hd_values) if hd_values else float("inf"),
            "dice_range": (min(dice_values), max(dice_values)),
        }

    def save_results(self, output_path: Path | str) -> None:
        """Save optimization results to JSON.

        Args:
            output_path: Path to save results.
        """
        output_path = Path(output_path)

        results = {
            "algorithm": self.algorithm,
            "gold_standard": self.gold_standard_name,
            "search_space": [
                {
                    "name": s.name,
                    "min_val": s.min_val,
                    "max_val": s.max_val,
                    "step": s.step,
                    "choices": s.choices,
                }
                for s in self.search_space
            ],
            "trials": [t.to_dict() for t in self.trials],
            "summary": self.get_summary(),
            "saved_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved optimization results to {output_path}")

    def load_results(self, input_path: Path | str) -> None:
        """Load previous optimization results.

        Args:
            input_path: Path to load results from.
        """
        input_path = Path(input_path)

        with open(input_path) as f:
            data = json.load(f)

        self.algorithm = data["algorithm"]
        self.gold_standard_name = data["gold_standard"]

        self.trials = [
            OptimizationTrial(
                trial_id=t["trial_id"],
                params=t["params"],
                dice=t["dice"],
                hausdorff_95=t["hausdorff_95"],
                strokes=t["strokes"],
                duration_ms=t["duration_ms"],
                voxels=t.get("voxels", 0),
                timestamp=t.get("timestamp", ""),
            )
            for t in data["trials"]
        ]

        logger.info(f"Loaded {len(self.trials)} trials from {input_path}")

    def analyze_parameter_sensitivity(self) -> dict[str, dict]:
        """Analyze how each parameter affects the Dice score.

        Returns:
            Dictionary mapping parameter names to sensitivity analysis.
        """
        if len(self.trials) < 5:
            logger.warning("Need at least 5 trials for sensitivity analysis")
            return {}

        results = {}

        for space in self.search_space:
            param_name = space.name

            # Group trials by parameter value
            param_values = []
            dice_values = []

            for trial in self.trials:
                if param_name in trial.params:
                    param_values.append(trial.params[param_name])
                    dice_values.append(trial.dice)

            if not param_values:
                continue

            # Simple correlation analysis
            import numpy as np

            if isinstance(param_values[0], (int, float)):
                correlation = np.corrcoef(param_values, dice_values)[0, 1]
                results[param_name] = {
                    "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                    "value_range": (min(param_values), max(param_values)),
                    "best_value": param_values[dice_values.index(max(dice_values))],
                }
            else:
                # Categorical - find best choice
                value_dice: dict[str, list[float]] = {}
                for v, d in zip(param_values, dice_values):
                    if v not in value_dice:
                        value_dice[v] = []
                    value_dice[v].append(d)

                avg_dice = {v: sum(ds) / len(ds) for v, ds in value_dice.items()}
                best_value = max(avg_dice.keys(), key=lambda k: avg_dice[k])

                results[param_name] = {
                    "type": "categorical",
                    "avg_dice_by_value": avg_dice,
                    "best_value": best_value,
                }

        return results
