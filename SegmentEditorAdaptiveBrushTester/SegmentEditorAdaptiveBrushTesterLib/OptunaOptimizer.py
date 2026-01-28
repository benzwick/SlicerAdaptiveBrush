"""Optuna-powered parameter optimization for segmentation algorithms.

Provides smart parameter optimization using TPE (Tree-structured Parzen Estimator)
sampling and HyperbandPruner for early stopping of poor trials.

See ADR-011 for architecture decisions.

Requires: pip install optuna
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Check for optuna availability
try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


@dataclass
class OptunaTrialResult:
    """Result from a single Optuna trial."""

    trial_number: int
    params: dict[str, Any]
    value: float  # Objective value (Dice for maximize, HD95 for minimize)
    duration_ms: float
    pruned: bool = False
    user_attrs: dict[str, Any] = field(default_factory=dict)
    intermediate_values: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trial_number": self.trial_number,
            "params": self.params,
            "value": self.value,
            "duration_ms": self.duration_ms,
            "pruned": self.pruned,
            "user_attrs": self.user_attrs,
            "intermediate_values": self.intermediate_values,
        }


@dataclass
class OptimizationResults:
    """Complete results from an optimization run."""

    config_name: str
    n_trials: int
    best_trial: OptunaTrialResult | None
    trials: list[OptunaTrialResult]
    parameter_importance: dict[str, float]
    study_name: str
    direction: str
    duration_seconds: float
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dicom_info: dict[str, Any] | None = None  # DICOM volume/study UIDs

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "config_name": self.config_name,
            "n_trials": self.n_trials,
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "trials": [t.to_dict() for t in self.trials],
            "parameter_importance": self.parameter_importance,
            "study_name": self.study_name,
            "direction": self.direction,
            "duration_seconds": self.duration_seconds,
            "completed_at": self.completed_at,
        }
        if self.dicom_info:
            result["dicom"] = self.dicom_info
        return result


class OptunaOptimizer:
    """Optuna-powered parameter optimization.

    Uses TPE sampler for intelligent parameter suggestion and
    HyperbandPruner for early stopping of poor trials.

    Example:
        from OptimizationConfig import OptimizationConfig

        config = OptimizationConfig.load("configs/tumor_optimization.yaml")
        optimizer = OptunaOptimizer(config)

        def objective(params):
            # Run segmentation with params...
            return dice_score

        results = optimizer.optimize(objective)
        print(f"Best Dice: {results.best_trial.value}")
    """

    def __init__(
        self,
        config: Any,  # OptimizationConfig
        output_dir: Path | str | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            config: OptimizationConfig instance.
            output_dir: Directory for outputs. If None, uses optimization_results/.

        Raises:
            ImportError: If optuna is not installed.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.config = config
        self.study: optuna.Study | None = None
        self.trial_results: list[OptunaTrialResult] = []
        self.dicom_info: dict[str, Any] | None = None

        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            output_dir = (
                Path("optimization_results") / f"{timestamp}_{config.name.replace(' ', '_')}"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config copy
        config_copy_path = self.output_dir / "config.yaml"
        config.save(config_copy_path)

    def set_dicom_info(self, dicom_info: dict[str, Any]) -> None:
        """Set DICOM volume/study information.

        Args:
            dicom_info: Dictionary with DICOM UIDs (patient_id, study_instance_uid,
                       volume_series_uid, volume_name).
        """
        self.dicom_info = dicom_info

    def create_study(self, study_name: str | None = None) -> optuna.Study:
        """Create Optuna study with configured sampler and pruner.

        Args:
            study_name: Optional study name. Defaults to config name.

        Returns:
            Created Optuna study.
        """
        if study_name is None:
            study_name = self.config.name.replace(" ", "_")

        # Create sampler
        sampler = self._create_sampler()

        # Create pruner
        pruner = self._create_pruner()

        # SQLite storage for persistence
        storage_path = self.output_dir / "optuna_study.db"
        storage = f"sqlite:///{storage_path}"

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.config.direction,
            load_if_exists=True,
        )

        logger.info(
            f"Created study '{study_name}' with {self.config.sampler} sampler "
            f"and {self.config.pruner} pruner"
        )

        return self.study

    def _create_sampler(self) -> Any:
        """Create sampler based on config."""
        if self.config.sampler == "tpe":
            return TPESampler(seed=42)
        elif self.config.sampler == "random":
            return RandomSampler(seed=42)
        elif self.config.sampler == "cmaes":
            return CmaEsSampler(seed=42)
        else:
            logger.warning(f"Unknown sampler '{self.config.sampler}', using TPE")
            return TPESampler(seed=42)

    def _create_pruner(self) -> Any:
        """Create pruner based on config."""
        if self.config.pruner == "hyperband":
            return HyperbandPruner()
        elif self.config.pruner == "median":
            return MedianPruner()
        elif self.config.pruner == "none":
            return optuna.pruners.NopPruner()
        else:
            logger.warning(f"Unknown pruner '{self.config.pruner}', using Hyperband")
            return HyperbandPruner()

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameters for a trial.

        Uses config's parameter space to suggest values.
        Handles hierarchical parameters (algorithm-specific).

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of suggested parameters.
        """
        params: dict[str, Any] = {}

        # Algorithm selection
        if self.config.algorithm_substitution_enabled and self.config.algorithm_candidates:
            # Multiple algorithms - let Optuna choose
            algo = trial.suggest_categorical("algorithm", self.config.algorithm_candidates)
            params["algorithm"] = algo
        elif self.config.fixed_algorithm:
            # Single fixed algorithm
            algo = self.config.fixed_algorithm
            params["algorithm"] = algo
        else:
            algo = None

        # Global parameters
        for name, spec in self.config.global_params.items():
            params[name] = self._suggest_param(trial, name, spec)

        # Algorithm-specific parameters (hierarchical)
        if algo and algo in self.config.algorithm_params:
            for name, spec in self.config.algorithm_params[algo].items():
                params[name] = self._suggest_param(trial, name, spec)

        return params

    def _suggest_param(self, trial: optuna.Trial, name: str, spec: Any) -> Any:
        """Suggest a single parameter value.

        Args:
            trial: Optuna trial.
            name: Parameter name.
            spec: ParameterSpec.

        Returns:
            Suggested value.
        """
        if spec.param_type == "categorical":
            return trial.suggest_categorical(name, spec.choices)

        elif spec.param_type == "int":
            return trial.suggest_int(
                name,
                int(spec.min_val),
                int(spec.max_val),
                step=int(spec.step) if spec.step else 1,
                log=spec.log_scale,
            )

        elif spec.param_type == "float":
            if spec.step:
                return trial.suggest_float(
                    name,
                    spec.min_val,
                    spec.max_val,
                    step=spec.step,
                    log=spec.log_scale,
                )
            else:
                return trial.suggest_float(
                    name,
                    spec.min_val,
                    spec.max_val,
                    log=spec.log_scale,
                )

        else:
            logger.warning(f"Unknown param type '{spec.param_type}', treating as float")
            return trial.suggest_float(name, spec.min_val or 0, spec.max_val or 1)

    def optimize(
        self,
        objective: Callable[[optuna.Trial, dict[str, Any]], float],
        n_trials: int | None = None,
        timeout: float | None = None,
        show_progress_bar: bool = True,
        callbacks: list[Callable] | None = None,
    ) -> OptimizationResults:
        """Run optimization.

        Args:
            objective: Objective function taking (trial, params) and returning metric.
            n_trials: Number of trials. Defaults to config value.
            timeout: Timeout in seconds. Defaults to config value.
            show_progress_bar: Whether to show progress bar.
            callbacks: Additional Optuna callbacks.

        Returns:
            OptimizationResults with all trial data.
        """
        import time

        if self.study is None:
            self.create_study()

        n_trials = n_trials or self.config.n_trials
        timeout_seconds = (
            (self.config.timeout_minutes * 60) if self.config.timeout_minutes else timeout
        )

        logger.info(f"Starting optimization: {n_trials} trials")
        start_time = time.time()

        def wrapped_objective(trial: optuna.Trial) -> float:
            """Wrapper that handles parameter suggestion and result recording."""
            params = self.suggest_params(trial)
            trial_start = time.time()

            try:
                value = objective(trial, params)

                # Record result
                # Note: intermediate_values removed in Optuna 4.x
                result = OptunaTrialResult(
                    trial_number=trial.number,
                    params=params,
                    value=value,
                    duration_ms=(time.time() - trial_start) * 1000,
                    pruned=False,
                    user_attrs=dict(trial.user_attrs),
                    intermediate_values={},  # Tracking intermediate values removed in Optuna 4.x
                )
                self.trial_results.append(result)

                return value

            except optuna.TrialPruned:
                result = OptunaTrialResult(
                    trial_number=trial.number,
                    params=params,
                    value=float("-inf") if self.config.direction == "maximize" else float("inf"),
                    duration_ms=(time.time() - trial_start) * 1000,
                    pruned=True,
                    user_attrs=dict(trial.user_attrs),
                    intermediate_values={},  # Tracking intermediate values removed in Optuna 4.x
                )
                self.trial_results.append(result)
                raise

        # Run optimization
        assert self.study is not None  # Created in create_study()
        self.study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks or [],
        )

        duration = time.time() - start_time

        # Build results
        results = self._build_results(duration)

        # Save results
        self._save_results(results)

        return results

    def _build_results(self, duration_seconds: float) -> OptimizationResults:
        """Build optimization results.

        Args:
            duration_seconds: Total optimization duration.

        Returns:
            OptimizationResults instance.
        """
        best_trial = None
        if self.study and self.study.best_trial:
            bt = self.study.best_trial
            best_trial = OptunaTrialResult(
                trial_number=bt.number,
                params=bt.params,
                value=bt.value,
                duration_ms=bt.duration.total_seconds() * 1000 if bt.duration else 0,
                pruned=False,
                user_attrs=dict(bt.user_attrs),
            )

        # Compute parameter importance
        importance = self.get_param_importance()

        return OptimizationResults(
            config_name=self.config.name,
            n_trials=len(self.trial_results),
            best_trial=best_trial,
            trials=self.trial_results,
            parameter_importance=importance,
            study_name=self.study.study_name if self.study else "",
            direction=self.config.direction,
            duration_seconds=duration_seconds,
            dicom_info=self.dicom_info,
        )

    def get_param_importance(self) -> dict[str, float]:
        """Compute parameter importance using FAnova.

        Returns:
            Dictionary mapping parameter names to importance (0-1).
        """
        if self.study is None or len(self.study.trials) < 5:
            logger.warning("Need at least 5 trials for importance analysis")
            return {}

        try:
            importance = optuna.importance.get_param_importances(
                self.study,
                evaluator=optuna.importance.FanovaImportanceEvaluator(),
            )
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            return {}

    def _save_results(self, results: OptimizationResults) -> None:
        """Save results to output directory.

        Args:
            results: Results to save.
        """
        # Save full results JSON
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        # Save parameter importance separately
        importance_path = self.output_dir / "parameter_importance.json"
        with open(importance_path, "w") as f:
            json.dump(results.parameter_importance, f, indent=2)

        logger.info(f"Saved optimization results to {self.output_dir}")

    def report_intermediate(self, trial: optuna.Trial, step: int, value: float) -> bool:
        """Report intermediate value for pruning.

        Args:
            trial: Current trial.
            step: Step number (e.g., stroke index).
            value: Intermediate metric value.

        Returns:
            True if trial should be pruned.
        """
        trial.report(value, step)
        return bool(trial.should_prune())

    def get_best_params(self) -> dict[str, Any]:
        """Get best parameters found.

        Returns:
            Best parameter dictionary.
        """
        if self.study and self.study.best_trial:
            return dict(self.study.best_params)
        return {}

    def get_best_value(self) -> float | None:
        """Get best objective value.

        Returns:
            Best value or None if no trials.
        """
        if self.study and self.study.best_trial:
            return float(self.study.best_value)
        return None

    def resume(self) -> None:
        """Resume optimization from saved state.

        Creates study with load_if_exists=True to continue from
        SQLite storage.
        """
        if self.study is None:
            self.create_study()

        # Load existing trial results
        assert self.study is not None  # Created by create_study()
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = OptunaTrialResult(
                    trial_number=trial.number,
                    params=trial.params,
                    value=trial.value if trial.value is not None else float("nan"),
                    duration_ms=trial.duration.total_seconds() * 1000 if trial.duration else 0,
                    pruned=False,
                    user_attrs=dict(trial.user_attrs),
                )
                self.trial_results.append(result)
            elif trial.state == optuna.trial.TrialState.PRUNED:
                result = OptunaTrialResult(
                    trial_number=trial.number,
                    params=trial.params,
                    value=float("-inf") if self.config.direction == "maximize" else float("inf"),
                    duration_ms=trial.duration.total_seconds() * 1000 if trial.duration else 0,
                    pruned=True,
                    user_attrs=dict(trial.user_attrs),
                )
                self.trial_results.append(result)

        logger.info(f"Resumed study with {len(self.trial_results)} existing trials")


def run_optimization_with_recipe(
    config_path: Path | str,
    recipe_path: Path | str | None = None,
    gold_standard_path: Path | str | None = None,
) -> OptimizationResults:
    """Convenience function to run optimization with a recipe.

    Args:
        config_path: Path to YAML config file.
        recipe_path: Optional recipe path override.
        gold_standard_path: Optional gold standard path override.

    Returns:
        OptimizationResults.
    """
    from .OptimizationConfig import OptimizationConfig
    from .Recipe import Recipe
    from .RecipeRunner import RecipeRunner
    from .SegmentationMetrics import SegmentationMetrics

    # Load config
    config = OptimizationConfig.load(config_path)

    # Override recipe if provided
    if recipe_path:
        config.recipes = [config.recipes[0]]  # Keep first
        config.recipes[0].path = Path(recipe_path)
    if gold_standard_path:
        config.recipes[0].gold_standard = Path(gold_standard_path)

    # Create optimizer
    optimizer = OptunaOptimizer(config)
    optimizer.create_study()

    # Define objective
    def objective(trial: optuna.Trial, params: dict[str, Any]) -> float:
        # Load recipe
        recipe_spec = config.recipes[0]
        recipe = Recipe.load(recipe_spec.path)

        # Run recipe - parameters are applied by the RecipeRunner
        # which sets up the effect before calling recipe.run(effect)
        runner = RecipeRunner(recipe)

        # TODO: Apply params to effect before running
        # The new recipe system uses native Python scripts that call
        # effect methods directly. Optimization would need to:
        # 1. Set effect parameters before recipe.run(effect)
        # 2. Or wrap the effect to override parameters

        result = runner.run()

        if not result.success:
            return float("-inf") if config.direction == "maximize" else float("inf")

        # Compute final metrics against gold standard
        if recipe_spec.gold_standard:
            # TODO: Load gold standard segmentation for comparison
            metrics = SegmentationMetrics.compute(
                result.segmentation_node,
                result.segment_id,
                None,  # gold_seg_node - would need to load
                "",  # gold_segment_id - placeholder until gold loading implemented
                result.volume_node,
            )

            if config.primary_metric == "dice":
                return metrics.dice
            else:
                return metrics.hausdorff_95

        # No gold standard - return voxel count as proxy
        return float(result.voxel_count)

    # Run optimization
    results = optimizer.optimize(objective)

    return results
