"""Algorithm characterization from optimization data.

Analyzes optimization trial results to generate comprehensive
algorithm profiles with strengths, weaknesses, and recommendations.

See ADR-011 for architecture decisions.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .OptunaOptimizer import OptimizationResults, OptunaTrialResult

from .AlgorithmProfile import (
    AlgorithmComparison,
    AlgorithmProfile,
    ExampleScreenshot,
    OptimalPreset,
    PerformanceMetrics,
    get_display_name,
)

logger = logging.getLogger(__name__)


class AlgorithmCharacterizer:
    """Generate algorithm profiles from optimization data.

    Analyzes optimization results to characterize each algorithm's
    performance, identify optimal parameters, and generate
    qualitative assessments.

    Example:
        results = OptunaOptimizer(...).optimize(...)
        characterizer = AlgorithmCharacterizer(results)
        profiles = characterizer.characterize_all()

        for profile in profiles:
            print(f"{profile.display_name}: {profile.strengths}")
    """

    # Speed thresholds (ms)
    SPEED_FAST_THRESHOLD = 100
    SPEED_SLOW_THRESHOLD = 500

    # Click efficiency thresholds
    CLICKS_VERY_EFFICIENT = 3
    CLICKS_EFFICIENT = 5
    CLICKS_MODERATE = 8

    def __init__(
        self,
        results: OptimizationResults,
        sample_data_name: str = "",
    ) -> None:
        """Initialize characterizer.

        Args:
            results: Optimization results to analyze.
            sample_data_name: Name of sample data used.
        """
        self.results = results
        self.sample_data_name = sample_data_name
        self._trials_by_algorithm: dict[str, list[OptunaTrialResult]] = {}
        self._group_trials_by_algorithm()

    def _group_trials_by_algorithm(self) -> None:
        """Group trials by algorithm."""
        self._trials_by_algorithm = {}

        for trial in self.results.trials:
            algo = trial.params.get("algorithm", "unknown")
            if algo not in self._trials_by_algorithm:
                self._trials_by_algorithm[algo] = []
            self._trials_by_algorithm[algo].append(trial)

        logger.debug(
            f"Grouped {len(self.results.trials)} trials into "
            f"{len(self._trials_by_algorithm)} algorithms"
        )

    def characterize(self, algorithm: str) -> AlgorithmProfile:
        """Generate profile for a single algorithm.

        Args:
            algorithm: Algorithm name to characterize.

        Returns:
            AlgorithmProfile with complete characterization.
        """
        trials = self._trials_by_algorithm.get(algorithm, [])
        if not trials:
            logger.warning(f"No trials found for algorithm: {algorithm}")
            return AlgorithmProfile(
                algorithm=algorithm,
                display_name=get_display_name(algorithm),
            )

        # Filter out pruned and failed trials for metrics
        completed_trials = [t for t in trials if not t.pruned and t.value > float("-inf")]

        if not completed_trials:
            logger.warning(f"No completed trials for algorithm: {algorithm}")
            return AlgorithmProfile(
                algorithm=algorithm,
                display_name=get_display_name(algorithm),
                performance=PerformanceMetrics(
                    trial_count=len(trials),
                    pruned_count=sum(1 for t in trials if t.pruned),
                ),
            )

        # Compute performance metrics
        performance = self._compute_performance(completed_trials, len(trials))

        # Find optimal preset
        optimal = self._find_optimal_preset(completed_trials)

        # Compute parameter importance for this algorithm
        param_importance = self._compute_param_importance(algorithm)

        # Generate qualitative assessments
        strengths = self._identify_strengths(performance, completed_trials)
        weaknesses = self._identify_weaknesses(performance, completed_trials)
        best_for = self._infer_best_use_cases(performance, completed_trials)
        avoid_for = self._infer_avoid_cases(performance, completed_trials)

        # Click efficiency
        click_efficiency = self._compute_click_efficiency(completed_trials)
        strokes_to_90 = self._compute_strokes_to_threshold(completed_trials, 0.9)

        # Segmentation tendency
        over_seg_score, tendency = self._compute_segmentation_tendency(completed_trials)

        # Select example screenshots
        examples = self._select_example_screenshots(completed_trials)

        profile = AlgorithmProfile(
            algorithm=algorithm,
            display_name=get_display_name(algorithm),
            performance=performance,
            optimal_preset=optimal,
            parameter_importance=param_importance,
            strengths=strengths,
            weaknesses=weaknesses,
            best_for=best_for,
            avoid_for=avoid_for,
            click_efficiency=click_efficiency,
            strokes_to_90_pct=strokes_to_90,
            over_segmentation_score=over_seg_score,
            segmentation_tendency=tendency,
            example_screenshots=examples,
            sample_data_used=self.sample_data_name,
        )

        return profile

    def characterize_all(self) -> list[AlgorithmProfile]:
        """Generate profiles for all algorithms in results.

        Returns:
            List of AlgorithmProfile objects.
        """
        profiles = []
        for algorithm in self._trials_by_algorithm:
            profile = self.characterize(algorithm)
            profiles.append(profile)

        # Sort by mean dice descending
        profiles.sort(key=lambda p: p.performance.mean_dice, reverse=True)

        return profiles

    def create_comparison(self) -> AlgorithmComparison:
        """Create comparison of all algorithms.

        Returns:
            AlgorithmComparison with rankings and recommendations.
        """
        profiles = self.characterize_all()

        # Rank by dice
        ranking_by_dice = [
            p.algorithm
            for p in sorted(profiles, key=lambda p: p.performance.mean_dice, reverse=True)
        ]

        # Rank by speed
        ranking_by_speed = [
            p.algorithm for p in sorted(profiles, key=lambda p: p.performance.mean_time_ms)
        ]

        # Recommend default (best balance of dice and speed)
        recommended = self._recommend_default(profiles)

        comparison = AlgorithmComparison(
            profiles=profiles,
            ranking_by_dice=ranking_by_dice,
            ranking_by_speed=ranking_by_speed,
            recommended_default=recommended,
            comparison_notes=self._generate_comparison_notes(profiles),
        )

        return comparison

    def _compute_performance(
        self,
        trials: list[OptunaTrialResult],
        total_trials: int,
    ) -> PerformanceMetrics:
        """Compute performance metrics from trials."""
        dices = [t.value for t in trials if t.value > 0]
        times = [t.duration_ms for t in trials if t.duration_ms > 0]

        if not dices:
            return PerformanceMetrics(trial_count=total_trials)

        metrics = PerformanceMetrics(
            best_dice=max(dices),
            mean_dice=statistics.mean(dices),
            std_dice=statistics.stdev(dices) if len(dices) > 1 else 0,
            median_dice=statistics.median(dices),
            worst_dice=min(dices),
            mean_time_ms=statistics.mean(times) if times else 0,
            std_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
            trial_count=total_trials,
            pruned_count=total_trials - len(trials),
        )

        # Categorize speed
        if metrics.mean_time_ms < self.SPEED_FAST_THRESHOLD:
            metrics.speed_category = "fast"
        elif metrics.mean_time_ms > self.SPEED_SLOW_THRESHOLD:
            metrics.speed_category = "slow"
        else:
            metrics.speed_category = "medium"

        return metrics

    def _find_optimal_preset(
        self,
        trials: list[OptunaTrialResult],
    ) -> OptimalPreset:
        """Find optimal parameter preset."""
        if not trials:
            return OptimalPreset()

        # Find best trial by dice
        best = max(trials, key=lambda t: t.value)

        # Filter params to remove non-parameter keys
        params = {k: v for k, v in best.params.items() if k != "algorithm"}

        return OptimalPreset(
            params=params,
            dice_achieved=best.value,
            description=f"Parameters from trial #{best.trial_number}",
        )

    def _compute_param_importance(self, algorithm: str) -> dict[str, float]:
        """Compute parameter importance for algorithm.

        Uses results-level importance, filtered to relevant params.
        """
        # Start with global importance from results
        importance = dict(self.results.parameter_importance)

        # Remove algorithm param itself
        importance.pop("algorithm", None)

        return importance

    def _identify_strengths(
        self,
        performance: PerformanceMetrics,
        trials: list[OptunaTrialResult],
    ) -> list[str]:
        """Identify algorithm strengths."""
        strengths = []

        # High Dice achievable
        if performance.best_dice >= 0.99:
            strengths.append("Perfect Dice achievable (1.0)")
        elif performance.best_dice >= 0.95:
            strengths.append(f"Excellent accuracy (best Dice: {performance.best_dice:.3f})")

        # Consistent performance
        if performance.std_dice < 0.05:
            strengths.append("Consistent performance across parameter variations")

        # Fast speed
        if performance.speed_category == "fast":
            strengths.append(f"Very fast (~{performance.mean_time_ms:.0f}ms)")

        # High worst case
        if performance.worst_dice >= 0.8:
            strengths.append("Robust - worst case still achieves good results")

        # Low pruning rate
        prune_rate = performance.pruned_count / max(performance.trial_count, 1)
        if prune_rate < 0.1:
            strengths.append("Few trials fail or require early stopping")

        return strengths

    def _identify_weaknesses(
        self,
        performance: PerformanceMetrics,
        trials: list[OptunaTrialResult],
    ) -> list[str]:
        """Identify algorithm weaknesses."""
        weaknesses = []

        # Slow speed
        if performance.speed_category == "slow":
            weaknesses.append(f"Slow computation (~{performance.mean_time_ms:.0f}ms)")

        # Inconsistent
        if performance.std_dice > 0.15:
            weaknesses.append("Parameter-sensitive - results vary significantly")

        # Low worst case
        if performance.worst_dice < 0.5 and performance.best_dice > 0.8:
            weaknesses.append("Can fail badly with wrong parameters")

        # High pruning rate
        prune_rate = performance.pruned_count / max(performance.trial_count, 1)
        if prune_rate > 0.3:
            weaknesses.append("Many trials fail or produce poor results")

        # Mediocre best case
        if performance.best_dice < 0.85:
            weaknesses.append(f"Limited accuracy ceiling (max Dice: {performance.best_dice:.3f})")

        return weaknesses

    def _infer_best_use_cases(
        self,
        performance: PerformanceMetrics,
        trials: list[OptunaTrialResult],
    ) -> list[str]:
        """Infer best use cases for algorithm."""
        use_cases = []

        # High accuracy algorithms
        if performance.best_dice >= 0.95:
            use_cases.append("High-precision segmentation tasks")

        # Fast algorithms
        if performance.speed_category == "fast":
            use_cases.append("Interactive segmentation (real-time feedback)")
            use_cases.append("Large volume processing")

        # Consistent algorithms
        if performance.std_dice < 0.05:
            use_cases.append("Automated pipelines (predictable results)")

        # Based on display name / algorithm type
        # (This would ideally come from algorithm metadata)

        return use_cases or ["General-purpose segmentation"]

    def _infer_avoid_cases(
        self,
        performance: PerformanceMetrics,
        trials: list[OptunaTrialResult],
    ) -> list[str]:
        """Infer cases to avoid for algorithm."""
        avoid = []

        # Slow algorithms
        if performance.speed_category == "slow":
            avoid.append("Interactive use where responsiveness matters")

        # Inconsistent algorithms
        if performance.std_dice > 0.15:
            avoid.append("Automated pipelines without parameter tuning")

        # Low accuracy
        if performance.best_dice < 0.85:
            avoid.append("Tasks requiring high precision")

        return avoid

    def _compute_click_efficiency(
        self,
        trials: list[OptunaTrialResult],
    ) -> str:
        """Compute click efficiency category."""
        # This would require stroke-level data from intermediate values
        # For now, use a heuristic based on trial performance

        if not trials:
            return "moderate"

        # Use intermediate values if available
        stroke_counts = []
        for trial in trials:
            if trial.intermediate_values:
                # Count strokes to reach 90% of final dice
                final = trial.value
                threshold = final * 0.9
                strokes = 0
                for step, dice in sorted(trial.intermediate_values.items()):
                    strokes = step + 1
                    if dice >= threshold:
                        break
                stroke_counts.append(strokes)

        if stroke_counts:
            avg_strokes = statistics.mean(stroke_counts)
            if avg_strokes <= self.CLICKS_VERY_EFFICIENT:
                return "very_efficient"
            elif avg_strokes <= self.CLICKS_EFFICIENT:
                return "efficient"
            elif avg_strokes <= self.CLICKS_MODERATE:
                return "moderate"
            else:
                return "inefficient"

        return "moderate"

    def _compute_strokes_to_threshold(
        self,
        trials: list[OptunaTrialResult],
        threshold_ratio: float,
    ) -> int:
        """Compute strokes needed to reach threshold of max dice.

        Args:
            trials: Trial results with intermediate values.
            threshold_ratio: Fraction of max dice to reach (e.g., 0.9).

        Returns:
            Average number of strokes, or 0 if unknown.
        """
        stroke_counts = []

        for trial in trials:
            if trial.intermediate_values:
                final = trial.value
                threshold = final * threshold_ratio

                for step, dice in sorted(trial.intermediate_values.items()):
                    if dice >= threshold:
                        stroke_counts.append(step + 1)
                        break

        if stroke_counts:
            return int(statistics.mean(stroke_counts))

        return 0

    def _compute_segmentation_tendency(
        self,
        trials: list[OptunaTrialResult],
    ) -> tuple[float, str]:
        """Compute over/under segmentation tendency.

        Returns:
            Tuple of (score, tendency) where score > 0 means over-segmentation.
        """
        # This would require volume comparison data
        # For now, return balanced
        return (0.0, "balanced")

    def _select_example_screenshots(
        self,
        trials: list[OptunaTrialResult],
        max_examples: int = 3,
    ) -> list[ExampleScreenshot]:
        """Select representative example screenshots."""
        examples: list[ExampleScreenshot] = []

        if not trials:
            return examples

        # Select best trial
        best = max(trials, key=lambda t: t.value)
        screenshots = best.user_attrs.get("screenshots", [])

        if screenshots:
            # Get final screenshot
            for ss in screenshots[-max_examples:]:
                examples.append(
                    ExampleScreenshot(
                        path=ss.get("path", ""),
                        dice=ss.get("dice", best.value),
                        trial_number=best.trial_number,
                    )
                )

        return examples

    def _recommend_default(self, profiles: list[AlgorithmProfile]) -> str:
        """Recommend default algorithm based on balance of factors."""
        if not profiles:
            return ""

        # Score each profile
        scores = {}
        for p in profiles:
            # Weighted score: dice most important, then speed, then consistency
            dice_score = p.performance.mean_dice * 100
            speed_score = 100 - min(p.performance.mean_time_ms / 10, 100)  # 0-100
            consistency_score = (1 - min(p.performance.std_dice, 0.5) * 2) * 50

            scores[p.algorithm] = dice_score * 0.5 + speed_score * 0.3 + consistency_score * 0.2

        return max(scores, key=lambda k: scores[k])

    def _generate_comparison_notes(self, profiles: list[AlgorithmProfile]) -> str:
        """Generate comparison notes."""
        if not profiles:
            return ""

        notes = []

        # Best by dice
        best_dice = max(profiles, key=lambda p: p.performance.mean_dice)
        notes.append(
            f"Best accuracy: {best_dice.display_name} "
            f"(mean Dice: {best_dice.performance.mean_dice:.3f})"
        )

        # Fastest
        fastest = min(profiles, key=lambda p: p.performance.mean_time_ms)
        notes.append(
            f"Fastest: {fastest.display_name} " f"(~{fastest.performance.mean_time_ms:.0f}ms)"
        )

        return " | ".join(notes)

    def save_profiles(self, output_path: Path | str) -> None:
        """Save all profiles to JSON file.

        Args:
            output_path: Path to save profiles.
        """
        comparison = self.create_comparison()
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2)

        logger.info(f"Saved algorithm profiles to {output_path}")

    @classmethod
    def load_profiles(cls, input_path: Path | str) -> AlgorithmComparison:
        """Load profiles from JSON file.

        Args:
            input_path: Path to load from.

        Returns:
            AlgorithmComparison with loaded profiles.
        """
        input_path = Path(input_path)

        with open(input_path) as f:
            data = json.load(f)

        return AlgorithmComparison.from_dict(data)
