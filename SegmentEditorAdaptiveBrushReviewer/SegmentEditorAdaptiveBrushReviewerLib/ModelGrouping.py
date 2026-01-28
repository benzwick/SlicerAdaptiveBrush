"""Trial-to-model mapping for CrossSegmentationExplorer-style comparison.

Maps optimization trials to "models" for side-by-side comparison,
similar to how CrossSegmentationExplorer groups AI segmentations.

See ADR-018 for design rationale.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ResultsLoader import TrialData

logger = logging.getLogger(__name__)


@dataclass
class ComparisonModel:
    """A model for comparison, containing one or more trials.

    In CrossSegmentationExplorer, a "model" is a group of segmentations
    from the same AI model. For optimization, we map this to groupings
    like "all watershed trials" or "best trial per algorithm".

    Attributes:
        name: Display name for this model (e.g., "watershed", "Gold Standard").
        trials: List of trials belonging to this model.
        color: Optional display color (R, G, B) for visualization.
        metadata: Additional metadata about this model.
    """

    name: str
    trials: list[TrialData] = field(default_factory=list)
    color: tuple[float, float, float] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def best_trial(self) -> TrialData | None:
        """Get the best performing trial in this model."""
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t.value)

    @property
    def best_score(self) -> float:
        """Get the best Dice score in this model."""
        best = self.best_trial
        if best is None:
            return 0.0
        return best.value

    @property
    def trial_count(self) -> int:
        """Get the number of trials in this model."""
        return len(self.trials)


class TrialModelMapper:
    """Map optimization trials to comparison models.

    Provides various grouping strategies for organizing trials
    into CrossSegmentationExplorer-compatible "models".

    Usage:
        mapper = TrialModelMapper()

        # Group by algorithm
        models = mapper.group_by_algorithm(trials)
        # Returns: {"watershed": ComparisonModel, "geodesic": ComparisonModel, ...}

        # Get top N per algorithm
        models = mapper.get_top_n_per_algorithm(trials, n=1)
        # Returns: {"watershed": ComparisonModel(trials=[best_watershed]), ...}

        # Group by Dice score range
        ranges = [(0.95, 1.0, "excellent"), (0.90, 0.95, "good"), (0.0, 0.90, "poor")]
        models = mapper.group_by_dice_range(trials, ranges)
    """

    # Default colors for common algorithms (RGB 0-1)
    DEFAULT_COLORS = {
        "watershed": (0.2, 0.6, 1.0),  # Blue
        "geodesic": (0.2, 0.8, 0.2),  # Green
        "level_set": (1.0, 0.4, 0.2),  # Orange
        "random_walker": (0.8, 0.2, 0.8),  # Purple
        "connected_threshold": (1.0, 0.8, 0.2),  # Yellow
        "region_growing": (0.2, 0.8, 0.8),  # Cyan
        "threshold_brush": (0.8, 0.8, 0.8),  # Gray
        "gold_standard": (1.0, 0.84, 0.0),  # Gold
    }

    def __init__(self) -> None:
        """Initialize trial-to-model mapper."""
        pass

    def group_by_algorithm(
        self, trials: list[TrialData], include_empty: bool = False
    ) -> dict[str, ComparisonModel]:
        """Group trials by algorithm parameter.

        Args:
            trials: List of TrialData objects to group.
            include_empty: If True, include algorithms with no trials.

        Returns:
            Dictionary mapping algorithm name to ComparisonModel.
        """
        groups: dict[str, list[TrialData]] = defaultdict(list)

        for trial in trials:
            algo = trial.params.get("algorithm", "unknown")
            groups[algo].append(trial)

        models = {}
        for algo, algo_trials in groups.items():
            models[algo] = ComparisonModel(
                name=algo,
                trials=sorted(algo_trials, key=lambda t: t.value, reverse=True),
                color=self.DEFAULT_COLORS.get(algo),
                metadata={"grouping": "algorithm", "algorithm": algo},
            )

        logger.debug(f"Grouped {len(trials)} trials into {len(models)} algorithm models")
        return models

    def get_top_n_per_algorithm(
        self, trials: list[TrialData], n: int = 1
    ) -> dict[str, ComparisonModel]:
        """Get top N trials by Dice score for each algorithm.

        Args:
            trials: List of TrialData objects.
            n: Number of top trials to include per algorithm.

        Returns:
            Dictionary mapping algorithm name to ComparisonModel with top N trials.
        """
        by_algo = self.group_by_algorithm(trials)

        models = {}
        for algo, model in by_algo.items():
            # Trials are already sorted by value (descending)
            top_trials = model.trials[:n]
            models[algo] = ComparisonModel(
                name=algo,
                trials=top_trials,
                color=model.color,
                metadata={"grouping": "top_n", "n": n, "algorithm": algo},
            )

        logger.debug(f"Selected top {n} trials for {len(models)} algorithms")
        return models

    def group_by_dice_range(
        self, trials: list[TrialData], ranges: list[tuple[float, float, str]]
    ) -> dict[str, ComparisonModel]:
        """Group trials by Dice score ranges.

        Args:
            trials: List of TrialData objects.
            ranges: List of (min, max, name) tuples defining score ranges.
                Example: [(0.95, 1.0, "excellent"), (0.90, 0.95, "good")]

        Returns:
            Dictionary mapping range name to ComparisonModel.
        """
        groups: dict[str, list[TrialData]] = {name: [] for _, _, name in ranges}

        for trial in trials:
            for min_val, max_val, name in ranges:
                if min_val <= trial.value < max_val or (max_val == 1.0 and trial.value == 1.0):
                    groups[name].append(trial)
                    break

        models = {}
        for name, range_trials in groups.items():
            if range_trials:  # Only include non-empty groups
                models[name] = ComparisonModel(
                    name=name,
                    trials=sorted(range_trials, key=lambda t: t.value, reverse=True),
                    metadata={"grouping": "dice_range", "range_name": name},
                )

        logger.debug(f"Grouped {len(trials)} trials into {len(models)} Dice ranges")
        return models

    def group_by_trial_numbers(
        self, trials: list[TrialData], trial_numbers: list[int]
    ) -> dict[str, ComparisonModel]:
        """Select specific trials by number for comparison.

        Args:
            trials: List of TrialData objects.
            trial_numbers: List of trial numbers to include.

        Returns:
            Dictionary mapping "trial_N" to ComparisonModel.
        """
        trial_map = {t.trial_number: t for t in trials}

        models = {}
        for num in trial_numbers:
            if num in trial_map:
                trial = trial_map[num]
                algo = trial.params.get("algorithm", "unknown")
                name = f"trial_{num}_{algo}"
                models[name] = ComparisonModel(
                    name=name,
                    trials=[trial],
                    color=self.DEFAULT_COLORS.get(algo),
                    metadata={"grouping": "manual", "trial_number": num},
                )

        logger.debug(f"Selected {len(models)} specific trials for comparison")
        return models

    def create_gold_standard_model(self, gold_name: str = "Gold Standard") -> ComparisonModel:
        """Create a placeholder model for gold standard segmentation.

        The gold standard is treated as a special "model" in comparison views.

        Args:
            gold_name: Display name for the gold standard.

        Returns:
            ComparisonModel for gold standard (trials list will be empty,
            segmentation loaded separately).
        """
        return ComparisonModel(
            name=gold_name,
            trials=[],
            color=self.DEFAULT_COLORS["gold_standard"],
            metadata={"grouping": "gold_standard", "is_gold": True},
        )

    def get_best_overall(self, trials: list[TrialData]) -> ComparisonModel:
        """Get a model containing only the single best trial.

        Args:
            trials: List of TrialData objects.

        Returns:
            ComparisonModel with the single best trial.
        """
        if not trials:
            return ComparisonModel(name="best", trials=[])

        best = max(trials, key=lambda t: t.value)
        algo = best.params.get("algorithm", "unknown")

        return ComparisonModel(
            name=f"best_{algo}",
            trials=[best],
            color=self.DEFAULT_COLORS.get(algo),
            metadata={
                "grouping": "best_overall",
                "trial_number": best.trial_number,
                "dice": best.value,
            },
        )

    def filter_by_algorithm(
        self, trials: list[TrialData], algorithms: list[str]
    ) -> list[TrialData]:
        """Filter trials to include only specified algorithms.

        Args:
            trials: List of TrialData objects.
            algorithms: List of algorithm names to include.

        Returns:
            Filtered list of trials.
        """
        return [t for t in trials if t.params.get("algorithm") in algorithms]

    def filter_by_min_dice(self, trials: list[TrialData], min_dice: float) -> list[TrialData]:
        """Filter trials to include only those above minimum Dice score.

        Args:
            trials: List of TrialData objects.
            min_dice: Minimum Dice score threshold.

        Returns:
            Filtered list of trials.
        """
        return [t for t in trials if t.value >= min_dice]


# Convenience functions


def quick_compare_algorithms(trials: list[TrialData]) -> dict[str, ComparisonModel]:
    """Quick comparison: best trial per algorithm.

    Args:
        trials: List of TrialData objects.

    Returns:
        Dictionary of algorithm name to ComparisonModel with best trial.
    """
    mapper = TrialModelMapper()
    return mapper.get_top_n_per_algorithm(trials, n=1)


def quick_compare_top_trials(trials: list[TrialData], n: int = 5) -> dict[str, ComparisonModel]:
    """Quick comparison: top N trials overall.

    Args:
        trials: List of TrialData objects.
        n: Number of top trials to include.

    Returns:
        Dictionary of trial identifiers to ComparisonModel.
    """
    mapper = TrialModelMapper()
    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)[:n]
    return mapper.group_by_trial_numbers(trials, [t.trial_number for t in sorted_trials])
