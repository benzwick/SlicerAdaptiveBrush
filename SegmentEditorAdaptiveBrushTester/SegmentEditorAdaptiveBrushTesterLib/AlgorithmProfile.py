"""Data structures for algorithm performance profiles.

Contains AlgorithmProfile and related classes for characterizing
algorithm strengths, weaknesses, and optimal parameters.

See ADR-011 for architecture decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PerformanceMetrics:
    """Performance metrics for an algorithm."""

    best_dice: float = 0.0
    mean_dice: float = 0.0
    std_dice: float = 0.0
    median_dice: float = 0.0
    worst_dice: float = 0.0

    best_hausdorff_95: float = float("inf")
    mean_hausdorff_95: float = float("inf")
    std_hausdorff_95: float = 0.0

    mean_time_ms: float = 0.0
    std_time_ms: float = 0.0
    speed_category: str = "medium"  # fast, medium, slow

    trial_count: int = 0
    pruned_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "best_dice": self.best_dice,
            "mean_dice": self.mean_dice,
            "std_dice": self.std_dice,
            "median_dice": self.median_dice,
            "worst_dice": self.worst_dice,
            "best_hausdorff_95": self.best_hausdorff_95
            if self.best_hausdorff_95 != float("inf")
            else None,
            "mean_hausdorff_95": self.mean_hausdorff_95
            if self.mean_hausdorff_95 != float("inf")
            else None,
            "std_hausdorff_95": self.std_hausdorff_95,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "speed_category": self.speed_category,
            "trial_count": self.trial_count,
            "pruned_count": self.pruned_count,
        }


@dataclass
class OptimalPreset:
    """Optimal parameter preset for an algorithm."""

    params: dict[str, Any] = field(default_factory=dict)
    dice_achieved: float = 0.0
    hausdorff_95_achieved: float = float("inf")
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "params": self.params,
            "dice_achieved": self.dice_achieved,
            "hausdorff_95_achieved": self.hausdorff_95_achieved
            if self.hausdorff_95_achieved != float("inf")
            else None,
            "description": self.description,
        }


@dataclass
class ExampleScreenshot:
    """Reference to an example screenshot."""

    path: str
    dice: float
    description: str = ""
    trial_number: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "dice": self.dice,
            "description": self.description,
            "trial_number": self.trial_number,
        }


@dataclass
class AlgorithmProfile:
    """Complete profile for a segmentation algorithm.

    Contains performance metrics, optimal parameters, strengths,
    weaknesses, and use case recommendations.

    Example:
        profile = AlgorithmProfile(
            algorithm="watershed",
            display_name="Watershed",
            performance=PerformanceMetrics(best_dice=1.0, mean_dice=0.95),
            optimal_preset=OptimalPreset(params={"edge_sensitivity": 40}),
            strengths=["Perfect Dice achievable", "Good boundary adherence"],
            weaknesses=["Medium speed"],
            best_for=["General-purpose segmentation"],
            avoid_for=["Speed-critical applications"],
        )
    """

    # Algorithm identification
    algorithm: str  # Internal name (watershed, level_set, etc.)
    display_name: str  # Human-readable name

    # Performance data
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    optimal_preset: OptimalPreset = field(default_factory=OptimalPreset)

    # Parameter importance (from FAnova)
    parameter_importance: dict[str, float] = field(default_factory=dict)

    # Qualitative assessment
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    best_for: list[str] = field(default_factory=list)
    avoid_for: list[str] = field(default_factory=list)

    # Click efficiency
    click_efficiency: str = "moderate"  # very_efficient, efficient, moderate, inefficient
    strokes_to_90_pct: int = 0  # Strokes needed to reach 90% of max Dice

    # Over/under segmentation tendency
    over_segmentation_score: float = 0.0  # Positive = over, negative = under
    segmentation_tendency: str = "balanced"  # over, under, balanced

    # Example screenshots
    example_screenshots: list[ExampleScreenshot] = field(default_factory=list)

    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    sample_data_used: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "algorithm": self.algorithm,
            "display_name": self.display_name,
            "performance": self.performance.to_dict(),
            "optimal_preset": self.optimal_preset.to_dict(),
            "parameter_importance": self.parameter_importance,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "best_for": self.best_for,
            "avoid_for": self.avoid_for,
            "click_efficiency": self.click_efficiency,
            "strokes_to_90_pct": self.strokes_to_90_pct,
            "over_segmentation_score": self.over_segmentation_score,
            "segmentation_tendency": self.segmentation_tendency,
            "example_screenshots": [s.to_dict() for s in self.example_screenshots],
            "generated_at": self.generated_at,
            "sample_data_used": self.sample_data_used,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AlgorithmProfile:
        """Create profile from dictionary."""
        performance = PerformanceMetrics(**data.get("performance", {}))
        optimal_preset = OptimalPreset(**data.get("optimal_preset", {}))
        example_screenshots = [ExampleScreenshot(**s) for s in data.get("example_screenshots", [])]

        return cls(
            algorithm=data["algorithm"],
            display_name=data.get("display_name", data["algorithm"]),
            performance=performance,
            optimal_preset=optimal_preset,
            parameter_importance=data.get("parameter_importance", {}),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            best_for=data.get("best_for", []),
            avoid_for=data.get("avoid_for", []),
            click_efficiency=data.get("click_efficiency", "moderate"),
            strokes_to_90_pct=data.get("strokes_to_90_pct", 0),
            over_segmentation_score=data.get("over_segmentation_score", 0.0),
            segmentation_tendency=data.get("segmentation_tendency", "balanced"),
            example_screenshots=example_screenshots,
            generated_at=data.get("generated_at", ""),
            sample_data_used=data.get("sample_data_used", ""),
            notes=data.get("notes", ""),
        )

    def get_summary(self) -> str:
        """Get one-line summary of algorithm."""
        return (
            f"{self.display_name}: Dice={self.performance.mean_dice:.3f} "
            f"({self.performance.speed_category}), {self.click_efficiency} clicks"
        )


@dataclass
class AlgorithmComparison:
    """Comparison of multiple algorithms."""

    profiles: list[AlgorithmProfile] = field(default_factory=list)
    ranking_by_dice: list[str] = field(default_factory=list)
    ranking_by_speed: list[str] = field(default_factory=list)
    recommended_default: str = ""
    comparison_notes: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "profiles": {p.algorithm: p.to_dict() for p in self.profiles},
            "ranking_by_dice": self.ranking_by_dice,
            "ranking_by_speed": self.ranking_by_speed,
            "recommended_default": self.recommended_default,
            "comparison_notes": self.comparison_notes,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AlgorithmComparison:
        """Create comparison from dictionary."""
        profiles = [AlgorithmProfile.from_dict(p) for p in data.get("profiles", {}).values()]
        return cls(
            profiles=profiles,
            ranking_by_dice=data.get("ranking_by_dice", []),
            ranking_by_speed=data.get("ranking_by_speed", []),
            recommended_default=data.get("recommended_default", ""),
            comparison_notes=data.get("comparison_notes", ""),
            generated_at=data.get("generated_at", ""),
        )

    def get_best_for_dice(self) -> AlgorithmProfile | None:
        """Get algorithm with best mean Dice."""
        if not self.profiles:
            return None
        return max(self.profiles, key=lambda p: p.performance.mean_dice)

    def get_fastest(self) -> AlgorithmProfile | None:
        """Get fastest algorithm."""
        if not self.profiles:
            return None
        return min(self.profiles, key=lambda p: p.performance.mean_time_ms)

    def get_most_efficient(self) -> AlgorithmProfile | None:
        """Get most click-efficient algorithm."""
        efficiency_order = {
            "very_efficient": 4,
            "efficient": 3,
            "moderate": 2,
            "inefficient": 1,
        }
        if not self.profiles:
            return None
        return max(
            self.profiles,
            key=lambda p: efficiency_order.get(p.click_efficiency, 0),
        )


# Algorithm display names
ALGORITHM_DISPLAY_NAMES = {
    "watershed": "Watershed",
    "level_set": "Level Set",
    "connected_threshold": "Connected Threshold",
    "region_growing": "Region Growing",
    "threshold_brush": "Threshold Brush",
    "geodesic_distance": "Geodesic Distance",
    "random_walker": "Random Walker",
}


def get_display_name(algorithm: str) -> str:
    """Get human-readable name for algorithm.

    Args:
        algorithm: Internal algorithm name.

    Returns:
        Display name.
    """
    return ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm.replace("_", " ").title())
