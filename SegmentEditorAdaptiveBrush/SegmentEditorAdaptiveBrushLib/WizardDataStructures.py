"""Data structures for the Parameter Wizard.

This module contains dataclasses used to collect and represent
samples, analysis results, and recommendations during the
Quick Select Parameters wizard workflow.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class WizardSamples:
    """Collected samples from wizard interaction.

    Stores foreground, background, and boundary samples collected
    during the wizard's interactive sampling phases.
    """

    foreground_points: list[tuple[int, int, int]] = field(default_factory=list)
    """IJK coordinates of sampled foreground voxels."""

    foreground_intensities: Optional[np.ndarray] = None
    """Intensity values at foreground points."""

    background_points: list[tuple[int, int, int]] = field(default_factory=list)
    """IJK coordinates of sampled background voxels."""

    background_intensities: Optional[np.ndarray] = None
    """Intensity values at background points."""

    boundary_points: list[tuple[int, int, int]] = field(default_factory=list)
    """IJK coordinates of traced boundary points (optional)."""

    volume_node: Any = None
    """Reference to the source volume node."""

    def has_foreground(self) -> bool:
        """Check if foreground samples have been collected."""
        return (
            len(self.foreground_points) > 0
            and self.foreground_intensities is not None
            and len(self.foreground_intensities) > 0
        )

    def has_background(self) -> bool:
        """Check if background samples have been collected."""
        return (
            len(self.background_points) > 0
            and self.background_intensities is not None
            and len(self.background_intensities) > 0
        )

    def has_boundary(self) -> bool:
        """Check if boundary samples have been collected."""
        return len(self.boundary_points) > 0

    @property
    def foreground_count(self) -> int:
        """Return the number of foreground samples."""
        return len(self.foreground_points)

    @property
    def background_count(self) -> int:
        """Return the number of background samples."""
        return len(self.background_points)

    def clear_foreground(self) -> None:
        """Clear all foreground samples."""
        self.foreground_points = []
        self.foreground_intensities = None

    def clear_background(self) -> None:
        """Clear all background samples."""
        self.background_points = []
        self.background_intensities = None

    def clear_boundary(self) -> None:
        """Clear all boundary samples."""
        self.boundary_points = []

    def clear_all(self) -> None:
        """Clear all samples."""
        self.clear_foreground()
        self.clear_background()
        self.clear_boundary()


@dataclass
class IntensityAnalysisResult:
    """Results from intensity distribution analysis.

    Contains statistics about foreground and background intensity
    distributions and measures of their separation.
    """

    foreground_min: float
    """Minimum intensity in foreground samples."""

    foreground_max: float
    """Maximum intensity in foreground samples."""

    foreground_mean: float
    """Mean intensity of foreground samples."""

    foreground_std: float
    """Standard deviation of foreground intensities."""

    background_min: float
    """Minimum intensity in background samples."""

    background_max: float
    """Maximum intensity in background samples."""

    background_mean: float
    """Mean intensity of background samples."""

    background_std: float
    """Standard deviation of background intensities."""

    separation_score: float
    """Score from 0-1 indicating how well-separated the distributions are.

    1.0 means perfect separation, 0.0 means complete overlap.
    """

    overlap_percentage: float
    """Percentage of intensity range where distributions overlap."""

    suggested_threshold_lower: float
    """Suggested lower intensity threshold for segmentation."""

    suggested_threshold_upper: float
    """Suggested upper intensity threshold for segmentation."""

    @property
    def threshold_range(self) -> float:
        """Return the threshold range width."""
        return self.suggested_threshold_upper - self.suggested_threshold_lower

    def is_well_separated(self, threshold: float = 0.7) -> bool:
        """Check if foreground and background are well-separated.

        Args:
            threshold: Separation score threshold for "well-separated" (default 0.7).

        Returns:
            True if separation_score >= threshold.
        """
        return self.separation_score >= threshold


@dataclass
class ShapeAnalysisResult:
    """Results from shape and boundary analysis.

    Contains metrics about the structure's size, shape, and
    boundary characteristics.
    """

    estimated_diameter_mm: float
    """Estimated diameter of the structure in millimeters."""

    circularity: float
    """Circularity metric from 0-1 (1.0 = perfect circle)."""

    convexity: float
    """Convexity metric from 0-1 (1.0 = perfectly convex)."""

    boundary_roughness: float
    """Boundary roughness metric from 0-1 (0.0 = smooth, 1.0 = rough)."""

    suggested_brush_radius_mm: float
    """Suggested brush radius based on structure size."""

    is_3d_structure: bool
    """True if structure spans multiple slices."""

    def is_small_structure(self, threshold_mm: float = 10.0) -> bool:
        """Check if structure is small.

        Args:
            threshold_mm: Diameter threshold in mm (default 10.0).

        Returns:
            True if diameter < threshold.
        """
        return self.estimated_diameter_mm < threshold_mm

    def is_large_structure(self, threshold_mm: float = 50.0) -> bool:
        """Check if structure is large.

        Args:
            threshold_mm: Diameter threshold in mm (default 50.0).

        Returns:
            True if diameter > threshold.
        """
        return self.estimated_diameter_mm > threshold_mm

    def has_smooth_boundary(self, threshold: float = 0.3) -> bool:
        """Check if structure has a smooth boundary.

        Args:
            threshold: Roughness threshold (default 0.3).

        Returns:
            True if boundary_roughness <= threshold.
        """
        return self.boundary_roughness <= threshold


@dataclass
class WizardRecommendation:
    """Final wizard recommendation with explanations.

    Contains the recommended algorithm and parameters along with
    reasoning for each choice.
    """

    algorithm: str
    """Recommended algorithm identifier."""

    algorithm_reason: str
    """Explanation for why this algorithm was chosen."""

    brush_radius_mm: float
    """Recommended brush radius in millimeters."""

    radius_reason: str
    """Explanation for the brush radius choice."""

    edge_sensitivity: int
    """Recommended edge sensitivity (0-100)."""

    sensitivity_reason: str
    """Explanation for the edge sensitivity choice."""

    threshold_lower: Optional[float] = None
    """Suggested lower threshold (if applicable)."""

    threshold_upper: Optional[float] = None
    """Suggested upper threshold (if applicable)."""

    threshold_reason: Optional[str] = None
    """Explanation for threshold values."""

    confidence: float = 0.5
    """Overall confidence in the recommendation (0-1)."""

    warnings: list[str] = field(default_factory=list)
    """List of warnings or caveats about the recommendation."""

    alternative_algorithms: list[tuple[str, str]] = field(default_factory=list)
    """List of (algorithm_id, reason) tuples for alternatives."""

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """Check if recommendation has high confidence.

        Args:
            threshold: Confidence threshold (default 0.75).

        Returns:
            True if confidence >= threshold.
        """
        return self.confidence >= threshold

    def has_warnings(self) -> bool:
        """Check if recommendation has any warnings."""
        return len(self.warnings) > 0

    def has_threshold_suggestion(self) -> bool:
        """Check if threshold values are suggested."""
        return self.threshold_lower is not None and self.threshold_upper is not None
