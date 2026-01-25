"""Wizard Analyzer for parameter wizard intensity and shape analysis.

This module provides analysis functions for the Quick Select Parameters
wizard, including intensity distribution analysis and shape estimation.
"""

from __future__ import annotations

import logging

import numpy as np
from WizardDataStructures import (
    IntensityAnalysisResult,
    ShapeAnalysisResult,
    WizardSamples,
)

logger = logging.getLogger(__name__)


class WizardAnalyzer:
    """Analyzes collected samples for the parameter wizard.

    This class provides methods to analyze foreground and background
    intensity distributions and estimate shape characteristics from
    sampled points.
    """

    def analyze_intensities(self, samples: WizardSamples) -> IntensityAnalysisResult:
        """Analyze foreground and background intensity distributions.

        Args:
            samples: WizardSamples containing foreground and background data.

        Returns:
            IntensityAnalysisResult with distribution statistics and thresholds.

        Raises:
            ValueError: If samples are missing foreground or background data.
        """
        if not samples.has_foreground():
            raise ValueError("Foreground samples required for intensity analysis")
        if not samples.has_background():
            raise ValueError("Background samples required for intensity analysis")

        fg = samples.foreground_intensities
        bg = samples.background_intensities

        # Compute basic statistics
        fg_min = float(np.min(fg))
        fg_max = float(np.max(fg))
        fg_mean = float(np.mean(fg))
        fg_std = float(np.std(fg))

        bg_min = float(np.min(bg))
        bg_max = float(np.max(bg))
        bg_mean = float(np.mean(bg))
        bg_std = float(np.std(bg))

        # Calculate separation score
        separation_score = self._calculate_separation_score(fg, bg)

        # Calculate overlap percentage
        overlap_percentage = self._calculate_overlap_percentage(fg, bg)

        # Suggest thresholds
        threshold_lower, threshold_upper = self._suggest_thresholds(fg, bg, fg_mean, bg_mean)

        return IntensityAnalysisResult(
            foreground_min=fg_min,
            foreground_max=fg_max,
            foreground_mean=fg_mean,
            foreground_std=fg_std,
            background_min=bg_min,
            background_max=bg_max,
            background_mean=bg_mean,
            background_std=bg_std,
            separation_score=separation_score,
            overlap_percentage=overlap_percentage,
            suggested_threshold_lower=threshold_lower,
            suggested_threshold_upper=threshold_upper,
        )

    def _calculate_separation_score(self, fg: np.ndarray, bg: np.ndarray) -> float:
        """Calculate separation score between two distributions.

        Uses a combination of:
        - Cohen's d effect size
        - Overlap percentage

        Args:
            fg: Foreground intensity values.
            bg: Background intensity values.

        Returns:
            Separation score from 0 (identical) to 1 (perfect separation).
        """
        fg_mean = np.mean(fg)
        bg_mean = np.mean(bg)
        fg_std = np.std(fg)
        bg_std = np.std(bg)

        # Pooled standard deviation
        pooled_std = np.sqrt((fg_std**2 + bg_std**2) / 2)

        if pooled_std < 1e-6:
            # Both distributions have near-zero variance
            if abs(fg_mean - bg_mean) < 1e-6:
                return 0.0  # Identical distributions
            else:
                return 1.0  # Different constant values

        # Cohen's d effect size
        cohens_d = abs(fg_mean - bg_mean) / pooled_std

        # Convert to 0-1 scale using sigmoid-like transformation
        # d > 2 is considered "very large" effect
        # d > 0.8 is "large", d > 0.5 is "medium", d > 0.2 is "small"
        separation_score = 1.0 - 1.0 / (1.0 + cohens_d / 2.0)

        # Ensure in [0, 1] range
        return float(np.clip(separation_score, 0.0, 1.0))

    def _calculate_overlap_percentage(self, fg: np.ndarray, bg: np.ndarray) -> float:
        """Calculate the percentage of intensity range that overlaps.

        Args:
            fg: Foreground intensity values.
            bg: Background intensity values.

        Returns:
            Percentage of overlap (0-100).
        """
        # Use percentiles to be robust to outliers
        fg_low, fg_high = np.percentile(fg, [5, 95])
        bg_low, bg_high = np.percentile(bg, [5, 95])

        # Find overlap region
        overlap_low = max(fg_low, bg_low)
        overlap_high = min(fg_high, bg_high)

        if overlap_low >= overlap_high:
            return 0.0  # No overlap

        # Total range covered by both distributions
        total_low = min(fg_low, bg_low)
        total_high = max(fg_high, bg_high)
        total_range = total_high - total_low

        if total_range < 1e-6:
            return 100.0  # Identical distributions

        overlap_range = overlap_high - overlap_low
        overlap_percentage = 100.0 * overlap_range / total_range

        return float(np.clip(overlap_percentage, 0.0, 100.0))

    def _suggest_thresholds(
        self, fg: np.ndarray, bg: np.ndarray, fg_mean: float, bg_mean: float
    ) -> tuple[float, float]:
        """Suggest intensity thresholds based on distributions.

        Args:
            fg: Foreground intensity values.
            bg: Background intensity values.
            fg_mean: Foreground mean intensity.
            bg_mean: Background mean intensity.

        Returns:
            Tuple of (lower_threshold, upper_threshold).
        """
        # Determine if foreground is brighter or darker than background
        fg_is_brighter = fg_mean > bg_mean

        # Use percentiles for robustness
        fg_p5, fg_p95 = np.percentile(fg, [5, 95])
        bg_p5, bg_p95 = np.percentile(bg, [5, 95])

        if fg_is_brighter:
            # Foreground is brighter - find threshold between them
            # Lower threshold should be above most of background
            # Upper threshold should capture most of foreground
            threshold_lower = max(bg_p95, fg_p5 - (fg_p95 - fg_p5) * 0.1)
            threshold_lower = min(threshold_lower, fg_p5)  # Don't exceed foreground range
            threshold_upper = fg_p95 + (fg_p95 - fg_p5) * 0.1
        else:
            # Foreground is darker
            threshold_lower = fg_p5 - (fg_p95 - fg_p5) * 0.1
            threshold_upper = min(bg_p5, fg_p95 + (fg_p95 - fg_p5) * 0.1)
            threshold_upper = max(threshold_upper, fg_p95)  # Don't go below foreground range

        return float(threshold_lower), float(threshold_upper)

    def analyze_shape(
        self, samples: WizardSamples, spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> ShapeAnalysisResult:
        """Analyze structure shape from sampled points.

        Args:
            samples: WizardSamples containing foreground points and/or boundary points.
            spacing_mm: Voxel spacing in mm (x, y, z).

        Returns:
            ShapeAnalysisResult with shape characteristics.
        """
        # Determine which points to use for shape analysis
        # Prefer boundary points if available, fall back to foreground points
        if samples.has_boundary() and len(samples.boundary_points) >= 10:
            shape_points = np.array(samples.boundary_points, dtype=np.float64)
            shape_points_mm = shape_points * np.array(spacing_mm)
            points_array = shape_points
        elif len(samples.foreground_points) >= 3:
            points_array = np.array(samples.foreground_points, dtype=np.float64)
            shape_points_mm = points_array * np.array(spacing_mm)
        else:
            # Not enough points for shape analysis - return defaults
            return ShapeAnalysisResult(
                estimated_diameter_mm=20.0,
                circularity=0.5,
                convexity=0.5,
                boundary_roughness=0.5,
                suggested_brush_radius_mm=10.0,
                is_3d_structure=False,
            )

        # Estimate diameter from point spread
        diameter_mm = self._estimate_diameter(shape_points_mm)

        # Calculate circularity from point distribution
        circularity = self._calculate_circularity(shape_points_mm)

        # Calculate convexity
        convexity = self._calculate_convexity(shape_points_mm)

        # Calculate boundary roughness if boundary points available
        if samples.has_boundary():
            boundary_roughness = self.estimate_boundary_roughness(samples.boundary_points)
        else:
            # Estimate from point distribution variance
            boundary_roughness = self._estimate_roughness_from_points(shape_points_mm)

        # Detect if 3D structure
        z_coords = points_array[:, 2]
        unique_slices = len(np.unique(z_coords))
        is_3d = unique_slices >= 3

        # Suggest brush radius (roughly diameter / 4, clamped to reasonable range)
        suggested_radius = np.clip(diameter_mm / 4.0, 3.0, 50.0)

        return ShapeAnalysisResult(
            estimated_diameter_mm=diameter_mm,
            circularity=circularity,
            convexity=convexity,
            boundary_roughness=boundary_roughness,
            suggested_brush_radius_mm=float(suggested_radius),
            is_3d_structure=is_3d,
        )

    def _estimate_diameter(self, points_mm: np.ndarray) -> float:
        """Estimate structure diameter from point spread.

        Uses the maximum extent across all dimensions.

        Args:
            points_mm: Points in mm coordinates.

        Returns:
            Estimated diameter in mm.
        """
        # Calculate extent in each dimension
        mins = np.min(points_mm, axis=0)
        maxs = np.max(points_mm, axis=0)
        extents = maxs - mins

        # Use the maximum of XY extents (most relevant for brush sizing)
        diameter = max(extents[0], extents[1])

        # Add some margin based on standard deviation
        std_xy = np.mean([np.std(points_mm[:, 0]), np.std(points_mm[:, 1])])
        diameter = max(diameter, 4 * std_xy)

        return float(max(diameter, 5.0))  # Minimum 5mm

    def _calculate_circularity(self, points_mm: np.ndarray) -> float:
        """Calculate circularity from point distribution.

        Compares standard deviations in X and Y directions.

        Args:
            points_mm: Points in mm coordinates.

        Returns:
            Circularity from 0 (elongated) to 1 (circular).
        """
        if len(points_mm) < 3:
            return 0.5

        # Calculate standard deviations in X and Y
        std_x = np.std(points_mm[:, 0])
        std_y = np.std(points_mm[:, 1])

        if max(std_x, std_y) < 1e-6:
            return 1.0  # All points at same location

        # Ratio of smaller to larger std
        circularity = min(std_x, std_y) / max(std_x, std_y)

        return float(circularity)

    def _calculate_convexity(self, points_mm: np.ndarray) -> float:
        """Estimate convexity from point distribution.

        Args:
            points_mm: Points in mm coordinates.

        Returns:
            Convexity estimate from 0 to 1.
        """
        if len(points_mm) < 4:
            return 0.5

        # Use a simple heuristic: check if points form a roughly convex shape
        # by comparing centroid distance distribution
        centroid = np.mean(points_mm[:, :2], axis=0)
        distances = np.linalg.norm(points_mm[:, :2] - centroid, axis=1)

        if np.max(distances) < 1e-6:
            return 1.0

        # Coefficient of variation of distances
        # Low CV = more convex (uniform distance from center)
        cv = np.std(distances) / np.mean(distances)

        # Convert to 0-1 scale (lower CV = higher convexity)
        convexity = 1.0 / (1.0 + cv)

        return float(np.clip(convexity, 0.0, 1.0))

    def _estimate_roughness_from_points(self, points_mm: np.ndarray) -> float:
        """Estimate boundary roughness from point distribution variance.

        Args:
            points_mm: Points in mm coordinates.

        Returns:
            Estimated roughness from 0 to 1.
        """
        if len(points_mm) < 5:
            return 0.5

        # Calculate variance of point distances from centroid
        centroid = np.mean(points_mm[:, :2], axis=0)
        distances = np.linalg.norm(points_mm[:, :2] - centroid, axis=1)

        if np.mean(distances) < 1e-6:
            return 0.5

        # Higher variance relative to mean = more roughness
        cv = np.std(distances) / np.mean(distances)

        # Scale to 0-1 range
        roughness = np.clip(cv, 0.0, 1.0)

        return float(roughness)

    def estimate_boundary_roughness(
        self, boundary_points: list[tuple[int | float, int | float, int | float]]
    ) -> float:
        """Estimate boundary roughness from traced points.

        Analyzes the deviation of boundary points from a smooth curve.
        For smooth curves (circles, ellipses), the curvature is consistent.
        For rough boundaries, curvature varies significantly.

        Args:
            boundary_points: List of (x, y, z) boundary coordinates.

        Returns:
            Roughness from 0 (smooth) to 1 (rough).
        """
        if len(boundary_points) < 3:
            return 0.5  # Default for insufficient data

        points = np.array(boundary_points, dtype=np.float64)

        # Focus on 2D (x, y) for boundary analysis
        xy_points = points[:, :2]

        if len(xy_points) < 5:
            return 0.5

        # Calculate local angle changes between consecutive points
        vectors = np.diff(xy_points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)

        # Filter out zero-length vectors
        valid_mask = lengths > 1e-6
        if np.sum(valid_mask) < 3:
            return 0.5

        valid_vectors = vectors[valid_mask]
        valid_lengths = lengths[valid_mask]

        # Normalize vectors
        unit_vectors = valid_vectors / valid_lengths[:, np.newaxis]

        # Calculate angles between consecutive vectors (turning angles)
        if len(unit_vectors) < 2:
            return 0.5

        dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)

        # For a smooth curve, angles should be consistent
        # Roughness is based on how much the turning angle varies
        angle_std = np.std(angles)
        median_angle: float = float(np.median(angles))

        # Use median-based metric (more robust to outliers)
        # For a perfect circle, all angles are equal, so std should be ~0
        # High std relative to the median indicates roughness

        if median_angle < 1e-6:
            # Nearly straight line - check for any large angle changes
            if np.max(angles) > 0.1:
                return 0.5  # Has some corners
            return 0.0  # Smooth straight line

        # Coefficient of variation using median
        # For smooth curves: cv should be small
        cv = angle_std / (median_angle + 1e-6)

        # Also consider absolute variation in segment lengths
        length_cv = np.std(valid_lengths) / (np.mean(valid_lengths) + 1e-6)

        # Combined roughness metric
        # Smooth curve: low angle cv, consistent segment lengths
        # Rough curve: high angle cv or inconsistent segments
        roughness = 0.7 * cv + 0.3 * length_cv

        # Scale to 0-1 with reasonable sensitivity
        # cv=0.5 -> roughness ~0.33
        # cv=1.0 -> roughness ~0.5
        # cv=2.0 -> roughness ~0.67
        roughness = roughness / (1.0 + roughness)

        return float(np.clip(roughness, 0.0, 1.0))
