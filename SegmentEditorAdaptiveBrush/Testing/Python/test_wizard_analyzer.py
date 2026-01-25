"""Tests for WizardAnalyzer - intensity and shape analysis for parameter wizard.

These tests verify that the wizard analyzer correctly analyzes sampled
foreground and background intensities and estimates shape characteristics.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
_THIS_DIR = Path(__file__).parent
_LIB_DIR = _THIS_DIR.parent.parent / "SegmentEditorAdaptiveBrushLib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Import will fail until we implement the module
try:
    from WizardAnalyzer import WizardAnalyzer
    from WizardDataStructures import (
        IntensityAnalysisResult,
        ShapeAnalysisResult,
        WizardSamples,
    )
except ImportError:
    WizardAnalyzer = None
    WizardSamples = None


def create_test_samples(
    fg_mean: float = 150.0,
    fg_std: float = 20.0,
    bg_mean: float = 50.0,
    bg_std: float = 15.0,
    n_fg: int = 500,
    n_bg: int = 500,
    seed: int = 42,
) -> "WizardSamples":
    """Create test samples with specified intensity distributions."""
    np.random.seed(seed)

    # Generate foreground samples
    fg_intensities = np.random.normal(fg_mean, fg_std, n_fg).astype(np.float32)
    # Create points in a rough cluster (for shape analysis)
    fg_points = [
        (50 + int(np.random.normal(0, 10)), 50 + int(np.random.normal(0, 10)), 5)
        for _ in range(n_fg)
    ]

    # Generate background samples
    bg_intensities = np.random.normal(bg_mean, bg_std, n_bg).astype(np.float32)
    bg_points = [
        (10 + int(np.random.normal(0, 5)), 10 + int(np.random.normal(0, 5)), 5) for _ in range(n_bg)
    ]

    return WizardSamples(
        foreground_points=fg_points,
        foreground_intensities=fg_intensities,
        background_points=bg_points,
        background_intensities=bg_intensities,
    )


def create_circular_boundary_points(
    center: tuple, radius: float, n_points: int = 100, use_float: bool = False
) -> list:
    """Create points along a circular boundary.

    Args:
        center: Center point (x, y, z).
        radius: Circle radius.
        n_points: Number of points to generate.
        use_float: If True, use floating point coordinates for accuracy.
                  If False, round to integers (more realistic but lossy).
    """
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        if not use_float:
            x, y = int(x), int(y)
        points.append((x, y, center[2]))
    return points


def create_irregular_boundary_points(
    center: tuple, base_radius: float, n_points: int = 100
) -> list:
    """Create points along an irregular boundary with noise."""
    np.random.seed(42)
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        # Add random variation to radius (irregular boundary)
        radius_variation = base_radius * (1 + 0.3 * np.random.randn())
        x = int(center[0] + radius_variation * np.cos(angle))
        y = int(center[1] + radius_variation * np.sin(angle))
        points.append((x, y, center[2]))
    return points


@unittest.skipIf(WizardAnalyzer is None, "WizardAnalyzer not importable")
class TestWizardAnalyzerIntensity(unittest.TestCase):
    """Tests for intensity analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WizardAnalyzer()

    def test_analyze_intensities_basic(self):
        """Should analyze foreground and background intensity distributions."""
        samples = create_test_samples(fg_mean=150.0, bg_mean=50.0)
        result = self.analyzer.analyze_intensities(samples)

        self.assertIsInstance(result, IntensityAnalysisResult)
        self.assertAlmostEqual(result.foreground_mean, 150.0, delta=5)
        self.assertAlmostEqual(result.background_mean, 50.0, delta=5)

    def test_analyze_intensities_high_separation(self):
        """Should detect high separation between distinct distributions."""
        # Very distinct distributions
        samples = create_test_samples(fg_mean=200.0, fg_std=10.0, bg_mean=50.0, bg_std=10.0)
        result = self.analyzer.analyze_intensities(samples)

        self.assertGreater(result.separation_score, 0.8)
        self.assertLess(result.overlap_percentage, 10.0)

    def test_analyze_intensities_low_separation(self):
        """Should detect low separation for overlapping distributions."""
        # Overlapping distributions
        samples = create_test_samples(fg_mean=100.0, fg_std=30.0, bg_mean=90.0, bg_std=30.0)
        result = self.analyzer.analyze_intensities(samples)

        self.assertLess(result.separation_score, 0.5)
        self.assertGreater(result.overlap_percentage, 30.0)

    def test_threshold_suggestion_for_separated(self):
        """Should suggest reasonable thresholds for separated distributions."""
        samples = create_test_samples(fg_mean=150.0, fg_std=10.0, bg_mean=50.0, bg_std=10.0)
        result = self.analyzer.analyze_intensities(samples)

        # Thresholds should be between background and foreground
        self.assertGreater(result.suggested_threshold_lower, result.background_mean)
        self.assertLess(result.suggested_threshold_upper, result.foreground_mean + 50)

        # Lower threshold should be less than upper
        self.assertLess(result.suggested_threshold_lower, result.suggested_threshold_upper)

    def test_min_max_values(self):
        """Should correctly compute min/max values."""
        samples = create_test_samples()
        result = self.analyzer.analyze_intensities(samples)

        self.assertLess(result.foreground_min, result.foreground_max)
        self.assertLess(result.background_min, result.background_max)
        self.assertLessEqual(result.foreground_min, result.foreground_mean)
        self.assertGreaterEqual(result.foreground_max, result.foreground_mean)

    def test_std_values(self):
        """Should correctly compute standard deviations."""
        samples = create_test_samples(fg_std=20.0, bg_std=15.0)
        result = self.analyzer.analyze_intensities(samples)

        self.assertAlmostEqual(result.foreground_std, 20.0, delta=5)
        self.assertAlmostEqual(result.background_std, 15.0, delta=5)

    def test_handles_small_sample_sizes(self):
        """Should handle small sample sizes gracefully."""
        samples = create_test_samples(n_fg=10, n_bg=10)
        result = self.analyzer.analyze_intensities(samples)

        # Should still produce valid results
        self.assertIsInstance(result, IntensityAnalysisResult)
        self.assertGreater(result.foreground_std, 0)

    def test_requires_both_foreground_and_background(self):
        """Should raise error if samples are missing."""
        # Only foreground
        samples_fg_only = WizardSamples(
            foreground_points=[(50, 50, 5)],
            foreground_intensities=np.array([100.0]),
        )

        with self.assertRaises(ValueError):
            self.analyzer.analyze_intensities(samples_fg_only)

        # Only background
        samples_bg_only = WizardSamples(
            background_points=[(50, 50, 5)],
            background_intensities=np.array([50.0]),
        )

        with self.assertRaises(ValueError):
            self.analyzer.analyze_intensities(samples_bg_only)


@unittest.skipIf(WizardAnalyzer is None, "WizardAnalyzer not importable")
class TestWizardAnalyzerSeparationScore(unittest.TestCase):
    """Tests for separation score calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WizardAnalyzer()

    def test_perfect_separation(self):
        """Should return high score for non-overlapping distributions."""
        fg = np.array([100.0, 101.0, 102.0, 99.0, 98.0])
        bg = np.array([10.0, 11.0, 12.0, 9.0, 8.0])

        score = self.analyzer._calculate_separation_score(fg, bg)
        self.assertGreater(score, 0.9)

    def test_complete_overlap(self):
        """Should return low score for identical distributions."""
        fg = np.array([100.0, 101.0, 102.0, 99.0, 98.0])
        bg = np.array([100.0, 101.0, 102.0, 99.0, 98.0])

        score = self.analyzer._calculate_separation_score(fg, bg)
        self.assertLess(score, 0.2)

    def test_partial_overlap(self):
        """Should return moderate score for partially overlapping distributions."""
        np.random.seed(42)
        fg = np.random.normal(100, 15, 100)
        bg = np.random.normal(70, 15, 100)

        score = self.analyzer._calculate_separation_score(fg, bg)
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.9)

    def test_score_in_valid_range(self):
        """Separation score should always be between 0 and 1."""
        for fg_mean, bg_mean in [(100, 50), (100, 100), (100, 200), (100, 95)]:
            np.random.seed(42)
            fg = np.random.normal(fg_mean, 20, 100)
            bg = np.random.normal(bg_mean, 20, 100)

            score = self.analyzer._calculate_separation_score(fg, bg)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


@unittest.skipIf(WizardAnalyzer is None, "WizardAnalyzer not importable")
class TestWizardAnalyzerShape(unittest.TestCase):
    """Tests for shape analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WizardAnalyzer()

    def test_analyze_shape_from_points(self):
        """Should analyze shape from sampled points."""
        # Create a roughly circular cluster of points
        np.random.seed(42)
        points = [
            (50 + int(10 * np.cos(a)), 50 + int(10 * np.sin(a)), 5)
            for a in np.linspace(0, 2 * np.pi, 50)
        ]

        samples = WizardSamples(
            foreground_points=points,
            foreground_intensities=np.ones(len(points)) * 100,
        )

        result = self.analyzer.analyze_shape(samples, spacing_mm=(1.0, 1.0, 2.0))

        self.assertIsInstance(result, ShapeAnalysisResult)
        self.assertGreater(result.estimated_diameter_mm, 0)
        self.assertGreater(result.suggested_brush_radius_mm, 0)

    def test_diameter_estimation(self):
        """Should estimate diameter reasonably from point spread."""
        # Create points with known spread using controlled range
        np.random.seed(42)
        # Use uniform distribution for predictable extent
        points = [
            (50 + int(np.random.uniform(-20, 20)), 50 + int(np.random.uniform(-20, 20)), 5)
            for _ in range(200)
        ]

        samples = WizardSamples(
            foreground_points=points,
            foreground_intensities=np.ones(len(points)) * 100,
        )

        result = self.analyzer.analyze_shape(samples, spacing_mm=(1.0, 1.0, 1.0))

        # Uniform distribution in [-20, 20] gives extent of ~40mm
        # With some margin, should be in reasonable range
        self.assertGreater(result.estimated_diameter_mm, 30)
        self.assertLess(result.estimated_diameter_mm, 60)

    def test_circularity_from_circular_boundary(self):
        """Should detect high circularity for circular boundary."""
        # Use floating point for accurate geometry (integer rounding causes issues)
        boundary = create_circular_boundary_points(
            (50, 50, 5), radius=20.0, n_points=100, use_float=True
        )

        samples = WizardSamples(
            foreground_points=[(50, 50, 5)],
            foreground_intensities=np.array([100.0]),
            boundary_points=boundary,
        )

        result = self.analyzer.analyze_shape(samples, spacing_mm=(1.0, 1.0, 1.0))

        self.assertGreater(result.circularity, 0.7)

    def test_brush_radius_suggestion(self):
        """Should suggest brush radius based on structure size."""
        # Small structure (20mm diameter)
        small_points = [
            (50 + dx, 50 + dy, 5) for dx in range(-10, 11, 5) for dy in range(-10, 11, 5)
        ]
        small_samples = WizardSamples(
            foreground_points=small_points,
            foreground_intensities=np.ones(len(small_points)) * 100,
        )
        small_result = self.analyzer.analyze_shape(small_samples, spacing_mm=(1.0, 1.0, 1.0))

        # Large structure (100mm diameter)
        large_points = [
            (50 + dx, 50 + dy, 5) for dx in range(-50, 51, 10) for dy in range(-50, 51, 10)
        ]
        large_samples = WizardSamples(
            foreground_points=large_points,
            foreground_intensities=np.ones(len(large_points)) * 100,
        )
        large_result = self.analyzer.analyze_shape(large_samples, spacing_mm=(1.0, 1.0, 1.0))

        # Larger structure should have larger suggested brush
        self.assertLess(
            small_result.suggested_brush_radius_mm, large_result.suggested_brush_radius_mm
        )

    def test_3d_structure_detection(self):
        """Should detect if structure spans multiple slices."""
        # 2D structure (single slice)
        points_2d = [(50 + dx, 50 + dy, 5) for dx in range(-10, 11, 5) for dy in range(-10, 11, 5)]
        samples_2d = WizardSamples(
            foreground_points=points_2d,
            foreground_intensities=np.ones(len(points_2d)) * 100,
        )
        result_2d = self.analyzer.analyze_shape(samples_2d, spacing_mm=(1.0, 1.0, 2.0))

        # 3D structure (multiple slices)
        points_3d = [
            (50 + dx, 50 + dy, z)
            for dx in range(-10, 11, 5)
            for dy in range(-10, 11, 5)
            for z in range(0, 10)
        ]
        samples_3d = WizardSamples(
            foreground_points=points_3d,
            foreground_intensities=np.ones(len(points_3d)) * 100,
        )
        result_3d = self.analyzer.analyze_shape(samples_3d, spacing_mm=(1.0, 1.0, 2.0))

        self.assertFalse(result_2d.is_3d_structure)
        self.assertTrue(result_3d.is_3d_structure)


@unittest.skipIf(WizardAnalyzer is None, "WizardAnalyzer not importable")
class TestWizardAnalyzerBoundaryRoughness(unittest.TestCase):
    """Tests for boundary roughness estimation."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WizardAnalyzer()

    def test_smooth_circular_boundary(self):
        """Should detect low roughness for smooth circular boundary."""
        # Use floating point for accurate geometry (integer rounding causes issues)
        boundary = create_circular_boundary_points(
            (50, 50, 5), radius=20.0, n_points=100, use_float=True
        )
        roughness = self.analyzer.estimate_boundary_roughness(boundary)

        self.assertLess(roughness, 0.3)

    def test_irregular_boundary(self):
        """Should detect high roughness for irregular boundary."""
        boundary = create_irregular_boundary_points((50, 50, 5), base_radius=20.0, n_points=100)
        roughness = self.analyzer.estimate_boundary_roughness(boundary)

        self.assertGreater(roughness, 0.2)

    def test_roughness_in_valid_range(self):
        """Roughness should always be between 0 and 1."""
        # Test various boundary types
        for n_points in [10, 50, 100]:
            circular = create_circular_boundary_points((50, 50, 5), radius=20.0, n_points=n_points)
            roughness = self.analyzer.estimate_boundary_roughness(circular)

            self.assertGreaterEqual(roughness, 0.0)
            self.assertLessEqual(roughness, 1.0)

    def test_handles_insufficient_points(self):
        """Should handle boundary with too few points."""
        # Only 2 points
        boundary = [(50, 50, 5), (60, 60, 5)]
        roughness = self.analyzer.estimate_boundary_roughness(boundary)

        # Should return a default value (moderate roughness)
        self.assertGreaterEqual(roughness, 0.0)
        self.assertLessEqual(roughness, 1.0)

    def test_handles_empty_boundary(self):
        """Should handle empty boundary gracefully."""
        roughness = self.analyzer.estimate_boundary_roughness([])

        # Should return default moderate roughness
        self.assertEqual(roughness, 0.5)


if __name__ == "__main__":
    unittest.main()
