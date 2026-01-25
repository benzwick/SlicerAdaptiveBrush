"""Tests for wizard data structures.

These tests verify the dataclasses used by the parameter wizard
for collecting samples and representing analysis results.
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
    from WizardDataStructures import (
        IntensityAnalysisResult,
        ShapeAnalysisResult,
        WizardRecommendation,
        WizardSamples,
    )
except ImportError:
    WizardSamples = None
    IntensityAnalysisResult = None
    ShapeAnalysisResult = None
    WizardRecommendation = None


@unittest.skipIf(WizardSamples is None, "WizardDataStructures not importable")
class TestWizardSamples(unittest.TestCase):
    """Tests for WizardSamples dataclass."""

    def test_create_empty_samples(self):
        """Should create WizardSamples with empty lists."""
        samples = WizardSamples()
        self.assertEqual(samples.foreground_points, [])
        self.assertEqual(samples.background_points, [])
        self.assertEqual(samples.boundary_points, [])
        self.assertIsNone(samples.foreground_intensities)
        self.assertIsNone(samples.background_intensities)
        self.assertIsNone(samples.volume_node)

    def test_create_with_foreground_data(self):
        """Should create WizardSamples with foreground data."""
        points = [(10, 20, 5), (11, 21, 5), (12, 22, 5)]
        intensities = np.array([100.0, 105.0, 98.0])

        samples = WizardSamples(
            foreground_points=points,
            foreground_intensities=intensities,
        )

        self.assertEqual(samples.foreground_points, points)
        np.testing.assert_array_equal(samples.foreground_intensities, intensities)

    def test_create_with_all_data(self):
        """Should create WizardSamples with all sample types."""
        fg_points = [(10, 20, 5), (11, 21, 5)]
        fg_intensities = np.array([100.0, 105.0])
        bg_points = [(50, 60, 5), (51, 61, 5)]
        bg_intensities = np.array([50.0, 55.0])
        boundary_points = [(30, 40, 5), (31, 41, 5)]

        samples = WizardSamples(
            foreground_points=fg_points,
            foreground_intensities=fg_intensities,
            background_points=bg_points,
            background_intensities=bg_intensities,
            boundary_points=boundary_points,
        )

        self.assertEqual(len(samples.foreground_points), 2)
        self.assertEqual(len(samples.background_points), 2)
        self.assertEqual(len(samples.boundary_points), 2)

    def test_has_foreground(self):
        """Should check if foreground samples exist."""
        empty = WizardSamples()
        self.assertFalse(empty.has_foreground())

        with_fg = WizardSamples(
            foreground_points=[(10, 20, 5)],
            foreground_intensities=np.array([100.0]),
        )
        self.assertTrue(with_fg.has_foreground())

    def test_has_background(self):
        """Should check if background samples exist."""
        empty = WizardSamples()
        self.assertFalse(empty.has_background())

        with_bg = WizardSamples(
            background_points=[(50, 60, 5)],
            background_intensities=np.array([50.0]),
        )
        self.assertTrue(with_bg.has_background())

    def test_has_boundary(self):
        """Should check if boundary samples exist."""
        empty = WizardSamples()
        self.assertFalse(empty.has_boundary())

        with_boundary = WizardSamples(boundary_points=[(30, 40, 5)])
        self.assertTrue(with_boundary.has_boundary())

    def test_foreground_count(self):
        """Should return count of foreground samples."""
        samples = WizardSamples(
            foreground_points=[(10, 20, 5), (11, 21, 5), (12, 22, 5)],
            foreground_intensities=np.array([100.0, 105.0, 98.0]),
        )
        self.assertEqual(samples.foreground_count, 3)

    def test_background_count(self):
        """Should return count of background samples."""
        samples = WizardSamples(
            background_points=[(50, 60, 5)],
            background_intensities=np.array([50.0]),
        )
        self.assertEqual(samples.background_count, 1)

    def test_clear_foreground(self):
        """Should clear foreground samples."""
        samples = WizardSamples(
            foreground_points=[(10, 20, 5)],
            foreground_intensities=np.array([100.0]),
        )
        samples.clear_foreground()
        self.assertEqual(samples.foreground_points, [])
        self.assertIsNone(samples.foreground_intensities)

    def test_clear_background(self):
        """Should clear background samples."""
        samples = WizardSamples(
            background_points=[(50, 60, 5)],
            background_intensities=np.array([50.0]),
        )
        samples.clear_background()
        self.assertEqual(samples.background_points, [])
        self.assertIsNone(samples.background_intensities)

    def test_clear_boundary(self):
        """Should clear boundary samples."""
        samples = WizardSamples(boundary_points=[(30, 40, 5)])
        samples.clear_boundary()
        self.assertEqual(samples.boundary_points, [])

    def test_clear_all(self):
        """Should clear all samples."""
        samples = WizardSamples(
            foreground_points=[(10, 20, 5)],
            foreground_intensities=np.array([100.0]),
            background_points=[(50, 60, 5)],
            background_intensities=np.array([50.0]),
            boundary_points=[(30, 40, 5)],
        )
        samples.clear_all()
        self.assertFalse(samples.has_foreground())
        self.assertFalse(samples.has_background())
        self.assertFalse(samples.has_boundary())


@unittest.skipIf(IntensityAnalysisResult is None, "WizardDataStructures not importable")
class TestIntensityAnalysisResult(unittest.TestCase):
    """Tests for IntensityAnalysisResult dataclass."""

    def test_create_result(self):
        """Should create IntensityAnalysisResult with all fields."""
        result = IntensityAnalysisResult(
            foreground_min=80.0,
            foreground_max=120.0,
            foreground_mean=100.0,
            foreground_std=10.0,
            background_min=30.0,
            background_max=70.0,
            background_mean=50.0,
            background_std=12.0,
            separation_score=0.85,
            overlap_percentage=5.0,
            suggested_threshold_lower=75.0,
            suggested_threshold_upper=125.0,
        )

        self.assertEqual(result.foreground_mean, 100.0)
        self.assertEqual(result.background_mean, 50.0)
        self.assertEqual(result.separation_score, 0.85)

    def test_threshold_range(self):
        """Should compute threshold range."""
        result = IntensityAnalysisResult(
            foreground_min=80.0,
            foreground_max=120.0,
            foreground_mean=100.0,
            foreground_std=10.0,
            background_min=30.0,
            background_max=70.0,
            background_mean=50.0,
            background_std=12.0,
            separation_score=0.85,
            overlap_percentage=5.0,
            suggested_threshold_lower=75.0,
            suggested_threshold_upper=125.0,
        )

        self.assertEqual(result.threshold_range, 50.0)

    def test_is_well_separated(self):
        """Should determine if intensities are well-separated."""
        well_separated = IntensityAnalysisResult(
            foreground_min=80.0,
            foreground_max=120.0,
            foreground_mean=100.0,
            foreground_std=10.0,
            background_min=30.0,
            background_max=70.0,
            background_mean=50.0,
            background_std=12.0,
            separation_score=0.85,
            overlap_percentage=5.0,
            suggested_threshold_lower=75.0,
            suggested_threshold_upper=125.0,
        )
        self.assertTrue(well_separated.is_well_separated())

        poorly_separated = IntensityAnalysisResult(
            foreground_min=80.0,
            foreground_max=150.0,
            foreground_mean=100.0,
            foreground_std=25.0,
            background_min=70.0,
            background_max=130.0,
            background_mean=90.0,
            background_std=20.0,
            separation_score=0.3,
            overlap_percentage=45.0,
            suggested_threshold_lower=85.0,
            suggested_threshold_upper=115.0,
        )
        self.assertFalse(poorly_separated.is_well_separated())


@unittest.skipIf(ShapeAnalysisResult is None, "WizardDataStructures not importable")
class TestShapeAnalysisResult(unittest.TestCase):
    """Tests for ShapeAnalysisResult dataclass."""

    def test_create_result(self):
        """Should create ShapeAnalysisResult with all fields."""
        result = ShapeAnalysisResult(
            estimated_diameter_mm=25.0,
            circularity=0.8,
            convexity=0.9,
            boundary_roughness=0.3,
            suggested_brush_radius_mm=12.0,
            is_3d_structure=True,
        )

        self.assertEqual(result.estimated_diameter_mm, 25.0)
        self.assertEqual(result.circularity, 0.8)
        self.assertTrue(result.is_3d_structure)

    def test_is_small_structure(self):
        """Should classify small structures (<10mm)."""
        small = ShapeAnalysisResult(
            estimated_diameter_mm=8.0,
            circularity=0.9,
            convexity=0.95,
            boundary_roughness=0.1,
            suggested_brush_radius_mm=4.0,
            is_3d_structure=False,
        )
        self.assertTrue(small.is_small_structure())

        large = ShapeAnalysisResult(
            estimated_diameter_mm=50.0,
            circularity=0.7,
            convexity=0.8,
            boundary_roughness=0.4,
            suggested_brush_radius_mm=25.0,
            is_3d_structure=True,
        )
        self.assertFalse(large.is_small_structure())

    def test_is_large_structure(self):
        """Should classify large structures (>50mm)."""
        large = ShapeAnalysisResult(
            estimated_diameter_mm=75.0,
            circularity=0.6,
            convexity=0.7,
            boundary_roughness=0.5,
            suggested_brush_radius_mm=30.0,
            is_3d_structure=True,
        )
        self.assertTrue(large.is_large_structure())

    def test_has_smooth_boundary(self):
        """Should classify smooth boundaries."""
        smooth = ShapeAnalysisResult(
            estimated_diameter_mm=25.0,
            circularity=0.9,
            convexity=0.95,
            boundary_roughness=0.15,
            suggested_brush_radius_mm=12.0,
            is_3d_structure=False,
        )
        self.assertTrue(smooth.has_smooth_boundary())

        rough = ShapeAnalysisResult(
            estimated_diameter_mm=25.0,
            circularity=0.6,
            convexity=0.7,
            boundary_roughness=0.6,
            suggested_brush_radius_mm=12.0,
            is_3d_structure=True,
        )
        self.assertFalse(rough.has_smooth_boundary())


@unittest.skipIf(WizardRecommendation is None, "WizardDataStructures not importable")
class TestWizardRecommendation(unittest.TestCase):
    """Tests for WizardRecommendation dataclass."""

    def test_create_recommendation(self):
        """Should create WizardRecommendation with all fields."""
        rec = WizardRecommendation(
            algorithm="watershed",
            algorithm_reason="Good balance of speed and precision",
            brush_radius_mm=12.0,
            radius_reason="Based on estimated structure diameter of 25mm",
            edge_sensitivity=65,
            sensitivity_reason="Moderate boundary roughness requires higher sensitivity",
            threshold_lower=75.0,
            threshold_upper=125.0,
            threshold_reason="Based on foreground/background intensity analysis",
            confidence=0.85,
            warnings=["High noise detected"],
            alternative_algorithms=[
                ("level_set_cpu", "Higher precision but slower"),
                ("connected_threshold", "Faster but less precise"),
            ],
        )

        self.assertEqual(rec.algorithm, "watershed")
        self.assertEqual(rec.brush_radius_mm, 12.0)
        self.assertEqual(rec.edge_sensitivity, 65)
        self.assertEqual(rec.confidence, 0.85)
        self.assertEqual(len(rec.warnings), 1)
        self.assertEqual(len(rec.alternative_algorithms), 2)

    def test_create_minimal_recommendation(self):
        """Should create recommendation with only required fields."""
        rec = WizardRecommendation(
            algorithm="connected_threshold",
            algorithm_reason="High intensity separation",
            brush_radius_mm=10.0,
            radius_reason="Default size",
            edge_sensitivity=50,
            sensitivity_reason="Default sensitivity",
        )

        self.assertEqual(rec.algorithm, "connected_threshold")
        self.assertIsNone(rec.threshold_lower)
        self.assertIsNone(rec.threshold_upper)
        self.assertIsNone(rec.threshold_reason)
        self.assertEqual(rec.confidence, 0.5)  # Default
        self.assertEqual(rec.warnings, [])
        self.assertEqual(rec.alternative_algorithms, [])

    def test_is_high_confidence(self):
        """Should identify high confidence recommendations."""
        high = WizardRecommendation(
            algorithm="watershed",
            algorithm_reason="test",
            brush_radius_mm=10.0,
            radius_reason="test",
            edge_sensitivity=50,
            sensitivity_reason="test",
            confidence=0.85,
        )
        self.assertTrue(high.is_high_confidence())

        low = WizardRecommendation(
            algorithm="watershed",
            algorithm_reason="test",
            brush_radius_mm=10.0,
            radius_reason="test",
            edge_sensitivity=50,
            sensitivity_reason="test",
            confidence=0.5,
        )
        self.assertFalse(low.is_high_confidence())

    def test_has_warnings(self):
        """Should check if recommendation has warnings."""
        with_warnings = WizardRecommendation(
            algorithm="watershed",
            algorithm_reason="test",
            brush_radius_mm=10.0,
            radius_reason="test",
            edge_sensitivity=50,
            sensitivity_reason="test",
            warnings=["Warning 1", "Warning 2"],
        )
        self.assertTrue(with_warnings.has_warnings())

        without_warnings = WizardRecommendation(
            algorithm="watershed",
            algorithm_reason="test",
            brush_radius_mm=10.0,
            radius_reason="test",
            edge_sensitivity=50,
            sensitivity_reason="test",
        )
        self.assertFalse(without_warnings.has_warnings())

    def test_has_threshold_suggestion(self):
        """Should check if threshold values are suggested."""
        with_threshold = WizardRecommendation(
            algorithm="threshold_brush",
            algorithm_reason="test",
            brush_radius_mm=10.0,
            radius_reason="test",
            edge_sensitivity=50,
            sensitivity_reason="test",
            threshold_lower=75.0,
            threshold_upper=125.0,
            threshold_reason="Based on analysis",
        )
        self.assertTrue(with_threshold.has_threshold_suggestion())

        without_threshold = WizardRecommendation(
            algorithm="watershed",
            algorithm_reason="test",
            brush_radius_mm=10.0,
            radius_reason="test",
            edge_sensitivity=50,
            sensitivity_reason="test",
        )
        self.assertFalse(without_threshold.has_threshold_suggestion())


if __name__ == "__main__":
    unittest.main()
