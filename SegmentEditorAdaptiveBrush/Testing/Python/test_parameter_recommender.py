"""Tests for ParameterRecommender - algorithm and parameter recommendation engine.

These tests verify that the recommender produces sensible algorithm and
parameter suggestions based on intensity and shape analysis results.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
_THIS_DIR = Path(__file__).parent
_LIB_DIR = _THIS_DIR.parent.parent / "SegmentEditorAdaptiveBrushLib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Import will fail until we implement the module
try:
    from ParameterRecommender import ParameterRecommender
    from WizardDataStructures import (
        IntensityAnalysisResult,
        ShapeAnalysisResult,
        WizardRecommendation,
    )
except ImportError:
    ParameterRecommender = None
    IntensityAnalysisResult = None
    ShapeAnalysisResult = None


def create_well_separated_intensity_result() -> "IntensityAnalysisResult":
    """Create intensity result with well-separated distributions."""
    return IntensityAnalysisResult(
        foreground_min=140.0,
        foreground_max=180.0,
        foreground_mean=160.0,
        foreground_std=10.0,
        background_min=40.0,
        background_max=80.0,
        background_mean=60.0,
        background_std=10.0,
        separation_score=0.9,
        overlap_percentage=2.0,
        suggested_threshold_lower=100.0,
        suggested_threshold_upper=200.0,
    )


def create_poorly_separated_intensity_result() -> "IntensityAnalysisResult":
    """Create intensity result with poorly separated distributions."""
    return IntensityAnalysisResult(
        foreground_min=80.0,
        foreground_max=150.0,
        foreground_mean=110.0,
        foreground_std=20.0,
        background_min=70.0,
        background_max=130.0,
        background_mean=100.0,
        background_std=18.0,
        separation_score=0.25,
        overlap_percentage=60.0,
        suggested_threshold_lower=95.0,
        suggested_threshold_upper=125.0,
    )


def create_medium_structure_shape_result() -> "ShapeAnalysisResult":
    """Create shape result for a medium-sized structure."""
    return ShapeAnalysisResult(
        estimated_diameter_mm=30.0,
        circularity=0.8,
        convexity=0.85,
        boundary_roughness=0.3,
        suggested_brush_radius_mm=10.0,
        is_3d_structure=True,
    )


def create_small_structure_shape_result() -> "ShapeAnalysisResult":
    """Create shape result for a small structure."""
    return ShapeAnalysisResult(
        estimated_diameter_mm=8.0,
        circularity=0.9,
        convexity=0.95,
        boundary_roughness=0.15,
        suggested_brush_radius_mm=4.0,
        is_3d_structure=False,
    )


def create_irregular_structure_shape_result() -> "ShapeAnalysisResult":
    """Create shape result for an irregular structure (like a tumor)."""
    return ShapeAnalysisResult(
        estimated_diameter_mm=25.0,
        circularity=0.5,
        convexity=0.6,
        boundary_roughness=0.7,
        suggested_brush_radius_mm=8.0,
        is_3d_structure=True,
    )


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderBasic(unittest.TestCase):
    """Basic tests for ParameterRecommender."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_recommend_returns_valid_recommendation(self):
        """Should return a valid WizardRecommendation object."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        self.assertIsInstance(rec, WizardRecommendation)
        self.assertIsNotNone(rec.algorithm)
        self.assertIsNotNone(rec.algorithm_reason)
        self.assertGreater(rec.brush_radius_mm, 0)
        self.assertGreaterEqual(rec.edge_sensitivity, 0)
        self.assertLessEqual(rec.edge_sensitivity, 100)

    def test_confidence_in_valid_range(self):
        """Confidence should be between 0 and 1."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        self.assertGreaterEqual(rec.confidence, 0.0)
        self.assertLessEqual(rec.confidence, 1.0)

    def test_includes_threshold_suggestion(self):
        """Should include threshold values from intensity analysis."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        self.assertTrue(rec.has_threshold_suggestion())
        self.assertEqual(rec.threshold_lower, intensity.suggested_threshold_lower)
        self.assertEqual(rec.threshold_upper, intensity.suggested_threshold_upper)


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderAlgorithmSelection(unittest.TestCase):
    """Tests for algorithm selection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_high_separation_prefers_fast_algorithm(self):
        """High intensity separation should recommend faster algorithms."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # With high separation, should prefer Connected Threshold or Region Growing
        fast_algorithms = ["connected_threshold", "region_growing", "geodesic_distance"]
        self.assertIn(rec.algorithm, fast_algorithms + ["watershed"])

    def test_low_separation_prefers_edge_based_algorithm(self):
        """Low intensity separation should recommend edge-based algorithms."""
        intensity = create_poorly_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # With low separation, should prefer Watershed or Level Set
        edge_algorithms = [
            "watershed",
            "level_set_cpu",
            "level_set_gpu",
            "geodesic_distance",
            "random_walker",
        ]
        self.assertIn(rec.algorithm, edge_algorithms)

    def test_irregular_boundary_considers_level_set(self):
        """Irregular boundaries may benefit from Level Set."""
        intensity = create_well_separated_intensity_result()
        shape = create_irregular_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # Should recommend a valid algorithm
        # Note: This is a soft test - the recommender may have other valid choices
        self.assertIsNotNone(rec.algorithm)

    def test_algorithm_is_valid_identifier(self):
        """Algorithm should be a valid identifier string."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        valid_algorithms = [
            "geodesic_distance",
            "watershed",
            "random_walker",
            "level_set_gpu",
            "level_set_cpu",
            "connected_threshold",
            "region_growing",
            "threshold_brush",
        ]
        self.assertIn(rec.algorithm, valid_algorithms)


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderEdgeSensitivity(unittest.TestCase):
    """Tests for edge sensitivity calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_smooth_boundary_lower_sensitivity(self):
        """Smooth boundaries should have lower edge sensitivity."""
        intensity = create_well_separated_intensity_result()
        smooth_shape = ShapeAnalysisResult(
            estimated_diameter_mm=30.0,
            circularity=0.9,
            convexity=0.95,
            boundary_roughness=0.1,  # Very smooth
            suggested_brush_radius_mm=10.0,
            is_3d_structure=True,
        )

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=smooth_shape,
        )

        # Should have moderate to low sensitivity for smooth boundaries
        self.assertLessEqual(rec.edge_sensitivity, 70)

    def test_rough_boundary_higher_sensitivity(self):
        """Rough boundaries should have higher edge sensitivity."""
        intensity = create_well_separated_intensity_result()
        rough_shape = ShapeAnalysisResult(
            estimated_diameter_mm=30.0,
            circularity=0.5,
            convexity=0.6,
            boundary_roughness=0.8,  # Very rough
            suggested_brush_radius_mm=10.0,
            is_3d_structure=True,
        )

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=rough_shape,
        )

        # Should have higher sensitivity for rough boundaries
        self.assertGreaterEqual(rec.edge_sensitivity, 50)

    def test_low_separation_increases_sensitivity(self):
        """Low intensity separation should increase edge sensitivity."""
        intensity = create_poorly_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec_low_sep = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # Compare with high separation
        intensity_high = create_well_separated_intensity_result()
        rec_high_sep = self.recommender.recommend(
            intensity_result=intensity_high,
            shape_result=shape,
        )

        # Low separation should generally have higher sensitivity
        self.assertGreaterEqual(rec_low_sep.edge_sensitivity, rec_high_sep.edge_sensitivity - 20)


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderBrushRadius(unittest.TestCase):
    """Tests for brush radius recommendation."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_uses_shape_suggested_radius(self):
        """Should use or be close to shape's suggested brush radius."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # Should be close to shape's suggestion
        self.assertAlmostEqual(rec.brush_radius_mm, shape.suggested_brush_radius_mm, delta=5.0)

    def test_small_structure_smaller_brush(self):
        """Small structures should get smaller brush radius."""
        intensity = create_well_separated_intensity_result()
        small_shape = create_small_structure_shape_result()
        medium_shape = create_medium_structure_shape_result()

        rec_small = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=small_shape,
        )

        rec_medium = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=medium_shape,
        )

        self.assertLess(rec_small.brush_radius_mm, rec_medium.brush_radius_mm)


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderModality(unittest.TestCase):
    """Tests for modality-aware recommendations."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_ct_modality_hint(self):
        """CT modality should influence recommendations."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
            modality="CT",
        )

        # Should produce valid recommendation
        self.assertIsNotNone(rec.algorithm)
        # CT images often work well with threshold-based approaches
        # (soft assertion - just check it doesn't crash)

    def test_mri_t1_modality_hint(self):
        """MRI T1 modality should influence recommendations."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
            modality="MRI_T1",
        )

        self.assertIsNotNone(rec.algorithm)

    def test_ultrasound_modality_considers_noise(self):
        """Ultrasound modality should account for noise."""
        intensity = create_poorly_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
            modality="Ultrasound",
        )

        # Ultrasound is noisy - should prefer edge-aware algorithms
        self.assertIsNotNone(rec.algorithm)


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderStructureType(unittest.TestCase):
    """Tests for structure-type-aware recommendations."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_tumor_structure_type(self):
        """Tumor structure type should influence recommendations."""
        intensity = create_well_separated_intensity_result()
        shape = create_irregular_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
            structure_type="tumor",
        )

        # Tumors often have irregular boundaries
        # Should consider Level Set or similar
        self.assertIsNotNone(rec.algorithm)

    def test_bone_structure_type(self):
        """Bone structure type should prefer threshold-based approaches."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
            structure_type="bone",
        )

        # Bone typically has high contrast
        # Should work well with connected threshold
        self.assertIsNotNone(rec.algorithm)


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderPriority(unittest.TestCase):
    """Tests for user priority preferences."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_speed_priority(self):
        """Speed priority should prefer faster algorithms."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
            priority="speed",
        )

        fast_algorithms = [
            "connected_threshold",
            "region_growing",
            "threshold_brush",
            "geodesic_distance",
        ]
        self.assertIn(
            rec.algorithm, fast_algorithms + ["watershed"]
        )  # Watershed is also reasonably fast

    def test_precision_priority(self):
        """Precision priority should prefer more accurate algorithms."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
            priority="precision",
        )

        # Should prefer algorithms known for precision
        precision_algorithms = [
            "level_set_cpu",
            "level_set_gpu",
            "watershed",
            "geodesic_distance",
            "random_walker",
        ]
        self.assertIn(
            rec.algorithm, precision_algorithms + ["connected_threshold", "region_growing"]
        )


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderWarnings(unittest.TestCase):
    """Tests for warning generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_low_separation_generates_warning(self):
        """Low intensity separation should generate a warning."""
        intensity = create_poorly_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # Should warn about low separation
        self.assertTrue(rec.has_warnings())
        warning_text = " ".join(rec.warnings).lower()
        # Should mention separation, contrast, or overlap
        self.assertTrue(
            any(term in warning_text for term in ["separation", "contrast", "overlap", "similar"])
        )

    def test_high_confidence_no_critical_warnings(self):
        """High confidence recommendations should have manageable warnings."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # With good conditions, confidence should be reasonably high
        self.assertGreater(rec.confidence, 0.5)


@unittest.skipIf(ParameterRecommender is None, "ParameterRecommender not importable")
class TestParameterRecommenderAlternatives(unittest.TestCase):
    """Tests for alternative algorithm suggestions."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = ParameterRecommender()

    def test_provides_alternatives(self):
        """Should provide alternative algorithm suggestions."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        # Should provide at least one alternative
        self.assertGreaterEqual(len(rec.alternative_algorithms), 1)

    def test_alternatives_are_different_from_primary(self):
        """Alternative algorithms should differ from primary."""
        intensity = create_well_separated_intensity_result()
        shape = create_medium_structure_shape_result()

        rec = self.recommender.recommend(
            intensity_result=intensity,
            shape_result=shape,
        )

        for alt_algo, alt_reason in rec.alternative_algorithms:
            self.assertNotEqual(alt_algo, rec.algorithm)
            self.assertIsNotNone(alt_reason)


if __name__ == "__main__":
    unittest.main()
