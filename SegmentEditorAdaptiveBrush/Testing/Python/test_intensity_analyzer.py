"""Tests for IntensityAnalyzer - GMM-based threshold estimation.

These tests verify that the intensity analyzer correctly identifies
intensity distributions and computes appropriate thresholds for
adaptive brush segmentation.
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

from test_fixtures.synthetic_image import (  # noqa: E402
    create_bimodal_image,
    create_noisy_sphere,
    create_uniform_image,
)

# Import markers from conftest
try:
    from conftest import requires_sklearn
except ImportError:
    import pytest

    requires_sklearn = pytest.mark.skip(reason="conftest not found")

# Import the actual IntensityAnalyzer
try:
    from IntensityAnalyzer import HAS_SKLEARN, IntensityAnalyzer
except ImportError:
    IntensityAnalyzer = None
    HAS_SKLEARN = False


@unittest.skipIf(IntensityAnalyzer is None, "IntensityAnalyzer not importable")
class TestIntensityAnalyzer(unittest.TestCase):
    """Tests for IntensityAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.analyzer = IntensityAnalyzer(use_gmm=False)  # Use simple stats for speed
        if HAS_SKLEARN:
            self.analyzer_gmm = IntensityAnalyzer(use_gmm=True)

    def test_uniform_image_single_component(self):
        """Uniform image should result in narrow threshold range."""
        image = create_uniform_image(size=(50, 50, 5), intensity=100.0, noise_std=5.0)
        seed_point = (25, 25, 2)  # (x, y, z)

        result = self.analyzer.analyze(image, seed_point, radius_voxels=(20, 20, 2))

        # For uniform image, thresholds should be around the mean ± some range
        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertIn("mean", result)
        self.assertIn("std", result)

        # Mean should be close to 100
        self.assertAlmostEqual(result["mean"], 100, delta=20)

        # Range should include the seed value
        self.assertLess(result["lower"], 100)
        self.assertGreater(result["upper"], 100)

    def test_bimodal_image_detects_two_components(self):
        """Bimodal image with GMM should detect two intensity components."""
        if not HAS_SKLEARN:
            self.skipTest("sklearn not available for GMM")

        image, mask = create_bimodal_image(
            size=(100, 100, 5), mean1=100.0, mean2=200.0, std1=10.0, std2=10.0
        )
        seed_point = (25, 50, 2)  # In region 1 (x < 50)

        result = self.analyzer_gmm.analyze(image, seed_point, radius_voxels=(30, 30, 2))

        # GMM should detect 2+ components
        self.assertGreaterEqual(result.get("n_components", 1), 2)

    def test_seed_in_low_region_gets_low_thresholds(self):
        """Seed in low-intensity region should get thresholds around that region."""
        image, mask = create_bimodal_image(
            size=(100, 100, 5), mean1=100.0, mean2=200.0, std1=10.0, std2=10.0
        )
        seed_point = (25, 50, 2)  # In low region (x < 50)

        result = self.analyzer.analyze(image, seed_point, radius_voxels=(20, 20, 2))

        # Thresholds should be centered around 100 (low region)
        # Upper bound should be less than 150 (below the transition to high region)
        self.assertGreater(result["lower"], 50)  # Above noise floor
        self.assertLess(result["upper"], 175)  # Below high region mean

    def test_seed_in_high_region_gets_high_thresholds(self):
        """Seed in high-intensity region should get thresholds around that region."""
        image, mask = create_bimodal_image(
            size=(100, 100, 5), mean1=100.0, mean2=200.0, std1=10.0, std2=10.0
        )
        seed_point = (75, 50, 2)  # In high region (x >= 50)

        result = self.analyzer.analyze(image, seed_point, radius_voxels=(20, 20, 2))

        # Thresholds should be centered around 200 (high region)
        self.assertGreater(result["lower"], 125)  # Above low region mean
        self.assertLess(result["upper"], 250)  # Reasonable upper bound

    def test_fallback_without_sklearn(self):
        """Should work with simple statistics when sklearn unavailable."""
        image = create_uniform_image(size=(50, 50, 5), intensity=100.0, noise_std=5.0)
        seed_point = (25, 25, 2)

        analyzer = IntensityAnalyzer(use_gmm=False)
        result = analyzer.analyze(image, seed_point)

        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertIn("mean", result)
        self.assertIn("std", result)
        self.assertEqual(result["n_components"], 1)

    def test_roi_extraction(self):
        """Should only analyze ROI around seed, not full image."""
        # Large bimodal image but small ROI
        image, mask = create_bimodal_image(
            size=(200, 200, 20), mean1=100.0, mean2=200.0, std1=5.0, std2=5.0
        )
        seed_point = (25, 100, 10)  # In low region (x < 100)
        radius = 15  # Small ROI, should only see low region

        result = self.analyzer.analyze(image, seed_point, radius_voxels=(radius, radius, 5))

        # With small ROI entirely in low region, upper should be low
        self.assertLess(result["upper"], 150)

    def test_handles_edge_of_image(self):
        """Should handle seed points near image boundaries."""
        image = create_uniform_image(size=(50, 50, 5), intensity=100.0, noise_std=5.0)
        seed_point = (2, 2, 0)  # Near corner

        # Should not crash
        result = self.analyzer.analyze(image, seed_point, radius_voxels=(10, 10, 2))
        self.assertIn("mean", result)

    def test_edge_sensitivity_affects_thresholds(self):
        """Edge sensitivity should affect threshold width."""
        image = create_uniform_image(size=(50, 50, 5), intensity=100.0, noise_std=10.0)
        seed_point = (25, 25, 2)

        # Permissive (wide thresholds)
        result_permissive = self.analyzer.analyze(
            image, seed_point, radius_voxels=(20, 20, 2), edge_sensitivity=0.0
        )

        # Strict (narrow thresholds)
        result_strict = self.analyzer.analyze(
            image, seed_point, radius_voxels=(20, 20, 2), edge_sensitivity=1.0
        )

        range_permissive = result_permissive["upper"] - result_permissive["lower"]
        range_strict = result_strict["upper"] - result_strict["lower"]

        # Permissive should have wider range than strict
        self.assertGreater(range_permissive, range_strict)


@unittest.skipIf(IntensityAnalyzer is None, "IntensityAnalyzer not importable")
class TestIntensityAnalyzerEdgeCases(unittest.TestCase):
    """Edge case tests for IntensityAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.analyzer = IntensityAnalyzer(use_gmm=False)

    def test_constant_intensity_region(self):
        """Should handle region with zero variance."""
        image = create_uniform_image(size=(50, 50, 5), intensity=100.0, noise_std=0.0)
        seed_point = (25, 25, 2)

        result = self.analyzer.analyze(image, seed_point)

        # Constant region should have zero std
        self.assertEqual(result["std"], 0.0)
        # Thresholds should still be valid
        self.assertLessEqual(result["lower"], 100)
        self.assertGreaterEqual(result["upper"], 100)

    def test_very_noisy_image(self):
        """Should handle images with high noise levels."""
        image, mask = create_bimodal_image(
            size=(100, 100, 5),
            mean1=100.0,
            mean2=200.0,
            std1=50.0,  # Very high noise
            std2=50.0,
        )
        seed_point = (25, 50, 2)

        result = self.analyzer.analyze(image, seed_point)

        # Thresholds should be wider due to noise
        threshold_range = result["upper"] - result["lower"]
        self.assertGreater(threshold_range, 50)  # Wide range due to noise

    def test_small_roi(self):
        """Should handle very small ROI."""
        image = create_uniform_image(size=(100, 100, 10), intensity=100.0, noise_std=5.0)
        seed_point = (50, 50, 5)

        # Very small ROI
        result = self.analyzer.analyze(image, seed_point, radius_voxels=(2, 2, 1))

        self.assertIn("mean", result)
        self.assertIn("lower", result)
        self.assertIn("upper", result)

    def test_3d_sphere_roi(self):
        """Should work with 3D sphere data."""
        image, ground_truth = create_noisy_sphere(
            size=(50, 50, 50),
            radius=15.0,
            inside_mean=200.0,
            outside_mean=50.0,
            inside_std=10.0,
            outside_std=10.0,
        )
        seed_point = (25, 25, 25)  # Center of sphere

        result = self.analyzer.analyze(image, seed_point, radius_voxels=(10, 10, 10))

        # Should identify the bright center region
        self.assertGreater(result["mean"], 150)

    def test_seed_intensity_out_of_bounds(self):
        """Should handle seed at image boundary gracefully."""
        image = create_uniform_image(size=(50, 50, 5), intensity=100.0, noise_std=5.0)

        # Seed at exact boundary
        result = self.analyzer.analyze(image, seed_point=(0, 0, 0))
        self.assertIn("mean", result)

        # Seed at other corner
        result = self.analyzer.analyze(image, seed_point=(49, 49, 4))
        self.assertIn("mean", result)


@unittest.skipIf(
    IntensityAnalyzer is None or not HAS_SKLEARN, "IntensityAnalyzer or sklearn not available"
)
class TestIntensityAnalyzerGMM(unittest.TestCase):
    """Tests specifically for GMM functionality."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.analyzer = IntensityAnalyzer(use_gmm=True)

    def test_gmm_bimodal_identifies_components(self):
        """GMM should identify two components in bimodal image."""
        image, mask = create_bimodal_image(
            size=(100, 100, 10),
            mean1=50.0,
            mean2=200.0,  # Clear separation
            std1=10.0,
            std2=10.0,
        )
        seed_point = (25, 50, 5)  # In low region

        result = self.analyzer.analyze(image, seed_point, radius_voxels=(40, 40, 5))

        self.assertGreaterEqual(result["n_components"], 2)

    def test_gmm_selects_correct_component(self):
        """GMM should select component containing seed intensity."""
        image, mask = create_bimodal_image(
            size=(100, 100, 10),
            mean1=50.0,
            mean2=200.0,
            std1=10.0,
            std2=10.0,
        )

        # Seed in low region
        result_low = self.analyzer.analyze(
            image, seed_point=(25, 50, 5), radius_voxels=(40, 40, 5)
        )

        # Seed in high region
        result_high = self.analyzer.analyze(
            image, seed_point=(75, 50, 5), radius_voxels=(40, 40, 5)
        )

        # Low region seed should get thresholds around 50
        self.assertLess(result_low["mean"], 100)

        # High region seed should get thresholds around 200
        self.assertGreater(result_high["mean"], 150)

    def test_gmm_component_range(self):
        """GMM should try range of component counts."""
        image = create_uniform_image(size=(100, 100, 10), intensity=100.0, noise_std=20.0)
        seed_point = (50, 50, 5)

        analyzer = IntensityAnalyzer(use_gmm=True, n_components_range=(1, 3))
        result = analyzer.analyze(image, seed_point)

        # Uniform image should prefer 1-2 components
        self.assertLessEqual(result["n_components"], 3)


if __name__ == "__main__":
    unittest.main()
