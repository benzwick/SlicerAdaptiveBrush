"""Tests for adaptive brush segmentation algorithms.

These tests verify that the segmentation algorithms correctly identify
regions based on intensity similarity and respect image boundaries.

Note: Many of these tests require SimpleITK which is only available
in the Slicer environment. Tests will skip gracefully when SimpleITK
is not available.
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

# Try to import SimpleITK
try:
    import SimpleITK as sitk

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    sitk = None

# Import PerformanceCache for cache tests
try:
    from PerformanceCache import CacheStats, PerformanceCache
except ImportError:
    PerformanceCache = None
    CacheStats = None


def requires_sitk(test_func):
    """Decorator to skip tests that require SimpleITK."""
    return unittest.skipIf(not HAS_SIMPLEITK, "SimpleITK not available")(test_func)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    return 2 * intersection / (np.sum(mask1 > 0) + np.sum(mask2 > 0) + 1e-8)


class TestConnectedThresholdAlgorithm(unittest.TestCase):
    """Tests for connected threshold segmentation (fast mode)."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_segments_uniform_region(self):
        """Connected threshold should segment uniform intensity region."""
        image = create_uniform_image(size=(50, 50, 10), intensity=100.0, noise_std=5.0)
        seed_point = (25, 25, 5)  # (x, y, z) -> localSeed (i, j, k)

        # Convert to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Run connected threshold (seed in sitk is x, y, z)
        result = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[seed_point],
            lower=80.0,
            upper=120.0,
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Should segment a significant portion
        self.assertGreater(np.sum(mask), 1000)

    @requires_sitk
    def test_stops_at_intensity_boundary(self):
        """Connected threshold should stop at intensity boundary."""
        image, ground_truth = create_bimodal_image(
            size=(100, 100, 10), mean1=100.0, mean2=200.0, std1=5.0, std2=5.0
        )
        seed_point = (25, 50, 5)  # In low region (x < 50)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        result = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[seed_point],
            lower=80.0,
            upper=130.0,  # Should not reach high region (200)
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Should segment low region, not high region
        # Check overlap with ground truth region 1 (low intensity)
        low_region = ground_truth == 1
        high_region = ground_truth == 2

        low_overlap = np.sum(mask & low_region)
        high_overlap = np.sum(mask & high_region)

        self.assertGreater(low_overlap, high_overlap * 10)  # Mostly in low region

    @requires_sitk
    def test_empty_result_for_out_of_range(self):
        """Should return empty mask when seed is outside threshold range."""
        image = create_uniform_image(size=(50, 50, 10), intensity=100.0, noise_std=5.0)
        seed_point = (25, 25, 5)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Threshold range that doesn't include the image intensity
        result = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[seed_point],
            lower=200.0,
            upper=300.0,  # Image is around 100
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        self.assertEqual(np.sum(mask), 0)


class TestWatershedAlgorithm(unittest.TestCase):
    """Tests for watershed-based segmentation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_watershed_segments_uniform_region(self):
        """Watershed should fill uniform intensity region."""
        image = create_uniform_image(size=(50, 50, 10), intensity=100.0, noise_std=5.0)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # First get initial region with connected threshold
        initial = sitk.ConnectedThreshold(
            sitk_image, seedList=[(25, 25, 5)], lower=80.0, upper=120.0, replaceValue=1
        )

        # Compute gradient
        gradient = sitk.GradientMagnitude(sitk_image)

        # Create markers
        foreground = sitk.BinaryErode(initial, [2, 2, 2])
        dilated = sitk.BinaryDilate(initial, [3, 3, 3])
        background = sitk.Subtract(dilated, sitk.BinaryDilate(initial, [1, 1, 1]))
        markers = sitk.Add(foreground, sitk.Multiply(background, 2))
        markers = sitk.Cast(markers, sitk.sitkUInt8)

        # Run watershed
        watershed = sitk.MorphologicalWatershedFromMarkers(
            gradient, markers, markWatershedLine=False, fullyConnected=True
        )

        # Extract foreground
        result = sitk.BinaryThreshold(watershed, 1, 1, 1, 0)
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Should have segmented something
        self.assertGreater(np.sum(mask), 100)

    @requires_sitk
    def test_watershed_stops_at_edge(self):
        """Watershed should stop at intensity boundary."""
        image, ground_truth = create_bimodal_image(
            size=(100, 100, 10), mean1=100.0, mean2=200.0, std1=5.0, std2=5.0
        )
        seed_point = (25, 50, 5)  # In low region

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Initial threshold
        initial = sitk.ConnectedThreshold(
            sitk_image, seedList=[seed_point], lower=80.0, upper=130.0, replaceValue=1
        )
        initial_mask = sitk.GetArrayFromImage(initial)

        if np.sum(initial_mask) == 0:
            self.skipTest("Initial segmentation empty")

        # Gradient
        gradient = sitk.GradientMagnitude(sitk_image)

        # Markers
        foreground = sitk.BinaryErode(initial, [2, 2, 2])
        dilated = sitk.BinaryDilate(initial, [3, 3, 3])
        background = sitk.Subtract(dilated, sitk.BinaryDilate(initial, [1, 1, 1]))
        markers = sitk.Add(foreground, sitk.Multiply(background, 2))
        markers = sitk.Cast(markers, sitk.sitkUInt8)

        # Watershed
        watershed = sitk.MorphologicalWatershedFromMarkers(
            gradient, markers, markWatershedLine=False, fullyConnected=True
        )
        result = sitk.BinaryThreshold(watershed, 1, 1, 1, 0)
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Should mostly stay in low region
        low_region = ground_truth == 1
        dice_with_low = compute_dice(mask, low_region)

        self.assertGreater(dice_with_low, 0.5)


class TestLevelSetAlgorithm(unittest.TestCase):
    """Tests for level-set-based segmentation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_level_set_segments_sphere(self):
        """Level set should segment spherical region."""
        image, ground_truth = create_noisy_sphere(
            size=(50, 50, 50),
            radius=15.0,
            inside_mean=200.0,
            outside_mean=50.0,
            inside_std=10.0,
            outside_std=10.0,
        )
        seed_point = (25, 25, 25)  # Center of sphere

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Initial segmentation
        initial = sitk.ConnectedThreshold(
            sitk_image, seedList=[seed_point], lower=150.0, upper=250.0, replaceValue=1
        )
        initial_mask = sitk.GetArrayFromImage(initial)

        if np.sum(initial_mask) < 100:
            self.skipTest("Initial segmentation too small")

        # Create distance map
        initial_sitk = sitk.GetImageFromArray(initial_mask.astype(np.uint8))
        distance_map = sitk.SignedMaurerDistanceMap(
            initial_sitk, insideIsPositive=True, squaredDistance=False, useImageSpacing=False
        )

        # Speed image from gradient
        gradient = sitk.GradientMagnitude(sitk_image)
        grad_array = sitk.GetArrayFromImage(gradient)
        grad_max = np.max(grad_array) + 1e-8
        speed_array = 1.0 - (grad_array / grad_max)
        speed = sitk.GetImageFromArray(speed_array.astype(np.float32))

        # Run geodesic active contour
        gac = sitk.GeodesicActiveContourLevelSetImageFilter()
        gac.SetPropagationScaling(1.0)
        gac.SetCurvatureScaling(0.5)
        gac.SetAdvectionScaling(1.0)
        gac.SetMaximumRMSError(0.01)
        gac.SetNumberOfIterations(50)

        level_set = gac.Execute(
            sitk.Cast(distance_map, sitk.sitkFloat32), sitk.Cast(speed, sitk.sitkFloat32)
        )

        result = sitk.BinaryThreshold(level_set, lowerThreshold=0)
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Calculate Dice with ground truth
        dice = compute_dice(mask, ground_truth)
        self.assertGreater(dice, 0.5)  # Reasonable overlap


class TestRegionGrowingAlgorithm(unittest.TestCase):
    """Tests for region growing segmentation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_region_growing_uniform_region(self):
        """Region growing should expand through homogeneous region."""
        image = create_uniform_image(size=(50, 50, 10), intensity=100.0, noise_std=5.0)
        seed_point = (25, 25, 5)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        result = sitk.ConfidenceConnected(
            sitk_image,
            seedList=[seed_point],
            numberOfIterations=3,
            multiplier=2.5,
            initialNeighborhoodRadius=2,
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Should grow through uniform region
        self.assertGreater(np.sum(mask), 1000)

    @requires_sitk
    def test_region_growing_stops_at_boundary(self):
        """Region growing should stop at intensity boundaries."""
        image, ground_truth = create_bimodal_image(
            size=(100, 100, 10), mean1=100.0, mean2=200.0, std1=10.0, std2=10.0
        )
        seed_point = (25, 50, 5)  # In low region

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        result = sitk.ConfidenceConnected(
            sitk_image,
            seedList=[seed_point],
            numberOfIterations=3,
            multiplier=2.0,  # Moderate tolerance
            initialNeighborhoodRadius=2,
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Check that it mostly stays in low region
        low_region = ground_truth == 1
        high_region = ground_truth == 2

        low_overlap = np.sum(mask & low_region)
        high_overlap = np.sum(mask & high_region)

        # Should have more overlap with low region
        self.assertGreater(low_overlap, high_overlap)


class TestThresholdBrushAlgorithm(unittest.TestCase):
    """Tests for threshold brush segmentation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_threshold_brush_simple_mask(self):
        """Threshold brush should create mask within intensity range."""
        image = create_uniform_image(size=(50, 50, 10), intensity=100.0, noise_std=5.0)

        # Simple threshold operation (doesn't require SimpleITK)
        lower, upper = 90.0, 110.0
        mask = ((image >= lower) & (image <= upper)).astype(np.uint8)

        # Most of the image should be within range
        total_voxels = image.size
        masked_voxels = np.sum(mask)

        self.assertGreater(masked_voxels / total_voxels, 0.8)

    def test_threshold_brush_excludes_out_of_range(self):
        """Threshold brush should exclude voxels outside range."""
        image, ground_truth = create_bimodal_image(
            size=(100, 100, 10), mean1=100.0, mean2=200.0, std1=5.0, std2=5.0
        )

        # Threshold for low region only
        lower, upper = 80.0, 130.0
        mask = ((image >= lower) & (image <= upper)).astype(np.uint8)

        # Should mostly match low region
        low_region = ground_truth == 1
        high_region = ground_truth == 2

        low_overlap = np.sum(mask & low_region)
        high_overlap = np.sum(mask & high_region)

        self.assertGreater(low_overlap, high_overlap * 5)


class TestBrushMaskApplication(unittest.TestCase):
    """Tests for brush radius mask application."""

    def test_circular_brush_mask(self):
        """Brush mask should create circular/spherical region."""
        shape = (10, 50, 50)  # (z, y, x)
        center = (5, 25, 25)  # (z, y, x)
        radius = 10

        # Create full mask
        full_mask = np.ones(shape, dtype=np.uint8)

        # Create brush mask (spherical)
        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
        distance = np.sqrt(
            (x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2
        )
        brush_mask = distance <= radius

        # Apply brush mask
        result = (full_mask & brush_mask).astype(np.uint8)

        # Volume should be approximately 4/3 * pi * r^3 for sphere
        # But since we have limited z extent, it's more like a cylinder cap
        expected_area = np.pi * radius**2 * min(shape[0], 2 * radius)
        actual_volume = np.sum(result)

        # Should be in reasonable range
        self.assertGreater(actual_volume, expected_area * 0.5)
        self.assertLess(actual_volume, expected_area * 2.0)

    def test_brush_mask_limits_segmentation(self):
        """Brush mask should limit segmentation to brush radius."""
        shape = (10, 100, 100)
        center = (5, 50, 50)  # (z, y, x) for mask center
        radius = 20

        # Full segmentation mask (entire image)
        seg_mask = np.ones(shape, dtype=np.uint8)

        # Brush radius mask
        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
        distance = np.sqrt(
            (x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2
        )
        brush_mask = distance <= radius

        # Apply brush constraint
        result = (seg_mask & brush_mask).astype(np.uint8)

        # Result should be limited to brush region
        total_brush = np.sum(brush_mask)
        total_result = np.sum(result)

        self.assertEqual(total_result, total_brush)


class TestAlgorithmComparison(unittest.TestCase):
    """Tests comparing different algorithms on same data."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_all_algorithms_produce_result(self):
        """All algorithms should produce non-empty results on simple data."""
        image = create_uniform_image(size=(50, 50, 10), intensity=100.0, noise_std=5.0)
        seed_point = (25, 25, 5)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Connected threshold
        ct_result = sitk.ConnectedThreshold(
            sitk_image, seedList=[seed_point], lower=80.0, upper=120.0, replaceValue=1
        )
        ct_mask = sitk.GetArrayFromImage(ct_result)
        self.assertGreater(np.sum(ct_mask), 0, "Connected threshold should produce result")

        # Confidence connected (region growing)
        cc_result = sitk.ConfidenceConnected(
            sitk_image, seedList=[seed_point], numberOfIterations=3, multiplier=2.5
        )
        cc_mask = sitk.GetArrayFromImage(cc_result)
        self.assertGreater(np.sum(cc_mask), 0, "Region growing should produce result")

        # Threshold (simple)
        thresh_mask = ((image >= 80) & (image <= 120)).astype(np.uint8)
        self.assertGreater(np.sum(thresh_mask), 0, "Threshold should produce result")

    @requires_sitk
    def test_connected_threshold_faster_than_watershed(self):
        """Connected threshold should be faster than watershed."""
        import time

        image = create_uniform_image(size=(100, 100, 20), intensity=100.0, noise_std=5.0)
        seed_point = (50, 50, 10)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Time connected threshold
        start = time.perf_counter()
        for _ in range(5):
            sitk.ConnectedThreshold(
                sitk_image, seedList=[seed_point], lower=80.0, upper=120.0, replaceValue=1
            )
        ct_time = time.perf_counter() - start

        # Time watershed (full pipeline)
        start = time.perf_counter()
        for _ in range(5):
            initial = sitk.ConnectedThreshold(
                sitk_image, seedList=[seed_point], lower=80.0, upper=120.0, replaceValue=1
            )
            gradient = sitk.GradientMagnitude(sitk_image)
            foreground = sitk.BinaryErode(initial, [2, 2, 2])
            dilated = sitk.BinaryDilate(initial, [3, 3, 3])
            background = sitk.Subtract(dilated, sitk.BinaryDilate(initial, [1, 1, 1]))
            markers = sitk.Add(foreground, sitk.Multiply(background, 2))
            markers = sitk.Cast(markers, sitk.sitkUInt8)
            sitk.MorphologicalWatershedFromMarkers(gradient, markers)
        ws_time = time.perf_counter() - start

        # Connected threshold should be faster
        self.assertLess(ct_time, ws_time)


@unittest.skipIf(PerformanceCache is None, "PerformanceCache not importable")
class TestPerformanceCache(unittest.TestCase):
    """Tests for caching during drag operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = PerformanceCache()

    def test_cache_initial_state(self):
        """Cache should start empty."""
        self.assertIsNone(self.cache.gradient_cache)
        self.assertIsNone(self.cache.roi_cache)
        self.assertIsNone(self.cache.threshold_cache)

    def test_cache_invalidation(self):
        """Cache invalidate should clear all caches."""
        # Set some values
        self.cache.gradient_cache = np.array([1, 2, 3])
        self.cache.roi_cache = np.array([4, 5, 6])
        self.cache.threshold_cache = {"lower": 0, "upper": 100}

        # Invalidate
        self.cache.invalidate()

        self.assertIsNone(self.cache.gradient_cache)
        self.assertIsNone(self.cache.roi_cache)
        self.assertIsNone(self.cache.threshold_cache)

    def test_cache_on_mouse_release(self):
        """Mouse release should clear short-lived caches but keep gradient."""
        # Set caches
        self.cache.gradient_cache = np.array([1, 2, 3])
        self.cache.roi_cache = np.array([4, 5, 6])
        self.cache.roi_bounds = ((0, 0, 0), (10, 10, 10))

        # Mouse release
        self.cache.onMouseRelease()

        # Gradient should persist
        self.assertIsNotNone(self.cache.gradient_cache)
        # ROI should be cleared
        self.assertIsNone(self.cache.roi_cache)
        self.assertIsNone(self.cache.roi_bounds)

    def test_cache_validity_check(self):
        """isValidFor should correctly check bounds."""
        # Set up bounds
        self.cache.roi_bounds = ((10, 10, 5), (50, 50, 15))

        # Seed inside bounds with small radius
        self.assertTrue(self.cache.isValidFor((30, 30, 10), (10, 10, 3)))

        # Seed outside bounds
        self.assertFalse(self.cache.isValidFor((5, 30, 10), (10, 10, 3)))

        # Seed inside but radius extends outside
        self.assertFalse(self.cache.isValidFor((30, 30, 10), (25, 25, 10)))

    def test_cache_clear(self):
        """Clear should reset everything including stats."""
        self.cache.gradient_cache = np.array([1, 2, 3])
        self.cache.stats.gradient_hits = 10

        self.cache.clear()

        self.assertIsNone(self.cache.gradient_cache)
        self.assertEqual(self.cache.stats.gradient_hits, 0)

    def test_threshold_caching_reuses_when_similar_intensity(self):
        """Threshold cache should be reused when seed intensity is similar."""
        # Set up cached threshold
        self.cache.threshold_cache = {"lower": 80, "upper": 120, "mean": 100, "std": 10}
        self.cache.threshold_seed_intensity = 100.0
        self.cache.threshold_tolerance = 15.0  # 1.5 * std

        # Check that similar intensity can reuse
        self.assertTrue(self.cache._canReuseThresholds(105.0))  # Within tolerance
        self.assertTrue(self.cache._canReuseThresholds(95.0))  # Within tolerance

        # Check that different intensity cannot reuse
        self.assertFalse(self.cache._canReuseThresholds(120.0))  # Outside tolerance
        self.assertFalse(self.cache._canReuseThresholds(80.0))  # Outside tolerance

    def test_threshold_caching_no_cache_returns_false(self):
        """Should not reuse when no cache exists."""
        self.assertFalse(self.cache._canReuseThresholds(100.0))

    def test_gradient_caching_reuses_same_slice(self):
        """Gradient cache should be reused for same slice."""
        # Create dummy volume and gradient
        volume = np.random.rand(10, 50, 50).astype(np.float32)
        gradient = np.random.rand(50, 50).astype(np.float32)

        # Mock compute function that should only be called once
        call_count = [0]

        def compute_gradient(slice_array):
            call_count[0] += 1
            return gradient

        # First call - should compute
        result1 = self.cache.getOrComputeGradient(volume, 5, "vol1", compute_gradient)
        self.assertEqual(call_count[0], 1)
        self.assertEqual(self.cache.stats.gradient_misses, 1)

        # Second call same slice - should reuse cache
        result2 = self.cache.getOrComputeGradient(volume, 5, "vol1", compute_gradient)
        self.assertEqual(call_count[0], 1)  # Still 1, no new computation
        self.assertEqual(self.cache.stats.gradient_hits, 1)

        np.testing.assert_array_equal(result1, result2)

    def test_gradient_caching_invalidates_on_slice_change(self):
        """Gradient cache should recompute when slice changes."""
        volume = np.random.rand(10, 50, 50).astype(np.float32)
        call_count = [0]

        def compute_gradient(slice_array):
            call_count[0] += 1
            return slice_array * 0.5  # Simple gradient mock

        # First call on slice 5
        self.cache.getOrComputeGradient(volume, 5, "vol1", compute_gradient)
        self.assertEqual(call_count[0], 1)

        # Second call on slice 7 - should recompute
        self.cache.getOrComputeGradient(volume, 7, "vol1", compute_gradient)
        self.assertEqual(call_count[0], 2)
        self.assertEqual(self.cache.stats.gradient_misses, 2)

    def test_gradient_caching_invalidates_on_volume_change(self):
        """Gradient cache should recompute when volume changes."""
        volume = np.random.rand(10, 50, 50).astype(np.float32)
        call_count = [0]

        def compute_gradient(slice_array):
            call_count[0] += 1
            return slice_array * 0.5

        # First call
        self.cache.getOrComputeGradient(volume, 5, "vol1", compute_gradient)
        self.assertEqual(call_count[0], 1)

        # Second call with different volume ID - should recompute
        self.cache.getOrComputeGradient(volume, 5, "vol2", compute_gradient)
        self.assertEqual(call_count[0], 2)


@unittest.skipIf(CacheStats is None, "CacheStats not importable")
class TestCacheStats(unittest.TestCase):
    """Tests for cache statistics tracking."""

    def test_stats_initial_state(self):
        """Stats should start at zero."""
        stats = CacheStats()

        self.assertEqual(stats.gradient_hits, 0)
        self.assertEqual(stats.gradient_misses, 0)
        self.assertEqual(stats.roi_hits, 0)
        self.assertEqual(stats.roi_misses, 0)

    def test_stats_reset(self):
        """Reset should zero all counters."""
        stats = CacheStats()
        stats.gradient_hits = 10
        stats.gradient_misses = 5
        stats.roi_hits = 8
        stats.roi_misses = 3

        stats.reset()

        self.assertEqual(stats.gradient_hits, 0)
        self.assertEqual(stats.gradient_misses, 0)
        self.assertEqual(stats.roi_hits, 0)
        self.assertEqual(stats.roi_misses, 0)


if __name__ == "__main__":
    unittest.main()
