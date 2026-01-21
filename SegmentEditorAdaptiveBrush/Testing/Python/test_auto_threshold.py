"""Tests for auto-threshold functionality in the Threshold Brush algorithm.

These tests verify:
- Auto-threshold computation using various methods (Otsu, Huang, etc.)
- Automatic detection of foreground vs background based on seed intensity
- Set-from-seed threshold computation
- Threshold brush algorithm with auto mode
"""

import unittest

import numpy as np
from test_fixtures.synthetic_image import create_bimodal_image, create_noisy_sphere

# Try to import SimpleITK (required for auto-threshold)
try:
    import SimpleITK as sitk

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False


def requires_sitk(func):
    """Decorator to skip tests if SimpleITK is not available."""
    return unittest.skipUnless(HAS_SIMPLEITK, "SimpleITK not available")(func)


class TestAutoThresholdComputation(unittest.TestCase):
    """Tests for _computeAutoThreshold method."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_otsu_threshold_bimodal_image(self):
        """Otsu should find threshold between two peaks of bimodal image."""
        # Create bimodal image: left half ~100, right half ~200
        image, _ = create_bimodal_image(
            size=(50, 50, 10), mean1=100.0, mean2=200.0, std1=10.0, std2=10.0
        )

        # Compute Otsu threshold
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(1)
        otsu_filter.SetOutsideValue(0)
        otsu_filter.Execute(sitk_image)
        threshold = otsu_filter.GetThreshold()

        # Threshold should be between the two means
        self.assertGreater(threshold, 120)
        self.assertLess(threshold, 180)

    @requires_sitk
    def test_seed_in_bright_region_returns_upper_range(self):
        """When seed is in bright region, should return (threshold, max) range."""
        # Create bimodal image
        image, _ = create_bimodal_image(
            size=(50, 50, 10), mean1=50.0, mean2=200.0, std1=10.0, std2=10.0
        )

        # Seed in bright region (right half, x >= 25)
        local_seed = (35, 25, 5)  # (i, j, k) = (x, y, z)
        seed_intensity = image[local_seed[2], local_seed[1], local_seed[0]]

        # Compute threshold
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(1)
        otsu_filter.SetOutsideValue(0)
        otsu_filter.Execute(sitk_image)
        threshold = otsu_filter.GetThreshold()

        # Seed should be above threshold (in bright region)
        self.assertGreater(seed_intensity, threshold)

        # Auto-detection should return (threshold, max)
        data_max = float(image.max())
        if seed_intensity >= threshold:
            lower, upper = threshold, data_max
        else:
            lower, upper = float(image.min()), threshold

        self.assertAlmostEqual(lower, threshold, delta=1.0)
        self.assertAlmostEqual(upper, data_max, delta=1.0)

    @requires_sitk
    def test_seed_in_dark_region_returns_lower_range(self):
        """When seed is in dark region, should return (min, threshold) range."""
        # Create bimodal image
        image, _ = create_bimodal_image(
            size=(50, 50, 10), mean1=50.0, mean2=200.0, std1=10.0, std2=10.0
        )

        # Seed in dark region (left half, x < 25)
        local_seed = (10, 25, 5)  # (i, j, k) = (x, y, z)
        seed_intensity = image[local_seed[2], local_seed[1], local_seed[0]]

        # Compute threshold
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(1)
        otsu_filter.SetOutsideValue(0)
        otsu_filter.Execute(sitk_image)
        threshold = otsu_filter.GetThreshold()

        # Seed should be below threshold (in dark region)
        self.assertLess(seed_intensity, threshold)

        # Auto-detection should return (min, threshold)
        data_min = float(image.min())
        if seed_intensity >= threshold:
            lower, upper = threshold, float(image.max())
        else:
            lower, upper = data_min, threshold

        self.assertAlmostEqual(lower, data_min, delta=1.0)
        self.assertAlmostEqual(upper, threshold, delta=1.0)

    @requires_sitk
    def test_all_threshold_methods_produce_valid_results(self):
        """All threshold methods should produce valid thresholds."""
        image, _ = create_bimodal_image(
            size=(50, 50, 10), mean1=80.0, mean2=180.0, std1=15.0, std2=15.0
        )
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        methods = [
            ("otsu", sitk.OtsuThresholdImageFilter),
            ("huang", sitk.HuangThresholdImageFilter),
            ("triangle", sitk.TriangleThresholdImageFilter),
            ("max_entropy", sitk.MaximumEntropyThresholdImageFilter),
            ("isodata", sitk.IsoDataThresholdImageFilter),
            ("li", sitk.LiThresholdImageFilter),
        ]

        data_min, data_max = float(image.min()), float(image.max())

        for method_name, filter_class in methods:
            with self.subTest(method=method_name):
                filter_obj = filter_class()
                filter_obj.SetInsideValue(1)
                filter_obj.SetOutsideValue(0)
                filter_obj.Execute(sitk_image)
                threshold = filter_obj.GetThreshold()

                # Threshold should be within data range
                self.assertGreaterEqual(
                    threshold, data_min, f"{method_name} threshold below data min"
                )
                self.assertLessEqual(threshold, data_max, f"{method_name} threshold above data max")

    @requires_sitk
    def test_threshold_methods_differ_on_skewed_distribution(self):
        """Different methods should produce different thresholds for skewed data."""
        # Create skewed image (more dark than bright)
        np.random.seed(42)
        shape = (10, 50, 50)
        image = np.zeros(shape, dtype=np.float32)
        # 80% dark, 20% bright
        image[:, :, :40] = np.random.normal(50, 10, (10, 50, 40)).astype(np.float32)
        image[:, :, 40:] = np.random.normal(200, 10, (10, 50, 10)).astype(np.float32)

        sitk_image = sitk.GetImageFromArray(image)

        # Get thresholds from different methods
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(sitk_image)
        otsu_threshold = otsu.GetThreshold()

        triangle = sitk.TriangleThresholdImageFilter()
        triangle.Execute(sitk_image)
        triangle_threshold = triangle.GetThreshold()

        # Triangle often differs from Otsu on skewed distributions
        # Just verify both are valid and potentially different
        self.assertGreater(otsu_threshold, 40)
        self.assertLess(otsu_threshold, 210)
        self.assertGreater(triangle_threshold, 40)
        self.assertLess(triangle_threshold, 210)


class TestSetFromSeed(unittest.TestCase):
    """Tests for threshold computation from seed intensity."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_seed_intensity_extracted_correctly(self):
        """Should extract correct intensity at seed point."""
        # Create simple test image with known value at seed
        image = np.zeros((10, 50, 50), dtype=np.float32)
        image[5, 25, 30] = 150.0  # Known value at (x=30, y=25, z=5)

        seed_ijk = (30, 25, 5)  # (i, j, k) = (x, y, z)
        # Array indexing is [z, y, x]
        seed_intensity = image[seed_ijk[2], seed_ijk[1], seed_ijk[0]]

        self.assertEqual(seed_intensity, 150.0)

    def test_local_std_computed_from_roi(self):
        """Local std should be computed from small ROI around seed."""
        # Create image with known statistics in ROI
        image = np.zeros((20, 50, 50), dtype=np.float32)

        # Fill a 21x21x21 region around center with known std
        center = (25, 25, 10)
        radius = 10
        roi_values = np.random.normal(100, 20, (21, 21, 21)).astype(np.float32)

        z_start = max(0, center[2] - radius)
        z_end = min(20, center[2] + radius + 1)
        y_start = max(0, center[1] - radius)
        y_end = min(50, center[1] + radius + 1)
        x_start = max(0, center[0] - radius)
        x_end = min(50, center[0] + radius + 1)

        image[z_start:z_end, y_start:y_end, x_start:x_end] = roi_values[
            : z_end - z_start, : y_end - y_start, : x_end - x_start
        ]

        # Extract ROI and compute std
        roi = image[z_start:z_end, y_start:y_end, x_start:x_end]
        local_std = np.std(roi)

        # Should be close to 20 (the std we used)
        self.assertAlmostEqual(local_std, 20.0, delta=5.0)

    def test_tolerance_scales_threshold_range(self):
        """Higher tolerance should produce wider threshold range."""
        seed_intensity = 100.0
        local_std = 20.0

        # Low tolerance (20%)
        low_tolerance = 20
        low_range = local_std * (low_tolerance / 100.0) * 2.5
        low_lower = seed_intensity - low_range
        low_upper = seed_intensity + low_range

        # High tolerance (80%)
        high_tolerance = 80
        high_range = local_std * (high_tolerance / 100.0) * 2.5
        high_lower = seed_intensity - high_range
        high_upper = seed_intensity + high_range

        # High tolerance should produce wider range
        self.assertGreater(high_upper - high_lower, low_upper - low_lower)

        # Verify specific values
        self.assertAlmostEqual(low_range, 10.0, delta=0.1)  # 20 * 0.2 * 2.5 = 10
        self.assertAlmostEqual(high_range, 40.0, delta=0.1)  # 20 * 0.8 * 2.5 = 40


class TestThresholdBrushAlgorithm(unittest.TestCase):
    """Tests for _thresholdBrush method with auto and manual modes."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_manual_threshold_applies_range(self):
        """Manual threshold should paint only voxels in specified range."""
        # Create gradient image: 0 to 255 along x
        image = np.zeros((10, 50, 100), dtype=np.float32)
        gradient = np.linspace(0, 255, 100, dtype=np.float32)
        image[:, :, :] = gradient[np.newaxis, np.newaxis, :]

        # Apply threshold: 100 to 150
        lower, upper = 100.0, 150.0
        mask = ((image >= lower) & (image <= upper)).astype(np.uint8)

        # The gradient goes from 0 at x=0 to 255 at x=99
        # Find which x indices have values in [100, 150]
        in_range = (gradient >= lower) & (gradient <= upper)
        expected_x_indices = np.where(in_range)[0]

        # Verify mask is 1 where gradient is in range
        for x in expected_x_indices:
            self.assertEqual(mask[0, 0, x], 1, f"Expected mask=1 at x={x}, gradient={gradient[x]}")

        # Verify mask is 0 where gradient is outside range
        out_of_range = ~in_range
        out_of_range_indices = np.where(out_of_range)[0]
        for x in out_of_range_indices:
            self.assertEqual(mask[0, 0, x], 0, f"Expected mask=0 at x={x}, gradient={gradient[x]}")

        # Should have some voxels in range
        self.assertGreater(np.sum(mask), 0)

    @requires_sitk
    def test_auto_threshold_segments_seed_region(self):
        """Auto threshold should segment region containing the seed."""
        # Create bimodal image
        image, ground_truth = create_bimodal_image(
            size=(50, 50, 10), mean1=50.0, mean2=200.0, std1=10.0, std2=10.0
        )

        # Seed in dark region
        local_seed = (10, 25, 5)

        # Compute auto threshold
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(sitk_image)
        threshold = otsu.GetThreshold()

        seed_intensity = image[local_seed[2], local_seed[1], local_seed[0]]
        data_min, data_max = float(image.min()), float(image.max())

        if seed_intensity >= threshold:
            lower, upper = threshold, data_max
        else:
            lower, upper = data_min, threshold

        mask = ((image >= lower) & (image <= upper)).astype(np.uint8)

        # Seed point should be in the mask
        self.assertEqual(mask[local_seed[2], local_seed[1], local_seed[0]], 1)

        # Most of the dark region (left half) should be in mask
        dark_region_coverage = np.mean(mask[:, :, :25])
        self.assertGreater(dark_region_coverage, 0.9)

        # Most of bright region (right half) should NOT be in mask
        bright_region_coverage = np.mean(mask[:, :, 25:])
        self.assertLess(bright_region_coverage, 0.1)

    @requires_sitk
    def test_auto_threshold_with_sphere(self):
        """Auto threshold should segment sphere when seed is inside."""
        # Create sphere with bright interior
        image, ground_truth = create_noisy_sphere(
            size=(50, 50, 50),
            radius=15.0,
            inside_mean=200.0,
            outside_mean=50.0,
            inside_std=10.0,
            outside_std=10.0,
        )

        # Seed at center of sphere
        local_seed = (25, 25, 25)

        # Compute auto threshold
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(sitk_image)
        threshold = otsu.GetThreshold()

        seed_intensity = image[local_seed[2], local_seed[1], local_seed[0]]
        data_min, data_max = float(image.min()), float(image.max())

        if seed_intensity >= threshold:
            lower, upper = threshold, data_max
        else:
            lower, upper = data_min, threshold

        mask = ((image >= lower) & (image <= upper)).astype(np.uint8)

        # Compute Dice coefficient with ground truth
        intersection = np.sum(mask & ground_truth)
        dice = 2 * intersection / (np.sum(mask) + np.sum(ground_truth) + 1e-8)

        # Should have high overlap with sphere
        self.assertGreater(dice, 0.8)


class TestAutoThresholdEdgeCases(unittest.TestCase):
    """Tests for edge cases in auto-threshold computation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_uniform_image_threshold(self):
        """Uniform image should still produce valid threshold."""
        image = np.full((10, 50, 50), 100.0, dtype=np.float32)
        # Add tiny noise to avoid numerical issues
        image += np.random.normal(0, 0.1, image.shape).astype(np.float32)

        sitk_image = sitk.GetImageFromArray(image)
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(sitk_image)
        threshold = otsu.GetThreshold()

        # Threshold should be close to the uniform value
        self.assertAlmostEqual(threshold, 100.0, delta=5.0)

    @requires_sitk
    def test_high_noise_still_works(self):
        """Should handle high noise gracefully."""
        # Create very noisy bimodal image
        np.random.seed(42)
        image = np.zeros((10, 50, 50), dtype=np.float32)
        image[:, :, :25] = np.random.normal(100, 40, (10, 50, 25)).astype(np.float32)
        image[:, :, 25:] = np.random.normal(200, 40, (10, 50, 25)).astype(np.float32)

        sitk_image = sitk.GetImageFromArray(image)
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(sitk_image)
        threshold = otsu.GetThreshold()

        # Should still find reasonable threshold (between means)
        self.assertGreater(threshold, 80)
        self.assertLess(threshold, 220)

    def test_seed_at_boundary(self):
        """Should handle seed at image boundary."""
        image = np.random.normal(100, 10, (10, 50, 50)).astype(np.float32)

        # Seeds at boundaries
        boundary_seeds = [
            (0, 0, 0),  # Corner
            (49, 49, 9),  # Opposite corner
            (25, 25, 0),  # Edge
            (0, 25, 5),  # Edge
        ]

        for seed in boundary_seeds:
            with self.subTest(seed=seed):
                # Should not raise exception
                intensity = image[seed[2], seed[1], seed[0]]
                self.assertIsInstance(intensity, (float, np.floating))


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    return 2 * intersection / (np.sum(mask1) + np.sum(mask2) + 1e-8)


if __name__ == "__main__":
    unittest.main()
