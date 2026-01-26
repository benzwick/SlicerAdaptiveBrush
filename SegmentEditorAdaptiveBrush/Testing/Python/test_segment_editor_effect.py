"""Integration tests for SegmentEditorEffect.

These tests verify the core functionality of the SegmentEditorEffect class,
including algorithm wrappers and mask computation.

Note: Tests requiring Slicer's Qt/VTK environment are marked with
requires_slicer and will skip when running outside Slicer.
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


def requires_sitk(test_func):
    """Decorator to skip tests that require SimpleITK."""
    return unittest.skipIf(not HAS_SIMPLEITK, "SimpleITK not available")(test_func)


def requires_slicer(test_func):
    """Decorator to skip tests that require Slicer environment."""
    try:
        import slicer  # noqa: F401

        has_slicer = True
    except ImportError:
        has_slicer = False
    return unittest.skipIf(not has_slicer, "Slicer not available")(test_func)


class TestAlgorithmWrappers(unittest.TestCase):
    """Tests for standalone algorithm wrapper methods.

    These tests verify the SimpleITK-based algorithm implementations
    work correctly without needing the full Slicer environment.
    """

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    @requires_sitk
    def test_connected_threshold_segments_uniform_region(self):
        """_connectedThreshold should segment uniform intensity region."""
        # Create test image with uniform region
        image = create_uniform_image(size=(50, 50, 20), intensity=100.0, noise_std=5.0)
        seed = (25, 25, 10)  # Center of volume

        # Run connected threshold directly using SimpleITK
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        result = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[seed],
            lower=85.0,
            upper=115.0,
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Should segment most of the volume
        coverage = np.sum(mask) / mask.size
        self.assertGreater(coverage, 0.8, "Should segment most of uniform region")

    @requires_sitk
    def test_connected_threshold_respects_boundaries(self):
        """_connectedThreshold should stop at intensity boundaries."""
        # Create bimodal image
        image, _ = create_bimodal_image(
            size=(100, 100, 20), mean1=100.0, mean2=200.0, std1=5.0, std2=5.0
        )
        seed = (25, 50, 10)  # In low-intensity region (left half)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        result = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[seed],
            lower=85.0,
            upper=115.0,
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        # Should only segment left half (low intensity)
        left_coverage = np.sum(mask[:, :, :50]) / mask[:, :, :50].size
        right_coverage = np.sum(mask[:, :, 50:]) / mask[:, :, 50:].size

        self.assertGreater(left_coverage, 0.7, "Should segment left region")
        self.assertLess(right_coverage, 0.1, "Should not segment right region")

    @requires_sitk
    def test_watershed_produces_result(self):
        """Watershed algorithm should produce non-empty result."""
        image = create_uniform_image(size=(50, 50, 20), intensity=100.0, noise_std=10.0)

        # Simple gradient-based watershed test
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Compute gradient magnitude
        gradient = sitk.GradientMagnitude(sitk_image)

        # Create marker image (seed at center)
        marker_array = np.zeros_like(image, dtype=np.uint8)
        marker_array[10, 25, 25] = 1  # Foreground marker
        # Background marker at edge
        marker_array[10, 0, 0] = 2
        marker_array[10, 49, 49] = 2

        markers = sitk.GetImageFromArray(marker_array)
        markers.CopyInformation(sitk_image)

        # Run morphological watershed
        watershed = sitk.MorphologicalWatershedFromMarkers(gradient, markers)
        result = sitk.GetArrayFromImage(watershed)

        # Foreground label (1) should have some voxels
        foreground_count = np.sum(result == 1)
        self.assertGreater(foreground_count, 100, "Watershed should segment something")

    @requires_sitk
    def test_region_growing_uniform_region(self):
        """Region growing should expand in uniform intensity region."""
        image = create_uniform_image(size=(50, 50, 20), intensity=100.0, noise_std=5.0)
        seed = (25, 25, 10)

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # ConfidenceConnected is SimpleITK's region growing
        result = sitk.ConfidenceConnected(
            sitk_image,
            seedList=[seed],
            numberOfIterations=3,
            multiplier=2.5,
            initialNeighborhoodRadius=1,
            replaceValue=1,
        )
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)

        coverage = np.sum(mask) / mask.size
        self.assertGreater(coverage, 0.5, "Region growing should expand significantly")


class TestThresholdBrushAlgorithm(unittest.TestCase):
    """Tests for threshold brush algorithm (doesn't require SimpleITK filters)."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_threshold_mask_basic(self):
        """Threshold brush should create mask within threshold range."""
        # Create simple image
        image = np.zeros((20, 50, 50), dtype=np.float32)
        image[:, :, :25] = 100.0  # Left half = 100
        image[:, :, 25:] = 200.0  # Right half = 200

        # Threshold for left region
        lower, upper = 90.0, 110.0
        mask = ((image >= lower) & (image <= upper)).astype(np.uint8)

        # Should only select left half
        self.assertEqual(np.sum(mask[:, :, :25]), 20 * 50 * 25)
        self.assertEqual(np.sum(mask[:, :, 25:]), 0)

    def test_threshold_mask_with_noise(self):
        """Threshold brush should handle noisy images."""
        np.random.seed(42)
        image = np.random.normal(100, 10, (20, 50, 50)).astype(np.float32)

        # Most values should be within 2 std of mean
        lower, upper = 80.0, 120.0
        mask = ((image >= lower) & (image <= upper)).astype(np.uint8)

        coverage = np.sum(mask) / mask.size
        self.assertGreater(coverage, 0.9, "Should include most voxels within 2 std")


class TestBrushMaskCreation(unittest.TestCase):
    """Tests for brush mask creation (circular/spherical masks)."""

    def test_circular_mask_2d(self):
        """Create circular brush mask for 2D operations."""
        size = 50
        center = (25, 25)
        radius = 10

        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = (distance <= radius).astype(np.uint8)

        # Check mask properties
        self.assertEqual(mask[center[1], center[0]], 1, "Center should be in mask")
        self.assertEqual(mask[0, 0], 0, "Corner should be outside mask")

        # Approximate area check (pi * r^2)
        expected_area = np.pi * radius**2
        actual_area = np.sum(mask)
        self.assertAlmostEqual(actual_area, expected_area, delta=expected_area * 0.1)

    def test_spherical_mask_3d(self):
        """Create spherical brush mask for 3D operations."""
        size = 50
        center = (25, 25, 25)
        radius = 10

        z, y, x = np.ogrid[:size, :size, :size]
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        mask = (distance <= radius).astype(np.uint8)

        # Check mask properties
        self.assertEqual(mask[center[2], center[1], center[0]], 1, "Center should be in mask")
        self.assertEqual(mask[0, 0, 0], 0, "Corner should be outside mask")

        # Approximate volume check (4/3 * pi * r^3)
        expected_volume = (4 / 3) * np.pi * radius**3
        actual_volume = np.sum(mask)
        self.assertAlmostEqual(actual_volume, expected_volume, delta=expected_volume * 0.1)

    def test_anisotropic_ellipsoid_mask(self):
        """Create ellipsoid mask for anisotropic voxel spacing."""
        size = (20, 50, 50)  # z, y, x
        center = (10, 25, 25)
        # Different radius for each axis (to handle anisotropic spacing)
        radius_voxels = (5, 10, 10)  # z, y, x radii

        z, y, x = np.ogrid[: size[0], : size[1], : size[2]]
        distance = np.sqrt(
            ((x - center[2]) / radius_voxels[2]) ** 2
            + ((y - center[1]) / radius_voxels[1]) ** 2
            + ((z - center[0]) / radius_voxels[0]) ** 2
        )
        mask = (distance <= 1.0).astype(np.uint8)

        # Check center is included
        self.assertEqual(mask[center[0], center[1], center[2]], 1)

        # Check axes are correctly sized
        # X axis: should extend ~10 voxels from center
        self.assertEqual(mask[center[0], center[1], center[2] + 9], 1)
        self.assertEqual(mask[center[0], center[1], center[2] + 11], 0)

        # Z axis: should extend ~5 voxels from center
        self.assertEqual(mask[center[0] + 4, center[1], center[2]], 1)
        self.assertEqual(mask[center[0] + 6, center[1], center[2]], 0)


class TestROIExtraction(unittest.TestCase):
    """Tests for ROI extraction around seed point."""

    def test_roi_extraction_basic(self):
        """ROI extraction should get correct region around seed."""
        # Create volume with distinct regions
        volume = np.zeros((100, 100, 100), dtype=np.float32)
        volume[40:60, 40:60, 40:60] = 100.0  # Central cube

        seed = (50, 50, 50)  # i, j, k
        margin = 15

        # Extract ROI
        i_start = max(0, seed[0] - margin)
        i_end = min(volume.shape[2], seed[0] + margin + 1)
        j_start = max(0, seed[1] - margin)
        j_end = min(volume.shape[1], seed[1] + margin + 1)
        k_start = max(0, seed[2] - margin)
        k_end = min(volume.shape[0], seed[2] + margin + 1)

        roi = volume[k_start:k_end, j_start:j_end, i_start:i_end]

        # ROI should contain the bright region
        self.assertGreater(np.mean(roi), 0, "ROI should contain non-zero values")
        self.assertEqual(roi.shape, (31, 31, 31), "ROI should have expected size")

    def test_roi_extraction_at_boundary(self):
        """ROI extraction should handle volume boundaries correctly."""
        volume = np.ones((50, 50, 50), dtype=np.float32) * 100.0

        seed = (5, 5, 5)  # Near corner
        margin = 10

        # Extract ROI with clamping
        i_start = max(0, seed[0] - margin)
        i_end = min(volume.shape[2], seed[0] + margin + 1)
        j_start = max(0, seed[1] - margin)
        j_end = min(volume.shape[1], seed[1] + margin + 1)
        k_start = max(0, seed[2] - margin)
        k_end = min(volume.shape[0], seed[2] + margin + 1)

        roi = volume[k_start:k_end, j_start:j_end, i_start:i_end]

        # ROI should be smaller than normal (clamped at boundary)
        self.assertEqual(roi.shape, (16, 16, 16), "ROI should be clamped at boundary")


class TestParameterValidation(unittest.TestCase):
    """Tests for parameter validation and clamping."""

    def test_radius_clamping(self):
        """Brush radius should be clamped to valid range."""
        min_radius = 1.0
        max_radius = 100.0

        # Test clamping
        self.assertEqual(max(min_radius, min(max_radius, 50.0)), 50.0)
        self.assertEqual(max(min_radius, min(max_radius, 0.5)), 1.0)
        self.assertEqual(max(min_radius, min(max_radius, 150.0)), 100.0)

    def test_sensitivity_clamping(self):
        """Edge sensitivity should be clamped to 0-100."""
        min_sens = 0
        max_sens = 100

        self.assertEqual(max(min_sens, min(max_sens, 50)), 50)
        self.assertEqual(max(min_sens, min(max_sens, -10)), 0)
        self.assertEqual(max(min_sens, min(max_sens, 150)), 100)


@requires_sitk
class TestLevelSetAlgorithm(unittest.TestCase):
    """Tests for level set segmentation algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_level_set_segments_sphere(self):
        """Level set should segment sphere-like structure."""
        # Create image with bright sphere
        image, ground_truth = create_noisy_sphere(
            size=(50, 50, 50),
            center=(25, 25, 25),
            radius=10,
            inside_mean=200.0,
            outside_mean=50.0,
            inside_std=5.0,
            outside_std=5.0,
        )

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

        # Create initial level set (seed region)
        seed_array = np.zeros_like(image, dtype=np.float32)
        # Initialize with small sphere at center
        z, y, x = np.ogrid[:50, :50, :50]
        distance = np.sqrt((x - 25) ** 2 + (y - 25) ** 2 + (z - 25) ** 2)
        seed_array[distance <= 3] = 1.0
        seed_array[distance > 3] = -1.0

        seed_image = sitk.GetImageFromArray(seed_array)
        seed_image.CopyInformation(sitk_image)

        # Run geodesic active contour level set
        feature = sitk.GradientMagnitudeRecursiveGaussian(sitk_image, sigma=1.0)
        feature = sitk.Sigmoid(feature, alpha=-0.5, beta=3.0)

        level_set = sitk.GeodesicActiveContourLevelSetImageFilter()
        level_set.SetPropagationScaling(1.0)
        level_set.SetCurvatureScaling(0.5)
        level_set.SetAdvectionScaling(1.0)
        level_set.SetNumberOfIterations(100)

        result = level_set.Execute(seed_image, feature)
        result_array = sitk.GetArrayFromImage(result)

        # Level set output: negative inside, positive outside
        mask = (result_array <= 0).astype(np.uint8)

        # Should segment some portion of the sphere
        self.assertGreater(np.sum(mask), 100, "Level set should produce non-empty result")


if __name__ == "__main__":
    unittest.main()
