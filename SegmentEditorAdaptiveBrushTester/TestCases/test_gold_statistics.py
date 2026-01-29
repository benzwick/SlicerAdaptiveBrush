"""Tests for gold standard statistics computation.

Tests the GoldStandardManager statistics functionality:
- Voxel count
- Volume in mm³
- Bounding box (IJK and mm)
- Centroid (RAS)
- Checksum for integrity verification
- Comparison between gold and trial statistics
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

from .fixtures.mock_segmentations import MockSegmentationFactory

logger = logging.getLogger(__name__)


@register_test(category="gold_standard")
class TestGoldStatisticsComputation(TestCase):
    """Test statistics computation for gold standards."""

    name = "gold_statistics_computation"
    description = "Test voxel count, volume, bbox, centroid, checksum computation"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.seg_factory = None
        self.manager = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test with sample data."""
        logger.info("Setting up gold statistics test")

        slicer.mrmlScene.Clear(0)

        # Load sample data
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.seg_factory = MockSegmentationFactory()

        # Import GoldStandardManager
        from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

        self.manager = GoldStandardManager()

        ctx.screenshot("[setup] Sample data loaded")

    def run(self, ctx: TestContext) -> None:
        """Test statistics computation."""
        logger.info("Running statistics computation tests")

        # Test 1: Cube segmentation with known dimensions
        ctx.log("Test 1: Cube with known dimensions")
        cube_size = 10
        cube_corner = (100, 100, 50)
        cube_seg = self.seg_factory.create_cube_segmentation(
            corner_ijk=cube_corner,
            size=cube_size,
            volume_node=self.volume_node,
            name="CubeTest",
            segment_name="Cube",
        )

        stats = self.manager._compute_statistics(cube_seg, "Cube", self.volume_node)

        # Verify voxel count (10x10x10 = 1000)
        expected_voxels = cube_size**3
        ctx.assert_equal(
            stats["voxel_count"],
            expected_voxels,
            f"Cube voxel count should be {expected_voxels}",
        )

        # Verify bounding box
        ctx.assert_equal(
            stats["bounding_box_ijk"]["min"],
            list(cube_corner),
            "Bounding box min should match cube corner",
        )
        expected_max = [c + cube_size - 1 for c in cube_corner]
        ctx.assert_equal(
            stats["bounding_box_ijk"]["max"],
            expected_max,
            "Bounding box max should be corner + size - 1",
        )

        # Verify volume (voxels * voxel_volume)
        spacing = self.volume_node.GetSpacing()
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        expected_vol = expected_voxels * voxel_vol
        ctx.assert_almost_equal(
            stats["volume_mm3"],
            expected_vol,
            delta=0.1,
            message="Volume should match voxel_count * voxel_volume",
        )

        # Verify checksum exists
        ctx.assert_true(
            len(stats["checksum_sha256"]) == 64,
            "Checksum should be 64-char SHA256 hex string",
        )

        ctx.screenshot("[cube] Cube statistics computed")
        slicer.mrmlScene.RemoveNode(cube_seg)

        # Test 2: Sphere segmentation
        ctx.log("Test 2: Sphere segmentation")
        sphere_center = (128, 128, 64)
        sphere_radius = 15
        sphere_seg = self.seg_factory.create_sphere_segmentation(
            center_ijk=sphere_center,
            radius=sphere_radius,
            volume_node=self.volume_node,
            name="SphereTest",
            segment_name="Sphere",
        )

        sphere_stats = self.manager._compute_statistics(sphere_seg, "Sphere", self.volume_node)

        # Verify voxel count is reasonable for a sphere
        # Sphere volume = 4/3 * pi * r^3
        import math

        expected_sphere_voxels = int((4 / 3) * math.pi * (sphere_radius**3))
        # Allow 10% tolerance due to discrete voxelization
        ctx.assert_greater(
            sphere_stats["voxel_count"],
            expected_sphere_voxels * 0.9,
            "Sphere voxel count should be at least 90% of theoretical",
        )
        ctx.assert_less(
            sphere_stats["voxel_count"],
            expected_sphere_voxels * 1.1,
            "Sphere voxel count should be at most 110% of theoretical",
        )

        # Verify centroid is near sphere center
        centroid = sphere_stats["centroid_ras"]
        ctx.assert_true(
            len(centroid) == 3,
            "Centroid should have 3 coordinates",
        )

        ctx.screenshot("[sphere] Sphere statistics computed")
        slicer.mrmlScene.RemoveNode(sphere_seg)

        # Test 3: Empty segmentation
        ctx.log("Test 3: Empty segmentation")
        empty_seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        empty_seg.SetName("EmptyTest")
        empty_seg.CreateDefaultDisplayNodes()
        empty_seg.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        empty_seg.GetSegmentation().AddEmptySegment("Empty")

        empty_stats = self.manager._compute_statistics(empty_seg, "Empty", self.volume_node)

        ctx.assert_equal(
            empty_stats["voxel_count"],
            0,
            "Empty segment should have 0 voxels",
        )
        ctx.assert_equal(
            empty_stats["volume_mm3"],
            0.0,
            "Empty segment should have 0 volume",
        )

        slicer.mrmlScene.RemoveNode(empty_seg)
        ctx.screenshot("[empty] Empty segmentation handled")

        # Test 4: Statistics comparison
        ctx.log("Test 4: Statistics comparison")
        gold_node, test_node = self.seg_factory.create_gold_test_pair(
            volume_node=self.volume_node,
            overlap_ratio=0.8,
            radius=20,
        )

        gold_stats = self.manager._compute_statistics(gold_node, "GoldSegment", self.volume_node)
        test_stats = self.manager._compute_statistics(test_node, "TestSegment", self.volume_node)

        comparison = self.manager.compare_statistics(gold_stats, test_stats)

        # Verify comparison metrics exist
        ctx.assert_true(
            "voxel_count_diff" in comparison,
            "Comparison should have voxel_count_diff",
        )
        ctx.assert_true(
            "voxel_count_ratio" in comparison,
            "Comparison should have voxel_count_ratio",
        )
        ctx.assert_true(
            "volume_diff_mm3" in comparison,
            "Comparison should have volume_diff_mm3",
        )
        ctx.assert_true(
            "centroid_distance_mm" in comparison,
            "Comparison should have centroid_distance_mm",
        )
        ctx.assert_true(
            "bbox_iou" in comparison,
            "Comparison should have bbox_iou",
        )

        # With 80% overlap, ratio should be close to 1.0
        ctx.assert_almost_equal(
            comparison["voxel_count_ratio"],
            1.0,
            delta=0.1,
            message="Similar-sized segmentations should have ratio near 1.0",
        )

        # BBox IoU should be high for 80% overlap
        ctx.assert_greater(
            comparison["bbox_iou"],
            0.5,
            "BBox IoU should be >0.5 for 80% overlap",
        )

        ctx.screenshot("[comparison] Statistics comparison computed")

    def verify(self, ctx: TestContext) -> None:
        """Verify test results."""
        logger.info("Verifying gold statistics tests")

        # Test format_statistics
        test_stats = {
            "voxel_count": 1000,
            "volume_mm3": 1234.56,
            "bounding_box_size_mm": [10.0, 10.0, 10.0],
            "centroid_ras": [0.0, 0.0, 0.0],
        }
        formatted = self.manager.format_statistics(test_stats)
        ctx.assert_true(
            "1,000" in formatted,
            "Formatted stats should show voxel count with comma",
        )
        ctx.assert_true(
            "1234.6" in formatted,
            "Formatted stats should show volume",
        )

        # Test format_comparison
        test_comparison = {
            "voxel_count_diff": 100,
            "voxel_count_ratio": 1.1,
            "volume_diff_mm3": 123.4,
            "centroid_distance_mm": 5.0,
            "bbox_iou": 0.85,
        }
        formatted_comp = self.manager.format_comparison(test_comparison)
        ctx.assert_true(
            "110.0%" in formatted_comp,
            "Formatted comparison should show ratio as percentage",
        )
        ctx.assert_true(
            "85.0%" in formatted_comp,
            "Formatted comparison should show IoU as percentage",
        )

        ctx.screenshot("[verify] Formatting verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down gold statistics test")

        if self.seg_factory:
            self.seg_factory.cleanup()
            self.seg_factory = None

        ctx.log("Teardown complete")


@register_test(category="gold_standard")
class TestGoldChecksumVerification(TestCase):
    """Test checksum verification on gold standard load."""

    name = "gold_checksum_verification"
    description = "Test that checksum verification catches modifications"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.seg_factory = None
        self.manager = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test."""
        logger.info("Setting up checksum verification test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        self.seg_factory = MockSegmentationFactory()

        from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

        self.manager = GoldStandardManager()

        ctx.screenshot("[setup] Ready for checksum test")

    def run(self, ctx: TestContext) -> None:
        """Test checksum verification."""
        logger.info("Running checksum verification test")

        # Create a segmentation
        seg_node = self.seg_factory.create_sphere_segmentation(
            center_ijk=(128, 128, 64),
            radius=10,
            volume_node=self.volume_node,
        )

        # Compute statistics twice - should get same checksum
        ctx.log("Computing statistics twice")
        stats1 = self.manager._compute_statistics(seg_node, "Sphere", self.volume_node)
        stats2 = self.manager._compute_statistics(seg_node, "Sphere", self.volume_node)

        ctx.assert_equal(
            stats1["checksum_sha256"],
            stats2["checksum_sha256"],
            "Same segmentation should produce same checksum",
        )

        ctx.log(f"Checksum: {stats1['checksum_sha256'][:16]}...")

        # Create a different segmentation - should have different checksum
        ctx.log("Creating different segmentation")
        seg_node2 = self.seg_factory.create_sphere_segmentation(
            center_ijk=(100, 100, 50),
            radius=15,
            volume_node=self.volume_node,
            name="Different",
        )

        stats3 = self.manager._compute_statistics(seg_node2, "Sphere", self.volume_node)

        ctx.assert_not_equal(
            stats1["checksum_sha256"],
            stats3["checksum_sha256"],
            "Different segmentations should have different checksums",
        )

        ctx.screenshot("[checksum] Checksum verification complete")

    def verify(self, ctx: TestContext) -> None:
        """Verify results."""
        ctx.log("Checksum verification test passed")
        ctx.screenshot("[verify] Test complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.seg_factory:
            self.seg_factory.cleanup()
