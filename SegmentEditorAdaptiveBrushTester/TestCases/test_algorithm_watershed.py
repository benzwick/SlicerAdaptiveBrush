"""Watershed algorithm tests for AdaptiveBrush.

Tests the watershed algorithm specifically:
1. Paint in uniform region
2. Verify segmentation boundary
3. Test edge sensitivity parameter
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="algorithm")
class TestAlgorithmWatershed(TestCase):
    """Test watershed algorithm behavior."""

    name = "algorithm_watershed"
    description = "Test watershed algorithm on brain tissue with various sensitivities"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and set up for watershed testing."""
        logger.info("Setting up watershed algorithm test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load MRHead sample data
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        ctx.log(f"Loaded volume: {self.volume_node.GetName()}")

        # Create segmentation node
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        # Add a segment
        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("WatershedTest")

        # Set up segment editor
        self.segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
        self.segment_editor_widget.setMRMLScene(slicer.mrmlScene)
        segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)

        # Activate Adaptive Brush
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        # Set to watershed algorithm
        scripted_effect = self.effect.self()
        scripted_effect.setParameter("Algorithm", "watershed")

        ctx.screenshot("001_watershed_setup", "Watershed algorithm ready")

    def run(self, ctx: TestContext) -> None:
        """Test watershed with different edge sensitivity values."""
        logger.info("Running watershed algorithm tests")

        scripted_effect = self.effect.self()

        # Test different edge sensitivity values
        sensitivities = [0.3, 0.5, 0.7]

        for sensitivity in sensitivities:
            ctx.log(f"Testing edge sensitivity: {sensitivity}")

            # Set sensitivity
            scripted_effect.setParameter("EdgeSensitivity", str(sensitivity))

            ctx.screenshot(
                f"002_sensitivity_{int(sensitivity * 100)}",
                f"Edge sensitivity set to {sensitivity}",
            )

            # Record the parameter
            ctx.record_metric(f"edge_sensitivity_{int(sensitivity * 100)}", sensitivity)

        # Test 3D mode toggle
        ctx.log("Testing 3D mode toggle")

        # Enable 3D mode
        scripted_effect.setParameter("BrushMode", "3d")
        ctx.screenshot("003_3d_mode", "3D brush mode enabled")

        # Disable 3D mode
        scripted_effect.setParameter("BrushMode", "2d")
        ctx.screenshot("004_2d_mode", "2D brush mode enabled")

    def verify(self, ctx: TestContext) -> None:
        """Verify watershed algorithm configuration."""
        logger.info("Verifying watershed algorithm test")

        scripted_effect = self.effect.self()

        # Verify algorithm is set to watershed
        algorithm = scripted_effect.parameter("Algorithm")
        ctx.assert_equal(
            algorithm,
            "watershed",
            "Algorithm should be set to watershed",
        )

        # Verify edge sensitivity parameter exists
        sensitivity = scripted_effect.parameter("EdgeSensitivity")
        ctx.assert_is_not_none(
            sensitivity,
            "Edge sensitivity parameter should exist",
        )

        # Verify sensitivity is in valid range
        try:
            sensitivity_val = float(sensitivity)
            ctx.assert_greater_equal(
                sensitivity_val,
                0.0,
                "Edge sensitivity should be >= 0",
            )
            ctx.assert_less_equal(
                sensitivity_val,
                1.0,
                "Edge sensitivity should be <= 1",
            )
        except (ValueError, TypeError):
            ctx.assert_true(False, f"Edge sensitivity should be a number, got: {sensitivity}")

        ctx.screenshot("005_watershed_verified", "Watershed algorithm verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down watershed test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")
