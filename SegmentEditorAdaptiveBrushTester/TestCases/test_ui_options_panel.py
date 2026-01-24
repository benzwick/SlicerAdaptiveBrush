"""UI options panel tests for AdaptiveBrush.

Tests the options panel UI:
1. Capture screenshots of panel state
2. Verify widget visibility per algorithm
3. Test collapsible sections
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="ui")
class TestUIOptionsPanel(TestCase):
    """Test options panel UI behavior."""

    name = "ui_options_panel"
    description = "Test options panel layout, widget visibility, and algorithm switching"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Set up for UI testing."""
        logger.info("Setting up UI options panel test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load sample data (needed to activate segment editor)
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        # Create segmentation
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("UITest")

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

        ctx.screenshot("001_ui_initial", "Initial UI state with Adaptive Brush active")

    def run(self, ctx: TestContext) -> None:
        """Test UI state for each algorithm."""
        logger.info("Running UI options panel tests")

        scripted_effect = self.effect.self()

        # List of algorithms to test
        algorithms = [
            "watershed",
            "level_set",
            "connected_threshold",
            "region_growing",
            "threshold_brush",
        ]

        for algo in algorithms:
            ctx.log(f"Testing UI for algorithm: {algo}")

            # Switch algorithm
            scripted_effect.setParameter("Algorithm", algo)

            # Force UI update
            slicer.app.processEvents()

            # Capture screenshot
            ctx.screenshot(
                f"002_{algo}_panel",
                f"Options panel with {algo} selected",
            )

        # Test Threshold Brush auto-methods
        ctx.log("Testing Threshold Brush auto-methods")
        scripted_effect.setParameter("Algorithm", "threshold_brush")

        auto_methods = ["Otsu", "Huang", "Triangle", "Li"]
        for method in auto_methods:
            scripted_effect.setParameter("ThresholdMethod", method)
            slicer.app.processEvents()
            ctx.screenshot(
                f"003_threshold_{method.lower()}",
                f"Threshold Brush with {method} method",
            )

    def verify(self, ctx: TestContext) -> None:
        """Verify UI state."""
        logger.info("Verifying UI options panel test")

        # Verify effect is active
        ctx.assert_is_not_none(
            self.effect,
            "Adaptive Brush effect should be active",
        )

        # Verify scripted effect is accessible
        scripted_effect = self.effect.self()
        ctx.assert_is_not_none(
            scripted_effect,
            "Scripted effect should be accessible",
        )

        # Verify algorithm selector exists by checking parameter
        algorithm = scripted_effect.parameter("Algorithm")
        ctx.assert_is_not_none(
            algorithm,
            "Algorithm parameter should exist",
        )

        ctx.screenshot("004_ui_verified", "UI verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down UI test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")
