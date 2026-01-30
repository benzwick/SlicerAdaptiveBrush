"""Documentation screenshots for workflow tutorials.

Captures step-by-step screenshots for tutorial documentation:
- Getting started workflow
- Brain tumor segmentation tutorial
- Multi-algorithm comparison workflow

These screenshots are tagged for documentation extraction.
"""

from __future__ import annotations

import logging

import slicer
import vtk
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="docs")
class TestDocsWorkflowGettingStarted(TestCase):
    """Generate documentation screenshots for Getting Started tutorial."""

    name = "docs_workflow_getting_started"
    description = "Capture step-by-step screenshots for Getting Started tutorial"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Prepare for getting started workflow documentation."""
        logger.info("Setting up Getting Started workflow documentation")

        # Start with clean scene
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

        ctx.screenshot(
            "Step 1: Start with empty Slicer",
            doc_tags=["getting_started", "empty_scene"],
        )

    def run(self, ctx: TestContext) -> None:
        """Capture getting started workflow steps."""
        logger.info("Running Getting Started workflow documentation")

        # Step 2: Load sample data
        ctx.log("Step 2: Loading sample data")
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        slicer.app.processEvents()

        ctx.screenshot(
            "Step 2: Load sample data (MRHead)",
            doc_tags=["getting_started", "load_data"],
        )

        # Step 3: Switch to Segment Editor
        ctx.log("Step 3: Opening Segment Editor")
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        ctx.screenshot(
            "Step 3: Open Segment Editor module",
            doc_tags=["getting_started", "segment_editor"],
        )

        # Step 4: Create segmentation
        ctx.log("Step 4: Creating segmentation")
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("Brain")

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        ctx.screenshot(
            "Step 4: Create new segmentation and segment",
            doc_tags=["getting_started", "create_segment"],
        )

        # Step 5: Select Adaptive Brush
        ctx.log("Step 5: Selecting Adaptive Brush effect")
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()
        slicer.app.processEvents()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        ctx.screenshot(
            "Step 5: Select Adaptive Brush effect",
            doc_tags=["getting_started", "select_effect"],
        )

        # Step 6: Configure brush
        ctx.log("Step 6: Configuring brush parameters")
        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 15.0

        # Select geodesic algorithm (good default)
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("geodesic")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        ctx.screenshot(
            "Step 6: Configure brush radius and algorithm",
            doc_tags=["getting_started", "configure"],
        )

        # Step 7: Position cursor (show brush preview)
        ctx.log("Step 7: Positioning cursor for painting")

        # Get Red slice widget
        layoutManager = slicer.app.layoutManager()
        red_widget = layoutManager.sliceWidget("Red")
        red_logic = red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        # Show brush at center
        center_xy = self._get_slice_center_xy(red_widget)
        scripted_effect._updateBrushPreview(center_xy, red_widget, eraseMode=False)
        red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot(
            "Step 7: Position cursor over target tissue",
            doc_tags=["getting_started", "position_cursor"],
        )

        # Note about painting
        ctx.log("Step 8: Click to paint (demonstration note)")
        ctx.screenshot(
            "Step 8: Click to paint - adaptive segmentation follows tissue boundaries",
            doc_tags=["getting_started", "paint"],
        )

    def _get_slice_center_xy(self, slice_widget):
        """Get XY coordinates for the center of a slice view."""
        slice_logic = slice_widget.sliceLogic()
        slice_node = slice_logic.GetSliceNode()

        slice_to_ras = slice_node.GetSliceToRAS()
        center_ras = [
            slice_to_ras.GetElement(0, 3),
            slice_to_ras.GetElement(1, 3),
            slice_to_ras.GetElement(2, 3),
        ]

        xy_to_ras = slice_node.GetXYToRAS()
        ras_to_xy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)

        ras_point = [center_ras[0], center_ras[1], center_ras[2], 1]
        xy_point = [0, 0, 0, 1]
        ras_to_xy.MultiplyPoint(ras_point, xy_point)

        return (int(xy_point[0]), int(xy_point[1]))

    def verify(self, ctx: TestContext) -> None:
        """Verify workflow steps captured."""
        logger.info("Verifying Getting Started workflow documentation")

        ctx.assert_greater_equal(
            len(ctx.screenshots),
            8,
            "Should have captured at least 8 workflow steps",
        )

        ctx.screenshot(
            "Getting Started workflow complete",
            doc_tags=["getting_started", "complete"],
        )

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down Getting Started workflow test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")


@register_test(category="docs")
class TestDocsWorkflowTumorSegmentation(TestCase):
    """Generate documentation screenshots for brain tumor segmentation tutorial."""

    name = "docs_workflow_tumor"
    description = "Capture screenshots for brain tumor segmentation tutorial"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Load brain tumor data."""
        logger.info("Setting up tumor segmentation workflow")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRBrainTumor1")
        if self.volume_node is None:
            # Fall back to MRHead if tumor sample not available
            self.volume_node = SampleData.downloadSample("MRHead")

        slicer.app.processEvents()

        ctx.screenshot(
            "Tumor workflow: Data loaded",
            doc_tags=["tumor", "data_loaded"],
        )

    def run(self, ctx: TestContext) -> None:
        """Capture tumor segmentation workflow steps."""
        logger.info("Running tumor segmentation workflow")

        # Create segmentation
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("Tumor")

        # Setup segment editor
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        ctx.screenshot(
            "Tumor workflow: Segment Editor ready",
            doc_tags=["tumor", "setup"],
        )

        # Activate Adaptive Brush
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()
        slicer.app.processEvents()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        # Configure for tumor segmentation
        scripted_effect = self.effect.self()

        # Use watershed for tumor (good at finding boundaries)
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("watershed")
        if idx >= 0:
            combo.setCurrentIndex(idx)

        # Higher sensitivity for tumor boundaries
        scripted_effect.sensitivitySlider.value = 60
        scripted_effect.radiusSlider.value = 10.0
        slicer.app.processEvents()

        ctx.screenshot(
            "Tumor workflow: Watershed algorithm with high sensitivity",
            doc_tags=["tumor", "configure"],
        )

        # Navigate to tumor region (if using MRBrainTumor1)
        layoutManager = slicer.app.layoutManager()
        red_widget = layoutManager.sliceWidget("Red")
        slicer.app.processEvents()

        # Show brush preview
        center_xy = self._get_slice_center_xy(red_widget)
        scripted_effect._updateBrushPreview(center_xy, red_widget, eraseMode=False)
        red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot(
            "Tumor workflow: Ready to paint tumor region",
            doc_tags=["tumor", "ready"],
        )

    def _get_slice_center_xy(self, slice_widget):
        """Get XY coordinates for the center of a slice view."""
        slice_logic = slice_widget.sliceLogic()
        slice_node = slice_logic.GetSliceNode()

        slice_to_ras = slice_node.GetSliceToRAS()
        center_ras = [
            slice_to_ras.GetElement(0, 3),
            slice_to_ras.GetElement(1, 3),
            slice_to_ras.GetElement(2, 3),
        ]

        xy_to_ras = slice_node.GetXYToRAS()
        ras_to_xy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)

        ras_point = [center_ras[0], center_ras[1], center_ras[2], 1]
        xy_point = [0, 0, 0, 1]
        ras_to_xy.MultiplyPoint(ras_point, xy_point)

        return (int(xy_point[0]), int(xy_point[1]))

    def verify(self, ctx: TestContext) -> None:
        """Verify workflow steps captured."""
        logger.info("Verifying tumor workflow documentation")

        ctx.assert_greater_equal(
            len(ctx.screenshots),
            4,
            "Should have captured tumor workflow steps",
        )

        ctx.screenshot(
            "Tumor workflow complete",
            doc_tags=["tumor", "complete"],
        )

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
