"""Basic workflow test for AdaptiveBrush.

Tests the basic workflow:
1. Load sample data (MRBrainTumour1 for better contrast)
2. Create segmentation
3. Activate Adaptive Brush
4. Paint with different algorithms
5. Verify non-empty segmentation
"""

from __future__ import annotations

import logging

import slicer
import vtk
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="workflow")
class TestWorkflowBasic(TestCase):
    """Test basic AdaptiveBrush workflow."""

    name = "workflow_basic"
    description = "Basic workflow: load data, create segmentation, paint, verify"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and create segmentation."""
        logger.info("Setting up basic workflow test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load MRBrainTumour1 sample data (has tumor with good contrast)
        import SampleData

        ctx.log("Loading MRBrainTumor1 sample data")
        self.volume_node = SampleData.downloadSample("MRBrainTumor1")

        if self.volume_node is None:
            raise RuntimeError("Failed to load MRBrainTumor1 sample data")

        ctx.log(f"Loaded volume: {self.volume_node.GetName()}")

        # Create segmentation node
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        # Add a segment
        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("TestSegment")

        ctx.log(f"Created segmentation with segment: {self.segment_id}")

        # Switch to Segment Editor module to see GUI
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        # Use the actual module's segment editor widget (visible in GUI)
        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        ctx.screenshot("[setup] MRBrainTumor1 loaded, segmentation created")

    def run(self, ctx: TestContext) -> None:
        """Activate Adaptive Brush and paint with each algorithm."""
        logger.info("Running basic workflow test")

        # Activate Adaptive Brush effect
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        effect = self.segment_editor_widget.activeEffect()

        if effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        ctx.log("Activated Adaptive Brush effect")

        # Get the scripted effect
        scripted_effect = effect.self()

        # Get the Red slice widget for painting
        layoutManager = slicer.app.layoutManager()
        redWidget = layoutManager.sliceWidget("Red")
        redLogic = redWidget.sliceLogic()

        # Paint location in RAS (near tumor in MRBrainTumor1)
        paint_ras = [5.6, -29.5, 28.4]

        # Navigate Red slice to this location
        redLogic.SetSliceOffset(paint_ras[2])
        slicer.app.processEvents()

        ctx.screenshot("[brush] Adaptive Brush activated")

        # Convert RAS to screen XY for the Red slice view
        xy = self._rasToXy(paint_ras, redWidget)
        if not xy:
            ctx.log("Could not convert RAS to screen coordinates")
            return

        # Test multiple algorithms
        algorithms_to_test = ["watershed", "connected_threshold", "threshold_brush"]

        for algo in algorithms_to_test:
            ctx.log(f"\nTesting algorithm: {algo}")

            # Select algorithm using the combo box (like a user would)
            combo = scripted_effect.algorithmCombo
            idx = combo.findData(algo)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            slicer.app.processEvents()

            ctx.screenshot(f"[{algo}] Before painting")

            ctx.log(f"Painting at RAS {paint_ras} (screen XY {xy})")

            with ctx.timing(f"paint_{algo}"):
                # Simulate brush stroke
                scripted_effect.scriptedEffect.saveStateForUndo()
                scripted_effect.isDrawing = True
                scripted_effect._currentStrokeEraseMode = False
                scripted_effect.processPoint(xy, redWidget)
                scripted_effect.isDrawing = False

            slicer.app.processEvents()
            ctx.screenshot(f"[{algo}] After paint stroke")

        ctx.screenshot("[complete] All algorithms tested")

    def _rasToXy(self, ras, sliceWidget):
        """Convert RAS coordinates to screen XY for a slice widget."""
        sliceLogic = sliceWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()

        # Get XY to RAS matrix and invert it
        xyToRas = sliceNode.GetXYToRAS()
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xyToRas, rasToXy)

        rasPoint = [ras[0], ras[1], ras[2], 1]
        xyPoint = [0, 0, 0, 1]
        rasToXy.MultiplyPoint(rasPoint, xyPoint)

        return (int(xyPoint[0]), int(xyPoint[1]))

    def verify(self, ctx: TestContext) -> None:
        """Verify the workflow completed successfully."""
        logger.info("Verifying basic workflow test")

        # Verify effect is still active
        effect = self.segment_editor_widget.activeEffect()
        ctx.assert_is_not_none(effect, "Adaptive Brush effect should still be active")

        # Verify effect name
        ctx.assert_equal(
            effect.name,
            "Adaptive Brush",
            "Active effect should be Adaptive Brush",
        )

        # Verify segmentation node exists
        ctx.assert_is_not_none(
            self.segmentation_node,
            "Segmentation node should exist",
        )

        # Verify segment exists and has voxels
        segmentation = self.segmentation_node.GetSegmentation()
        segment = segmentation.GetSegment(self.segment_id)
        ctx.assert_is_not_none(segment, "Test segment should exist")

        # Check if segment has any voxels (painting worked)
        import numpy as np

        labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, self.segment_id, self.volume_node
        )
        voxel_count = int(np.sum(labelmap > 0))
        ctx.metric("segmented_voxels", voxel_count)
        ctx.log(f"Segmented {voxel_count} voxels")

        # We expect some segmentation if painting worked
        ctx.assert_greater(voxel_count, 0, "Should have segmented some voxels")

        ctx.screenshot(f"[verify] {voxel_count:,} voxels segmented")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down basic workflow test")

        # Clean up segment editor widget
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        # Don't clear the scene - leave it for manual inspection
        ctx.log("Teardown complete. Scene preserved for inspection.")
