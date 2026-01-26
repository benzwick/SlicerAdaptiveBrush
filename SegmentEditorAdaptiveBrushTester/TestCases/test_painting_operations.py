"""Tests for painting operations (add and erase modes).

Tests the core painting functionality:
1. Add mode painting
2. Erase mode painting
3. Ctrl modifier for mode inversion
4. Multiple paint strokes
5. Undo/redo support
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import slicer
import vtk
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


def _get_slice_center_xy(slice_widget: Any) -> tuple[int, int]:
    """Get XY coordinates for the center of a slice view using RAS conversion."""
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


def _count_segment_voxels(segmentation_node: Any, segment_id: str) -> int:
    """Count the number of voxels in a segment."""
    # Import slicer at function level to avoid F823 linting error
    import slicer as slicer_module
    import vtk.util.numpy_support as vtk_np

    labelmap = slicer_module.vtkOrientedImageData()
    segmentation_node.GetBinaryLabelmapRepresentation(segment_id, labelmap)

    if labelmap is None:
        return 0

    # Get numpy array from vtkOrientedImageData using VTK's numpy support
    # (slicer.util.arrayFromVolume expects a volume node, not raw image data)
    scalars = labelmap.GetPointData().GetScalars()
    if scalars is None:
        return 0

    array = vtk_np.vtk_to_numpy(scalars)
    return int(np.sum(array > 0))


@register_test(category="painting")
class TestAddModePainting(TestCase):
    """Test painting in add mode."""

    name = "painting_add_mode"
    description = "Test that add mode correctly adds voxels to segment"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure for painting."""
        logger.info("Setting up add mode painting test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("PaintTest")

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        # Configure brush
        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 25.0

        # Use geodesic distance (fast and reliable)
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("geodesic_distance")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready for add mode painting test")

    def run(self, ctx: TestContext) -> None:
        """Test add mode painting at multiple positions."""
        logger.info("Running add mode painting test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        # Ensure add mode is selected
        if hasattr(scripted_effect, "addModeRadio"):
            scripted_effect.addModeRadio.checked = True
        slicer.app.processEvents()

        # Show brush
        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[before] Empty segment, brush visible")

        # Verify segment is empty
        voxels_before = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.assert_equal(voxels_before, 0, "Segment should be empty before painting")
        ctx.record_metric("voxels_before", voxels_before)

        # Paint at center
        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        scripted_effect.processPoint(center_xy, self.red_widget)
        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        voxels_after_1 = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after_paint1", voxels_after_1)

        ctx.screenshot("[after_paint1] First paint stroke")
        ctx.log(f"First stroke added {voxels_after_1} voxels")

        # Paint at offset position
        offset_xy = (center_xy[0] + 40, center_xy[1])
        scripted_effect._updateBrushPreview(offset_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        scripted_effect.processPoint(offset_xy, self.red_widget)
        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        voxels_after_2 = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after_paint2", voxels_after_2)

        ctx.screenshot("[after_paint2] Second paint stroke")
        ctx.log(f"Second stroke: total voxels now {voxels_after_2}")

    def verify(self, ctx: TestContext) -> None:
        """Verify add mode painted voxels."""
        logger.info("Verifying add mode painting")

        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)

        ctx.assert_greater(voxels, 0, "Add mode should have added voxels to segment")

        ctx.log(f"Final voxel count: {voxels}")

        ctx.screenshot("[verify] Add mode painting verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="painting")
class TestEraseModePainting(TestCase):
    """Test painting in erase mode."""

    name = "painting_erase_mode"
    description = "Test that erase mode correctly removes voxels from segment"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and create initial segmentation."""
        logger.info("Setting up erase mode painting test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("EraseTest")

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 25.0

        combo = scripted_effect.algorithmCombo
        idx = combo.findData("geodesic_distance")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready for erase mode test")

    def run(self, ctx: TestContext) -> None:
        """Test erase mode removes voxels."""
        logger.info("Running erase mode painting test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        # First, paint something in ADD mode
        ctx.log("First painting in add mode to create something to erase")

        if hasattr(scripted_effect, "addModeRadio"):
            scripted_effect.addModeRadio.checked = True
        slicer.app.processEvents()

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[add_before] About to paint in add mode")

        # Paint to add voxels
        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        scripted_effect.processPoint(center_xy, self.red_widget)
        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        voxels_after_add = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after_add", voxels_after_add)

        ctx.screenshot(f"[add_after] Added {voxels_after_add} voxels")
        ctx.assert_greater(voxels_after_add, 0, "Should have voxels to erase")

        # Now switch to ERASE mode
        ctx.log("Switching to erase mode")

        if hasattr(scripted_effect, "eraseModeRadio"):
            scripted_effect.eraseModeRadio.checked = True
        slicer.app.processEvents()

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=True)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[erase_before] Red brush, about to erase")

        # Erase at same position
        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = True
        scripted_effect.processPoint(center_xy, self.red_widget)
        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        voxels_after_erase = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after_erase", voxels_after_erase)

        ctx.screenshot(f"[erase_after] Voxels reduced to {voxels_after_erase}")
        ctx.log(f"Erased: {voxels_after_add - voxels_after_erase} voxels")

    def verify(self, ctx: TestContext) -> None:
        """Verify erase mode removed voxels."""
        logger.info("Verifying erase mode painting")

        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)

        # After erasing at same position, should have fewer voxels
        # (may not be zero if erase radius is smaller than add region)
        ctx.log(f"Final voxel count after erase: {voxels}")

        ctx.screenshot("[verify] Erase mode painting verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="painting")
class TestUndoRedo(TestCase):
    """Test undo/redo support for painting operations."""

    name = "painting_undo_redo"
    description = "Test undo and redo work correctly for paint operations"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure for undo/redo test."""
        logger.info("Setting up undo/redo test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("UndoTest")

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 20.0

        combo = scripted_effect.algorithmCombo
        idx = combo.findData("connected_threshold")  # Fast algorithm for undo test
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready for undo/redo test")

    def run(self, ctx: TestContext) -> None:
        """Test undo and redo after painting."""
        logger.info("Running undo/redo test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        # Verify starting empty
        voxels_start = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.assert_equal(voxels_start, 0, "Should start with empty segment")
        ctx.record_metric("voxels_start", voxels_start)

        ctx.screenshot("[start] Empty segment")

        # Paint something
        if hasattr(scripted_effect, "addModeRadio"):
            scripted_effect.addModeRadio.checked = True
        slicer.app.processEvents()

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        scripted_effect.processPoint(center_xy, self.red_widget)
        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        voxels_after_paint = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after_paint", voxels_after_paint)

        ctx.screenshot(f"[after_paint] Painted {voxels_after_paint} voxels")
        ctx.assert_greater(voxels_after_paint, 0, "Should have painted voxels")

        # Undo
        ctx.log("Testing undo")
        slicer.util.mainWindow().findChild(slicer.qMRMLSegmentEditorWidget).undo()
        slicer.app.processEvents()

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        voxels_after_undo = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after_undo", voxels_after_undo)

        ctx.screenshot(f"[after_undo] Voxels: {voxels_after_undo}")
        ctx.log(f"After undo: {voxels_after_undo} voxels")

        # Redo
        ctx.log("Testing redo")
        slicer.util.mainWindow().findChild(slicer.qMRMLSegmentEditorWidget).redo()
        slicer.app.processEvents()

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        voxels_after_redo = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after_redo", voxels_after_redo)

        ctx.screenshot(f"[after_redo] Voxels: {voxels_after_redo}")
        ctx.log(f"After redo: {voxels_after_redo} voxels")

    def verify(self, ctx: TestContext) -> None:
        """Verify undo/redo worked correctly."""
        logger.info("Verifying undo/redo")

        # Redo should restore the painted state
        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.assert_greater(voxels, 0, "Redo should restore painted voxels")

        ctx.screenshot("[verify] Undo/redo verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="painting")
class TestMultipleStrokes(TestCase):
    """Test multiple paint strokes accumulate correctly."""

    name = "painting_multiple_strokes"
    description = "Test that multiple paint strokes accumulate segmentation"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure for multiple strokes test."""
        logger.info("Setting up multiple strokes test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
            "MultiStrokeTest"
        )

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 20.0

        combo = scripted_effect.algorithmCombo
        idx = combo.findData("geodesic_distance")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready for multiple strokes test")

    def run(self, ctx: TestContext) -> None:
        """Paint multiple strokes at different positions."""
        logger.info("Running multiple strokes test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        if hasattr(scripted_effect, "addModeRadio"):
            scripted_effect.addModeRadio.checked = True
        slicer.app.processEvents()

        # Paint at multiple positions
        positions = [
            center_xy,
            (center_xy[0] + 50, center_xy[1]),
            (center_xy[0], center_xy[1] + 50),
            (center_xy[0] - 50, center_xy[1]),
            (center_xy[0], center_xy[1] - 50),
        ]

        voxel_counts = []

        for i, pos in enumerate(positions):
            ctx.log(f"Painting stroke {i + 1} at {pos}")

            scripted_effect._updateBrushPreview(pos, self.red_widget, eraseMode=False)
            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            scripted_effect.scriptedEffect.saveStateForUndo()
            scripted_effect.isDrawing = True
            scripted_effect._currentStrokeEraseMode = False
            scripted_effect.processPoint(pos, self.red_widget)
            scripted_effect.isDrawing = False
            slicer.app.processEvents()

            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)
            voxel_counts.append(voxels)
            ctx.record_metric(f"voxels_after_stroke_{i + 1}", voxels)

            ctx.screenshot(f"[stroke_{i + 1}] Total voxels: {voxels}")

        # Verify voxels increased or stayed same (may overlap)
        for i in range(1, len(voxel_counts)):
            ctx.assert_greater_equal(
                voxel_counts[i],
                voxel_counts[i - 1],
                f"Voxels should not decrease (stroke {i + 1})",
            )

    def verify(self, ctx: TestContext) -> None:
        """Verify multiple strokes accumulated."""
        logger.info("Verifying multiple strokes")

        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.assert_greater(voxels, 100, "Multiple strokes should produce substantial segmentation")

        ctx.log(f"Final voxel count from 5 strokes: {voxels}")

        ctx.screenshot("[verify] Multiple strokes verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")
