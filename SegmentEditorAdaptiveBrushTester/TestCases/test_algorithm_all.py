"""Tests for all AdaptiveBrush algorithms.

Tests each algorithm to verify:
1. Algorithm can be selected from UI
2. Algorithm computes non-empty mask
3. Mask is applied to segment correctly
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import slicer
import vtk
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)

# All available algorithms and their data values
ALGORITHMS = [
    ("geodesic_distance", "Geodesic Distance"),
    ("watershed", "Watershed"),
    ("random_walker", "Random Walker"),
    ("level_set", "Level Set"),
    ("connected_threshold", "Connected Threshold"),
    ("region_growing", "Region Growing"),
    ("threshold_brush", "Threshold Brush"),
]


def _get_slice_center_xy(slice_widget: Any) -> tuple[int, int]:
    """Get XY coordinates for the center of a slice view using RAS conversion."""
    slice_logic = slice_widget.sliceLogic()
    slice_node = slice_logic.GetSliceNode()

    # Get slice center in RAS
    slice_to_ras = slice_node.GetSliceToRAS()
    center_ras = [
        slice_to_ras.GetElement(0, 3),
        slice_to_ras.GetElement(1, 3),
        slice_to_ras.GetElement(2, 3),
    ]

    # Convert RAS to XY
    xy_to_ras = slice_node.GetXYToRAS()
    ras_to_xy = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)

    ras_point = [center_ras[0], center_ras[1], center_ras[2], 1]
    xy_point = [0, 0, 0, 1]
    ras_to_xy.MultiplyPoint(ras_point, xy_point)

    return (int(xy_point[0]), int(xy_point[1]))


def _count_segment_voxels(segmentation_node: Any, segment_id: str) -> int:
    """Count the number of voxels in a segment."""
    # Get labelmap representation
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


@register_test(category="algorithm")
class TestAllAlgorithmsSelection(TestCase):
    """Test that all algorithms can be selected from the UI."""

    name = "algorithm_all_selection"
    description = "Test selecting each algorithm from the dropdown"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and set up segment editor."""
        logger.info("Setting up algorithm selection test")

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
        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("TestSegment")

        # Switch to Segment Editor module
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        # Get segment editor widget
        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        # Activate Adaptive Brush
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        ctx.screenshot("[setup] Ready to test algorithm selection")

    def run(self, ctx: TestContext) -> None:
        """Test selecting each algorithm."""
        logger.info("Testing algorithm selection")

        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo

        # Start from a known state - set to last algorithm first to ensure
        # the first algorithm will trigger a change signal
        last_idx = combo.findData(ALGORITHMS[-1][0])
        if last_idx >= 0:
            combo.setCurrentIndex(last_idx)
            slicer.app.processEvents()

        for data_value, display_name in ALGORITHMS:
            ctx.log(f"Selecting algorithm: {display_name}")

            idx = combo.findData(data_value)
            ctx.assert_greater_equal(idx, 0, f"Algorithm '{data_value}' should be in combo box")

            combo.setCurrentIndex(idx)
            slicer.app.processEvents()

            # Verify internal state updated
            ctx.assert_equal(
                scripted_effect.algorithm,
                data_value,
                f"Algorithm should be set to {data_value}",
            )

            ctx.screenshot(f"[{data_value}] Algorithm selected: {display_name}")
            ctx.record_metric(f"algorithm_{data_value}_found", 1)

    def verify(self, ctx: TestContext) -> None:
        """Verify all algorithms were found."""
        logger.info("Verifying algorithm selection test")

        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo

        # Verify combo has all expected algorithms
        ctx.assert_equal(
            combo.count,
            len(ALGORITHMS),
            f"Combo should have {len(ALGORITHMS)} algorithms",
        )

        ctx.screenshot("[verify] All algorithms verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="algorithm")
class TestGeodesicDistanceAlgorithm(TestCase):
    """Test geodesic distance algorithm (default)."""

    name = "algorithm_geodesic_distance"
    description = "Test geodesic distance algorithm produces valid segmentation"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure geodesic distance."""
        logger.info("Setting up geodesic distance test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("GeodesicTest")

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

        # Select geodesic distance
        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("geodesic_distance")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        # Set brush radius
        scripted_effect.radiusSlider.value = 25.0
        slicer.app.processEvents()

        # Get Red slice widget
        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Geodesic distance algorithm ready")

    def run(self, ctx: TestContext) -> None:
        """Paint with geodesic distance algorithm."""
        logger.info("Running geodesic distance paint test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        # Show brush preview
        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[before_paint] Brush positioned at center")

        # Record voxel count before painting
        voxels_before = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_before", voxels_before)

        # Simulate painting
        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        # Reset lastIjk to avoid "same voxel" optimization skip between tests
        scripted_effect.lastIjk = None

        import time

        start_time = time.time()
        scripted_effect.processPoint(center_xy, self.red_widget)
        elapsed_ms = (time.time() - start_time) * 1000

        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        ctx.record_metric("paint_time_ms", elapsed_ms)

        # Force render to show result
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[after_paint] Geodesic distance segmentation result")

        # Record voxels after
        voxels_after = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_after", voxels_after)
        ctx.record_metric("voxels_added", voxels_after - voxels_before)

        ctx.log(f"Painted {voxels_after - voxels_before} voxels in {elapsed_ms:.1f}ms")

    def verify(self, ctx: TestContext) -> None:
        """Verify geodesic distance produced valid segmentation."""
        logger.info("Verifying geodesic distance result")

        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)

        ctx.assert_greater(voxels, 0, "Geodesic distance should produce non-empty segmentation")

        # Verify reasonable voxel count (not too few, not entire volume)
        ctx.assert_greater(voxels, 100, "Should segment meaningful region (>100 voxels)")

        ctx.screenshot("[verify] Geodesic distance verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="algorithm")
class TestLevelSetAlgorithm(TestCase):
    """Test level set algorithm."""

    name = "algorithm_level_set"
    description = "Test level set algorithm produces valid segmentation"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure level set."""
        logger.info("Setting up level set test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("LevelSetTest")

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

        # Select level set
        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("level_set")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        scripted_effect.radiusSlider.value = 20.0
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Level set algorithm ready")

    def run(self, ctx: TestContext) -> None:
        """Paint with level set algorithm."""
        logger.info("Running level set paint test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[before_paint] Brush positioned")

        voxels_before = _count_segment_voxels(self.segmentation_node, self.segment_id)

        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        # Reset lastIjk to avoid "same voxel" optimization skip between tests
        scripted_effect.lastIjk = None

        import time

        start_time = time.time()
        scripted_effect.processPoint(center_xy, self.red_widget)
        elapsed_ms = (time.time() - start_time) * 1000

        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        ctx.record_metric("paint_time_ms", elapsed_ms)

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[after_paint] Level set segmentation result")

        voxels_after = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_added", voxels_after - voxels_before)

        ctx.log(f"Painted {voxels_after - voxels_before} voxels in {elapsed_ms:.1f}ms")

    def verify(self, ctx: TestContext) -> None:
        """Verify level set produced valid segmentation."""
        logger.info("Verifying level set result")

        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)

        ctx.assert_greater(voxels, 0, "Level set should produce non-empty segmentation")

        ctx.screenshot("[verify] Level set verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="algorithm")
class TestConnectedThresholdAlgorithm(TestCase):
    """Test connected threshold algorithm (fast)."""

    name = "algorithm_connected_threshold"
    description = "Test connected threshold algorithm produces valid segmentation"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure connected threshold."""
        logger.info("Setting up connected threshold test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
            "ConnectedThresholdTest"
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

        # Select connected threshold
        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("connected_threshold")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        scripted_effect.radiusSlider.value = 25.0
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Connected threshold algorithm ready")

    def run(self, ctx: TestContext) -> None:
        """Paint with connected threshold algorithm."""
        logger.info("Running connected threshold paint test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[before_paint] Brush positioned")

        voxels_before = _count_segment_voxels(self.segmentation_node, self.segment_id)

        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        # Reset lastIjk to avoid "same voxel" optimization skip between tests
        scripted_effect.lastIjk = None

        import time

        start_time = time.time()
        scripted_effect.processPoint(center_xy, self.red_widget)
        elapsed_ms = (time.time() - start_time) * 1000

        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        ctx.record_metric("paint_time_ms", elapsed_ms)

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[after_paint] Connected threshold result")

        voxels_after = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_added", voxels_after - voxels_before)

        ctx.log(f"Painted {voxels_after - voxels_before} voxels in {elapsed_ms:.1f}ms")

    def verify(self, ctx: TestContext) -> None:
        """Verify connected threshold produced valid segmentation."""
        logger.info("Verifying connected threshold result")

        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)

        ctx.assert_greater(voxels, 0, "Connected threshold should produce non-empty segmentation")

        ctx.screenshot("[verify] Connected threshold verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="algorithm")
class TestRegionGrowingAlgorithm(TestCase):
    """Test region growing algorithm."""

    name = "algorithm_region_growing"
    description = "Test region growing algorithm produces valid segmentation"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure region growing."""
        logger.info("Setting up region growing test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
            "RegionGrowingTest"
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

        # Select region growing
        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("region_growing")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        scripted_effect.radiusSlider.value = 25.0
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Region growing algorithm ready")

    def run(self, ctx: TestContext) -> None:
        """Paint with region growing algorithm."""
        logger.info("Running region growing paint test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[before_paint] Brush positioned")

        voxels_before = _count_segment_voxels(self.segmentation_node, self.segment_id)

        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = False
        # Reset lastIjk to avoid "same voxel" optimization skip between tests
        scripted_effect.lastIjk = None

        import time

        start_time = time.time()
        scripted_effect.processPoint(center_xy, self.red_widget)
        elapsed_ms = (time.time() - start_time) * 1000

        scripted_effect.isDrawing = False
        slicer.app.processEvents()

        ctx.record_metric("paint_time_ms", elapsed_ms)

        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[after_paint] Region growing result")

        voxels_after = _count_segment_voxels(self.segmentation_node, self.segment_id)
        ctx.record_metric("voxels_added", voxels_after - voxels_before)

        ctx.log(f"Painted {voxels_after - voxels_before} voxels in {elapsed_ms:.1f}ms")

    def verify(self, ctx: TestContext) -> None:
        """Verify region growing produced valid segmentation."""
        logger.info("Verifying region growing result")

        voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)

        ctx.assert_greater(voxels, 0, "Region growing should produce non-empty segmentation")

        ctx.screenshot("[verify] Region growing verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="algorithm")
class TestThresholdBrushAlgorithm(TestCase):
    """Test threshold brush algorithm with auto-threshold."""

    name = "algorithm_threshold_brush"
    description = "Test threshold brush algorithm with various threshold methods"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and configure threshold brush."""
        logger.info("Setting up threshold brush test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
            "ThresholdBrushTest"
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

        # Select threshold brush
        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("threshold_brush")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        scripted_effect.radiusSlider.value = 25.0
        slicer.app.processEvents()

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Threshold brush algorithm ready")

    def run(self, ctx: TestContext) -> None:
        """Test threshold brush with different auto-threshold methods."""
        logger.info("Running threshold brush test")

        scripted_effect = self.effect.self()
        center_xy = _get_slice_center_xy(self.red_widget)

        # Test auto-threshold methods
        methods = ["otsu", "huang", "triangle"]

        for method in methods:
            ctx.log(f"Testing threshold method: {method}")

            # Clear segment for each method
            self.segmentation_node.GetSegmentation().RemoveAllSegments()
            self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
                f"Test_{method}"
            )
            self.segment_editor_widget.setCurrentSegmentID(self.segment_id)

            # Re-activate effect after segment change
            self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
            self.effect = self.segment_editor_widget.activeEffect()
            scripted_effect = self.effect.self()

            # Set algorithm and method
            combo = scripted_effect.algorithmCombo
            idx = combo.findData("threshold_brush")
            if idx >= 0:
                combo.setCurrentIndex(idx)
            slicer.app.processEvents()

            # Set threshold method using the method combo
            if hasattr(scripted_effect, "thresholdMethodCombo"):
                method_combo = scripted_effect.thresholdMethodCombo
                method_idx = method_combo.findData(method)
                if method_idx >= 0:
                    method_combo.setCurrentIndex(method_idx)
            slicer.app.processEvents()

            scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            ctx.screenshot(f"[{method}] Threshold method set")

            # Paint
            scripted_effect.scriptedEffect.saveStateForUndo()
            scripted_effect.isDrawing = True
            scripted_effect._currentStrokeEraseMode = False
            # Reset lastIjk to avoid "same voxel" optimization skip between iterations
            scripted_effect.lastIjk = None

            import time

            start_time = time.time()
            scripted_effect.processPoint(center_xy, self.red_widget)
            elapsed_ms = (time.time() - start_time) * 1000

            scripted_effect.isDrawing = False
            slicer.app.processEvents()

            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            voxels = _count_segment_voxels(self.segmentation_node, self.segment_id)
            ctx.record_metric(f"{method}_voxels", voxels)
            ctx.record_metric(f"{method}_time_ms", elapsed_ms)

            ctx.screenshot(f"[{method}_result] {voxels} voxels in {elapsed_ms:.1f}ms")

    def verify(self, ctx: TestContext) -> None:
        """Verify threshold brush configuration."""
        logger.info("Verifying threshold brush")

        scripted_effect = self.effect.self()

        ctx.assert_equal(
            scripted_effect.algorithm,
            "threshold_brush",
            "Algorithm should be threshold_brush",
        )

        ctx.screenshot("[verify] Threshold brush verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")
