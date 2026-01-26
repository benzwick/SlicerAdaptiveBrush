"""Tests for brush outline visualization (BrushOutlinePipeline).

Tests the visual feedback components:
1. Brush outline visibility and positioning
2. Color changes for add/erase modes
3. Inner zone circle visibility
4. Crosshair styles
5. Preview overlay rendering
"""

from __future__ import annotations

import logging
from typing import Any

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


@register_test(category="visualization")
class TestBrushOutlineVisibility(TestCase):
    """Test brush outline visibility and basic rendering."""

    name = "brush_outline_visibility"
    description = "Test that brush outline appears and updates position correctly"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up brush outline visibility test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("BrushTest")

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

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready to test brush outline")

    def run(self, ctx: TestContext) -> None:
        """Test brush outline visibility at different positions."""
        logger.info("Running brush outline visibility test")

        scripted_effect = self.effect.self()

        # Set large brush radius for visibility
        scripted_effect.radiusSlider.value = 30.0
        slicer.app.processEvents()

        center_xy = _get_slice_center_xy(self.red_widget)

        # Show brush at center
        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[center] Brush outline at slice center")

        # Move brush to different position (offset from center)
        offset_xy = (center_xy[0] + 50, center_xy[1] + 30)
        scripted_effect._updateBrushPreview(offset_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[offset] Brush outline at offset position")

        # Test different brush sizes
        for radius in [15.0, 25.0, 40.0]:
            scripted_effect.radiusSlider.value = radius
            slicer.app.processEvents()

            scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            ctx.screenshot(f"[radius_{int(radius)}] Brush radius {radius}mm")
            ctx.record_metric(f"radius_{int(radius)}_tested", 1)

    def verify(self, ctx: TestContext) -> None:
        """Verify brush outline is configured correctly."""
        logger.info("Verifying brush outline")

        scripted_effect = self.effect.self()

        # Verify outline pipelines dictionary exists
        ctx.assert_is_not_none(
            scripted_effect.outlinePipelines,
            "Outline pipelines dictionary should exist",
        )

        # Check if Red widget has a pipeline (key is view name string, e.g., "Red")
        red_key = "Red"
        if red_key in scripted_effect.outlinePipelines:
            pipeline = scripted_effect.outlinePipelines[red_key]
            ctx.assert_is_not_none(pipeline, "Red widget should have outline pipeline")
            ctx.log("Outline pipeline verified for Red widget")

        ctx.screenshot("[verify] Brush outline verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="visualization")
class TestBrushOutlineColors(TestCase):
    """Test brush outline color changes for add/erase modes."""

    name = "brush_outline_colors"
    description = "Test brush outline changes color between add and erase modes"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up brush outline colors test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("ColorTest")

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

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready to test brush colors")

    def run(self, ctx: TestContext) -> None:
        """Test brush outline colors in add and erase modes."""
        logger.info("Running brush color test")

        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 30.0
        slicer.app.processEvents()

        center_xy = _get_slice_center_xy(self.red_widget)

        # Test ADD mode (default) - should be yellow
        ctx.log("Testing ADD mode (yellow brush)")

        # Ensure add mode is selected
        if hasattr(scripted_effect, "addModeRadio"):
            scripted_effect.addModeRadio.checked = True
        slicer.app.processEvents()

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[add_mode] Yellow brush outline (add mode)")

        # Test ERASE mode - should be red/orange
        ctx.log("Testing ERASE mode (red brush)")

        if hasattr(scripted_effect, "eraseModeRadio"):
            scripted_effect.eraseModeRadio.checked = True
        slicer.app.processEvents()

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=True)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[erase_mode] Red/orange brush outline (erase mode)")

        # Switch back to add mode
        if hasattr(scripted_effect, "addModeRadio"):
            scripted_effect.addModeRadio.checked = True
        slicer.app.processEvents()

        scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[add_mode_restored] Yellow brush restored")

    def verify(self, ctx: TestContext) -> None:
        """Verify mode colors are correct."""
        logger.info("Verifying brush colors")

        scripted_effect = self.effect.self()

        # Verify add mode colors
        add_color = scripted_effect.BRUSH_COLOR_ADD
        ctx.assert_equal(len(add_color), 3, "Add mode color should have 3 components")
        ctx.log(f"Add mode color: RGB({add_color[0]:.2f}, {add_color[1]:.2f}, {add_color[2]:.2f})")

        # Verify erase mode colors
        erase_color = scripted_effect.BRUSH_COLOR_ERASE
        ctx.assert_equal(len(erase_color), 3, "Erase mode color should have 3 components")
        ctx.log(
            f"Erase mode color: RGB({erase_color[0]:.2f}, {erase_color[1]:.2f}, {erase_color[2]:.2f})"
        )

        # Colors should be different
        colors_different = (
            add_color[0] != erase_color[0]
            or add_color[1] != erase_color[1]
            or add_color[2] != erase_color[2]
        )
        ctx.assert_true(colors_different, "Add and erase colors should be different")

        ctx.screenshot("[verify] Brush colors verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="visualization")
class TestInnerZoneCircle(TestCase):
    """Test inner zone (threshold sampling) circle visibility."""

    name = "brush_inner_zone"
    description = "Test inner zone circle visibility and size adjustment"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up inner zone test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("InnerZoneTest")

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

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready to test inner zone circle")

    def run(self, ctx: TestContext) -> None:
        """Test inner zone circle at different sizes."""
        logger.info("Running inner zone test")

        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 35.0
        slicer.app.processEvents()

        center_xy = _get_slice_center_xy(self.red_widget)

        # Test different threshold zone percentages
        zone_values = [20, 40, 60, 80]

        for zone in zone_values:
            ctx.log(f"Testing threshold zone: {zone}%")

            scripted_effect.zoneSlider.value = zone
            slicer.app.processEvents()

            scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            ctx.screenshot(f"[zone_{zone}] Inner zone at {zone}%")
            ctx.record_metric(f"zone_{zone}_tested", 1)

    def verify(self, ctx: TestContext) -> None:
        """Verify inner zone configuration."""
        logger.info("Verifying inner zone")

        scripted_effect = self.effect.self()

        # Verify zone slider exists and has valid range
        zone_value = scripted_effect.zoneSlider.value
        ctx.assert_greater_equal(zone_value, 0, "Zone value should be >= 0")
        ctx.assert_less_equal(zone_value, 100, "Zone value should be <= 100")

        ctx.log(f"Threshold zone value: {zone_value}%")

        ctx.screenshot("[verify] Inner zone verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="visualization")
class TestCrosshairStyles(TestCase):
    """Test crosshair style options."""

    name = "brush_crosshair_styles"
    description = "Test different crosshair styles in brush visualization"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up crosshair styles test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("CrosshairTest")

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

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready to test crosshair styles")

    def run(self, ctx: TestContext) -> None:
        """Test different crosshair styles."""
        logger.info("Running crosshair styles test")

        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 35.0
        slicer.app.processEvents()

        center_xy = _get_slice_center_xy(self.red_widget)

        # Check if crosshair combo exists
        if not hasattr(scripted_effect, "crosshairCombo"):
            ctx.log("Crosshair combo not found - may be in collapsed section")
            scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()
            ctx.screenshot("[default] Default crosshair style")
            return

        crosshair_combo = scripted_effect.crosshairCombo

        # Test each crosshair style
        for i in range(crosshair_combo.count):
            style_text = crosshair_combo.itemText(i)
            style_data = crosshair_combo.itemData(i)

            ctx.log(f"Testing crosshair style: {style_text}")

            crosshair_combo.setCurrentIndex(i)
            slicer.app.processEvents()

            scripted_effect._updateBrushPreview(center_xy, self.red_widget, eraseMode=False)
            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            ctx.screenshot(f"[{style_data}] Crosshair style: {style_text}")
            ctx.record_metric(f"crosshair_{style_data}_tested", 1)

    def verify(self, ctx: TestContext) -> None:
        """Verify crosshair configuration."""
        logger.info("Verifying crosshair styles")

        scripted_effect = self.effect.self()

        # Verify crosshair style property exists
        if hasattr(scripted_effect, "crosshairStyle"):
            style = scripted_effect.crosshairStyle
            ctx.log(f"Current crosshair style: {style}")
        else:
            ctx.log("Crosshair style attribute not found")

        ctx.screenshot("[verify] Crosshair verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")
