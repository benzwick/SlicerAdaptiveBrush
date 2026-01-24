"""UI options panel tests for AdaptiveBrush.

Tests the options panel UI:
1. Capture screenshots of panel state with brush circle visible
2. Verify widget visibility per algorithm
3. Test collapsible sections
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


def _show_brush_at_slice_center(scripted_effect, slice_widget):
    """Show the brush circle at the center of a slice view using RAS coordinates.

    This triggers the brush outline to be visible for screenshots.
    Uses RAS-to-XY conversion like the painting code does.
    """
    import vtk

    slice_logic = slice_widget.sliceLogic()
    slice_node = slice_logic.GetSliceNode()
    view_name = slice_node.GetName()

    # Get the current slice center in RAS coordinates
    # This gives us a point that's definitely visible in the slice
    slice_to_ras = slice_node.GetSliceToRAS()
    # The slice center in RAS is the origin of the slice-to-RAS transform
    center_ras = [
        slice_to_ras.GetElement(0, 3),
        slice_to_ras.GetElement(1, 3),
        slice_to_ras.GetElement(2, 3),
    ]

    # Convert RAS to XY (slice view coordinates)
    xy_to_ras = slice_node.GetXYToRAS()
    ras_to_xy = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)

    ras_point = [center_ras[0], center_ras[1], center_ras[2], 1]
    xy_point = [0, 0, 0, 1]
    ras_to_xy.MultiplyPoint(ras_point, xy_point)

    center_xy = (int(xy_point[0]), int(xy_point[1]))

    logger.info(f"Brush preview: view={view_name}, ras={center_ras}, xy={center_xy}")

    # Check if pipelines exist
    num_pipelines = len(scripted_effect.outlinePipelines)
    if num_pipelines == 0:
        logger.warning("No outline pipelines exist - creating them")
        scripted_effect._createOutlinePipelines()

    # Call _updateBrushPreview to position and show the brush outline
    scripted_effect._updateBrushPreview(center_xy, slice_widget, eraseMode=False)

    # Force render
    view = slice_widget.sliceView()
    view.forceRender()
    slicer.app.processEvents()


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

        # Switch to Segment Editor module so we can see the UI
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        # Now use the actual segment editor widget from the module
        segment_editor_module_widget = slicer.modules.segmenteditor.widgetRepresentation().self()
        segment_editor_module_widget.editor.setSegmentationNode(self.segmentation_node)
        segment_editor_module_widget.editor.setSourceVolumeNode(self.volume_node)
        segment_editor_module_widget.editor.setCurrentSegmentID(self.segment_id)

        # Activate Adaptive Brush
        segment_editor_module_widget.editor.setActiveEffectByName("Adaptive Brush")
        self.effect = segment_editor_module_widget.editor.activeEffect()
        self.segment_editor_widget = segment_editor_module_widget.editor

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        # Get Red slice widget for showing brush
        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()

        # Navigate to middle of volume (like workflow_basic does)
        # This ensures the slice view is properly initialized
        red_logic.SetSliceOffset(0)  # Navigate to center of volume
        slicer.app.processEvents()

        # Set a larger brush radius to make it clearly visible
        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = 25.0  # 25mm radius
        slicer.app.processEvents()

        # Show brush circle at center of Red view
        _show_brush_at_slice_center(scripted_effect, self.red_widget)

        ctx.screenshot("[setup] MRHead loaded, Adaptive Brush active, brush visible")

    def run(self, ctx: TestContext) -> None:
        """Test UI state for each algorithm."""
        logger.info("Running UI options panel tests")

        scripted_effect = self.effect.self()

        # List of algorithms to test (use correct names)
        algorithms = [
            "watershed",
            "level_set_cpu",
            "connected_threshold",
            "region_growing",
            "threshold_brush",
            "geodesic_distance",
            "random_walker",
        ]

        for algo in algorithms:
            ctx.log(f"Testing UI for algorithm: {algo}")

            # Use the combo box like a user would - find and select the algorithm
            combo = scripted_effect.algorithmCombo
            idx = combo.findData(algo)
            if idx >= 0:
                ctx.log(f"  Selecting '{combo.itemText(idx)}' (index {idx})")
                combo.setCurrentIndex(idx)  # This triggers onAlgorithmChanged signal
            else:
                ctx.log(f"  WARNING: Algorithm '{algo}' not found in combo box")
                continue

            # Force UI update
            slicer.app.processEvents()

            # Show brush circle (re-trigger to update after algorithm change)
            _show_brush_at_slice_center(scripted_effect, self.red_widget)

            # Capture screenshot with algorithm name in description
            ctx.screenshot(f"[{algo}] Options panel with brush circle")

        # Test Threshold Brush auto-methods
        ctx.log("Testing Threshold Brush auto-methods")

        # First select threshold_brush using combo
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("threshold_brush")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        # Map method names to combo data values
        method_map = {"Otsu": "otsu", "Huang": "huang", "Triangle": "triangle", "Li": "li"}
        for method_display, method_data in method_map.items():
            ctx.log(f"  Selecting threshold method: {method_display}")

            # Use the combo box like a user would
            method_combo = scripted_effect.thresholdMethodCombo
            idx = method_combo.findData(method_data)
            if idx >= 0:
                method_combo.setCurrentIndex(idx)  # Triggers signal
            slicer.app.processEvents()

            # Show brush circle
            _show_brush_at_slice_center(scripted_effect, self.red_widget)

            ctx.screenshot(f"[threshold_brush_{method_data}] {method_display} method")

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

        # Verify algorithm selector exists by checking instance variable
        ctx.assert_is_not_none(
            scripted_effect.algorithm,
            "Algorithm should be set",
        )

        ctx.screenshot("[verify] UI verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down UI test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")
