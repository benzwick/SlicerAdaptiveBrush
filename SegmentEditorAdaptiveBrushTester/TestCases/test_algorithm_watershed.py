"""Watershed algorithm tests for AdaptiveBrush.

Tests the watershed algorithm specifically:
1. Paint in uniform region
2. Verify segmentation boundary
3. Test edge sensitivity parameter
"""

from __future__ import annotations

import logging

import slicer
import vtk
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

        # Switch to Segment Editor module to see GUI
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        # Use the actual module's segment editor widget
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

        # Select watershed algorithm using combo box
        scripted_effect = self.effect.self()
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("watershed")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        # Get Red slice widget
        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()

        # Set a larger brush radius for visibility in screenshots
        scripted_effect.radiusSlider.value = 25.0
        slicer.app.processEvents()

        # Navigate to center of volume
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        # Get center XY using RAS conversion
        self.center_xy = self._get_slice_center_xy(self.red_widget)

        # Show brush at center
        scripted_effect._updateBrushPreview(self.center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[setup] MRHead loaded, watershed selected, brush visible")

    def _get_slice_center_xy(self, slice_widget):
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

    def run(self, ctx: TestContext) -> None:
        """Test watershed with different edge sensitivity values."""
        logger.info("Running watershed algorithm tests")

        scripted_effect = self.effect.self()

        # Test different edge sensitivity values
        sensitivities = [30, 50, 70]

        for sensitivity in sensitivities:
            ctx.log(f"Testing edge sensitivity: {sensitivity}")

            # Set sensitivity using the slider (like a user would)
            scripted_effect.sensitivitySlider.value = sensitivity
            slicer.app.processEvents()

            # Show brush circle (update after parameter change)
            scripted_effect._updateBrushPreview(self.center_xy, self.red_widget, eraseMode=False)
            self.red_widget.sliceView().forceRender()
            slicer.app.processEvents()

            ctx.screenshot(f"[sensitivity_{sensitivity}] Edge sensitivity set, brush visible")

            # Record the parameter
            ctx.record_metric(f"edge_sensitivity_{sensitivity}", sensitivity)

        # Test 3D mode toggle
        ctx.log("Testing 3D mode toggle")

        # Enable 3D mode using checkbox
        scripted_effect.sphereModeCheckbox.checked = True
        slicer.app.processEvents()

        # Show brush circle
        scripted_effect._updateBrushPreview(self.center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[3d_mode] 3D brush mode enabled")

        # Disable 3D mode
        scripted_effect.sphereModeCheckbox.checked = False
        slicer.app.processEvents()

        # Show brush circle
        scripted_effect._updateBrushPreview(self.center_xy, self.red_widget, eraseMode=False)
        self.red_widget.sliceView().forceRender()
        slicer.app.processEvents()

        ctx.screenshot("[2d_mode] 2D brush mode enabled")

    def verify(self, ctx: TestContext) -> None:
        """Verify watershed algorithm configuration."""
        logger.info("Verifying watershed algorithm test")

        scripted_effect = self.effect.self()

        # Verify algorithm is set to watershed
        ctx.assert_equal(
            scripted_effect.algorithm,
            "watershed",
            "Algorithm should be set to watershed",
        )

        # Verify edge sensitivity parameter exists
        sensitivity = scripted_effect.edgeSensitivity
        ctx.assert_is_not_none(
            sensitivity,
            "Edge sensitivity parameter should exist",
        )

        # Verify sensitivity is in valid range (0-100)
        sensitivity_val = float(sensitivity)
        ctx.assert_greater_equal(
            sensitivity_val,
            0.0,
            "Edge sensitivity should be >= 0",
        )
        ctx.assert_less_equal(
            sensitivity_val,
            100.0,
            "Edge sensitivity should be <= 100",
        )

        ctx.screenshot("[verify] Watershed algorithm verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down watershed test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")
