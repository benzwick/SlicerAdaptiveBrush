"""Documentation screenshots for UI reference.

Captures comprehensive screenshots of all UI elements for auto-generated documentation.
Covers:
- Main options panel
- Brush settings (radius, modes)
- Threshold settings
- Post-processing options
- Algorithm-specific panels
- Parameter Wizard steps

These screenshots are tagged for documentation extraction.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="docs")
class TestDocsUIReference(TestCase):
    """Generate documentation screenshots for UI reference."""

    name = "docs_ui_reference"
    description = "Capture documentation screenshots for all UI elements"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and prepare for UI documentation."""
        logger.info("Setting up UI reference documentation test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load sample data
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        # Create segmentation node
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        # Add a segment
        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
            "UIDocumentation"
        )

        # Switch to Segment Editor module
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        # Get the segment editor widget
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

        ctx.screenshot(
            "Adaptive Brush effect activated",
            doc_tags=["ui", "effect_activated"],
        )

    def run(self, ctx: TestContext) -> None:
        """Capture screenshots for all UI elements."""
        logger.info("Running UI reference documentation capture")

        scripted_effect = self.effect.self()

        # Main options panel (default state)
        ctx.screenshot(
            "Main options panel - default state",
            doc_tags=["ui", "options_panel", "default"],
        )

        # Brush settings section
        self._capture_brush_settings(ctx, scripted_effect)

        # Algorithm selection
        self._capture_algorithm_selection(ctx, scripted_effect)

        # Threshold settings
        self._capture_threshold_settings(ctx, scripted_effect)

        # Post-processing options
        self._capture_post_processing(ctx, scripted_effect)

        # Advanced settings (collapsed section)
        self._capture_advanced_settings(ctx, scripted_effect)

        # Brush modes (Add/Erase)
        self._capture_brush_modes(ctx, scripted_effect)

    def _capture_brush_settings(self, ctx: TestContext, scripted_effect) -> None:
        """Capture brush settings UI screenshots."""
        ctx.log("Capturing brush settings")

        # Default brush radius
        ctx.screenshot(
            "Brush settings - default radius",
            doc_tags=["ui", "brush_settings", "radius"],
        )

        # Small brush
        scripted_effect.radiusSlider.value = 5.0
        slicer.app.processEvents()
        ctx.screenshot(
            "Brush settings - small radius (5mm)",
            doc_tags=["ui", "brush_settings", "radius_small"],
        )

        # Large brush
        scripted_effect.radiusSlider.value = 40.0
        slicer.app.processEvents()
        ctx.screenshot(
            "Brush settings - large radius (40mm)",
            doc_tags=["ui", "brush_settings", "radius_large"],
        )

        # Reset to default
        scripted_effect.radiusSlider.value = 20.0
        slicer.app.processEvents()

        # 3D mode toggle
        scripted_effect.sphereModeCheckbox.checked = True
        slicer.app.processEvents()
        ctx.screenshot(
            "Brush settings - 3D sphere mode enabled",
            doc_tags=["ui", "brush_settings", "3d_mode"],
        )

        scripted_effect.sphereModeCheckbox.checked = False
        slicer.app.processEvents()

    def _capture_algorithm_selection(self, ctx: TestContext, scripted_effect) -> None:
        """Capture algorithm selection UI screenshots."""
        ctx.log("Capturing algorithm selection")

        # Show algorithm dropdown
        ctx.screenshot(
            "Algorithm selection dropdown",
            doc_tags=["ui", "algorithm_selection"],
        )

        # Different algorithms
        algorithms = [
            ("geodesic", "Geodesic Distance"),
            ("watershed", "Watershed"),
            ("threshold_brush", "Threshold Brush"),
        ]

        for algo_id, algo_name in algorithms:
            combo = scripted_effect.algorithmCombo
            idx = combo.findData(algo_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            slicer.app.processEvents()

            ctx.screenshot(
                f"Algorithm selected: {algo_name}",
                doc_tags=["ui", "algorithm_selection", algo_id],
            )

    def _capture_threshold_settings(self, ctx: TestContext, scripted_effect) -> None:
        """Capture threshold settings UI screenshots."""
        ctx.log("Capturing threshold settings")

        # Select threshold brush to show threshold controls
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("threshold_brush")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        ctx.screenshot(
            "Threshold settings - auto mode",
            doc_tags=["ui", "threshold_settings", "auto"],
        )

        # Manual threshold mode (if available)
        if hasattr(scripted_effect, "thresholdModeCombo"):
            mode_combo = scripted_effect.thresholdModeCombo
            manual_idx = mode_combo.findText("Manual")
            if manual_idx >= 0:
                mode_combo.setCurrentIndex(manual_idx)
                slicer.app.processEvents()

                ctx.screenshot(
                    "Threshold settings - manual mode",
                    doc_tags=["ui", "threshold_settings", "manual"],
                )

    def _capture_post_processing(self, ctx: TestContext, scripted_effect) -> None:
        """Capture post-processing options UI screenshots."""
        ctx.log("Capturing post-processing options")

        # Fill holes toggle
        if hasattr(scripted_effect, "fillHolesCheckbox"):
            scripted_effect.fillHolesCheckbox.checked = True
            slicer.app.processEvents()
            ctx.screenshot(
                "Post-processing - fill holes enabled",
                doc_tags=["ui", "post_processing", "fill_holes"],
            )

        # Closing radius
        if hasattr(scripted_effect, "closingRadiusSlider"):
            scripted_effect.closingRadiusSlider.value = 2.0
            slicer.app.processEvents()
            ctx.screenshot(
                "Post-processing - closing radius set",
                doc_tags=["ui", "post_processing", "closing"],
            )

            scripted_effect.closingRadiusSlider.value = 0.0
            slicer.app.processEvents()

    def _capture_advanced_settings(self, ctx: TestContext, scripted_effect) -> None:
        """Capture advanced settings UI screenshots."""
        ctx.log("Capturing advanced settings")

        # Look for advanced collapsible section
        if hasattr(scripted_effect, "advancedCollapsibleButton"):
            scripted_effect.advancedCollapsibleButton.collapsed = False
            slicer.app.processEvents()

            ctx.screenshot(
                "Advanced settings expanded",
                doc_tags=["ui", "advanced_settings", "expanded"],
            )

            scripted_effect.advancedCollapsibleButton.collapsed = True
            slicer.app.processEvents()

    def _capture_brush_modes(self, ctx: TestContext, scripted_effect) -> None:
        """Capture brush mode (Add/Erase) UI screenshots."""
        ctx.log("Capturing brush modes")

        # Add mode (default)
        if hasattr(scripted_effect, "addModeRadio"):
            scripted_effect.addModeRadio.checked = True
            slicer.app.processEvents()
            ctx.screenshot(
                "Brush mode - Add (paint)",
                doc_tags=["ui", "brush_mode", "add"],
            )

        # Erase mode
        if hasattr(scripted_effect, "eraseModeRadio"):
            scripted_effect.eraseModeRadio.checked = True
            slicer.app.processEvents()
            ctx.screenshot(
                "Brush mode - Erase",
                doc_tags=["ui", "brush_mode", "erase"],
            )

            # Reset to add mode
            if hasattr(scripted_effect, "addModeRadio"):
                scripted_effect.addModeRadio.checked = True
                slicer.app.processEvents()

    def verify(self, ctx: TestContext) -> None:
        """Verify we captured UI screenshots."""
        logger.info("Verifying UI reference documentation")

        # Check we have a reasonable number of screenshots
        ctx.assert_greater(
            len(ctx.screenshots),
            10,
            "Should have captured multiple UI screenshots",
        )

        ctx.screenshot(
            "UI reference documentation complete",
            doc_tags=["ui", "complete"],
        )

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down UI reference documentation test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")
