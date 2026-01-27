"""UI tests for Reviewer visualization controls.

Tests layout buttons, view mode toggles, and visibility controls.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_ui")
class TestReviewerUIVisualization(TestCase):
    """UI tests for visualization controls in Reviewer module."""

    name = "ui_reviewer_visualization"
    description = "Test layout buttons, view mode toggles, and visibility controls"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up Reviewer module UI."""
        logger.info("Setting up Reviewer UI visualization test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load sample data
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        # Switch to Reviewer module
        slicer.util.selectModule("SegmentEditorAdaptiveBrushReviewer")
        slicer.app.processEvents()

        # Get widget reference
        module_widget = slicer.modules.segmenteditoradaptivebrushreviewer.widgetRepresentation()
        self.widget = module_widget.self()

        if self.widget is None:
            raise RuntimeError("Failed to get Reviewer widget")

        slicer.app.processEvents()
        ctx.screenshot("[setup] Reviewer module loaded")

    def run(self, ctx: TestContext) -> None:
        """Test visualization UI controls."""
        logger.info("Running visualization UI tests")

        layoutManager = slicer.app.layoutManager()

        # Test layout buttons
        ctx.log("Testing layout buttons")

        # Test Conventional layout (3)
        ctx.log("Testing Conventional layout")
        self.widget.layoutConventionalButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            layoutManager.layout,
            3,
            "Layout should be Conventional (3)",
        )
        ctx.screenshot("[layout_conventional] Conventional layout")

        # Test Four-Up layout (4)
        ctx.log("Testing Four-Up layout")
        self.widget.layoutFourUpButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            layoutManager.layout,
            4,
            "Layout should be Four-Up (4)",
        )
        ctx.screenshot("[layout_fourup] Four-Up layout")

        # Test 3D Only layout (6)
        ctx.log("Testing 3D Only layout")
        self.widget.layout3DOnlyButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            layoutManager.layout,
            6,
            "Layout should be 3D Only (6)",
        )
        ctx.screenshot("[layout_3donly] 3D Only layout")

        # Test Dual 3D layout (15)
        ctx.log("Testing Dual 3D layout")
        self.widget.layoutDual3DButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            layoutManager.layout,
            15,
            "Layout should be Dual 3D (15)",
        )
        ctx.screenshot("[layout_dual3d] Dual 3D layout")

        # Return to conventional for remaining tests
        self.widget.layoutConventionalButton.click()
        slicer.app.processEvents()

        # Test view mode buttons
        ctx.log("Testing view mode buttons")

        # Get the button group
        buttons = self.widget.viewModeGroup.buttons()
        ctx.assert_greater(len(buttons), 0, "Should have view mode buttons")

        # Test each view mode
        mode_names = ["Outline", "Transparent", "Fill"]
        for i, mode in enumerate(mode_names):
            ctx.log(f"Testing {mode} mode")

            # Find and click the button
            for button in buttons:
                if button.text == mode:
                    button.click()
                    slicer.app.processEvents()
                    break

            ctx.assert_equal(
                self.widget.viewModeGroup.checkedId(),
                i,
                f"{mode} mode should be selected (id {i})",
            )
            ctx.screenshot(f"[viewmode_{mode.lower()}] {mode} mode selected")

        # Test toggleGoldCheck
        ctx.log("Testing toggleGoldCheck")
        initial_gold_state = self.widget.toggleGoldCheck.isChecked()
        ctx.log(f"Initial gold visibility: {initial_gold_state}")

        # Toggle to opposite state
        self.widget.toggleGoldCheck.setChecked(not initial_gold_state)
        slicer.app.processEvents()
        ctx.assert_not_equal(
            self.widget.toggleGoldCheck.isChecked(),
            initial_gold_state,
            "Gold visibility should toggle",
        )
        ctx.screenshot("[gold_toggled] Gold visibility toggled")

        # Toggle back
        self.widget.toggleGoldCheck.setChecked(initial_gold_state)
        slicer.app.processEvents()

        # Test toggleTestCheck
        ctx.log("Testing toggleTestCheck")
        initial_test_state = self.widget.toggleTestCheck.isChecked()
        ctx.log(f"Initial test visibility: {initial_test_state}")

        # Toggle to opposite state
        self.widget.toggleTestCheck.setChecked(not initial_test_state)
        slicer.app.processEvents()
        ctx.assert_not_equal(
            self.widget.toggleTestCheck.isChecked(),
            initial_test_state,
            "Test visibility should toggle",
        )
        ctx.screenshot("[test_toggled] Test visibility toggled")

        # Toggle back
        self.widget.toggleTestCheck.setChecked(initial_test_state)
        slicer.app.processEvents()

        # Test both visibility toggles
        ctx.log("Testing combined visibility toggles")
        self.widget.toggleGoldCheck.setChecked(True)
        self.widget.toggleTestCheck.setChecked(True)
        slicer.app.processEvents()
        ctx.screenshot("[both_visible] Both gold and test visible")

        self.widget.toggleGoldCheck.setChecked(False)
        self.widget.toggleTestCheck.setChecked(False)
        slicer.app.processEvents()
        ctx.screenshot("[both_hidden] Both gold and test hidden")

        # Restore
        self.widget.toggleGoldCheck.setChecked(True)
        self.widget.toggleTestCheck.setChecked(True)
        slicer.app.processEvents()

        # Test _set_layout directly
        ctx.log("Testing _set_layout method")
        self.widget._set_layout(3)
        slicer.app.processEvents()
        ctx.assert_equal(
            layoutManager.layout,
            3,
            "_set_layout(3) should set conventional",
        )

        self.widget._set_layout(4)
        slicer.app.processEvents()
        ctx.assert_equal(
            layoutManager.layout,
            4,
            "_set_layout(4) should set four-up",
        )
        ctx.screenshot("[set_layout] _set_layout tested")

    def verify(self, ctx: TestContext) -> None:
        """Verify visualization test results."""
        logger.info("Verifying visualization UI test")

        # Verify layout buttons exist
        ctx.assert_is_not_none(self.widget.layoutConventionalButton, "Conventional button exists")
        ctx.assert_is_not_none(self.widget.layoutFourUpButton, "Four-Up button exists")
        ctx.assert_is_not_none(self.widget.layout3DOnlyButton, "3D Only button exists")
        ctx.assert_is_not_none(self.widget.layoutDual3DButton, "Dual 3D button exists")

        # Verify view mode group exists
        ctx.assert_is_not_none(self.widget.viewModeGroup, "View mode group exists")

        # Verify toggle checkboxes exist
        ctx.assert_is_not_none(self.widget.toggleGoldCheck, "Gold toggle exists")
        ctx.assert_is_not_none(self.widget.toggleTestCheck, "Test toggle exists")

        ctx.screenshot("[verify] Verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down visualization UI test")

        # Reset to conventional layout
        slicer.app.layoutManager().setLayout(3)

        self.widget = None
        ctx.log("Teardown complete")
