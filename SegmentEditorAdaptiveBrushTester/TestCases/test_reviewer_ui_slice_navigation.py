"""UI tests for Reviewer slice navigation controls.

Tests the slice slider, navigation buttons, and view linking functionality.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_ui")
class TestReviewerUISliceNavigation(TestCase):
    """UI tests for slice navigation controls in Reviewer module."""

    name = "ui_reviewer_slice_navigation"
    description = "Test slice slider, navigation buttons, and view linking UI"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up Reviewer module UI."""
        logger.info("Setting up Reviewer UI slice navigation test")

        # Enable quiet mode to suppress popups during testing
        import SegmentEditorAdaptiveBrushReviewer

        SegmentEditorAdaptiveBrushReviewer.set_quiet_mode(True)

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load sample data to have a volume to navigate
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

        # Initialize slice count for navigation
        self.widget.total_slices = 100
        self.widget.sliceSlider.setMaximum(99)
        self.widget.sliceSlider.setValue(0)
        self.widget.current_slice_index = 0
        self.widget._update_slice_display()

        slicer.app.processEvents()
        ctx.screenshot("[setup] Reviewer module loaded with MRHead")

    def run(self, ctx: TestContext) -> None:
        """Test slice navigation UI controls."""
        logger.info("Running slice navigation UI tests")

        # Test initial slider state
        ctx.log("Testing initial slider state")
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            0,
            "Slider should start at 0",
        )
        ctx.assert_equal(
            self.widget.sliceSlider.maximum,
            99,
            "Slider max should be 99 (100 slices)",
        )
        ctx.screenshot("[initial] Initial slider state")

        # Test slider value change
        ctx.log("Testing slider value change")
        self.widget.sliceSlider.setValue(50)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.current_slice_index,
            50,
            "Current slice index should update with slider",
        )
        ctx.assert_equal(
            self.widget.sliceLabel.text,
            "51/100",
            "Slice label should show 51/100 (1-indexed display)",
        )
        ctx.screenshot("[slider50] Slider moved to 50")

        # Test firstSliceButton (|<)
        ctx.log("Testing firstSliceButton")
        self.widget.sliceSlider.setValue(50)  # Start in middle
        slicer.app.processEvents()

        self.widget.firstSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            0,
            "First button should go to slice 0",
        )
        ctx.screenshot("[first] After first slice button")

        # Test lastSliceButton (>|)
        ctx.log("Testing lastSliceButton")
        self.widget.lastSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            99,
            "Last button should go to slice 99",
        )
        ctx.screenshot("[last] After last slice button")

        # Test prevSliceButton (<)
        ctx.log("Testing prevSliceButton")
        self.widget.sliceSlider.setValue(50)
        slicer.app.processEvents()

        self.widget.prevSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            49,
            "Prev button should decrease by 1",
        )
        ctx.screenshot("[prev] After prev slice button")

        # Test nextSliceButton (>)
        ctx.log("Testing nextSliceButton")
        self.widget.nextSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            50,
            "Next button should increase by 1",
        )
        ctx.screenshot("[next] After next slice button")

        # Test prevFastButton (<<) - jump 10 slices
        ctx.log("Testing prevFastButton")
        self.widget.sliceSlider.setValue(50)
        slicer.app.processEvents()

        self.widget.prevFastButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            40,
            "Prev fast button should decrease by 10",
        )
        ctx.screenshot("[prevfast] After prev fast button")

        # Test nextFastButton (>>) - jump 10 slices
        ctx.log("Testing nextFastButton")
        self.widget.nextFastButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            50,
            "Next fast button should increase by 10",
        )
        ctx.screenshot("[nextfast] After next fast button")

        # Test boundary conditions - prev at start
        ctx.log("Testing boundary - prev at start")
        self.widget.sliceSlider.setValue(0)
        slicer.app.processEvents()

        self.widget.prevSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            0,
            "Prev at start should stay at 0",
        )

        self.widget.prevFastButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            0,
            "Prev fast at start should stay at 0",
        )
        ctx.screenshot("[boundary_start] Boundary at start")

        # Test boundary conditions - next at end
        ctx.log("Testing boundary - next at end")
        self.widget.sliceSlider.setValue(99)
        slicer.app.processEvents()

        self.widget.nextSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            99,
            "Next at end should stay at 99",
        )

        self.widget.nextFastButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            99,
            "Next fast at end should stay at 99",
        )
        ctx.screenshot("[boundary_end] Boundary at end")

        # Test linkViewsCheck
        ctx.log("Testing linkViewsCheck")
        initial_link_state = self.widget.linkViewsCheck.isChecked()
        ctx.log(f"Initial link state: {initial_link_state}")

        # Toggle to opposite state
        self.widget.linkViewsCheck.setChecked(not initial_link_state)
        slicer.app.processEvents()
        ctx.assert_not_equal(
            self.widget.linkViewsCheck.isChecked(),
            initial_link_state,
            "Link checkbox should toggle",
        )
        ctx.screenshot("[link_toggled] Link views toggled")

        # Toggle back
        self.widget.linkViewsCheck.setChecked(initial_link_state)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.linkViewsCheck.isChecked(),
            initial_link_state,
            "Link checkbox should return to initial state",
        )

        # Test _jump_slices directly
        ctx.log("Testing _jump_slices")
        self.widget.sliceSlider.setValue(50)
        slicer.app.processEvents()

        self.widget._jump_slices(-5)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            45,
            "_jump_slices(-5) should move to 45",
        )

        self.widget._jump_slices(10)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            55,
            "_jump_slices(10) should move to 55",
        )
        ctx.screenshot("[jump_slices] Direct jump slices tested")

    def verify(self, ctx: TestContext) -> None:
        """Verify slice navigation test results."""
        logger.info("Verifying slice navigation UI test")

        # Verify widget is still responsive
        ctx.assert_is_not_none(self.widget, "Widget should still exist")
        ctx.assert_is_not_none(self.widget.sliceSlider, "Slider should exist")

        # Verify we can still set values
        self.widget.sliceSlider.setValue(25)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            25,
            "Slider should still be functional",
        )

        ctx.screenshot("[verify] Verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down slice navigation UI test")
        self.widget = None
        ctx.log("Teardown complete")
