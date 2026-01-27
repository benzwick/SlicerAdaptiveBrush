"""UI tests for Reviewer keyboard shortcuts.

Tests all keyboard shortcuts defined in ReviewShortcutFilter.
"""

from __future__ import annotations

import logging

import qt
import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_ui")
class TestReviewerUIKeyboard(TestCase):
    """UI tests for keyboard shortcuts in Reviewer module."""

    name = "ui_reviewer_keyboard"
    description = "Test all keyboard shortcuts (navigation, rating, bookmarks)"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up Reviewer module UI."""
        logger.info("Setting up Reviewer UI keyboard test")

        # Enable quiet mode to suppress popups during testing
        import SegmentEditorAdaptiveBrushReviewer

        SegmentEditorAdaptiveBrushReviewer.set_quiet_mode(True)

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

        # Initialize slice navigation
        self.widget.total_slices = 100
        self.widget.sliceSlider.setMaximum(99)
        self.widget.current_slice_index = 50
        self.widget.sliceSlider.setValue(50)
        self.widget._update_slice_display()

        slicer.app.processEvents()
        ctx.screenshot("[setup] Reviewer module ready for keyboard tests")

    def _simulate_key(
        self,
        key: int,
        ctrl: bool = False,
        shift: bool = False,
    ) -> bool:
        """Simulate a key press event.

        Args:
            key: Qt key code.
            ctrl: Whether Ctrl is held.
            shift: Whether Shift is held.

        Returns:
            True if the event was handled by the filter.
        """
        modifiers = qt.Qt.NoModifier
        if ctrl:
            modifiers |= qt.Qt.ControlModifier
        if shift:
            modifiers |= qt.Qt.ShiftModifier

        event = qt.QKeyEvent(qt.QEvent.KeyPress, key, modifiers)

        # Send to the shortcut filter
        if self.widget._shortcut_filter:
            handled = self.widget._shortcut_filter.eventFilter(slicer.util.mainWindow(), event)
            slicer.app.processEvents()
            return handled
        return False

    def run(self, ctx: TestContext) -> None:
        """Test keyboard shortcuts."""
        logger.info("Running keyboard shortcut tests")

        # Verify shortcut filter exists
        ctx.assert_is_not_none(
            self.widget._shortcut_filter,
            "Shortcut filter should exist",
        )
        ctx.screenshot("[initial] Shortcut filter ready")

        # Test slice navigation shortcuts
        ctx.log("Testing slice navigation shortcuts")

        # Right arrow - next slice
        ctx.log("Testing Right arrow")
        self.widget.sliceSlider.setValue(50)
        slicer.app.processEvents()

        handled = self._simulate_key(qt.Qt.Key_Right)
        ctx.assert_true(handled, "Right arrow should be handled")
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            51,
            "Right arrow should move to next slice",
        )
        ctx.screenshot("[right_arrow] After Right arrow")

        # Left arrow - previous slice
        ctx.log("Testing Left arrow")
        handled = self._simulate_key(qt.Qt.Key_Left)
        ctx.assert_true(handled, "Left arrow should be handled")
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            50,
            "Left arrow should move to previous slice",
        )
        ctx.screenshot("[left_arrow] After Left arrow")

        # Ctrl+Right - jump 10 slices forward
        ctx.log("Testing Ctrl+Right")
        handled = self._simulate_key(qt.Qt.Key_Right, ctrl=True)
        ctx.assert_true(handled, "Ctrl+Right should be handled")
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            60,
            "Ctrl+Right should jump 10 slices",
        )
        ctx.screenshot("[ctrl_right] After Ctrl+Right")

        # Ctrl+Left - jump 10 slices back
        ctx.log("Testing Ctrl+Left")
        handled = self._simulate_key(qt.Qt.Key_Left, ctrl=True)
        ctx.assert_true(handled, "Ctrl+Left should be handled")
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            50,
            "Ctrl+Left should jump 10 slices back",
        )
        ctx.screenshot("[ctrl_left] After Ctrl+Left")

        # Home - first slice
        ctx.log("Testing Home key")
        self.widget.sliceSlider.setValue(50)
        slicer.app.processEvents()

        handled = self._simulate_key(qt.Qt.Key_Home)
        ctx.assert_true(handled, "Home should be handled")
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            0,
            "Home should go to first slice",
        )
        ctx.screenshot("[home] After Home key")

        # End - last slice
        ctx.log("Testing End key")
        handled = self._simulate_key(qt.Qt.Key_End)
        ctx.assert_true(handled, "End should be handled")
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            99,
            "End should go to last slice",
        )
        ctx.screenshot("[end] After End key")

        # Test rating shortcuts (1-4)
        ctx.log("Testing rating shortcuts")
        from SegmentEditorAdaptiveBrushReviewerLib import Rating

        # Key 1 - Accept
        ctx.log("Testing Key 1 (Accept)")
        handled = self._simulate_key(qt.Qt.Key_1)
        ctx.assert_true(handled, "Key 1 should be handled")
        ctx.assert_equal(
            self.widget.ratingGroup.checkedId(),
            Rating.ACCEPT.value,
            "Key 1 should set Accept rating",
        )
        ctx.screenshot("[key1] After Key 1 (Accept)")

        # Key 2 - Minor
        ctx.log("Testing Key 2 (Minor)")
        handled = self._simulate_key(qt.Qt.Key_2)
        ctx.assert_true(handled, "Key 2 should be handled")
        ctx.assert_equal(
            self.widget.ratingGroup.checkedId(),
            Rating.MINOR.value,
            "Key 2 should set Minor rating",
        )
        ctx.screenshot("[key2] After Key 2 (Minor)")

        # Key 3 - Major
        ctx.log("Testing Key 3 (Major)")
        handled = self._simulate_key(qt.Qt.Key_3)
        ctx.assert_true(handled, "Key 3 should be handled")
        ctx.assert_equal(
            self.widget.ratingGroup.checkedId(),
            Rating.MAJOR.value,
            "Key 3 should set Major rating",
        )
        ctx.screenshot("[key3] After Key 3 (Major)")

        # Key 4 - Reject
        ctx.log("Testing Key 4 (Reject)")
        handled = self._simulate_key(qt.Qt.Key_4)
        ctx.assert_true(handled, "Key 4 should be handled")
        ctx.assert_equal(
            self.widget.ratingGroup.checkedId(),
            Rating.REJECT.value,
            "Key 4 should set Reject rating",
        )
        ctx.screenshot("[key4] After Key 4 (Reject)")

        # Test Space - toggle view mode
        ctx.log("Testing Space key (view mode toggle)")
        initial_mode = self.widget.viewModeGroup.checkedId()
        handled = self._simulate_key(qt.Qt.Key_Space)
        ctx.assert_true(handled, "Space should be handled")

        new_mode = self.widget.viewModeGroup.checkedId()
        expected_mode = (initial_mode + 1) % 3
        ctx.assert_equal(
            new_mode,
            expected_mode,
            f"Space should toggle view mode from {initial_mode} to {expected_mode}",
        )
        ctx.screenshot("[space] After Space key (view mode toggle)")

        # Test Ctrl+B - add bookmark
        ctx.log("Testing Ctrl+B (add bookmark)")
        initial_count = self.widget.bookmarks.count
        handled = self._simulate_key(qt.Qt.Key_B, ctrl=True)
        ctx.assert_true(handled, "Ctrl+B should be handled")
        ctx.assert_equal(
            self.widget.bookmarks.count,
            initial_count + 1,
            "Ctrl+B should add a bookmark",
        )
        ctx.screenshot("[ctrl_b] After Ctrl+B (add bookmark)")

        # Test Ctrl+R - restore last bookmark
        ctx.log("Testing Ctrl+R (restore bookmark)")
        if self.widget.bookmarks.count > 0:
            handled = self._simulate_key(qt.Qt.Key_R, ctrl=True)
            ctx.assert_true(handled, "Ctrl+R should be handled")
            ctx.screenshot("[ctrl_r] After Ctrl+R (restore bookmark)")

        # Test P - prev trial (if trials available)
        ctx.log("Testing P key (prev trial)")
        handled = self._simulate_key(qt.Qt.Key_P)
        ctx.assert_true(handled, "P key should be handled")
        ctx.screenshot("[key_p] After P key")

        # Test N - next trial (if trials available)
        ctx.log("Testing N key (next trial)")
        handled = self._simulate_key(qt.Qt.Key_N)
        ctx.assert_true(handled, "N key should be handled")
        ctx.screenshot("[key_n] After N key")

        # Test S - save rating (without Ctrl)
        ctx.log("Testing S key (save rating)")
        handled = self._simulate_key(qt.Qt.Key_S)
        ctx.assert_true(handled, "S key should be handled")
        ctx.screenshot("[key_s] After S key (save)")

    def verify(self, ctx: TestContext) -> None:
        """Verify keyboard test results."""
        logger.info("Verifying keyboard UI test")

        # Verify shortcut filter still works
        ctx.assert_is_not_none(
            self.widget._shortcut_filter,
            "Shortcut filter should still exist",
        )

        # Test that filter is properly installed
        # Filter should be installed but we can't easily verify without private API

        ctx.screenshot("[verify] Verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down keyboard UI test")

        # Clean up bookmarks
        if self.widget and self.widget.bookmarks:
            self.widget.bookmarks.clear_all()

        self.widget = None
        ctx.log("Teardown complete")
