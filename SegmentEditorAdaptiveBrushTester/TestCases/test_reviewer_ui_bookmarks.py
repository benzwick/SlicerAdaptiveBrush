"""UI tests for Reviewer bookmarks controls.

Tests the bookmark add, restore, delete buttons and combo box.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_ui")
class TestReviewerUIBookmarks(TestCase):
    """UI tests for bookmarks controls in Reviewer module."""

    name = "ui_reviewer_bookmarks"
    description = "Test bookmark add, restore, delete buttons and combo box"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up Reviewer module UI."""
        logger.info("Setting up Reviewer UI bookmarks test")

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

        # Clear any existing bookmarks
        if self.widget.bookmarks:
            self.widget.bookmarks.clear_all()
            self.widget._update_bookmark_combo()

        slicer.app.processEvents()
        ctx.screenshot("[setup] Reviewer module loaded, bookmarks cleared")

    def run(self, ctx: TestContext) -> None:
        """Test bookmark UI controls."""
        logger.info("Running bookmark UI tests")

        # Test initial state
        ctx.log("Testing initial bookmark state")
        ctx.assert_equal(
            self.widget.bookmarkCombo.count,
            0,
            "Bookmark combo should be empty initially",
        )
        ctx.assert_equal(
            self.widget.bookmarks.count,
            0,
            "Should have 0 bookmarks",
        )
        ctx.screenshot("[initial] Empty bookmarks state")

        # Test addBookmarkButton
        ctx.log("Testing addBookmarkButton")

        # Navigate to a position and add bookmark
        layoutManager = slicer.app.layoutManager()
        red_widget = layoutManager.sliceWidget("Red")
        red_logic = red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        # Set description
        self.widget.bookmarkDescEdit.setText("First test bookmark")

        # Click add button
        self.widget.addBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            1,
            "Should have 1 bookmark after add",
        )
        ctx.assert_equal(
            self.widget.bookmarkCombo.count,
            1,
            "Combo should have 1 item",
        )
        ctx.assert_equal(
            self.widget.bookmarkDescEdit.text,
            "",
            "Description field should be cleared after add",
        )
        ctx.screenshot("[bookmark1] First bookmark added")

        # Add second bookmark at different position
        ctx.log("Adding second bookmark")
        red_logic.SetSliceOffset(30)
        slicer.app.processEvents()

        self.widget.bookmarkDescEdit.setText("Second bookmark - different slice")
        self.widget.addBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            2,
            "Should have 2 bookmarks",
        )
        ctx.assert_equal(
            self.widget.bookmarkCombo.count,
            2,
            "Combo should have 2 items",
        )
        ctx.screenshot("[bookmark2] Second bookmark added")

        # Add third bookmark without description (should auto-generate)
        ctx.log("Adding third bookmark without description")
        red_logic.SetSliceOffset(-20)
        slicer.app.processEvents()

        self.widget.bookmarkDescEdit.clear()
        self.widget.addBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            3,
            "Should have 3 bookmarks",
        )
        ctx.screenshot("[bookmark3] Third bookmark with auto description")

        # Test bookmark combo selection
        ctx.log("Testing bookmark combo selection")
        self.widget.bookmarkCombo.setCurrentIndex(0)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.bookmarkCombo.currentIndex,
            0,
            "Should be able to select first bookmark",
        )

        self.widget.bookmarkCombo.setCurrentIndex(1)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.bookmarkCombo.currentIndex,
            1,
            "Should be able to select second bookmark",
        )
        ctx.screenshot("[combo_select] Bookmark combo selection")

        # Test restoreBookmarkButton
        ctx.log("Testing restoreBookmarkButton")

        # Move to a different position
        red_logic.SetSliceOffset(50)
        slicer.app.processEvents()
        ctx.screenshot("[before_restore] Position before restore")

        # Select and restore first bookmark
        self.widget.bookmarkCombo.setCurrentIndex(0)
        self.widget.restoreBookmarkButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[after_restore] Position after restoring bookmark 0")

        # Restore second bookmark
        self.widget.bookmarkCombo.setCurrentIndex(1)
        self.widget.restoreBookmarkButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[restore2] Position after restoring bookmark 1")

        # Test deleteBookmarkButton
        ctx.log("Testing deleteBookmarkButton")

        # Select and delete second bookmark
        self.widget.bookmarkCombo.setCurrentIndex(1)
        self.widget.deleteBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            2,
            "Should have 2 bookmarks after delete",
        )
        ctx.assert_equal(
            self.widget.bookmarkCombo.count,
            2,
            "Combo should have 2 items after delete",
        )
        ctx.screenshot("[after_delete] After deleting bookmark 1")

        # Delete another bookmark
        self.widget.bookmarkCombo.setCurrentIndex(0)
        self.widget.deleteBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            1,
            "Should have 1 bookmark remaining",
        )
        ctx.screenshot("[after_delete2] After deleting another bookmark")

        # Test restore when no bookmarks selected (edge case)
        ctx.log("Testing edge case - restore with valid selection")
        if self.widget.bookmarkCombo.count > 0:
            self.widget.bookmarkCombo.setCurrentIndex(0)
            self.widget.restoreBookmarkButton.click()
            slicer.app.processEvents()
            # Should not crash

        # Delete last bookmark
        if self.widget.bookmarkCombo.count > 0:
            self.widget.bookmarkCombo.setCurrentIndex(0)
            self.widget.deleteBookmarkButton.click()
            slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            0,
            "Should have 0 bookmarks after deleting all",
        )
        ctx.screenshot("[all_deleted] All bookmarks deleted")

    def verify(self, ctx: TestContext) -> None:
        """Verify bookmark UI test results."""
        logger.info("Verifying bookmark UI test")

        # Verify controls are still accessible
        ctx.assert_is_not_none(self.widget.addBookmarkButton, "Add button should exist")
        ctx.assert_is_not_none(self.widget.restoreBookmarkButton, "Restore button should exist")
        ctx.assert_is_not_none(self.widget.deleteBookmarkButton, "Delete button should exist")
        ctx.assert_is_not_none(self.widget.bookmarkCombo, "Combo should exist")
        ctx.assert_is_not_none(self.widget.bookmarkDescEdit, "Description edit should exist")

        ctx.screenshot("[verify] Verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down bookmark UI test")

        # Clear bookmarks
        if self.widget and self.widget.bookmarks:
            self.widget.bookmarks.clear_all()

        self.widget = None
        ctx.log("Teardown complete")
