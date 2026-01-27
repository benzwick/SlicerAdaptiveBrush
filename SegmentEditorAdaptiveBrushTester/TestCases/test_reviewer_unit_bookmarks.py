"""Unit tests for SceneViewBookmarks class.

Tests the bookmark management functionality for saving and restoring views.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_unit")
class TestSceneViewBookmarksUnit(TestCase):
    """Unit tests for SceneViewBookmarks class."""

    name = "unit_scene_view_bookmarks"
    description = "Test SceneViewBookmarks add, restore, list, and delete operations"

    def __init__(self) -> None:
        super().__init__()
        self.bookmarks = None
        self.volume_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test with sample data."""
        logger.info("Setting up SceneViewBookmarks unit test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Import bookmarks class
        from SegmentEditorAdaptiveBrushReviewerLib import SceneViewBookmarks

        self.bookmarks = SceneViewBookmarks()

        # Load sample data
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        slicer.app.processEvents()
        ctx.screenshot("[setup] SceneViewBookmarks initialized with MRHead")

    def run(self, ctx: TestContext) -> None:
        """Test SceneViewBookmarks operations."""
        logger.info("Running SceneViewBookmarks unit tests")

        # Test initial state
        ctx.log("Testing initial state")
        ctx.assert_equal(self.bookmarks.count, 0, "Should have 0 bookmarks initially")
        bookmarks_list = self.bookmarks.list_bookmarks()
        ctx.assert_equal(len(bookmarks_list), 0, "list_bookmarks should return empty list")
        ctx.screenshot("[initial] Empty bookmarks state")

        # Test add_bookmark with auto-generated name
        ctx.log("Testing add_bookmark with auto name")
        idx1 = self.bookmarks.add_bookmark("First interesting slice")
        ctx.assert_equal(idx1, 0, "First bookmark should be index 0")
        ctx.assert_equal(self.bookmarks.count, 1, "Should have 1 bookmark")
        ctx.screenshot("[bookmark1] First bookmark added")

        # Test add_bookmark with custom name
        ctx.log("Testing add_bookmark with custom name")

        # Navigate to a different position first
        layoutManager = slicer.app.layoutManager()
        red_widget = layoutManager.sliceWidget("Red")
        red_logic = red_widget.sliceLogic()
        red_logic.SetSliceOffset(20)  # Different slice position
        slicer.app.processEvents()

        idx2 = self.bookmarks.add_bookmark(
            description="Boundary issue found",
            name="BoundaryBookmark",
        )
        ctx.assert_equal(idx2, 1, "Second bookmark should be index 1")
        ctx.assert_equal(self.bookmarks.count, 2, "Should have 2 bookmarks")
        ctx.screenshot("[bookmark2] Second bookmark with custom name")

        # Add a third bookmark
        red_logic.SetSliceOffset(-10)
        slicer.app.processEvents()

        idx3 = self.bookmarks.add_bookmark("Third position", name="ThirdBookmark")
        ctx.assert_equal(idx3, 2, "Third bookmark should be index 2")
        ctx.assert_equal(self.bookmarks.count, 3, "Should have 3 bookmarks")
        ctx.screenshot("[bookmark3] Third bookmark added")

        # Test list_bookmarks
        ctx.log("Testing list_bookmarks")
        bookmarks_list = self.bookmarks.list_bookmarks()
        ctx.assert_equal(len(bookmarks_list), 3, "Should list 3 bookmarks")

        # Check bookmark contents
        name1, desc1 = bookmarks_list[0]
        ctx.assert_equal(desc1, "First interesting slice", "First description should match")

        name2, desc2 = bookmarks_list[1]
        ctx.assert_equal(name2, "BoundaryBookmark", "Second name should match")
        ctx.assert_equal(desc2, "Boundary issue found", "Second description should match")

        ctx.log(f"Bookmarks: {bookmarks_list}")

        # Test get_bookmark_name
        ctx.log("Testing get_bookmark_name")
        name = self.bookmarks.get_bookmark_name(1)
        ctx.assert_equal(name, "BoundaryBookmark", "Should get correct name")

        invalid_name = self.bookmarks.get_bookmark_name(99)
        ctx.assert_is_none(invalid_name, "Invalid index should return None")

        # Test restore_bookmark
        ctx.log("Testing restore_bookmark")

        # First, move to a different position
        red_logic.SetSliceOffset(50)
        slicer.app.processEvents()
        ctx.screenshot("[before_restore] Position before restore")

        # Restore first bookmark
        result = self.bookmarks.restore_bookmark(0)
        ctx.assert_true(result, "restore_bookmark should return True")
        slicer.app.processEvents()
        ctx.screenshot("[after_restore] Position after restoring bookmark 0")

        # Restore second bookmark
        result = self.bookmarks.restore_bookmark(1)
        ctx.assert_true(result, "restore_bookmark(1) should return True")
        slicer.app.processEvents()
        ctx.screenshot("[restore2] Position after restoring bookmark 1")

        # Test restore with invalid index
        result = self.bookmarks.restore_bookmark(99)
        ctx.assert_false(result, "restore with invalid index should return False")

        result = self.bookmarks.restore_bookmark(-1)
        ctx.assert_false(result, "restore with negative index should return False")

        # Test remove_bookmark
        ctx.log("Testing remove_bookmark")
        result = self.bookmarks.remove_bookmark(1)  # Remove "BoundaryBookmark"
        ctx.assert_true(result, "remove_bookmark should return True")
        ctx.assert_equal(self.bookmarks.count, 2, "Should have 2 bookmarks after remove")

        # Verify the remaining bookmarks
        bookmarks_list = self.bookmarks.list_bookmarks()
        ctx.assert_equal(len(bookmarks_list), 2, "Should list 2 bookmarks")
        name0, _ = bookmarks_list[0]
        name1, _ = bookmarks_list[1]
        ctx.log(f"Remaining bookmarks: {name0}, {name1}")

        # Test remove with invalid index
        result = self.bookmarks.remove_bookmark(99)
        ctx.assert_false(result, "remove with invalid index should return False")

        ctx.screenshot("[after_remove] After removing one bookmark")

        # Test clear_all
        ctx.log("Testing clear_all")
        self.bookmarks.clear_all()
        ctx.assert_equal(self.bookmarks.count, 0, "Should have 0 bookmarks after clear")
        bookmarks_list = self.bookmarks.list_bookmarks()
        ctx.assert_equal(len(bookmarks_list), 0, "list should be empty after clear")
        ctx.screenshot("[cleared] All bookmarks cleared")

    def verify(self, ctx: TestContext) -> None:
        """Verify test results."""
        logger.info("Verifying SceneViewBookmarks unit test")

        # Add some bookmarks for cleanup test
        self.bookmarks.add_bookmark("Verify bookmark 1")
        self.bookmarks.add_bookmark("Verify bookmark 2")
        ctx.assert_equal(self.bookmarks.count, 2, "Should have 2 bookmarks")

        # Test cleanup
        ctx.log("Testing cleanup")
        self.bookmarks.cleanup()
        ctx.assert_equal(self.bookmarks.count, 0, "Should have 0 bookmarks after cleanup")

        ctx.screenshot("[verify] Cleanup complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down SceneViewBookmarks unit test")

        if self.bookmarks:
            self.bookmarks.cleanup()
            self.bookmarks = None

        ctx.log("Teardown complete")
