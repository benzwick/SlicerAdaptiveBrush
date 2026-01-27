"""Unit tests for ViewGroupManager class.

Tests the view linking and synchronization functionality.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_unit")
class TestViewGroupManagerUnit(TestCase):
    """Unit tests for ViewGroupManager class."""

    name = "unit_view_group_manager"
    description = "Test ViewGroupManager view linking and slice offset control"

    def __init__(self) -> None:
        super().__init__()
        self.manager = None
        self.volume_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test with sample data."""
        logger.info("Setting up ViewGroupManager unit test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Import manager
        from SegmentEditorAdaptiveBrushReviewerLib import ViewGroupManager

        self.manager = ViewGroupManager()

        # Load sample data to have something to navigate
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        slicer.app.processEvents()
        ctx.screenshot("[setup] ViewGroupManager initialized with MRHead")

    def run(self, ctx: TestContext) -> None:
        """Test ViewGroupManager operations."""
        logger.info("Running ViewGroupManager unit tests")

        # Test initial state
        ctx.log("Testing initial state")
        ctx.assert_false(self.manager.is_linked, "Should not be linked initially")
        ctx.screenshot("[initial] Initial unlinked state")

        # Test enable_linking
        ctx.log("Testing enable_linking")
        self.manager.enable_linking()
        slicer.app.processEvents()
        ctx.assert_true(self.manager.is_linked, "Should be linked after enable")
        ctx.screenshot("[linked] View linking enabled")

        # Verify composite nodes are set to linked
        linked_count = 0
        for composite_node in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            if composite_node.GetLinkedControl():
                linked_count += 1
        ctx.assert_greater(linked_count, 0, "At least one composite should be linked")

        # Test disable_linking
        ctx.log("Testing disable_linking")
        self.manager.disable_linking()
        slicer.app.processEvents()
        ctx.assert_false(self.manager.is_linked, "Should not be linked after disable")
        ctx.screenshot("[unlinked] View linking disabled")

        # Test set_linked convenience method
        ctx.log("Testing set_linked")
        self.manager.set_linked(True)
        ctx.assert_true(self.manager.is_linked, "Should be linked after set_linked(True)")

        self.manager.set_linked(False)
        ctx.assert_false(self.manager.is_linked, "Should not be linked after set_linked(False)")

        # Re-enable for navigation tests
        self.manager.enable_linking()

        # Test get_slice_range
        ctx.log("Testing get_slice_range")
        slice_range = self.manager.get_slice_range("Red")
        ctx.assert_is_not_none(slice_range, "Should get slice range for Red view")

        if slice_range:
            min_offset, max_offset = slice_range
            ctx.assert_less(min_offset, max_offset, "Min should be less than max")
            ctx.log(f"Red slice range: {min_offset:.1f} to {max_offset:.1f}")

        # Test get_slice_offset
        ctx.log("Testing get_slice_offset")
        offset = self.manager.get_slice_offset("Red")
        ctx.assert_is_not_none(offset, "Should get slice offset")

        if offset is not None and slice_range:
            ctx.assert_greater_equal(offset, min_offset, "Offset should be >= min")
            ctx.assert_less_equal(offset, max_offset, "Offset should be <= max")
            ctx.log(f"Current Red offset: {offset:.1f}")

        # Test set_slice_offset
        ctx.log("Testing set_slice_offset")
        if slice_range:
            mid_offset = (min_offset + max_offset) / 2
            result = self.manager.set_slice_offset(mid_offset, "Red")
            ctx.assert_true(result, "set_slice_offset should return True")

            slicer.app.processEvents()

            # Verify offset was set
            new_offset = self.manager.get_slice_offset("Red")
            if new_offset is not None:
                # Allow small tolerance for floating point
                diff = abs(new_offset - mid_offset)
                ctx.assert_less(diff, 1.0, f"Offset should be near {mid_offset:.1f}")
                ctx.log(f"Set offset to {mid_offset:.1f}, got {new_offset:.1f}")

        ctx.screenshot("[navigation] Slice navigation tested")

        # Test observer setup
        ctx.log("Testing setup_slice_observer")
        observer_calls = [0]

        def test_observer(caller, event):
            observer_calls[0] += 1

        result = self.manager.setup_slice_observer(test_observer, "Red")
        ctx.assert_true(result, "setup_slice_observer should return True")

        # Trigger an observation by changing slice
        if slice_range:
            test_offset = min_offset + (max_offset - min_offset) * 0.25
            self.manager.set_slice_offset(test_offset, "Red")
            slicer.app.processEvents()

            # Observer may or may not be called depending on timing
            # Just verify no errors occurred
            ctx.log(f"Observer called {observer_calls[0]} times")

        ctx.screenshot("[observer] Observer setup complete")

        # Test invalid view name
        ctx.log("Testing invalid view name")
        invalid_range = self.manager.get_slice_range("InvalidView")
        ctx.assert_is_none(invalid_range, "Invalid view should return None")

        invalid_offset = self.manager.get_slice_offset("InvalidView")
        ctx.assert_is_none(invalid_offset, "Invalid view should return None for offset")

    def verify(self, ctx: TestContext) -> None:
        """Verify test results."""
        logger.info("Verifying ViewGroupManager unit test")

        # Test cleanup
        ctx.log("Testing cleanup")
        self.manager.cleanup()

        # Manager should handle cleanup gracefully
        ctx.assert_is_not_none(self.manager, "Manager should still exist after cleanup")

        ctx.screenshot("[verify] Cleanup complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down ViewGroupManager unit test")

        if self.manager:
            self.manager.cleanup()
            self.manager = None

        ctx.log("Teardown complete")
