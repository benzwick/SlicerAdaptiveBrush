"""Unit tests for SequenceRecorder class.

Tests the sequence-based workflow recording functionality
without requiring full module setup.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_unit")
class TestSequenceRecorderUnit(TestCase):
    """Unit tests for SequenceRecorder class."""

    name = "unit_sequence_recorder"
    description = "Test SequenceRecorder initialization, recording, and playback"

    def __init__(self) -> None:
        super().__init__()
        self.recorder = None
        self.volume_node = None
        self.seg_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test with sample data."""
        logger.info("Setting up SequenceRecorder unit test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Import here to avoid issues when Slicer not available
        from SegmentEditorAdaptiveBrushReviewerLib import SequenceRecorder

        self.recorder = SequenceRecorder()

        # Load sample data for recording
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        # Create segmentation node
        self.seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.seg_node.SetName("TestSegmentation")
        self.seg_node.CreateDefaultDisplayNodes()
        self.seg_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        self.seg_node.GetSegmentation().AddEmptySegment("TestSegment")

        ctx.screenshot("[setup] Sample data loaded, recorder initialized")

    def run(self, ctx: TestContext) -> None:
        """Test SequenceRecorder operations."""
        logger.info("Running SequenceRecorder unit tests")

        # Test initial state
        ctx.log("Testing initial state")
        ctx.assert_false(self.recorder.is_recording, "Should not be recording initially")
        ctx.assert_equal(self.recorder.step_count, 0, "Step count should be 0 initially")
        ctx.assert_is_none(
            self.recorder.get_browser_node(),
            "Browser node should be None before recording",
        )
        ctx.screenshot("[initial] Recorder initial state")

        # Test start recording
        ctx.log("Testing start_recording")
        result = self.recorder.start_recording(self.seg_node, self.volume_node)
        ctx.assert_true(result, "start_recording should return True")
        ctx.assert_true(self.recorder.is_recording, "Should be recording after start")
        ctx.assert_is_not_none(
            self.recorder.get_browser_node(),
            "Browser node should exist after start",
        )
        slicer.app.processEvents()
        ctx.screenshot("[started] Recording started")

        # Test record_step
        ctx.log("Testing record_step")
        step1 = self.recorder.record_step("First brush stroke")
        ctx.assert_equal(step1, 0, "First step should be 0")
        ctx.assert_equal(self.recorder.step_count, 1, "Step count should be 1")

        step2 = self.recorder.record_step("Second brush stroke")
        ctx.assert_equal(step2, 1, "Second step should be 1")
        ctx.assert_equal(self.recorder.step_count, 2, "Step count should be 2")

        step3 = self.recorder.record_step("Third action")
        ctx.assert_equal(step3, 2, "Third step should be 2")
        ctx.assert_equal(self.recorder.step_count, 3, "Step count should be 3")

        slicer.app.processEvents()
        ctx.screenshot("[recorded] Three steps recorded")

        # Test duplicate start (should fail)
        ctx.log("Testing duplicate start_recording")
        result2 = self.recorder.start_recording(self.seg_node, self.volume_node)
        ctx.assert_false(result2, "Starting while recording should return False")

        # Test add_note
        ctx.log("Testing add_note")
        self.recorder.add_note("Important observation about boundary")
        # Note is added to current step (step 2)

        # Test get_note_at_step
        note = self.recorder.get_note_at_step(0)
        ctx.assert_equal(note, "First brush stroke", "Should get note from step 0")

        # Test goto_step
        ctx.log("Testing goto_step")
        result = self.recorder.goto_step(0)
        ctx.assert_true(result, "goto_step(0) should succeed")

        result = self.recorder.goto_step(2)
        ctx.assert_true(result, "goto_step(2) should succeed")

        result = self.recorder.goto_step(99)
        ctx.assert_false(result, "goto_step(99) should fail (out of range)")

        result = self.recorder.goto_step(-1)
        ctx.assert_false(result, "goto_step(-1) should fail (negative)")

        slicer.app.processEvents()
        ctx.screenshot("[navigation] Step navigation tested")

        # Test get_sequence
        ctx.log("Testing get_sequence")
        seg_seq = self.recorder.get_sequence("segmentation")
        ctx.assert_is_not_none(seg_seq, "Should have segmentation sequence")

        notes_seq = self.recorder.get_sequence("notes")
        ctx.assert_is_not_none(notes_seq, "Should have notes sequence")

        invalid_seq = self.recorder.get_sequence("invalid")
        ctx.assert_is_none(invalid_seq, "Invalid sequence name should return None")

        # Test stop_recording
        ctx.log("Testing stop_recording")
        self.recorder.stop_recording()
        ctx.assert_false(self.recorder.is_recording, "Should not be recording after stop")
        ctx.assert_equal(self.recorder.step_count, 3, "Step count preserved after stop")

        # Can still navigate after stopping
        result = self.recorder.goto_step(1)
        ctx.assert_true(result, "Should still be able to navigate after stop")

        ctx.screenshot("[stopped] Recording stopped, still navigable")

    def verify(self, ctx: TestContext) -> None:
        """Verify test results."""
        logger.info("Verifying SequenceRecorder unit test")

        # Final verification
        ctx.assert_is_not_none(self.recorder, "Recorder should exist")

        # Test cleanup
        self.recorder.cleanup()
        ctx.assert_false(self.recorder.is_recording, "Not recording after cleanup")
        ctx.assert_equal(self.recorder.step_count, 0, "Step count 0 after cleanup")
        ctx.assert_is_none(
            self.recorder.get_browser_node(),
            "Browser node None after cleanup",
        )

        ctx.screenshot("[verify] Cleanup complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down SequenceRecorder unit test")

        if self.recorder:
            self.recorder.cleanup()
            self.recorder = None

        ctx.log("Teardown complete")
