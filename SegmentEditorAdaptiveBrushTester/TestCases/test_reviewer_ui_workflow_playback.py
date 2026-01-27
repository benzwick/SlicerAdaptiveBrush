"""UI tests for Reviewer workflow playback controls.

Tests the recording start/stop/step and playback navigation.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="reviewer_ui")
class TestReviewerUIWorkflowPlayback(TestCase):
    """UI tests for workflow playback controls in Reviewer module."""

    name = "ui_reviewer_workflow_playback"
    description = "Test recording start/stop/step buttons and playback controls"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None
        self.seg_node = None

    def setup(self, ctx: TestContext) -> None:
        """Set up Reviewer module UI with segmentation."""
        logger.info("Setting up Reviewer UI workflow playback test")

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

        # Create a segmentation node for recording
        self.seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.seg_node.SetName("WorkflowTestSegmentation")
        self.seg_node.CreateDefaultDisplayNodes()
        self.seg_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        self.seg_node.GetSegmentation().AddEmptySegment("TestSegment")

        # Switch to Reviewer module
        slicer.util.selectModule("SegmentEditorAdaptiveBrushReviewer")
        slicer.app.processEvents()

        # Get widget reference
        module_widget = slicer.modules.segmenteditoradaptivebrushreviewer.widgetRepresentation()
        self.widget = module_widget.self()

        if self.widget is None:
            raise RuntimeError("Failed to get Reviewer widget")

        # Set up viz controller to have a test node
        self.widget.viz_controller.test_seg_node = self.seg_node

        slicer.app.processEvents()
        ctx.screenshot("[setup] Reviewer module loaded with segmentation")

    def run(self, ctx: TestContext) -> None:
        """Test workflow playback UI controls."""
        logger.info("Running workflow playback UI tests")

        # Test initial state
        ctx.log("Testing initial recording state")
        ctx.assert_true(
            self.widget.startRecordingButton.isEnabled(),
            "Start Recording should be enabled initially",
        )
        ctx.assert_false(
            self.widget.stopRecordingButton.isEnabled(),
            "Stop Recording should be disabled initially",
        )
        ctx.assert_false(
            self.widget.recordStepButton.isEnabled(),
            "Record Step should be disabled initially",
        )
        ctx.assert_equal(
            self.widget.workflowStepLabel.text,
            "0/0",
            "Step label should show 0/0",
        )
        ctx.screenshot("[initial] Initial recording state")

        # Test startRecordingButton
        ctx.log("Testing startRecordingButton")
        self.widget.startRecordingButton.click()
        slicer.app.processEvents()

        ctx.assert_false(
            self.widget.startRecordingButton.isEnabled(),
            "Start Recording should be disabled after starting",
        )
        ctx.assert_true(
            self.widget.stopRecordingButton.isEnabled(),
            "Stop Recording should be enabled after starting",
        )
        ctx.assert_true(
            self.widget.recordStepButton.isEnabled(),
            "Record Step should be enabled after starting",
        )
        ctx.screenshot("[recording] Recording started")

        # Test recordStepButton
        ctx.log("Testing recordStepButton")
        self.widget.recordStepButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.sequence_recorder.step_count,
            1,
            "Should have 1 step after first record",
        )
        ctx.screenshot("[step1] First step recorded")

        # Record more steps
        self.widget.recordStepButton.click()
        slicer.app.processEvents()
        self.widget.recordStepButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.sequence_recorder.step_count,
            3,
            "Should have 3 steps",
        )
        ctx.screenshot("[step3] Three steps recorded")

        # Test stopRecordingButton
        ctx.log("Testing stopRecordingButton")
        self.widget.stopRecordingButton.click()
        slicer.app.processEvents()

        ctx.assert_true(
            self.widget.startRecordingButton.isEnabled(),
            "Start Recording should be enabled after stopping",
        )
        ctx.assert_false(
            self.widget.stopRecordingButton.isEnabled(),
            "Stop Recording should be disabled after stopping",
        )
        ctx.assert_false(
            self.widget.recordStepButton.isEnabled(),
            "Record Step should be disabled after stopping",
        )
        ctx.screenshot("[stopped] Recording stopped")

        # Test workflow navigation buttons
        ctx.log("Testing workflow navigation buttons")

        # Verify step count is maintained
        ctx.assert_equal(
            self.widget.sequence_recorder.step_count,
            3,
            "Step count should be preserved after stop",
        )

        # Update workflow UI
        self.widget._update_workflow_ui()
        slicer.app.processEvents()

        # Test workflowSlider
        ctx.log("Testing workflowSlider")
        ctx.assert_equal(
            self.widget.workflowSlider.maximum,
            2,
            "Slider max should be 2 (0-indexed, 3 steps)",
        )

        self.widget.workflowSlider.setValue(1)
        slicer.app.processEvents()
        ctx.screenshot("[slider1] Workflow slider at step 1")

        # Test workflowFirstButton
        ctx.log("Testing workflowFirstButton")
        self.widget.workflowSlider.setValue(2)  # Start at end
        slicer.app.processEvents()

        self.widget.workflowFirstButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[first] After workflow first button")

        # Test workflowLastButton
        ctx.log("Testing workflowLastButton")
        self.widget.workflowLastButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[last] After workflow last button")

        # Test workflowPrevButton
        ctx.log("Testing workflowPrevButton")
        self.widget.workflowPrevButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[prev] After workflow prev button")

        # Test workflowNextButton
        ctx.log("Testing workflowNextButton")
        self.widget.workflowNextButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[next] After workflow next button")

        # Test addNoteButton
        ctx.log("Testing addNoteButton")
        self.widget.workflowNoteEdit.setText("Test annotation note")
        self.widget.addNoteButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.workflowNoteEdit.text,
            "",
            "Note edit should be cleared after adding",
        )
        ctx.screenshot("[note_added] Note added to step")

        # Test workflowNotesLabel updates
        self.widget._update_workflow_ui()
        slicer.app.processEvents()

        # Note label should contain something
        notes_text = self.widget.workflowNotesLabel.text
        ctx.assert_true(
            len(notes_text) > 0,
            "Notes label should have content",
        )
        ctx.log(f"Notes label: {notes_text}")
        ctx.screenshot("[notes_display] Notes label updated")

    def verify(self, ctx: TestContext) -> None:
        """Verify workflow playback test results."""
        logger.info("Verifying workflow playback UI test")

        # Verify controls exist
        ctx.assert_is_not_none(self.widget.startRecordingButton, "Start button exists")
        ctx.assert_is_not_none(self.widget.stopRecordingButton, "Stop button exists")
        ctx.assert_is_not_none(self.widget.recordStepButton, "Record step button exists")
        ctx.assert_is_not_none(self.widget.workflowSlider, "Workflow slider exists")
        ctx.assert_is_not_none(self.widget.workflowFirstButton, "First button exists")
        ctx.assert_is_not_none(self.widget.workflowPrevButton, "Prev button exists")
        ctx.assert_is_not_none(self.widget.workflowNextButton, "Next button exists")
        ctx.assert_is_not_none(self.widget.workflowLastButton, "Last button exists")

        ctx.screenshot("[verify] Verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down workflow playback UI test")

        # Clean up sequence recorder
        if self.widget and self.widget.sequence_recorder:
            self.widget.sequence_recorder.cleanup()

        self.widget = None
        ctx.log("Teardown complete")
