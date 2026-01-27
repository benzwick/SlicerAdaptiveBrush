"""Integration tests for Reviewer module.

Tests the full workflow: load run, select trial, navigate, bookmark, rate, export.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

from .fixtures import MockOptimizationRunFactory, MockSegmentationFactory

logger = logging.getLogger(__name__)


@register_test(category="reviewer_integration")
class TestReviewerIntegration(TestCase):
    """Full integration test for Reviewer module workflow."""

    name = "integration_reviewer_workflow"
    description = "Test complete workflow: load, navigate, bookmark, rate, export"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None
        self.mock_run_factory = None
        self.mock_seg_factory = None
        self.run_path = None

    def setup(self, ctx: TestContext) -> None:
        """Set up complete test environment."""
        logger.info("Setting up Reviewer integration test")

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

        # Create mock optimization run
        self.mock_run_factory = MockOptimizationRunFactory()
        self.run_path = self.mock_run_factory.create_run(
            name="integration_test_run",
            num_trials=10,
        )

        # Create mock segmentations for testing
        self.mock_seg_factory = MockSegmentationFactory()

        # Switch to Reviewer module
        slicer.util.selectModule("SegmentEditorAdaptiveBrushReviewer")
        slicer.app.processEvents()

        # Get widget reference
        module_widget = slicer.modules.segmenteditoradaptivebrushreviewer.widgetRepresentation()
        self.widget = module_widget.self()

        if self.widget is None:
            raise RuntimeError("Failed to get Reviewer widget")

        # Configure results loader to use our mock directory
        self.widget.results_loader.results_dir = self.run_path.parent

        slicer.app.processEvents()
        ctx.screenshot("[setup] Integration test environment ready")

    def run(self, ctx: TestContext) -> None:
        """Run complete integration workflow."""
        logger.info("Running Reviewer integration tests")

        # ================================================================
        # Phase 1: Load and select optimization run
        # ================================================================
        ctx.log("Phase 1: Load and select optimization run")

        # Refresh run list
        self.widget._refresh_run_list()
        slicer.app.processEvents()

        ctx.assert_greater(
            self.widget.runComboBox.count,
            0,
            "Should have at least one run after refresh",
        )
        ctx.screenshot("[phase1_runs] Run list populated")

        # Select the run
        self.widget.runComboBox.setCurrentIndex(0)
        slicer.app.processEvents()

        ctx.assert_is_not_none(
            self.widget.current_run,
            "current_run should be set after selection",
        )
        ctx.log(f"Selected run: {self.widget.current_run.name}")
        ctx.screenshot("[phase1_selected] Run selected")

        # Verify trials loaded
        ctx.assert_greater(
            self.widget.trialComboBox.count,
            0,
            "Should have trials after run selection",
        )
        ctx.log(f"Loaded {self.widget.trialComboBox.count} trials")

        # ================================================================
        # Phase 2: Select and display trial
        # ================================================================
        ctx.log("Phase 2: Select and display trial")

        # Select first trial
        self.widget.trialComboBox.setCurrentIndex(0)
        slicer.app.processEvents()

        ctx.assert_is_not_none(
            self.widget.current_trial,
            "current_trial should be set after selection",
        )
        ctx.log(f"Selected trial #{self.widget.current_trial.trial_number}")
        ctx.screenshot("[phase2_trial] Trial selected")

        # Verify parameters displayed
        params_text = self.widget.paramsText.toPlainText()
        ctx.assert_true(
            len(params_text) > 0,
            "Parameters text should be populated",
        )
        ctx.log(f"Parameters: {params_text[:100]}...")

        # Verify metrics displayed
        metrics_text = self.widget.metricsText.toPlainText()
        ctx.assert_true(
            len(metrics_text) > 0,
            "Metrics text should be populated",
        )

        # ================================================================
        # Phase 3: Navigate slices
        # ================================================================
        ctx.log("Phase 3: Navigate slices")

        # Initialize slice navigation
        self.widget.total_slices = 50
        self.widget.sliceSlider.setMaximum(49)
        self.widget.sliceSlider.setValue(0)
        slicer.app.processEvents()

        # Navigate using buttons
        self.widget.nextSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            1,
            "Slice should advance to 1",
        )

        # Jump to middle
        self.widget.sliceSlider.setValue(25)
        slicer.app.processEvents()
        ctx.screenshot("[phase3_middle] Navigated to middle slice")

        # Use fast navigation
        self.widget.nextFastButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            35,
            "Should jump 10 slices forward",
        )

        # Go to end
        self.widget.lastSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            49,
            "Should be at last slice",
        )
        ctx.screenshot("[phase3_end] Navigated to last slice")

        # ================================================================
        # Phase 4: Add bookmarks at interesting locations
        # ================================================================
        ctx.log("Phase 4: Add bookmarks")

        # Clear any existing bookmarks
        self.widget.bookmarks.clear_all()
        self.widget._update_bookmark_combo()

        # Navigate to first location and bookmark
        self.widget.sliceSlider.setValue(10)
        slicer.app.processEvents()

        self.widget.bookmarkDescEdit.setText("Potential boundary issue")
        self.widget.addBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            1,
            "Should have 1 bookmark",
        )
        ctx.screenshot("[phase4_bookmark1] First bookmark added")

        # Navigate to second location and bookmark
        self.widget.sliceSlider.setValue(30)
        slicer.app.processEvents()

        self.widget.bookmarkDescEdit.setText("Good segmentation area")
        self.widget.addBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            2,
            "Should have 2 bookmarks",
        )
        ctx.screenshot("[phase4_bookmark2] Second bookmark added")

        # Test bookmark restoration
        self.widget.sliceSlider.setValue(45)  # Move away
        slicer.app.processEvents()

        self.widget.bookmarkCombo.setCurrentIndex(0)
        self.widget.restoreBookmarkButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[phase4_restored] Bookmark restored")

        # ================================================================
        # Phase 5: Change visualization settings
        # ================================================================
        ctx.log("Phase 5: Visualization settings")

        # Test layout changes
        self.widget.layoutFourUpButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[phase5_fourup] Four-Up layout")

        self.widget.layoutConventionalButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[phase5_conventional] Conventional layout")

        # Test view mode changes
        for button in self.widget.viewModeGroup.buttons():
            if button.text == "Transparent":
                button.click()
                slicer.app.processEvents()
                break
        ctx.screenshot("[phase5_transparent] Transparent view mode")

        for button in self.widget.viewModeGroup.buttons():
            if button.text == "Outline":
                button.click()
                slicer.app.processEvents()
                break
        ctx.screenshot("[phase5_outline] Outline view mode")

        # ================================================================
        # Phase 6: Rate trial
        # ================================================================
        ctx.log("Phase 6: Rate trial")

        from SegmentEditorAdaptiveBrushReviewerLib import Rating

        # Select a rating
        for button in self.widget.ratingGroup.buttons():
            if self.widget.ratingGroup.id(button) == Rating.ACCEPT.value:
                button.click()
                break
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.ratingGroup.checkedId(),
            Rating.ACCEPT.value,
            "Accept rating should be selected",
        )

        # Add notes
        self.widget.notesEdit.setText("Good segmentation quality, minor edge artifacts")
        slicer.app.processEvents()

        # Save rating
        self.widget.saveRatingButton.click()
        slicer.app.processEvents()

        # Verify status updated
        status = self.widget.currentRatingLabel.text
        ctx.assert_true(
            "Accept" in status or "saved" in status.lower(),
            f"Status should indicate saved: {status}",
        )
        ctx.screenshot("[phase6_rated] Trial rated and saved")

        # ================================================================
        # Phase 7: Navigate between trials
        # ================================================================
        ctx.log("Phase 7: Navigate between trials")

        initial_trial_idx = self.widget.trialComboBox.currentIndex

        # Go to next trial
        self.widget.nextTrialButton.click()
        slicer.app.processEvents()

        if self.widget.trialComboBox.count > 1:
            ctx.assert_not_equal(
                self.widget.trialComboBox.currentIndex,
                initial_trial_idx,
                "Should move to different trial",
            )

        ctx.screenshot("[phase7_next_trial] Next trial")

        # Rate this trial differently
        for button in self.widget.ratingGroup.buttons():
            if self.widget.ratingGroup.id(button) == Rating.MINOR.value:
                button.click()
                break
        slicer.app.processEvents()

        self.widget.notesEdit.setText("Minor corrections needed at boundaries")
        self.widget.saveRatingButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[phase7_rated2] Second trial rated")

        # ================================================================
        # Phase 8: Workflow recording (if available)
        # ================================================================
        ctx.log("Phase 8: Workflow recording")

        # Create a test segmentation for recording
        gold_node, test_node = self.mock_seg_factory.create_gold_test_pair(
            self.volume_node,
            overlap_ratio=0.85,
            center_ijk=(128, 128, 64),
            radius=15,
        )

        # Set up viz controller with the test node
        self.widget.viz_controller.test_seg_node = test_node
        self.widget.viz_controller.gold_seg_node = gold_node

        # Start recording
        self.widget.startRecordingButton.click()
        slicer.app.processEvents()

        ctx.assert_true(
            self.widget.sequence_recorder.is_recording,
            "Should be recording after start",
        )
        ctx.screenshot("[phase8_recording] Recording started")

        # Record a few steps
        for _ in range(3):
            self.widget.recordStepButton.click()
            slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.sequence_recorder.step_count,
            3,
            "Should have 3 recorded steps",
        )
        ctx.screenshot("[phase8_steps] Steps recorded")

        # Stop recording
        self.widget.stopRecordingButton.click()
        slicer.app.processEvents()

        ctx.assert_false(
            self.widget.sequence_recorder.is_recording,
            "Should not be recording after stop",
        )
        ctx.screenshot("[phase8_stopped] Recording stopped")

        # Test playback navigation
        self.widget.workflowFirstButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[phase8_playback_first] Playback at first step")

        self.widget.workflowNextButton.click()
        slicer.app.processEvents()
        ctx.screenshot("[phase8_playback_next] Playback at next step")

        # ================================================================
        # Phase 9: Final verification
        # ================================================================
        ctx.log("Phase 9: Final verification")

        # Verify all components are still functional
        ctx.assert_is_not_none(self.widget.current_run, "Run still loaded")
        ctx.assert_is_not_none(self.widget.current_trial, "Trial still selected")
        ctx.assert_greater(self.widget.bookmarks.count, 0, "Bookmarks exist")

        # Verify ratings were saved
        rating_count = len(self.widget.rating_manager.get_all_ratings())
        ctx.assert_greater(rating_count, 0, "Ratings should be saved")
        ctx.log(f"Total ratings saved: {rating_count}")

        ctx.screenshot("[phase9_complete] Integration test complete")

    def verify(self, ctx: TestContext) -> None:
        """Verify integration test results."""
        logger.info("Verifying Reviewer integration test")

        # Verify all major components are functional
        ctx.assert_is_not_none(self.widget, "Widget exists")
        ctx.assert_is_not_none(self.widget.results_loader, "Results loader exists")
        ctx.assert_is_not_none(self.widget.viz_controller, "Viz controller exists")
        ctx.assert_is_not_none(self.widget.rating_manager, "Rating manager exists")
        ctx.assert_is_not_none(self.widget.bookmarks, "Bookmarks manager exists")
        ctx.assert_is_not_none(self.widget.sequence_recorder, "Sequence recorder exists")
        ctx.assert_is_not_none(self.widget.view_group_manager, "View group manager exists")

        ctx.screenshot("[verify] Final verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down Reviewer integration test")

        # Clean up sequence recorder
        if self.widget and self.widget.sequence_recorder:
            self.widget.sequence_recorder.cleanup()

        # Clean up bookmarks
        if self.widget and self.widget.bookmarks:
            self.widget.bookmarks.clear_all()

        # Clean up mock factories
        if self.mock_run_factory:
            self.mock_run_factory.cleanup()
            self.mock_run_factory = None

        if self.mock_seg_factory:
            self.mock_seg_factory.cleanup()
            self.mock_seg_factory = None

        self.widget = None
        ctx.log("Teardown complete")
