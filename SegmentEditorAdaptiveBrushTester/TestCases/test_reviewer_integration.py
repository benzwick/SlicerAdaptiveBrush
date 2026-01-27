"""Integration tests for Reviewer module.

Tests the full workflow: load run, select trial, navigate, bookmark, rate, export.
Uses REAL optimization runs from optimization_results/ for realistic testing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

from .fixtures import MockSegmentationFactory

logger = logging.getLogger(__name__)


def get_optimization_results_dir() -> Path:
    """Get the optimization_results directory path."""
    # Navigate from test file to SlicerAdaptiveBrush directory
    # test file: TestCases/test_reviewer_integration.py
    # TestCases -> SegmentEditorAdaptiveBrushTester -> SlicerAdaptiveBrush
    test_dir = Path(__file__).parent  # TestCases
    slicer_extension_dir = test_dir.parent.parent  # SlicerAdaptiveBrush (inner)
    return slicer_extension_dir / "optimization_results"


@register_test(category="reviewer_integration")
class TestReviewerIntegration(TestCase):
    """Full integration test for Reviewer module workflow.

    Uses real optimization runs from optimization_results/ directory.
    This tests against actual data that varies between runs, providing
    more realistic and comprehensive test coverage.
    """

    name = "integration_reviewer_workflow"
    description = "Test complete workflow with REAL optimization runs"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None
        self.mock_seg_factory = None
        self.results_dir = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test environment using real optimization runs."""
        logger.info("Setting up Reviewer integration test with real runs")

        # Enable quiet mode to suppress popups during testing
        import SegmentEditorAdaptiveBrushReviewer

        SegmentEditorAdaptiveBrushReviewer.set_quiet_mode(True)

        # Check for real optimization runs
        self.results_dir = get_optimization_results_dir()
        if not self.results_dir.exists():
            raise RuntimeError(
                f"optimization_results directory not found: {self.results_dir}\n"
                "Run an optimization first to generate test data."
            )

        # Find runs with results.json
        runs = [
            p for p in self.results_dir.iterdir() if p.is_dir() and (p / "results.json").exists()
        ]
        if not runs:
            raise RuntimeError(
                f"No optimization runs found in {self.results_dir}\n"
                "Run an optimization first to generate test data."
            )

        # Sort by name (timestamp) descending to get latest
        runs.sort(reverse=True)
        latest_run = runs[0]
        ctx.log(f"Using latest optimization run: {latest_run.name}")
        ctx.log(f"Found {len(runs)} total runs in optimization_results/")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load sample data (for workflow recording phase)
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        # Create mock segmentations for workflow recording test (Phase 8)
        self.mock_seg_factory = MockSegmentationFactory()

        # Switch to Reviewer module
        slicer.util.selectModule("SegmentEditorAdaptiveBrushReviewer")
        slicer.app.processEvents()

        # Get widget reference
        module_widget = slicer.modules.segmenteditoradaptivebrushreviewer.widgetRepresentation()
        self.widget = module_widget.self()

        if self.widget is None:
            raise RuntimeError("Failed to get Reviewer widget")

        # Configure results loader to use the real optimization_results directory
        self.widget.results_loader.results_dir = self.results_dir

        slicer.app.processEvents()
        ctx.screenshot("[setup] Integration test with real optimization runs")

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

        # Get actual slice count from volume (use axial dimension)
        if self.volume_node:
            dims = self.volume_node.GetImageData().GetDimensions()
            total_slices = dims[2]  # Axial slices
        else:
            total_slices = 50  # Fallback

        # Initialize slice navigation with actual dimensions
        self.widget.total_slices = total_slices
        self.widget.sliceSlider.setMaximum(total_slices - 1)
        self.widget.sliceSlider.setValue(0)
        slicer.app.processEvents()
        ctx.log(f"Volume has {total_slices} axial slices")

        # Navigate using buttons
        self.widget.nextSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            1,
            "Slice should advance to 1",
        )

        # Jump to middle
        middle_slice = total_slices // 2
        self.widget.sliceSlider.setValue(middle_slice)
        slicer.app.processEvents()
        ctx.screenshot(f"[phase3_middle] Navigated to slice {middle_slice}")

        # Use fast navigation (jumps 10 slices)
        expected_after_fast = min(middle_slice + 10, total_slices - 1)
        self.widget.nextFastButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            expected_after_fast,
            f"Should jump to slice {expected_after_fast}",
        )

        # Go to end
        self.widget.lastSliceButton.click()
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.sliceSlider.value,
            total_slices - 1,
            f"Should be at last slice ({total_slices - 1})",
        )
        ctx.screenshot("[phase3_end] Navigated to last slice")

        # ================================================================
        # Phase 4: Add bookmarks at interesting locations
        # ================================================================
        ctx.log("Phase 4: Add bookmarks")

        # Clear any existing bookmarks
        self.widget.bookmarks.clear_all()
        self.widget._update_bookmark_combo()

        # Navigate to first location (20% through volume) and bookmark
        bookmark1_slice = total_slices // 5
        self.widget.sliceSlider.setValue(bookmark1_slice)
        slicer.app.processEvents()

        self.widget.bookmarkDescEdit.setText("Potential boundary issue")
        self.widget.addBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            1,
            "Should have 1 bookmark",
        )
        ctx.screenshot(f"[phase4_bookmark1] Bookmark at slice {bookmark1_slice}")

        # Navigate to second location (60% through volume) and bookmark
        bookmark2_slice = (total_slices * 3) // 5
        self.widget.sliceSlider.setValue(bookmark2_slice)
        slicer.app.processEvents()

        self.widget.bookmarkDescEdit.setText("Good segmentation area")
        self.widget.addBookmarkButton.click()
        slicer.app.processEvents()

        ctx.assert_equal(
            self.widget.bookmarks.count,
            2,
            "Should have 2 bookmarks",
        )
        ctx.screenshot(f"[phase4_bookmark2] Bookmark at slice {bookmark2_slice}")

        # Test bookmark restoration - move to 90% through volume first
        away_slice = (total_slices * 9) // 10
        self.widget.sliceSlider.setValue(away_slice)
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

        # Clean up mock segmentation factory (used for workflow recording test)
        if self.mock_seg_factory:
            self.mock_seg_factory.cleanup()
            self.mock_seg_factory = None

        self.widget = None
        ctx.log("Teardown complete")
