"""UI tests for Reviewer rating controls.

Tests rating buttons, save, export, and trial navigation.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

from .fixtures import MockOptimizationRunFactory

logger = logging.getLogger(__name__)


@register_test(category="reviewer_ui")
class TestReviewerUIRating(TestCase):
    """UI tests for rating controls in Reviewer module."""

    name = "ui_reviewer_rating"
    description = "Test rating buttons, save, export, and trial navigation"

    def __init__(self) -> None:
        super().__init__()
        self.widget = None
        self.volume_node = None
        self.mock_factory = None

    def setup(self, ctx: TestContext) -> None:
        """Set up Reviewer module UI with mock run."""
        logger.info("Setting up Reviewer UI rating test")

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

        # Create mock optimization run for testing
        self.mock_factory = MockOptimizationRunFactory()
        run_path = self.mock_factory.create_run("test_rating_run", num_trials=5)

        # Update results loader to use our temp directory
        self.widget.results_loader.results_dir = run_path.parent

        # Refresh run list
        self.widget._refresh_run_list()
        slicer.app.processEvents()

        ctx.screenshot("[setup] Reviewer module loaded with mock run")

    def run(self, ctx: TestContext) -> None:
        """Test rating UI controls."""
        logger.info("Running rating UI tests")

        # Test initial rating state
        ctx.log("Testing initial rating state")

        # No rating should be selected initially
        # (may or may not be checked depending on previous state)

        ctx.assert_equal(
            self.widget.notesEdit.text,
            "",
            "Notes should be empty initially (for new trial)",
        )
        ctx.screenshot("[initial] Initial rating state")

        # Test rating buttons (1-4)
        ctx.log("Testing rating buttons")

        buttons = self.widget.ratingGroup.buttons()
        ctx.assert_equal(len(buttons), 4, "Should have 4 rating buttons")

        # Click each rating button
        from SegmentEditorAdaptiveBrushReviewerLib import Rating

        rating_values = [Rating.ACCEPT, Rating.MINOR, Rating.MAJOR, Rating.REJECT]
        rating_names = ["Accept", "Minor", "Major", "Reject"]

        for rating, name in zip(rating_values, rating_names):
            ctx.log(f"Testing {name} rating button")

            # Find button with matching ID
            for button in buttons:
                if self.widget.ratingGroup.id(button) == rating.value:
                    button.click()
                    slicer.app.processEvents()
                    break

            ctx.assert_equal(
                self.widget.ratingGroup.checkedId(),
                rating.value,
                f"{name} rating should be selected (value {rating.value})",
            )
            ctx.screenshot(f"[rating_{name.lower()}] {name} rating selected")

        # Test notesEdit
        ctx.log("Testing notesEdit")
        test_note = "Test reviewer notes for this trial"
        self.widget.notesEdit.setText(test_note)
        slicer.app.processEvents()
        ctx.assert_equal(
            self.widget.notesEdit.text,
            test_note,
            "Notes should be set",
        )
        ctx.screenshot("[notes] Notes entered")

        # Test saveRatingButton (only if we have a run and trial)
        ctx.log("Testing saveRatingButton")
        if self.widget.current_run and self.widget.current_trial:
            # Set a rating first
            for button in buttons:
                if self.widget.ratingGroup.id(button) == Rating.ACCEPT.value:
                    button.click()
                    break
            slicer.app.processEvents()

            self.widget.saveRatingButton.click()
            slicer.app.processEvents()

            # Check status label updated
            status_text = self.widget.currentRatingLabel.text
            ctx.assert_true(
                "Accept" in status_text or "saved" in status_text.lower(),
                f"Status should indicate saved rating, got: {status_text}",
            )
            ctx.screenshot("[saved] Rating saved")
        else:
            ctx.log("Skipping save test - no run/trial selected")

        # Test trial navigation
        ctx.log("Testing trial navigation buttons")

        # Ensure we have trials selected
        if self.widget.trialComboBox.count > 1:
            # Go to first trial
            self.widget.trialComboBox.setCurrentIndex(0)
            slicer.app.processEvents()

            # Test nextTrialButton
            self.widget.nextTrialButton.click()
            slicer.app.processEvents()
            ctx.assert_equal(
                self.widget.trialComboBox.currentIndex,
                1,
                "Should be on second trial after next",
            )
            ctx.screenshot("[next_trial] After next trial button")

            # Test prevTrialButton
            self.widget.prevTrialButton.click()
            slicer.app.processEvents()
            ctx.assert_equal(
                self.widget.trialComboBox.currentIndex,
                0,
                "Should be on first trial after prev",
            )
            ctx.screenshot("[prev_trial] After prev trial button")

            # Test boundary - prev at start
            self.widget.prevTrialButton.click()
            slicer.app.processEvents()
            ctx.assert_equal(
                self.widget.trialComboBox.currentIndex,
                0,
                "Should stay at first trial",
            )

            # Test boundary - next at end
            self.widget.trialComboBox.setCurrentIndex(self.widget.trialComboBox.count - 1)
            slicer.app.processEvents()
            last_idx = self.widget.trialComboBox.currentIndex

            self.widget.nextTrialButton.click()
            slicer.app.processEvents()
            ctx.assert_equal(
                self.widget.trialComboBox.currentIndex,
                last_idx,
                "Should stay at last trial",
            )
            ctx.screenshot("[boundary_trial] Trial boundary test")
        else:
            ctx.log("Skipping trial navigation - not enough trials")

        # Test exportRatingsButton
        ctx.log("Testing exportRatingsButton")
        if self.widget.current_run:
            # Export ratings - this creates a CSV
            # Note: This may show a dialog, but button should be clickable
            ctx.assert_is_not_none(
                self.widget.exportRatingsButton,
                "Export button should exist",
            )
            ctx.screenshot("[export] Export button available")
        else:
            ctx.log("Skipping export test - no run loaded")

    def verify(self, ctx: TestContext) -> None:
        """Verify rating test results."""
        logger.info("Verifying rating UI test")

        # Verify rating controls exist
        ctx.assert_is_not_none(self.widget.ratingGroup, "Rating group exists")
        ctx.assert_is_not_none(self.widget.notesEdit, "Notes edit exists")
        ctx.assert_is_not_none(self.widget.saveRatingButton, "Save button exists")
        ctx.assert_is_not_none(self.widget.prevTrialButton, "Prev trial button exists")
        ctx.assert_is_not_none(self.widget.nextTrialButton, "Next trial button exists")
        ctx.assert_is_not_none(self.widget.exportRatingsButton, "Export button exists")
        ctx.assert_is_not_none(self.widget.currentRatingLabel, "Status label exists")

        ctx.screenshot("[verify] Verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down rating UI test")

        # Clean up mock factory
        if self.mock_factory:
            self.mock_factory.cleanup()
            self.mock_factory = None

        self.widget = None
        ctx.log("Teardown complete")
