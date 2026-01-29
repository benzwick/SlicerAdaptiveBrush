"""Documentation screenshots for Reviewer module.

Captures screenshots of the Reviewer module UI for documentation:
- Run selection
- Trial comparison view
- Rating interface
- Metrics display
- Export functionality

These screenshots are tagged for documentation extraction.
"""

from __future__ import annotations

import logging

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="docs")
class TestDocsReviewer(TestCase):
    """Generate documentation screenshots for Reviewer module."""

    name = "docs_reviewer"
    description = "Capture documentation screenshots for Reviewer module UI"

    def __init__(self) -> None:
        super().__init__()
        self.reviewer_widget = None
        self.mock_run_path = None

    def setup(self, ctx: TestContext) -> None:
        """Open Reviewer module."""
        logger.info("Setting up Reviewer module documentation")

        # Check if Reviewer module is available
        try:
            slicer.util.selectModule("SegmentEditorAdaptiveBrushReviewer")
            slicer.app.processEvents()

            # Get the module widget
            module = slicer.modules.segmenteditoradaptivebrushreviewer
            self.reviewer_widget = module.widgetRepresentation()
            slicer.app.processEvents()

            ctx.screenshot(
                "Reviewer module - initial state",
                doc_tags=["reviewer", "overview", "initial"],
            )

        except Exception as e:
            logger.warning(f"Reviewer module not available: {e}")
            ctx.log("Reviewer module not available - skipping detailed screenshots")

    def run(self, ctx: TestContext) -> None:
        """Capture Reviewer module UI screenshots."""
        logger.info("Running Reviewer module documentation capture")

        if self.reviewer_widget is None:
            ctx.log("Reviewer module not available - capturing placeholder")
            ctx.screenshot(
                "Reviewer module - not available in this configuration",
                doc_tags=["reviewer", "unavailable"],
            )
            return

        # Run selection section
        self._capture_run_selection(ctx)

        # Trial list section
        self._capture_trial_list(ctx)

        # Visualization controls
        self._capture_visualization(ctx)

        # Rating interface
        self._capture_rating(ctx)

        # Metrics display
        self._capture_metrics(ctx)

    def _capture_run_selection(self, ctx: TestContext) -> None:
        """Capture run selection UI."""
        ctx.log("Capturing run selection UI")

        ctx.screenshot(
            "Reviewer - Run selection area",
            doc_tags=["reviewer", "run_selection"],
        )

    def _capture_trial_list(self, ctx: TestContext) -> None:
        """Capture trial list UI."""
        ctx.log("Capturing trial list UI")

        ctx.screenshot(
            "Reviewer - Trial list",
            doc_tags=["reviewer", "trial_list"],
        )

    def _capture_visualization(self, ctx: TestContext) -> None:
        """Capture visualization controls."""
        ctx.log("Capturing visualization controls")

        # Look for visualization mode buttons
        widget = self.reviewer_widget.self() if hasattr(self.reviewer_widget, "self") else None

        if widget and hasattr(widget, "outlineModeButton"):
            ctx.screenshot(
                "Reviewer - Visualization mode controls",
                doc_tags=["reviewer", "visualization", "modes"],
            )

        # Layout buttons
        if widget and hasattr(widget, "layoutButtonGroup"):
            ctx.screenshot(
                "Reviewer - Layout options",
                doc_tags=["reviewer", "visualization", "layout"],
            )

    def _capture_rating(self, ctx: TestContext) -> None:
        """Capture rating interface."""
        ctx.log("Capturing rating interface")

        widget = self.reviewer_widget.self() if hasattr(self.reviewer_widget, "self") else None

        if widget and hasattr(widget, "ratingButtonGroup"):
            ctx.screenshot(
                "Reviewer - Rating interface (Good/Acceptable/Poor)",
                doc_tags=["reviewer", "rating", "buttons"],
            )

        ctx.screenshot(
            "Reviewer - Rating and notes section",
            doc_tags=["reviewer", "rating", "notes"],
        )

    def _capture_metrics(self, ctx: TestContext) -> None:
        """Capture metrics display."""
        ctx.log("Capturing metrics display")

        widget = self.reviewer_widget.self() if hasattr(self.reviewer_widget, "self") else None

        if widget and hasattr(widget, "metricsTable"):
            ctx.screenshot(
                "Reviewer - Metrics table (Dice, HD95, etc.)",
                doc_tags=["reviewer", "metrics", "table"],
            )

    def verify(self, ctx: TestContext) -> None:
        """Verify Reviewer screenshots captured."""
        logger.info("Verifying Reviewer documentation")

        ctx.assert_greater(
            len(ctx.screenshots),
            0,
            "Should have captured at least one screenshot",
        )

        ctx.screenshot(
            "Reviewer documentation complete",
            doc_tags=["reviewer", "complete"],
        )

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down Reviewer documentation test")

        # Switch back to a neutral module
        try:
            slicer.util.selectModule("Welcome")
        except Exception:
            pass

        self.reviewer_widget = None
        ctx.log("Teardown complete")
