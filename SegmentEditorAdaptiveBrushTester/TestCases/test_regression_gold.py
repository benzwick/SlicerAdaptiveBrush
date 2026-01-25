"""Gold standard regression test.

Runs recipes with gold standards and verifies Dice scores meet threshold.
"""

from __future__ import annotations

import logging

from SegmentEditorAdaptiveBrushTesterLib import (
    RecipeTestRunner,
    TestCase,
    TestContext,
    register_test,
)

logger = logging.getLogger(__name__)

# Pass/fail thresholds
DICE_THRESHOLD = 0.80


@register_test(category="regression")
class TestRegressionGold(TestCase):
    """Run recipes against gold standards."""

    name = "regression_gold"
    description = "Run recipes and compare to gold standards"

    def __init__(self) -> None:
        super().__init__()
        self.suite_result = None

    def setup(self, ctx: TestContext) -> None:
        """Initialize test runner."""
        ctx.log("Discovering recipes with gold standards...")
        runner = RecipeTestRunner()
        recipes = runner.discover_recipes()
        ctx.log(f"Found {len(recipes)} recipe(s) with gold standards")
        for path, gold in recipes:
            ctx.log(f"  - {path.stem} -> {gold}")

    def run(self, ctx: TestContext) -> None:
        """Run all recipe tests."""
        runner = RecipeTestRunner(output_base=ctx.output_folder.parent)
        self.suite_result = runner.run_all()

        # Log results
        for result in self.suite_result.results:
            status = "PASS" if result.passed else "FAIL"
            ctx.log(
                f"[{status}] {result.recipe_name}: "
                f"Dice={result.dice:.3f}, HD95={result.hausdorff_95:.1f}mm"
            )
            if result.error:
                ctx.log(f"       Error: {result.error}")

            ctx.metric(f"{result.recipe_name}_dice", result.dice)
            ctx.metric(f"{result.recipe_name}_hausdorff_95", result.hausdorff_95)

    def verify(self, ctx: TestContext) -> None:
        """Check all tests passed."""
        if self.suite_result is None:
            ctx.assert_true(False, "No test results")
            return

        ctx.log(f"\n{'=' * 50}")
        ctx.log("REGRESSION TEST SUMMARY")
        ctx.log(f"{'=' * 50}")
        ctx.log(f"Dice threshold: >= {DICE_THRESHOLD}")
        ctx.log(f"Passed: {self.suite_result.passed_count}/{self.suite_result.total_count}")

        if self.suite_result.passed:
            ctx.log("ALL TESTS PASSED")
            ctx.assert_true(True, "All recipe tests passed")
        else:
            failed = [r for r in self.suite_result.results if not r.passed]
            ctx.log(f"FAILED: {len(failed)}")
            for r in failed:
                ctx.log(f"  - {r.recipe_name}: Dice={r.dice:.3f}")
            ctx.assert_true(False, f"{len(failed)} recipe test(s) failed")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        ctx.log("Teardown complete")
