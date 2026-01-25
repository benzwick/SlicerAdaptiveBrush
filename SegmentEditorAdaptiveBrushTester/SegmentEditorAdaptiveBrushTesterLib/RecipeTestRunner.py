"""Recipe-based test runner.

Discovers recipes with gold standards and runs them as regression tests.
This replaces the complex TestCase approach with a simple:
  run recipe → compare to gold → pass/fail on Dice
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .GoldStandardManager import GoldStandardManager
from .Recipe import Recipe
from .RecipeRunner import RecipeRunner
from .ScreenshotCapture import ScreenshotCapture
from .SegmentationMetrics import SegmentationMetrics
from .TestRunFolder import TestRunFolder

logger = logging.getLogger(__name__)

# Pass/fail thresholds
DICE_THRESHOLD = 0.80
HAUSDORFF_THRESHOLD = 10.0  # mm


@dataclass
class RecipeTestResult:
    """Result of a single recipe test."""

    recipe_name: str
    gold_standard: str
    passed: bool
    dice: float = 0.0
    hausdorff_95: float = 0.0
    duration_ms: float = 0.0
    voxels_test: int = 0
    voxels_gold: int = 0
    error: str | None = None


@dataclass
class RecipeTestSuiteResult:
    """Result of running all recipe tests."""

    output_folder: Path
    results: list[RecipeTestResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)


class RecipeTestRunner:
    """Run recipes as regression tests.

    Discovers all recipes that have a gold_standard field and runs them,
    comparing output to the gold standard segmentation.

    Usage:
        runner = RecipeTestRunner()
        result = runner.run_all()
        print(f"Passed: {result.passed_count}/{result.total_count}")
    """

    def __init__(self, output_base: Path | None = None):
        """Initialize test runner.

        Args:
            output_base: Base directory for test output.
        """
        if output_base is None:
            output_base = Path(__file__).parent.parent.parent / "test_runs"
        self._output_base = output_base
        self._gold_manager = GoldStandardManager()
        self._screenshot_capture = ScreenshotCapture()

    def discover_recipes(self) -> list[tuple[Path, str]]:
        """Find all recipes with gold standards.

        Returns:
            List of (recipe_path, gold_standard_name) tuples.
        """
        recipes_dir = Path(__file__).parent.parent / "recipes"
        if not recipes_dir.exists():
            return []

        results = []
        for recipe_file in recipes_dir.glob("*.py"):
            if recipe_file.name.startswith("_") or recipe_file.name == "template.py":
                continue

            # Load recipe to check for gold_standard
            try:
                recipe = Recipe.load(recipe_file)
                if recipe.gold_standard:
                    # Verify gold standard exists
                    if self._gold_manager.gold_exists(recipe.gold_standard):
                        results.append((recipe_file, recipe.gold_standard))
                    else:
                        logger.warning(
                            f"Recipe {recipe.name} references missing gold standard: "
                            f"{recipe.gold_standard}"
                        )
            except Exception as e:
                logger.warning(f"Could not load recipe {recipe_file}: {e}")

        return results

    def run_recipe_test(
        self,
        recipe_path: Path,
        gold_name: str,
        test_run_folder: TestRunFolder | None = None,
    ) -> RecipeTestResult:
        """Run a single recipe test.

        Args:
            recipe_path: Path to recipe file.
            gold_name: Name of gold standard to compare against.
            test_run_folder: Optional test run folder for output.

        Returns:
            RecipeTestResult with pass/fail and metrics.
        """

        recipe = Recipe.load(recipe_path)
        result = RecipeTestResult(
            recipe_name=recipe.name,
            gold_standard=gold_name,
            passed=False,
        )

        start_time = time.time()

        try:
            # Load gold standard first (before clearing scene)
            gold_seg_node, gold_metadata = self._gold_manager.load_gold(gold_name)
            gold_segment_id = gold_metadata.get("segment_id", "Segment_1")
            result.voxels_gold = gold_metadata.get("voxel_count", 0)

            # Screenshot before
            if test_run_folder:
                self._screenshot_capture.set_base_folder(test_run_folder.screenshots_folder)
                self._screenshot_capture.set_group(recipe.name)
                self._screenshot_capture.screenshot(f"[1_before] Before: {recipe.name}")

            # Run recipe (don't clear scene - we need gold standard)
            runner = RecipeRunner(recipe)
            run_result = runner.run(clear_scene=False)

            if not run_result.success:
                result.error = run_result.error or "Recipe execution failed"
                return result

            result.voxels_test = run_result.voxel_count
            result.duration_ms = run_result.duration_ms

            # Get volume node for metrics
            volume_node = run_result.volume_node

            # Compute metrics
            metrics = SegmentationMetrics.compute(
                run_result.segmentation_node,
                run_result.segment_id,
                gold_seg_node,
                gold_segment_id,
                volume_node,
            )

            result.dice = metrics.dice
            result.hausdorff_95 = metrics.hausdorff_95

            # Screenshot after
            if test_run_folder:
                self._screenshot_capture.screenshot(
                    f"[2_after] Dice={metrics.dice:.3f}, HD95={metrics.hausdorff_95:.1f}mm"
                )

            # Determine pass/fail
            result.passed = (
                metrics.dice >= DICE_THRESHOLD and metrics.hausdorff_95 <= HAUSDORFF_THRESHOLD
            )

            logger.info(
                f"Recipe test '{recipe.name}': "
                f"Dice={metrics.dice:.3f}, HD95={metrics.hausdorff_95:.1f}mm "
                f"-> {'PASS' if result.passed else 'FAIL'}"
            )

        except Exception as e:
            logger.exception(f"Error testing recipe {recipe.name}: {e}")
            result.error = str(e)

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def run_all(self) -> RecipeTestSuiteResult:
        """Run all recipe tests.

        Returns:
            RecipeTestSuiteResult with all test results.
        """
        import slicer

        recipes = self.discover_recipes()
        if not recipes:
            logger.warning("No recipes with gold standards found")
            return RecipeTestSuiteResult(
                output_folder=self._output_base,
                results=[],
            )

        logger.info(f"Running {len(recipes)} recipe test(s)")

        # Create test run folder
        test_run_folder = TestRunFolder.create(
            base_path=self._output_base,
            run_name="recipe_tests",
        )

        start_time = time.time()
        results = []

        for recipe_path, gold_name in recipes:
            # Clear scene between tests
            slicer.mrmlScene.Clear(0)
            slicer.app.processEvents()

            result = self.run_recipe_test(recipe_path, gold_name, test_run_folder)
            results.append(result)

        duration = time.time() - start_time

        suite_result = RecipeTestSuiteResult(
            output_folder=test_run_folder.path,
            results=results,
            duration_seconds=duration,
        )

        # Save results
        self._save_results(test_run_folder, suite_result)

        # Save screenshot manifest
        self._screenshot_capture.save_manifest()

        logger.info(
            f"Recipe tests complete: {suite_result.passed_count}/{suite_result.total_count} passed "
            f"in {duration:.2f}s"
        )

        return suite_result

    def _save_results(
        self, test_run_folder: TestRunFolder, suite_result: RecipeTestSuiteResult
    ) -> None:
        """Save test results to files."""
        import json

        # Convert to serializable format
        results_data = [
            {
                "recipe": r.recipe_name,
                "gold_standard": r.gold_standard,
                "passed": r.passed,
                "dice": r.dice,
                "hausdorff_95": r.hausdorff_95,
                "duration_ms": r.duration_ms,
                "voxels_test": r.voxels_test,
                "voxels_gold": r.voxels_gold,
                "error": r.error,
            }
            for r in suite_result.results
        ]

        # Save results.json
        results_file = test_run_folder.path / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "suite": "recipe_tests",
                    "passed": suite_result.passed,
                    "passed_count": suite_result.passed_count,
                    "total_count": suite_result.total_count,
                    "duration_seconds": suite_result.duration_seconds,
                    "results": results_data,
                },
                f,
                indent=2,
            )

        # Save metadata
        test_run_folder.save_metadata(
            {
                "suite": "recipe_tests",
                "total_tests": suite_result.total_count,
                "passed": suite_result.passed_count,
                "failed": suite_result.total_count - suite_result.passed_count,
                "duration_seconds": suite_result.duration_seconds,
            }
        )


def run_recipe_tests() -> RecipeTestSuiteResult:
    """Convenience function to run all recipe tests.

    Returns:
        RecipeTestSuiteResult with all test results.
    """
    runner = RecipeTestRunner()
    return runner.run_all()
