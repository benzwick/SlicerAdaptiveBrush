"""Tests for SteppingRecipeRunner and SegmentationCheckpoint classes.

These tests verify the step-by-step recipe execution, checkpointing,
and branch recording functionality.

Note: Most SteppingRecipeRunner methods require Slicer. Tests here focus on
logic that can run standalone, with Slicer-dependent tests marked to skip.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Add Tester library to path
_THIS_DIR = Path(__file__).parent
_TESTER_LIB_DIR = (
    _THIS_DIR.parent.parent.parent
    / "SegmentEditorAdaptiveBrushTester"
    / "SegmentEditorAdaptiveBrushTesterLib"
)
if str(_TESTER_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTER_LIB_DIR))

try:
    from ActionRecipe import ActionRecipe, RecipeAction
    from SteppingRecipeRunner import SegmentationCheckpoint, SteppingRecipeRunner
except ImportError:
    ActionRecipe = None
    RecipeAction = None
    SegmentationCheckpoint = None
    SteppingRecipeRunner = None


@unittest.skipIf(SegmentationCheckpoint is None, "SteppingRecipeRunner module not importable")
class TestSegmentationCheckpoint(unittest.TestCase):
    """Tests for SegmentationCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """Should create checkpoint with correct attributes."""
        labelmap = np.zeros((10, 50, 50), dtype=np.uint8)
        checkpoint = SegmentationCheckpoint(
            step_index=3,
            labelmap_array=labelmap,
        )

        self.assertEqual(checkpoint.step_index, 3)
        self.assertEqual(checkpoint.labelmap_array.shape, (10, 50, 50))
        self.assertGreater(checkpoint.timestamp, 0)

    def test_checkpoint_size_bytes(self):
        """Should calculate correct size in bytes."""
        # uint8 array: 10 * 50 * 50 = 25000 bytes
        labelmap = np.zeros((10, 50, 50), dtype=np.uint8)
        checkpoint = SegmentationCheckpoint(step_index=0, labelmap_array=labelmap)

        self.assertEqual(checkpoint.size_bytes, 25000)

    def test_checkpoint_size_bytes_different_dtypes(self):
        """Should handle different dtypes correctly."""
        # uint16 array: 10 * 50 * 50 * 2 = 50000 bytes
        labelmap_16 = np.zeros((10, 50, 50), dtype=np.uint16)
        checkpoint_16 = SegmentationCheckpoint(step_index=0, labelmap_array=labelmap_16)

        self.assertEqual(checkpoint_16.size_bytes, 50000)


@unittest.skipIf(SteppingRecipeRunner is None, "SteppingRecipeRunner module not importable")
class TestSteppingRecipeRunnerInit(unittest.TestCase):
    """Tests for SteppingRecipeRunner initialization."""

    def setUp(self):
        """Create sample recipe for testing."""
        self.actions = [
            RecipeAction.paint(ras=(10.0, 20.0, 30.0), algorithm="watershed"),
            RecipeAction.paint(ras=(15.0, 25.0, 35.0), algorithm="watershed"),
            RecipeAction.erase(ras=(12.0, 22.0, 32.0), algorithm="watershed"),
        ]
        self.recipe = ActionRecipe(
            name="test_recipe",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )

    def test_initialization(self):
        """Should initialize with correct default state."""
        runner = SteppingRecipeRunner(self.recipe)

        self.assertEqual(runner.recipe, self.recipe)
        self.assertEqual(runner.current_step, -1)
        self.assertEqual(runner.checkpoints, [])
        self.assertFalse(runner.is_branching)

    def test_total_steps(self):
        """Should return correct total steps."""
        runner = SteppingRecipeRunner(self.recipe)

        self.assertEqual(runner.total_steps, 3)

    def test_is_at_start(self):
        """Should detect start position correctly."""
        runner = SteppingRecipeRunner(self.recipe)

        self.assertTrue(runner.is_at_start)

        runner.current_step = 0
        self.assertFalse(runner.is_at_start)

    def test_is_at_end(self):
        """Should detect end position correctly."""
        runner = SteppingRecipeRunner(self.recipe)

        self.assertFalse(runner.is_at_end)

        runner.current_step = 2  # Last step (0-indexed)
        self.assertTrue(runner.is_at_end)


@unittest.skipIf(SteppingRecipeRunner is None, "SteppingRecipeRunner module not importable")
class TestSteppingRecipeRunnerNavigation(unittest.TestCase):
    """Tests for step navigation logic."""

    def setUp(self):
        """Create sample recipe for testing."""
        self.actions = [
            RecipeAction.paint(ras=(10.0, 20.0, 30.0), algorithm="watershed"),
            RecipeAction.paint(ras=(15.0, 25.0, 35.0), algorithm="watershed"),
            RecipeAction.paint(ras=(20.0, 30.0, 40.0), algorithm="level_set"),
            RecipeAction.erase(ras=(12.0, 22.0, 32.0), algorithm="watershed"),
        ]
        self.recipe = ActionRecipe(
            name="test_recipe",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )

    def test_get_current_action_at_start(self):
        """Should return None when at start."""
        runner = SteppingRecipeRunner(self.recipe)

        self.assertIsNone(runner.get_current_action())

    def test_get_current_action_after_step(self):
        """Should return current action after stepping."""
        runner = SteppingRecipeRunner(self.recipe)
        runner.current_step = 1

        action = runner.get_current_action()

        self.assertIsNotNone(action)
        self.assertEqual(action.ras, (15.0, 25.0, 35.0))

    def test_get_next_action(self):
        """Should return next action."""
        runner = SteppingRecipeRunner(self.recipe)

        next_action = runner.get_next_action()

        self.assertIsNotNone(next_action)
        self.assertEqual(next_action.ras, (10.0, 20.0, 30.0))

    def test_get_next_action_at_end(self):
        """Should return None when at end."""
        runner = SteppingRecipeRunner(self.recipe)
        runner.current_step = 3  # Last step

        self.assertIsNone(runner.get_next_action())


@unittest.skipIf(SteppingRecipeRunner is None, "SteppingRecipeRunner module not importable")
class TestSteppingRecipeRunnerCheckpoints(unittest.TestCase):
    """Tests for checkpoint management."""

    def setUp(self):
        """Create sample recipe and runner for testing."""
        self.actions = [
            RecipeAction.paint(ras=(10.0, 20.0, 30.0)),
            RecipeAction.paint(ras=(15.0, 25.0, 35.0)),
        ]
        self.recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )
        self.runner = SteppingRecipeRunner(self.recipe)

    def test_find_checkpoint_returns_match(self):
        """Should find checkpoint for exact step."""
        checkpoint = SegmentationCheckpoint(
            step_index=1,
            labelmap_array=np.zeros((5, 10, 10), dtype=np.uint8),
        )
        self.runner.checkpoints.append(checkpoint)

        found = self.runner._find_checkpoint(1)

        self.assertEqual(found, checkpoint)

    def test_find_checkpoint_returns_none_for_missing(self):
        """Should return None when checkpoint not found."""
        found = self.runner._find_checkpoint(5)

        self.assertIsNone(found)

    def test_find_nearest_checkpoint_before(self):
        """Should find nearest checkpoint at or before step."""
        cp0 = SegmentationCheckpoint(
            step_index=0, labelmap_array=np.zeros((5, 10, 10), dtype=np.uint8)
        )
        cp2 = SegmentationCheckpoint(
            step_index=2, labelmap_array=np.zeros((5, 10, 10), dtype=np.uint8)
        )
        self.runner.checkpoints.extend([cp0, cp2])

        # Looking for step 3 should find checkpoint at step 2
        found = self.runner._find_nearest_checkpoint_before(3)
        self.assertEqual(found.step_index, 2)

        # Looking for step 1 should find checkpoint at step 0
        found = self.runner._find_nearest_checkpoint_before(1)
        self.assertEqual(found.step_index, 0)

    def test_find_nearest_checkpoint_before_returns_none(self):
        """Should return None when no checkpoint exists before step."""
        cp5 = SegmentationCheckpoint(
            step_index=5, labelmap_array=np.zeros((5, 10, 10), dtype=np.uint8)
        )
        self.runner.checkpoints.append(cp5)

        found = self.runner._find_nearest_checkpoint_before(3)

        self.assertIsNone(found)

    def test_get_checkpoint_stats_empty(self):
        """Should return empty stats when no checkpoints."""
        stats = self.runner.get_checkpoint_stats()

        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["total_bytes"], 0)
        self.assertEqual(stats["steps_covered"], [])

    def test_get_checkpoint_stats_with_data(self):
        """Should return correct stats."""
        cp0 = SegmentationCheckpoint(
            step_index=0, labelmap_array=np.zeros((10, 10, 10), dtype=np.uint8)
        )
        cp2 = SegmentationCheckpoint(
            step_index=2, labelmap_array=np.zeros((10, 10, 10), dtype=np.uint8)
        )
        self.runner.checkpoints.extend([cp0, cp2])

        stats = self.runner.get_checkpoint_stats()

        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["total_bytes"], 2000)  # 2 * 10*10*10 bytes
        self.assertEqual(stats["steps_covered"], [0, 2])


@unittest.skipIf(SteppingRecipeRunner is None, "SteppingRecipeRunner module not importable")
class TestSteppingRecipeRunnerBranching(unittest.TestCase):
    """Tests for branch recording functionality."""

    def setUp(self):
        """Create sample recipe and runner for testing."""
        self.actions = [
            RecipeAction.paint(ras=(10.0, 20.0, 30.0), algorithm="watershed"),
            RecipeAction.paint(ras=(15.0, 25.0, 35.0), algorithm="watershed"),
            RecipeAction.paint(ras=(20.0, 30.0, 40.0), algorithm="level_set"),
        ]
        self.recipe = ActionRecipe(
            name="original",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )
        self.runner = SteppingRecipeRunner(self.recipe)

    def test_start_branch(self):
        """Should start branch recording."""
        self.runner.current_step = 1
        self.runner.start_branch()

        self.assertTrue(self.runner.is_branching)
        self.assertEqual(self.runner._branch_start_step, 1)
        self.assertEqual(self.runner._branch_actions, [])

    def test_stop_branch(self):
        """Should stop branch recording."""
        self.runner.start_branch()
        self.runner.stop_branch()

        self.assertFalse(self.runner.is_branching)

    def test_add_manual_action(self):
        """Should add actions when branching."""
        self.runner.current_step = 1
        self.runner.start_branch()

        new_action = RecipeAction.paint(ras=(0.0, 0.0, 0.0))
        self.runner.add_manual_action(new_action)

        self.assertEqual(len(self.runner._branch_actions), 1)
        self.assertEqual(self.runner._branch_actions[0], new_action)

    def test_add_manual_action_without_branch_logs_warning(self):
        """Should not add action when not branching."""
        new_action = RecipeAction.paint(ras=(0.0, 0.0, 0.0))
        self.runner.add_manual_action(new_action)

        self.assertEqual(len(self.runner._branch_actions), 0)

    def test_save_branch_creates_recipe(self):
        """Should create branched recipe with correct actions."""
        self.runner.current_step = 1
        self.runner.start_branch()

        # Add a manual action
        new_action = RecipeAction.paint(ras=(100.0, 100.0, 100.0))
        self.runner.add_manual_action(new_action)

        branched = self.runner.save_branch("test_branch")

        # Should have actions 0, 1 from original + 1 new action = 3 total
        self.assertEqual(len(branched.actions), 3)
        self.assertEqual(branched.name, "test_branch")
        self.assertEqual(branched.sample_data, "MRHead")
        self.assertEqual(branched.segment_name, "Segment")

        # Last action should be the manually added one
        self.assertEqual(branched.actions[2].ras, (100.0, 100.0, 100.0))

        # Branching should be stopped after save
        self.assertFalse(self.runner.is_branching)

    def test_save_branch_metadata(self):
        """Should include branch metadata."""
        self.runner.current_step = 1
        self.runner.start_branch()
        self.runner.add_manual_action(RecipeAction.paint(ras=(0, 0, 0)))

        branched = self.runner.save_branch("test_branch")

        self.assertEqual(branched.metadata["source_recipe"], "original")
        self.assertEqual(branched.metadata["branch_start_step"], 1)
        self.assertEqual(branched.metadata["branch_actions_count"], 1)

    def test_save_branch_without_branching(self):
        """Should return sliced recipe when not branching."""
        self.runner.current_step = 1

        # Not in branch mode - should return sliced copy
        result = self.runner.save_branch()

        self.assertEqual(len(result.actions), 2)  # Actions 0 and 1


@unittest.skipIf(SteppingRecipeRunner is None, "SteppingRecipeRunner module not importable")
class TestSteppingRecipeRunnerCallbacks(unittest.TestCase):
    """Tests for callback functionality."""

    def setUp(self):
        """Create sample recipe and runner for testing."""
        self.recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=[RecipeAction.paint(ras=(0, 0, 0))],
        )
        self.runner = SteppingRecipeRunner(self.recipe)

    def test_set_step_callback(self):
        """Should set step callback."""
        callback = MagicMock()
        self.runner.set_step_callback(callback)

        self.assertEqual(self.runner._step_callback, callback)

    def test_set_checkpoint_callback(self):
        """Should set checkpoint callback."""
        callback = MagicMock()
        self.runner.set_checkpoint_callback(callback)

        self.assertEqual(self.runner._checkpoint_callback, callback)

    def test_clear_callbacks(self):
        """Should clear callbacks when set to None."""
        self.runner.set_step_callback(MagicMock())
        self.runner.set_checkpoint_callback(MagicMock())

        self.runner.set_step_callback(None)
        self.runner.set_checkpoint_callback(None)

        self.assertIsNone(self.runner._step_callback)
        self.assertIsNone(self.runner._checkpoint_callback)


@unittest.skipIf(SteppingRecipeRunner is None, "SteppingRecipeRunner module not importable")
class TestSteppingRecipeRunnerCleanup(unittest.TestCase):
    """Tests for cleanup functionality."""

    def test_cleanup_resets_state(self):
        """Should reset all state on cleanup."""
        self.actions = [RecipeAction.paint(ras=(0, 0, 0))]
        recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )
        runner = SteppingRecipeRunner(recipe)

        # Set some state
        runner.current_step = 2
        runner.checkpoints.append(
            SegmentationCheckpoint(step_index=0, labelmap_array=np.zeros((5, 5, 5), dtype=np.uint8))
        )
        runner.start_branch()

        # Cleanup
        runner.cleanup()

        self.assertEqual(runner.current_step, -1)
        self.assertEqual(runner.checkpoints, [])
        self.assertFalse(runner.is_branching)
        self.assertIsNone(runner._effect)
        self.assertIsNone(runner._segmentation_node)
        self.assertIsNone(runner._volume_node)


if __name__ == "__main__":
    unittest.main()
