"""Tests for ActionRecipe and RecipeAction classes.

These tests verify the action-based recipe format for step-by-step
recipe execution, serialization, and branching.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
    from ActionRecipe import ActionRecipe, RecipeAction, list_action_recipes
except ImportError:
    ActionRecipe = None
    RecipeAction = None
    list_action_recipes = None


@unittest.skipIf(RecipeAction is None, "ActionRecipe module not importable")
class TestRecipeAction(unittest.TestCase):
    """Tests for RecipeAction dataclass."""

    def test_paint_action_creation(self):
        """Should create a paint action with correct attributes."""
        action = RecipeAction.paint(
            ras=(10.0, 20.0, 30.0),
            algorithm="watershed",
            brush_radius_mm=15.0,
            edge_sensitivity=45,
        )

        self.assertEqual(action.type, "paint")
        self.assertEqual(action.ras, (10.0, 20.0, 30.0))
        self.assertEqual(action.params["algorithm"], "watershed")
        self.assertEqual(action.params["brush_radius_mm"], 15.0)
        self.assertEqual(action.params["edge_sensitivity"], 45)
        self.assertGreater(action.timestamp, 0)

    def test_erase_action_creation(self):
        """Should create an erase action with correct attributes."""
        action = RecipeAction.erase(
            ras=(-5.0, 10.0, 15.0),
            algorithm="connected_threshold",
            brush_radius_mm=10.0,
        )

        self.assertEqual(action.type, "erase")
        self.assertEqual(action.ras, (-5.0, 10.0, 15.0))
        self.assertEqual(action.params["algorithm"], "connected_threshold")

    def test_set_param_action(self):
        """Should create a parameter change action."""
        action = RecipeAction.set_param("brush_radius_mm", 25.0, "Increase radius")

        self.assertEqual(action.type, "set_param")
        self.assertEqual(action.params["name"], "brush_radius_mm")
        self.assertEqual(action.params["value"], 25.0)
        self.assertEqual(action.description, "Increase radius")

    def test_set_algorithm_action(self):
        """Should create an algorithm change action."""
        action = RecipeAction.set_algorithm("level_set")

        self.assertEqual(action.type, "set_algorithm")
        self.assertEqual(action.params["algorithm"], "level_set")
        self.assertIn("level_set", action.description)

    def test_to_dict_serialization(self):
        """Should serialize action to dictionary."""
        action = RecipeAction.paint(
            ras=(1.0, 2.0, 3.0),
            algorithm="watershed",
            brush_radius_mm=10.0,
        )

        d = action.to_dict()

        self.assertEqual(d["type"], "paint")
        self.assertEqual(d["ras"], [1.0, 2.0, 3.0])
        self.assertIn("algorithm", d["params"])
        self.assertIn("timestamp", d)

    def test_from_dict_deserialization(self):
        """Should deserialize action from dictionary."""
        d = {
            "type": "paint",
            "ras": [5.0, 10.0, 15.0],
            "params": {"algorithm": "watershed", "brush_radius_mm": 20.0},
            "timestamp": 1234567890.0,
            "description": "Test action",
        }

        action = RecipeAction.from_dict(d)

        self.assertEqual(action.type, "paint")
        self.assertEqual(action.ras, (5.0, 10.0, 15.0))
        self.assertEqual(action.params["algorithm"], "watershed")
        self.assertEqual(action.timestamp, 1234567890.0)
        self.assertEqual(action.description, "Test action")

    def test_from_dict_handles_none_ras(self):
        """Should handle None ras in deserialization."""
        d = {
            "type": "set_param",
            "ras": None,
            "params": {"name": "radius", "value": 10},
        }

        action = RecipeAction.from_dict(d)

        self.assertIsNone(action.ras)

    def test_roundtrip_serialization(self):
        """Should survive serialization roundtrip."""
        original = RecipeAction.paint(
            ras=(1.5, 2.5, 3.5),
            algorithm="watershed",
            brush_radius_mm=12.5,
            edge_sensitivity=50,
            custom_param="test",
        )

        d = original.to_dict()
        restored = RecipeAction.from_dict(d)

        self.assertEqual(original.type, restored.type)
        self.assertEqual(original.ras, restored.ras)
        self.assertEqual(original.params, restored.params)


@unittest.skipIf(ActionRecipe is None, "ActionRecipe module not importable")
class TestActionRecipe(unittest.TestCase):
    """Tests for ActionRecipe dataclass."""

    def setUp(self):
        """Create sample actions for testing."""
        self.actions = [
            RecipeAction.paint(
                ras=(10.0, 20.0, 30.0),
                algorithm="watershed",
                brush_radius_mm=20.0,
            ),
            RecipeAction.paint(
                ras=(15.0, 25.0, 35.0),
                algorithm="watershed",
                brush_radius_mm=15.0,
            ),
            RecipeAction.erase(
                ras=(12.0, 22.0, 32.0),
                algorithm="watershed",
                brush_radius_mm=10.0,
            ),
        ]

    def test_recipe_creation(self):
        """Should create a recipe with correct attributes."""
        recipe = ActionRecipe(
            name="test_recipe",
            sample_data="MRBrainTumor1",
            segment_name="Tumor",
            actions=self.actions,
            gold_standard="test_gold",
            description="Test recipe description",
        )

        self.assertEqual(recipe.name, "test_recipe")
        self.assertEqual(recipe.sample_data, "MRBrainTumor1")
        self.assertEqual(recipe.segment_name, "Tumor")
        self.assertEqual(len(recipe.actions), 3)
        self.assertEqual(recipe.gold_standard, "test_gold")

    def test_len_returns_action_count(self):
        """Should return number of actions from len()."""
        recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )

        self.assertEqual(len(recipe), 3)

    def test_getitem_returns_action(self):
        """Should return action by index."""
        recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )

        self.assertEqual(recipe[0].type, "paint")
        self.assertEqual(recipe[2].type, "erase")

    def test_to_dict_serialization(self):
        """Should serialize recipe to dictionary."""
        recipe = ActionRecipe(
            name="test_recipe",
            sample_data="MRBrainTumor1",
            segment_name="Tumor",
            actions=self.actions,
            gold_standard="test_gold",
        )

        d = recipe.to_dict()

        self.assertEqual(d["name"], "test_recipe")
        self.assertEqual(d["sample_data"], "MRBrainTumor1")
        self.assertEqual(d["segment_name"], "Tumor")
        self.assertEqual(len(d["actions"]), 3)
        self.assertEqual(d["gold_standard"], "test_gold")

    def test_save_creates_json_file(self):
        """Should save recipe to JSON file."""
        recipe = ActionRecipe(
            name="save_test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_recipe.json"
            recipe.save(path)

            self.assertTrue(path.exists())

            with open(path) as f:
                data = json.load(f)

            self.assertEqual(data["name"], "save_test")
            self.assertEqual(len(data["actions"]), 3)

    def test_save_adds_json_extension(self):
        """Should add .json extension if missing."""
        recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_recipe"  # No extension
            recipe.save(path)

            json_path = Path(tmpdir) / "test_recipe.json"
            self.assertTrue(json_path.exists())

    def test_load_reads_json_file(self):
        """Should load recipe from JSON file."""
        recipe = ActionRecipe(
            name="load_test",
            sample_data="MRBrainTumor1",
            segment_name="Tumor",
            actions=self.actions,
            gold_standard="test_gold",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_recipe.json"
            recipe.save(path)

            loaded = ActionRecipe.load(path)

            self.assertEqual(loaded.name, "load_test")
            self.assertEqual(loaded.sample_data, "MRBrainTumor1")
            self.assertEqual(loaded.segment_name, "Tumor")
            self.assertEqual(len(loaded.actions), 3)
            self.assertEqual(loaded.gold_standard, "test_gold")

    def test_load_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            ActionRecipe.load("/nonexistent/path/recipe.json")

    def test_roundtrip_preserves_data(self):
        """Should preserve all data through save/load cycle."""
        original = ActionRecipe(
            name="roundtrip_test",
            sample_data="MRBrainTumor1",
            segment_name="Tumor",
            actions=self.actions,
            gold_standard="gold",
            description="Test description",
            metadata={"key": "value", "number": 42},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original.save(path)
            loaded = ActionRecipe.load(path)

            self.assertEqual(original.name, loaded.name)
            self.assertEqual(original.sample_data, loaded.sample_data)
            self.assertEqual(original.segment_name, loaded.segment_name)
            self.assertEqual(original.gold_standard, loaded.gold_standard)
            self.assertEqual(original.description, loaded.description)
            self.assertEqual(original.metadata, loaded.metadata)
            self.assertEqual(len(original.actions), len(loaded.actions))

            for orig_action, loaded_action in zip(original.actions, loaded.actions):
                self.assertEqual(orig_action.type, loaded_action.type)
                self.assertEqual(orig_action.ras, loaded_action.ras)
                self.assertEqual(orig_action.params, loaded_action.params)

    def test_slice_to_creates_subset(self):
        """Should create recipe with actions up to given step."""
        recipe = ActionRecipe(
            name="original",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions,
        )

        sliced = recipe.slice_to(1)

        self.assertEqual(len(sliced.actions), 2)  # Steps 0 and 1
        self.assertEqual(sliced.actions[0].type, "paint")
        self.assertEqual(sliced.actions[1].type, "paint")
        self.assertIn("step1", sliced.name)
        self.assertIn("sliced_from", sliced.metadata)

    def test_slice_to_preserves_sample_data(self):
        """Should preserve sample data and segment name in sliced recipe."""
        recipe = ActionRecipe(
            name="original",
            sample_data="MRBrainTumor1",
            segment_name="Tumor",
            actions=self.actions,
            gold_standard="gold",
        )

        sliced = recipe.slice_to(0)

        self.assertEqual(sliced.sample_data, "MRBrainTumor1")
        self.assertEqual(sliced.segment_name, "Tumor")
        self.assertEqual(sliced.gold_standard, "gold")

    def test_append_actions_creates_extended_recipe(self):
        """Should create recipe with additional actions appended."""
        recipe = ActionRecipe(
            name="original",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions[:2],
        )

        new_actions = [
            RecipeAction.erase(ras=(0.0, 0.0, 0.0), brush_radius_mm=5.0),
        ]

        extended = recipe.append_actions(new_actions)

        self.assertEqual(len(extended.actions), 3)
        self.assertEqual(extended.actions[2].type, "erase")
        self.assertIn("extended", extended.name)

    def test_append_actions_preserves_original(self):
        """Should not modify original recipe."""
        recipe = ActionRecipe(
            name="original",
            sample_data="MRHead",
            segment_name="Segment",
            actions=self.actions[:2],
        )

        new_actions = [RecipeAction.erase(ras=(0.0, 0.0, 0.0))]
        recipe.append_actions(new_actions)

        # Original should be unchanged
        self.assertEqual(len(recipe.actions), 2)


@unittest.skipIf(ActionRecipe is None, "ActionRecipe module not importable")
class TestActionRecipeMetadata(unittest.TestCase):
    """Tests for recipe metadata handling."""

    def test_metadata_default_empty(self):
        """Should have empty metadata by default."""
        recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=[],
        )

        self.assertEqual(recipe.metadata, {})

    def test_metadata_preserved_in_serialization(self):
        """Should preserve metadata through serialization."""
        recipe = ActionRecipe(
            name="test",
            sample_data="MRHead",
            segment_name="Segment",
            actions=[],
            metadata={
                "author": "test",
                "version": 1,
                "tags": ["brain", "tumor"],
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            recipe.save(path)
            loaded = ActionRecipe.load(path)

            self.assertEqual(loaded.metadata["author"], "test")
            self.assertEqual(loaded.metadata["version"], 1)
            self.assertEqual(loaded.metadata["tags"], ["brain", "tumor"])

    def test_slice_to_tracks_source(self):
        """Should track source recipe in metadata when slicing."""
        recipe = ActionRecipe(
            name="original",
            sample_data="MRHead",
            segment_name="Segment",
            actions=[RecipeAction.paint(ras=(0, 0, 0))],
        )

        sliced = recipe.slice_to(0)

        self.assertEqual(sliced.metadata["sliced_from"], "original")
        self.assertEqual(sliced.metadata["sliced_at_step"], 0)


if __name__ == "__main__":
    unittest.main()
