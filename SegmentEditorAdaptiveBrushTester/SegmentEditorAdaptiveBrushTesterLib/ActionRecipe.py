"""Action-based recipe format for step-by-step execution.

Unlike function-based recipes (Recipe.py) which execute all-at-once,
ActionRecipe stores a list of discrete actions that can be:
- Stepped through one at a time
- Rewound to previous states
- Branched to create new variations

See ADR-014 for architecture decisions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .Recipe import Recipe

logger = logging.getLogger(__name__)


@dataclass
class RecipeAction:
    """A single discrete action in a recipe.

    Attributes:
        type: Action type ("paint", "erase", "set_param", "set_algorithm").
        ras: World coordinates (R, A, S) for paint/erase actions.
        params: Parameters for the action.
        timestamp: When the action was recorded (seconds since epoch).
        description: Human-readable description.
    """

    type: str
    ras: tuple[float, float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "ras": list(self.ras) if self.ras else None,
            "params": self.params,
            "timestamp": self.timestamp,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeAction:
        """Create from dictionary."""
        ras = tuple(data["ras"]) if data.get("ras") else None
        return cls(
            type=data["type"],
            ras=ras,  # type: ignore
            params=data.get("params", {}),
            timestamp=data.get("timestamp", 0.0),
            description=data.get("description", ""),
        )

    @classmethod
    def paint(
        cls,
        ras: tuple[float, float, float],
        algorithm: str | None = None,
        brush_radius_mm: float | None = None,
        edge_sensitivity: int | None = None,
        **kwargs: Any,
    ) -> RecipeAction:
        """Create a paint action.

        Args:
            ras: World coordinates (R, A, S).
            algorithm: Algorithm to use (e.g., "watershed").
            brush_radius_mm: Brush radius in mm.
            edge_sensitivity: Edge sensitivity (0-100).
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            A paint RecipeAction.
        """
        params: dict[str, Any] = {}
        if algorithm is not None:
            params["algorithm"] = algorithm
        if brush_radius_mm is not None:
            params["brush_radius_mm"] = brush_radius_mm
        if edge_sensitivity is not None:
            params["edge_sensitivity"] = edge_sensitivity
        params.update(kwargs)

        return cls(
            type="paint",
            ras=ras,
            params=params,
            timestamp=time.time(),
        )

    @classmethod
    def erase(
        cls,
        ras: tuple[float, float, float],
        algorithm: str | None = None,
        brush_radius_mm: float | None = None,
        edge_sensitivity: int | None = None,
        **kwargs: Any,
    ) -> RecipeAction:
        """Create an erase action.

        Args:
            ras: World coordinates (R, A, S).
            algorithm: Algorithm to use (e.g., "watershed").
            brush_radius_mm: Brush radius in mm.
            edge_sensitivity: Edge sensitivity (0-100).
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            An erase RecipeAction.
        """
        params: dict[str, Any] = {}
        if algorithm is not None:
            params["algorithm"] = algorithm
        if brush_radius_mm is not None:
            params["brush_radius_mm"] = brush_radius_mm
        if edge_sensitivity is not None:
            params["edge_sensitivity"] = edge_sensitivity
        params.update(kwargs)

        return cls(
            type="erase",
            ras=ras,
            params=params,
            timestamp=time.time(),
        )

    @classmethod
    def set_param(cls, name: str, value: Any, description: str = "") -> RecipeAction:
        """Create a parameter change action.

        Args:
            name: Parameter name.
            value: New value.
            description: Human-readable description.

        Returns:
            A set_param RecipeAction.
        """
        return cls(
            type="set_param",
            params={"name": name, "value": value},
            timestamp=time.time(),
            description=description or f"Set {name} to {value}",
        )

    @classmethod
    def set_algorithm(cls, algorithm: str, description: str = "") -> RecipeAction:
        """Create an algorithm change action.

        Args:
            algorithm: Algorithm name.
            description: Human-readable description.

        Returns:
            A set_algorithm RecipeAction.
        """
        return cls(
            type="set_algorithm",
            params={"algorithm": algorithm},
            timestamp=time.time(),
            description=description or f"Switch to {algorithm}",
        )


@dataclass
class ActionRecipe:
    """A recipe as a list of discrete actions.

    This format supports step-by-step execution, rewinding, and branching.

    Attributes:
        name: Recipe name.
        sample_data: Slicer SampleData name to load.
        segment_name: Name for the segment to create.
        actions: List of actions to execute.
        gold_standard: Name of gold standard to compare against (optional).
        description: Human-readable description.
        metadata: Additional metadata (e.g., source recipe, creation date).
    """

    name: str
    sample_data: str
    segment_name: str
    actions: list[RecipeAction]
    gold_standard: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of actions."""
        return len(self.actions)

    def __getitem__(self, index: int) -> RecipeAction:
        """Get action by index."""
        return self.actions[index]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "sample_data": self.sample_data,
            "segment_name": self.segment_name,
            "actions": [a.to_dict() for a in self.actions],
            "gold_standard": self.gold_standard,
            "description": self.description,
            "metadata": self.metadata,
        }

    def save(self, path: Path | str) -> None:
        """Save recipe to JSON file.

        Args:
            path: Path to save to (will add .json extension if missing).
        """
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved action recipe to {path}")

    @classmethod
    def load(cls, path: Path | str) -> ActionRecipe:
        """Load recipe from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Loaded ActionRecipe.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe not found: {path}")

        with open(path) as f:
            data = json.load(f)

        actions = [RecipeAction.from_dict(a) for a in data.get("actions", [])]

        recipe = cls(
            name=data.get("name", path.stem),
            sample_data=data.get("sample_data", ""),
            segment_name=data.get("segment_name", "Segment"),
            actions=actions,
            gold_standard=data.get("gold_standard"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

        logger.info(f"Loaded action recipe: {recipe.name} ({len(actions)} actions)")
        return recipe

    @classmethod
    def from_function_recipe(cls, recipe: Recipe) -> ActionRecipe:
        """Convert a function-based Recipe to ActionRecipe by recording execution.

        This executes the recipe's run() function with a proxy effect that
        records all paint operations as actions.

        Args:
            recipe: The function-based Recipe to convert.

        Returns:
            ActionRecipe with recorded actions.

        Note:
            This requires Slicer to be running as it needs the effect API.
        """
        import slicer

        # Create a recording proxy
        recorded_actions: list[RecipeAction] = []

        class RecordingEffectProxy:
            """Proxy that records calls to the effect."""

            def __init__(self, real_effect: Any):
                self._real_effect = real_effect
                self._current_algorithm: str | None = None
                self._current_radius: float | None = None
                self._current_sensitivity: int | None = None

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real_effect, name)

            def __setattr__(self, name: str, value: Any) -> None:
                if name.startswith("_"):
                    object.__setattr__(self, name, value)
                    return

                # Record parameter changes
                if name == "algorithm":
                    self._current_algorithm = value
                    recorded_actions.append(RecipeAction.set_algorithm(value))
                elif name == "brushRadiusMm":
                    self._current_radius = value
                    recorded_actions.append(RecipeAction.set_param("brush_radius_mm", value))
                elif name == "edgeSensitivity":
                    self._current_sensitivity = value
                    recorded_actions.append(RecipeAction.set_param("edge_sensitivity", value))
                else:
                    recorded_actions.append(RecipeAction.set_param(name, value))

                setattr(self._real_effect, name, value)

            def paintAt(self, r: float, a: float, s: float) -> None:
                """Record paint action."""
                recorded_actions.append(
                    RecipeAction.paint(
                        ras=(r, a, s),
                        algorithm=self._current_algorithm,
                        brush_radius_mm=self._current_radius,
                        edge_sensitivity=self._current_sensitivity,
                    )
                )
                self._real_effect.paintAt(r, a, s)

            def eraseAt(self, r: float, a: float, s: float) -> None:
                """Record erase action."""
                recorded_actions.append(
                    RecipeAction.erase(
                        ras=(r, a, s),
                        algorithm=self._current_algorithm,
                        brush_radius_mm=self._current_radius,
                        edge_sensitivity=self._current_sensitivity,
                    )
                )
                self._real_effect.eraseAt(r, a, s)

        # Set up scene for recipe execution
        # Clear scene and load sample data
        slicer.mrmlScene.Clear(0)

        volume_node = None
        if recipe.sample_data:
            import SampleData

            volume_node = SampleData.downloadSample(recipe.sample_data)

        # Create segmentation
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation_node.CreateDefaultDisplayNodes()
        if volume_node:
            segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

        # Add segment
        segmentation = segmentation_node.GetSegmentation()
        segment_id = segmentation.AddEmptySegment(recipe.segment_name)

        # Switch to Segment Editor and set up
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        editor_widget.setSegmentationNode(segmentation_node)
        if volume_node:
            editor_widget.setSourceVolumeNode(volume_node)
        editor_widget.setCurrentSegmentID(segment_id)
        slicer.app.processEvents()

        # Activate Adaptive Brush
        editor_widget.setActiveEffectByName("Adaptive Brush")
        effect = editor_widget.activeEffect()
        if effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        scripted_effect = effect.self()
        slicer.app.processEvents()

        # Execute the recipe with the recording proxy
        proxy = RecordingEffectProxy(scripted_effect)
        recipe.run(proxy)
        slicer.app.processEvents()

        return cls(
            name=recipe.name,
            sample_data=recipe.sample_data,
            segment_name=recipe.segment_name,
            actions=recorded_actions,
            gold_standard=recipe.gold_standard,
            description=f"Converted from {recipe.path}",
            metadata={
                "source_recipe": str(recipe.path),
                "converted_at": time.time(),
            },
        )

    def slice_to(self, step_index: int) -> ActionRecipe:
        """Create a new recipe with actions up to the given step.

        Args:
            step_index: Last step to include (inclusive).

        Returns:
            New ActionRecipe with actions[0:step_index+1].
        """
        return ActionRecipe(
            name=f"{self.name}_step{step_index}",
            sample_data=self.sample_data,
            segment_name=self.segment_name,
            actions=self.actions[: step_index + 1],
            gold_standard=self.gold_standard,
            description=f"{self.description} (sliced to step {step_index})",
            metadata={
                **self.metadata,
                "sliced_from": self.name,
                "sliced_at_step": step_index,
            },
        )

    def append_actions(self, new_actions: list[RecipeAction]) -> ActionRecipe:
        """Create a new recipe with additional actions appended.

        Args:
            new_actions: Actions to append.

        Returns:
            New ActionRecipe with additional actions.
        """
        return ActionRecipe(
            name=f"{self.name}_extended",
            sample_data=self.sample_data,
            segment_name=self.segment_name,
            actions=self.actions + new_actions,
            gold_standard=self.gold_standard,
            description=f"{self.description} (extended with {len(new_actions)} actions)",
            metadata={
                **self.metadata,
                "extended_from": self.name,
                "added_actions": len(new_actions),
            },
        )


def list_action_recipes(recipes_dir: Path | str | None = None) -> list[Path]:
    """List available action recipe files (JSON format).

    Args:
        recipes_dir: Directory to search. If None, uses default location.

    Returns:
        List of recipe file paths.
    """
    if recipes_dir is None:
        recipes_dir = Path(__file__).parent.parent / "recipes"

    recipes_dir = Path(recipes_dir)
    if not recipes_dir.exists():
        return []

    return sorted(recipes_dir.glob("*.json"))
