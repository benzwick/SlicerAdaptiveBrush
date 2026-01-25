"""Segmentation Recipe data structures.

Recipes capture complete segmentation workflows as Python objects,
enabling exact reproduction, optimization, and version control.

See ADR-013 for architecture decisions.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """A single segmentation action.

    Actions represent discrete steps in a segmentation workflow.
    Each action has an effect type and parameters specific to that effect.

    Example:
        action = Action.adaptive_brush(
            ras=(-5.31, 34.77, 20.83),
            algorithm="watershed",
            brush_radius_mm=25.0,
            edge_sensitivity=40,
        )
    """

    effect: str
    params: dict[str, Any]

    @classmethod
    def adaptive_brush(
        cls,
        ras: tuple[float, float, float],
        algorithm: str,
        brush_radius_mm: float,
        edge_sensitivity: int,
        mode: str = "add",
        threshold_zone: int = 50,
        **kwargs: Any,
    ) -> Action:
        """Create Adaptive Brush action.

        Args:
            ras: RAS (Right-Anterior-Superior) coordinates for the stroke.
            algorithm: Algorithm name (watershed, level_set_cpu, etc.).
            brush_radius_mm: Brush radius in millimeters.
            edge_sensitivity: Edge sensitivity (0-100).
            mode: "add" or "erase".
            threshold_zone: Threshold zone percentage (0-100).
            **kwargs: Algorithm-specific parameters.

        Returns:
            Action configured for Adaptive Brush effect.
        """
        return cls(
            effect="adaptive_brush",
            params={
                "ras": ras,
                "algorithm": algorithm,
                "brush_radius_mm": brush_radius_mm,
                "edge_sensitivity": edge_sensitivity,
                "mode": mode,
                "threshold_zone": threshold_zone,
                **kwargs,
            },
        )

    @classmethod
    def paint(
        cls,
        ras: tuple[float, float, float],
        radius_mm: float,
        mode: str = "add",
        sphere: bool = False,
    ) -> Action:
        """Create standard Paint effect action.

        Args:
            ras: RAS coordinates for the stroke.
            radius_mm: Paint brush radius in millimeters.
            mode: "add" or "erase".
            sphere: Whether to use 3D sphere brush.

        Returns:
            Action configured for Paint effect.
        """
        return cls(
            effect="paint",
            params={
                "ras": ras,
                "radius_mm": radius_mm,
                "mode": mode,
                "sphere": sphere,
            },
        )

    @classmethod
    def threshold(
        cls,
        min_value: float,
        max_value: float,
    ) -> Action:
        """Create Threshold effect action.

        Args:
            min_value: Minimum intensity threshold.
            max_value: Maximum intensity threshold.

        Returns:
            Action configured for Threshold effect.
        """
        return cls(
            effect="threshold",
            params={
                "min_value": min_value,
                "max_value": max_value,
            },
        )

    @classmethod
    def grow_from_seeds(cls) -> Action:
        """Create Grow from Seeds effect action.

        Returns:
            Action configured for Grow from Seeds effect.
        """
        return cls(effect="grow_from_seeds", params={})

    @classmethod
    def islands(
        cls,
        operation: str = "KEEP_LARGEST",
        min_size: int = 1000,
    ) -> Action:
        """Create Islands operation action.

        Args:
            operation: Operation type (KEEP_LARGEST, REMOVE_SMALL, etc.).
            min_size: Minimum island size in voxels.

        Returns:
            Action configured for Islands effect.
        """
        return cls(
            effect="islands",
            params={
                "operation": operation,
                "min_size": min_size,
            },
        )

    @classmethod
    def smoothing(
        cls,
        method: str = "MEDIAN",
        kernel_size_mm: float = 3.0,
    ) -> Action:
        """Create Smoothing effect action.

        Args:
            method: Smoothing method (MEDIAN, GAUSSIAN, etc.).
            kernel_size_mm: Kernel size in millimeters.

        Returns:
            Action configured for Smoothing effect.
        """
        return cls(
            effect="smoothing",
            params={
                "method": method,
                "kernel_size_mm": kernel_size_mm,
            },
        )

    @classmethod
    def scissors(
        cls,
        ras_points: list[tuple[float, float, float]],
        operation: str = "EraseInside",
        slice_cut: bool = True,
    ) -> Action:
        """Create Scissors effect action.

        Args:
            ras_points: List of RAS coordinates forming the cut path.
            operation: Operation type (EraseInside, EraseOutside, FillInside, etc.).
            slice_cut: Whether to cut in current slice only.

        Returns:
            Action configured for Scissors effect.
        """
        return cls(
            effect="scissors",
            params={
                "ras_points": ras_points,
                "operation": operation,
                "slice_cut": slice_cut,
            },
        )

    def with_overrides(self, overrides: dict[str, Any]) -> Action:
        """Create copy with parameter overrides applied.

        Args:
            overrides: Dictionary of parameter overrides.

        Returns:
            New Action with overrides applied.
        """
        new_params = {**self.params}
        for key, value in overrides.items():
            if key in new_params:
                new_params[key] = value
        return Action(self.effect, new_params)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "effect": self.effect,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        """Create Action from dictionary."""
        return cls(effect=data["effect"], params=data["params"])


@dataclass
class Recipe:
    """A complete segmentation recipe.

    Recipes capture the full workflow for creating a segmentation,
    including sample data, segment name, and sequence of actions.

    Example:
        recipe = Recipe(
            name="brain_tumor_1",
            description="5-click watershed segmentation",
            sample_data="MRBrainTumor1",
            segment_name="Tumor",
            actions=[
                Action.adaptive_brush(ras=(-5.31, 34.77, 20.83), ...),
                Action.adaptive_brush(ras=(-5.31, 25.12, 35.97), ...),
            ],
        )
    """

    name: str
    description: str
    sample_data: str  # Slicer SampleData name
    segment_name: str
    actions: list[Action]
    optimization_hints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> Recipe:
        """Load recipe from Python file.

        Args:
            path: Path to recipe Python file.

        Returns:
            Loaded Recipe object.

        Raises:
            FileNotFoundError: If recipe file doesn't exist.
            AttributeError: If recipe file doesn't define 'recipe' variable.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe file not found: {path}")

        spec = importlib.util.spec_from_file_location("recipe_module", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load recipe from: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "recipe"):
            raise AttributeError(f"Recipe file must define 'recipe' variable: {path}")

        recipe: Recipe = module.recipe

        # Copy optimization hints if present
        if hasattr(module, "optimization_hints"):
            recipe.optimization_hints = module.optimization_hints

        logger.info(f"Loaded recipe '{recipe.name}' from {path}")
        return recipe

    def with_overrides(self, overrides: dict[str, Any]) -> Recipe:
        """Create copy with parameter overrides applied.

        Supports global overrides (applied to all actions) and
        algorithm substitution.

        Args:
            overrides: Dictionary with optional keys:
                - "global": dict of params to override in all actions
                - "algorithm": algorithm to use for all adaptive_brush actions
                - "per_action": list of per-action override dicts

        Returns:
            New Recipe with overrides applied.
        """
        new_actions = []
        global_overrides = overrides.get("global", {})
        algorithm_override = overrides.get("algorithm")
        per_action = overrides.get("per_action", [])

        for i, action in enumerate(self.actions):
            action_overrides = {**global_overrides}

            # Apply algorithm override for adaptive_brush actions
            if algorithm_override and action.effect == "adaptive_brush":
                action_overrides["algorithm"] = algorithm_override

            # Apply per-action overrides if available
            if i < len(per_action):
                action_overrides.update(per_action[i])

            if action_overrides:
                new_actions.append(action.with_overrides(action_overrides))
            else:
                new_actions.append(action)

        return Recipe(
            name=self.name,
            description=self.description,
            sample_data=self.sample_data,
            segment_name=self.segment_name,
            actions=new_actions,
            optimization_hints=self.optimization_hints,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "sample_data": self.sample_data,
            "segment_name": self.segment_name,
            "actions": [a.to_dict() for a in self.actions],
            "optimization_hints": self.optimization_hints,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Recipe:
        """Create Recipe from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            sample_data=data["sample_data"],
            segment_name=data["segment_name"],
            actions=[Action.from_dict(a) for a in data["actions"]],
            optimization_hints=data.get("optimization_hints", {}),
            metadata=data.get("metadata", {}),
        )

    def save(self, output_path: Path | str) -> None:
        """Save recipe as Python file.

        Args:
            output_path: Path where to save the recipe file.
        """
        output_path = Path(output_path)
        code = self._generate_python_code()
        output_path.write_text(code)
        logger.info(f"Saved recipe '{self.name}' to {output_path}")

    def _generate_python_code(self) -> str:
        """Generate Python code for this recipe."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            '"""',
            f"Segmentation recipe: {self.name}",
            "",
            f"Description: {self.description}",
            f"Created: {timestamp}",
            f"Sample Data: {self.sample_data}",
            '"""',
            "from SegmentEditorAdaptiveBrushTesterLib.Recipe import Action, Recipe",
            "",
            "recipe = Recipe(",
            f'    name="{self.name}",',
            f'    description="{self.description}",',
            f'    sample_data="{self.sample_data}",',
            f'    segment_name="{self.segment_name}",',
            "    actions=[",
        ]

        for i, action in enumerate(self.actions):
            lines.append(f"        # Action {i + 1}")
            lines.append(self._format_action(action))

        lines.extend(
            [
                "    ],",
                ")",
                "",
                "# Optimization hints for parameter tuning",
                "optimization_hints = {",
            ]
        )

        # Add optimization hints
        for key, value in self.optimization_hints.items():
            lines.append(f'    "{key}": {repr(value)},')

        lines.extend(["}", ""])

        return "\n".join(lines)

    def _format_action(self, action: Action) -> str:
        """Format a single action as Python code."""
        if action.effect == "adaptive_brush":
            params = action.params
            ras = params["ras"]
            lines = [
                "        Action.adaptive_brush(",
                f"            ras=({ras[0]}, {ras[1]}, {ras[2]}),",
                f'            algorithm="{params["algorithm"]}",',
                f"            brush_radius_mm={params['brush_radius_mm']},",
                f"            edge_sensitivity={params['edge_sensitivity']},",
            ]

            # Add optional params
            if "mode" in params and params["mode"] != "add":
                lines.append(f'            mode="{params["mode"]}",')
            if "threshold_zone" in params:
                lines.append(f"            threshold_zone={params['threshold_zone']},")

            # Add algorithm-specific params
            for key, value in params.items():
                if key not in [
                    "ras",
                    "algorithm",
                    "brush_radius_mm",
                    "edge_sensitivity",
                    "mode",
                    "threshold_zone",
                ]:
                    if isinstance(value, str):
                        lines.append(f'            {key}="{value}",')
                    else:
                        lines.append(f"            {key}={value},")

            lines.append("        ),")
            return "\n".join(lines)

        elif action.effect == "paint":
            params = action.params
            ras = params["ras"]
            return (
                f"        Action.paint(\n"
                f"            ras=({ras[0]}, {ras[1]}, {ras[2]}),\n"
                f"            radius_mm={params['radius_mm']},\n"
                f'            mode="{params.get("mode", "add")}",\n'
                f"            sphere={params.get('sphere', False)},\n"
                f"        ),"
            )

        elif action.effect == "threshold":
            params = action.params
            return (
                f"        Action.threshold(\n"
                f"            min_value={params['min_value']},\n"
                f"            max_value={params['max_value']},\n"
                f"        ),"
            )

        elif action.effect == "grow_from_seeds":
            return "        Action.grow_from_seeds(),"

        elif action.effect == "islands":
            params = action.params
            return (
                f"        Action.islands(\n"
                f'            operation="{params.get("operation", "KEEP_LARGEST")}",\n'
                f"            min_size={params.get('min_size', 1000)},\n"
                f"        ),"
            )

        elif action.effect == "smoothing":
            params = action.params
            return (
                f"        Action.smoothing(\n"
                f'            method="{params.get("method", "MEDIAN")}",\n'
                f"            kernel_size_mm={params.get('kernel_size_mm', 3.0)},\n"
                f"        ),"
            )

        else:
            # Generic formatting
            return f"        Action(effect={repr(action.effect)}, params={repr(action.params)}),"

    def get_action_count(self) -> int:
        """Get number of actions in recipe."""
        return len(self.actions)

    def get_adaptive_brush_actions(self) -> list[Action]:
        """Get only adaptive brush actions."""
        return [a for a in self.actions if a.effect == "adaptive_brush"]

    def get_click_positions(self) -> list[tuple[float, float, float]]:
        """Get RAS positions of all click-based actions."""
        positions = []
        for action in self.actions:
            if "ras" in action.params:
                positions.append(tuple(action.params["ras"]))
        return positions
