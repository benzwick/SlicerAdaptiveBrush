"""Segmentation Recipe loader.

Recipes are Python scripts that define a `run(effect)` function.
This module loads recipes and provides utilities for running them.

Example recipe (brain_tumor_1.py):

    sample_data = "MRBrainTumor1"
    segment_name = "Tumor"

    def run(effect):
        effect.applyPreset("tumor_lesion")
        effect.brushRadiusMm = 20.0
        effect.paintAt(-5.31, 34.77, 20.83)
        effect.paintAt(-5.31, 25.12, 35.97)
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    """A loaded recipe ready to execute.

    Attributes:
        name: Recipe name (from filename).
        path: Path to the recipe file.
        sample_data: Slicer SampleData name to load.
        segment_name: Name for the segment to create.
        run: The run(effect) function to execute.
        module: The loaded Python module.
    """

    name: str
    path: Path
    sample_data: str
    segment_name: str
    run: Callable[[Any], None]
    module: ModuleType

    @classmethod
    def load(cls, path: Path | str) -> Recipe:
        """Load a recipe from a Python file.

        Args:
            path: Path to the recipe file.

        Returns:
            Loaded Recipe object.

        Raises:
            FileNotFoundError: If recipe file doesn't exist.
            AttributeError: If recipe doesn't define required attributes.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe not found: {path}")

        # Load the module
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load recipe: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract required attributes
        if not hasattr(module, "run"):
            raise AttributeError(f"Recipe must define run(effect) function: {path}")

        sample_data = getattr(module, "sample_data", "")
        segment_name = getattr(module, "segment_name", "Segment")

        logger.info(f"Loaded recipe: {path.stem}")

        return cls(
            name=path.stem,
            path=path,
            sample_data=sample_data,
            segment_name=segment_name,
            run=module.run,
            module=module,
        )

    def execute(self, effect: Any) -> None:
        """Execute the recipe with the given effect.

        Args:
            effect: The Adaptive Brush scripted effect instance.
        """
        logger.info(f"Executing recipe: {self.name}")
        self.run(effect)
        logger.info(f"Recipe complete: {self.name}")


def list_recipes(recipes_dir: Path | str | None = None) -> list[Path]:
    """List available recipe files.

    Args:
        recipes_dir: Directory to search. If None, uses default location.

    Returns:
        List of recipe file paths.
    """
    if recipes_dir is None:
        # Default to recipes/ in the same package
        recipes_dir = Path(__file__).parent.parent / "recipes"

    recipes_dir = Path(recipes_dir)
    if not recipes_dir.exists():
        return []

    return sorted(
        p
        for p in recipes_dir.glob("*.py")
        if not p.name.startswith("_") and p.name != "template.py"
    )
