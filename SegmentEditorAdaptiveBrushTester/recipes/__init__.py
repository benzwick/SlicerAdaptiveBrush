"""Segmentation recipes package.

This package contains segmentation recipes - Python files that define
complete segmentation workflows for reproducibility and optimization.

See ADR-013 for architecture decisions.
"""

from pathlib import Path

RECIPES_DIR = Path(__file__).parent


def list_recipes() -> list[Path]:
    """List all recipe files in this directory.

    Returns:
        List of paths to .py recipe files.
    """
    return [
        p
        for p in RECIPES_DIR.glob("*.py")
        if p.name != "__init__.py" and not p.name.startswith("_")
    ]


def get_recipe_path(name: str) -> Path:
    """Get path to a recipe by name.

    Args:
        name: Recipe name (without .py extension).

    Returns:
        Path to recipe file.

    Raises:
        FileNotFoundError: If recipe doesn't exist.
    """
    path = RECIPES_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Recipe not found: {name}")
    return path
