"""YAML configuration loader for optimization runs.

Loads and validates optimization configuration from YAML files,
supporting parameter spaces, recipe references, and settings.

See ADR-011 for architecture decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _ensure_yaml() -> bool:
    """Ensure PyYAML is available, installing if needed.

    Returns:
        True if yaml is available, False if installation failed.
    """
    global YAML_AVAILABLE, yaml

    if YAML_AVAILABLE:
        return True

    try:
        import slicer

        logger.info("Installing PyYAML...")
        slicer.util.pip_install("pyyaml")
        import yaml as _yaml

        yaml = _yaml
        YAML_AVAILABLE = True
        logger.info("PyYAML installed successfully")
        return True
    except Exception as e:
        logger.warning(f"Could not install PyYAML: {e}")
        return False


@dataclass
class ParameterSpec:
    """Specification for a parameter to optimize.

    Supports int, float, and categorical parameter types with
    ranges, steps, and choices.
    """

    name: str
    param_type: str  # "int", "float", "categorical"
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False

    @classmethod
    def from_dict(cls, name: str, spec: dict) -> ParameterSpec:
        """Create ParameterSpec from config dictionary.

        Args:
            name: Parameter name.
            spec: Dictionary with type, range/choices, etc.

        Returns:
            ParameterSpec instance.
        """
        param_type = spec.get("type", "float")

        if param_type == "categorical":
            return cls(
                name=name,
                param_type=param_type,
                choices=spec.get("choices", []),
            )

        # Numeric type
        range_vals = spec.get("range", [0, 1])
        return cls(
            name=name,
            param_type=param_type,
            min_val=range_vals[0] if len(range_vals) > 0 else 0,
            max_val=range_vals[1] if len(range_vals) > 1 else 1,
            step=spec.get("step"),
            log_scale=spec.get("log", False),
        )


@dataclass
class RecipeSpec:
    """Specification for a recipe to optimize."""

    path: Path
    gold_standard: Path | None = None
    weight: float = 1.0


@dataclass
class OptimizationConfig:
    """Complete optimization configuration.

    Loaded from YAML config files, this class contains all settings
    needed to run an optimization study.

    Example YAML:
        version: "1.0"
        name: "Tumor Optimization"
        settings:
          n_trials: 100
          sampler: "tpe"
          pruner: "hyperband"
        recipes:
          - path: "recipes/brain_tumor_1.py"
            gold_standard: "gold_standards/tumor.seg.nrrd"
        parameter_space:
          global:
            edge_sensitivity: {type: int, range: [20, 80], step: 10}
    """

    # Metadata
    version: str = "1.0"
    name: str = "Optimization"
    description: str = ""

    # Settings
    n_trials: int = 100
    timeout_minutes: int | None = None
    sampler: str = "tpe"  # tpe, random, cmaes
    pruner: str = "hyperband"  # hyperband, median, none
    primary_metric: str = "dice"  # dice, hausdorff_95
    direction: str = "maximize"  # maximize for dice, minimize for hausdorff
    save_segmentations: bool = True
    save_screenshots: bool = True

    # Recipes
    recipes: list[RecipeSpec] = field(default_factory=list)

    # Parameter spaces
    global_params: dict[str, ParameterSpec] = field(default_factory=dict)
    algorithm_params: dict[str, dict[str, ParameterSpec]] = field(default_factory=dict)

    # Algorithm substitution
    algorithm_substitution_enabled: bool = False
    algorithm_candidates: list[str] = field(default_factory=list)

    # Output settings
    output_formats: list[str] = field(default_factory=lambda: ["json", "markdown"])
    generate_algorithm_profiles: bool = True

    # Source path (set when loading)
    source_path: Path | None = None

    @classmethod
    def load(cls, config_path: Path | str) -> OptimizationConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Loaded OptimizationConfig.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.
            ImportError: If PyYAML is not installed.
        """
        if not _ensure_yaml():
            raise ImportError(
                "PyYAML is required for loading config files. Install with: pip install pyyaml"
            )

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        config = cls._parse_config(data)
        config.source_path = config_path

        logger.info(f"Loaded optimization config '{config.name}' from {config_path}")
        return config

    @classmethod
    def _parse_config(cls, data: dict) -> OptimizationConfig:
        """Parse configuration dictionary.

        Args:
            data: Parsed YAML dictionary.

        Returns:
            OptimizationConfig instance.
        """
        config = cls()

        # Metadata
        config.version = data.get("version", "1.0")
        config.name = data.get("name", "Optimization")
        config.description = data.get("description", "")

        # Settings
        settings = data.get("settings", {})
        config.n_trials = settings.get("n_trials", 100)
        config.timeout_minutes = settings.get("timeout_minutes")
        config.sampler = settings.get("sampler", "tpe")
        config.pruner = settings.get("pruner", "hyperband")
        config.primary_metric = settings.get("primary_metric", "dice")
        config.save_segmentations = settings.get("save_segmentations", True)
        config.save_screenshots = settings.get("save_screenshots", True)

        # Direction based on metric
        if config.primary_metric in ("hausdorff_95", "hausdorff"):
            config.direction = "minimize"
        else:
            config.direction = "maximize"

        # Recipes
        config.recipes = []
        for recipe_data in data.get("recipes", []):
            recipe_spec = RecipeSpec(
                path=Path(recipe_data["path"]),
                gold_standard=Path(recipe_data["gold_standard"])
                if "gold_standard" in recipe_data
                else None,
                weight=recipe_data.get("weight", 1.0),
            )
            config.recipes.append(recipe_spec)

        # Parameter space
        param_space = data.get("parameter_space", {})

        # Global parameters
        global_params = param_space.get("global", {})
        for name, spec in global_params.items():
            config.global_params[name] = ParameterSpec.from_dict(name, spec)

        # Algorithm substitution
        algo_sub = param_space.get("algorithm_substitution", {})
        config.algorithm_substitution_enabled = algo_sub.get("enabled", False)
        config.algorithm_candidates = algo_sub.get("candidates", [])

        # Algorithm-specific parameters
        algo_params = param_space.get("algorithms", {})
        for algo_name, params in algo_params.items():
            config.algorithm_params[algo_name] = {}
            for param_name, spec in params.items():
                config.algorithm_params[algo_name][param_name] = ParameterSpec.from_dict(
                    param_name, spec
                )

        # Output settings
        output = data.get("output", {})
        config.output_formats = output.get("reports", ["json", "markdown"])
        config.generate_algorithm_profiles = output.get("algorithm_profiles", True)

        return config

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        if self.n_trials <= 0:
            errors.append("n_trials must be positive")

        if self.sampler not in ("tpe", "random", "cmaes"):
            errors.append(f"Unknown sampler: {self.sampler}")

        if self.pruner not in ("hyperband", "median", "none"):
            errors.append(f"Unknown pruner: {self.pruner}")

        if self.primary_metric not in ("dice", "hausdorff_95", "hausdorff"):
            errors.append(f"Unknown primary_metric: {self.primary_metric}")

        if not self.recipes:
            errors.append("At least one recipe must be specified")

        for recipe in self.recipes:
            if not recipe.path.suffix == ".py":
                errors.append(f"Recipe path must be a .py file: {recipe.path}")

        return errors

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Configuration as dictionary.
        """
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "settings": {
                "n_trials": self.n_trials,
                "timeout_minutes": self.timeout_minutes,
                "sampler": self.sampler,
                "pruner": self.pruner,
                "primary_metric": self.primary_metric,
                "save_segmentations": self.save_segmentations,
                "save_screenshots": self.save_screenshots,
            },
            "recipes": [
                {
                    "path": str(r.path),
                    "gold_standard": str(r.gold_standard) if r.gold_standard else None,
                    "weight": r.weight,
                }
                for r in self.recipes
            ],
            "parameter_space": {
                "global": {
                    name: self._param_spec_to_dict(spec)
                    for name, spec in self.global_params.items()
                },
                "algorithm_substitution": {
                    "enabled": self.algorithm_substitution_enabled,
                    "candidates": self.algorithm_candidates,
                },
                "algorithms": {
                    algo: {name: self._param_spec_to_dict(spec) for name, spec in params.items()}
                    for algo, params in self.algorithm_params.items()
                },
            },
            "output": {
                "reports": self.output_formats,
                "algorithm_profiles": self.generate_algorithm_profiles,
            },
        }

    @staticmethod
    def _param_spec_to_dict(spec: ParameterSpec) -> dict:
        """Convert ParameterSpec to dictionary."""
        if spec.param_type == "categorical":
            return {"type": "categorical", "choices": spec.choices}

        result: dict[str, Any] = {
            "type": spec.param_type,
            "range": [spec.min_val, spec.max_val],
        }
        if spec.step is not None:
            result["step"] = spec.step
        if spec.log_scale:
            result["log"] = True
        return result

    def save(self, output_path: Path | str) -> None:
        """Save configuration to YAML file.

        Args:
            output_path: Path where to save config.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not _ensure_yaml():
            raise ImportError(
                "PyYAML is required for saving config files. Install with: pip install pyyaml"
            )

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved optimization config to {output_path}")


def create_default_config() -> OptimizationConfig:
    """Create a default optimization configuration.

    Returns:
        Default OptimizationConfig with reasonable settings.
    """
    config = OptimizationConfig(
        name="Default Optimization",
        description="Default parameter optimization configuration",
        n_trials=50,
        sampler="tpe",
        pruner="hyperband",
    )

    # Add default global parameters
    config.global_params = {
        "edge_sensitivity": ParameterSpec(
            name="edge_sensitivity",
            param_type="int",
            min_val=20,
            max_val=80,
            step=10,
        ),
        "threshold_zone": ParameterSpec(
            name="threshold_zone",
            param_type="int",
            min_val=30,
            max_val=70,
            step=10,
        ),
        "brush_radius_mm": ParameterSpec(
            name="brush_radius_mm",
            param_type="float",
            min_val=10.0,
            max_val=40.0,
            step=5.0,
        ),
    }

    # Add algorithm-specific parameters
    config.algorithm_params = {
        "watershed": {
            "watershedGradientScale": ParameterSpec(
                name="watershedGradientScale",
                param_type="float",
                min_val=0.5,
                max_val=2.5,
            ),
            "watershedSmoothing": ParameterSpec(
                name="watershedSmoothing",
                param_type="float",
                min_val=0.2,
                max_val=1.0,
            ),
        },
        "level_set_cpu": {
            "levelSetIterations": ParameterSpec(
                name="levelSetIterations",
                param_type="int",
                min_val=30,
                max_val=150,
                step=20,
            ),
            "levelSetPropagation": ParameterSpec(
                name="levelSetPropagation",
                param_type="float",
                min_val=0.5,
                max_val=2.0,
            ),
        },
    }

    return config
