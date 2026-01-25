# ADR-013: Segmentation Recipes

## Status

Implemented

The recipe system is fully functional with:
- Recipe class for complete segmentation workflows
- Action class for individual operations
- RecipeRunner for executing recipes in Slicer
- RecipeRecorder for capturing manual sessions
- RecipeTestRunner for regression testing against gold standards
- Example recipes (brain_tumor_1.py, template.py)
- Gold standard: MRBrainTumor1_tumor (Dice: 0.858, HD95: 3.28mm)

## Context

A "gold standard" segmentation is not just a final result - it's created by a **sequence of actions**, where:

- Each click can have **different parameters** (brush size, algorithm, sensitivity)
- Multiple **different tools/effects** may be used (Adaptive Brush, Paint, Threshold, Grow from Seeds)
- The sequence must be **recordable** from manual sessions
- Recipes must be **editable** by humans and AI agents
- Recipes enable **exact reproducibility** and **parameter optimization**

### Current Limitations

The existing `test_optimization_tumor.py` test hardcodes:
- Click locations in Python code
- Single parameter set for all clicks
- Single algorithm per test run

This makes it difficult to:
- Record and replay manual sessions
- Optimize parameters per-click
- Mix different effects in one workflow
- Share and version segmentation approaches

## Decision

Implement **Segmentation Recipes** as Python files that capture complete segmentation workflows.

### Why Python (Not YAML)?

| Aspect | Python | YAML |
|--------|--------|------|
| Flexibility | Full programming | Limited |
| Validation | Type hints, IDE support | Schema required |
| Editability | Very familiar | Also familiar |
| Execution | Direct import | Needs parser |
| Optimization hints | Natural Python dict | Complex nesting |

Python files provide maximum flexibility while remaining human-readable and AI-editable.

### Recipe File Structure

```python
# recipes/brain_tumor_1.py
"""
Segmentation recipe for MRBrainTumor1 tumor.

Created: 2026-01-25 (recorded from manual session)
Gold Standard: gold_standards/MRBrainTumor1_tumor.seg.nrrd
"""
from segmentation_recipes import Recipe, Action

recipe = Recipe(
    name="brain_tumor_1",
    description="5-click watershed segmentation of brain tumor",
    sample_data="MRBrainTumor1",  # Slicer SampleData name
    segment_name="Tumor",

    actions=[
        # First stroke - large brush, low sensitivity
        Action.adaptive_brush(
            ras=(-5.31, 34.77, 20.83),
            algorithm="watershed",
            brush_radius_mm=25.0,
            edge_sensitivity=40,
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),

        # Second stroke - same params
        Action.adaptive_brush(
            ras=(-5.31, 25.12, 35.97),
            algorithm="watershed",
            brush_radius_mm=25.0,
            edge_sensitivity=40,
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),

        # Third stroke - smaller brush for detail
        Action.adaptive_brush(
            ras=(-5.31, 20.70, 22.17),
            algorithm="watershed",
            brush_radius_mm=15.0,  # Smaller!
            edge_sensitivity=50,   # Higher!
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),

        # Clean up with erase
        Action.adaptive_brush(
            ras=(-1.35, 28.65, 18.90),
            algorithm="watershed",
            brush_radius_mm=10.0,
            edge_sensitivity=60,
            mode="erase",
        ),

        # Optional: Use standard paint for touch-up
        Action.paint(
            ras=(-2.0, 30.0, 20.0),
            radius_mm=3.0,
            mode="erase",
        ),
    ]
)

# Optimization hints (which params to vary)
optimization_hints = {
    "vary_globally": ["edge_sensitivity", "threshold_zone"],
    "vary_per_action": ["brush_radius_mm"],
    "algorithm_options": ["watershed", "level_set_cpu", "connected_threshold"],
}
```

### Core Classes

#### Recipe

```python
@dataclass
class Recipe:
    """A complete segmentation recipe."""
    name: str
    description: str
    sample_data: str         # Slicer SampleData name
    segment_name: str
    actions: list[Action]
    optimization_hints: dict = None

    @classmethod
    def load(cls, path: Path) -> "Recipe":
        """Load recipe from Python file."""
        spec = importlib.util.spec_from_file_location("recipe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.recipe

    def with_overrides(self, overrides: dict) -> "Recipe":
        """Create copy with parameter overrides applied."""
        new_actions = []
        for action in self.actions:
            new_params = {**action.params}
            # Apply global overrides
            for key, value in overrides.get("global", {}).items():
                if key in new_params:
                    new_params[key] = value
            # Apply algorithm override
            if "algorithm" in overrides:
                new_params["algorithm"] = overrides["algorithm"]
            new_actions.append(Action(action.effect, new_params))

        return Recipe(
            name=self.name,
            description=self.description,
            sample_data=self.sample_data,
            segment_name=self.segment_name,
            actions=new_actions,
            optimization_hints=self.optimization_hints,
        )
```

#### Action

```python
@dataclass
class Action:
    """A single segmentation action."""
    effect: str              # "adaptive_brush", "paint", "threshold", etc.
    params: dict             # All parameters for this action

    @classmethod
    def adaptive_brush(
        cls,
        ras: tuple[float, float, float],
        algorithm: str,
        brush_radius_mm: float,
        edge_sensitivity: int,
        mode: str = "add",
        **kwargs
    ) -> "Action":
        """Create Adaptive Brush action."""
        return cls(
            effect="adaptive_brush",
            params={
                "ras": ras,
                "algorithm": algorithm,
                "brush_radius_mm": brush_radius_mm,
                "edge_sensitivity": edge_sensitivity,
                "mode": mode,
                **kwargs
            }
        )

    @classmethod
    def paint(
        cls,
        ras: tuple[float, float, float],
        radius_mm: float,
        mode: str = "add",
        sphere: bool = False,
    ) -> "Action":
        """Create standard Paint action."""
        return cls(
            effect="paint",
            params={
                "ras": ras,
                "radius_mm": radius_mm,
                "mode": mode,
                "sphere": sphere,
            }
        )

    @classmethod
    def threshold(
        cls,
        min_value: float,
        max_value: float,
    ) -> "Action":
        """Create Threshold action."""
        return cls(
            effect="threshold",
            params={
                "min_value": min_value,
                "max_value": max_value,
            }
        )

    @classmethod
    def grow_from_seeds(cls) -> "Action":
        """Create Grow from Seeds action."""
        return cls(effect="grow_from_seeds", params={})

    @classmethod
    def islands(
        cls,
        operation: str = "KEEP_LARGEST",
        min_size: int = 1000,
    ) -> "Action":
        """Create Islands operation action."""
        return cls(
            effect="islands",
            params={
                "operation": operation,
                "min_size": min_size,
            }
        )

    @classmethod
    def smoothing(
        cls,
        method: str = "MEDIAN",
        kernel_size_mm: float = 3.0,
    ) -> "Action":
        """Create Smoothing action."""
        return cls(
            effect="smoothing",
            params={
                "method": method,
                "kernel_size_mm": kernel_size_mm,
            }
        )
```

### RecipeRunner

Executes recipes in Slicer:

```python
class RecipeRunner:
    """Execute a recipe in Slicer."""

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self.segmentation_node = None
        self.volume_node = None

    def run(
        self,
        progress_callback: Callable[[int, Action], None] = None,
        screenshot_callback: Callable[[int, str], None] = None,
    ) -> "SegmentationNode":
        """Execute recipe, optionally with callbacks."""
        # Load sample data
        self.volume_node = self._load_sample_data(self.recipe.sample_data)

        # Create segmentation
        self.segmentation_node = self._create_segmentation(self.recipe.segment_name)

        # Execute each action
        for i, action in enumerate(self.recipe.actions):
            if progress_callback:
                progress_callback(i, action)

            self._execute_action(action)

            if screenshot_callback:
                screenshot_callback(i, f"After action {i+1}")

        return self.segmentation_node

    def _execute_action(self, action: Action):
        """Dispatch to appropriate Slicer effect."""
        dispatch = {
            "adaptive_brush": self._execute_adaptive_brush,
            "paint": self._execute_paint,
            "threshold": self._execute_threshold,
            "grow_from_seeds": self._execute_grow_from_seeds,
            "islands": self._execute_islands,
            "smoothing": self._execute_smoothing,
        }

        handler = dispatch.get(action.effect)
        if handler is None:
            raise ValueError(f"Unknown effect: {action.effect}")

        handler(action.params)

    def _execute_adaptive_brush(self, params: dict):
        """Execute Adaptive Brush stroke."""
        # Activate effect
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        effect = self.segment_editor_widget.activeEffect().self()

        # Set parameters
        effect.radiusSlider.value = params["brush_radius_mm"]
        effect.sensitivitySlider.value = params["edge_sensitivity"]

        # Select algorithm
        idx = effect.algorithmCombo.findData(params["algorithm"])
        effect.algorithmCombo.setCurrentIndex(idx)

        # Set algorithm-specific params
        self._set_algorithm_params(effect, params)

        # Execute stroke at RAS position
        ras = params["ras"]
        xy = self._ras_to_xy(ras)
        is_erase = params.get("mode") == "erase"

        effect.scriptedEffect.saveStateForUndo()
        effect.isDrawing = True
        effect._currentStrokeEraseMode = is_erase
        effect.processPoint(xy, self.slice_widget)
        effect.isDrawing = False

        slicer.app.processEvents()
```

### RecipeRecorder

Records manual sessions into recipe files:

```python
class RecipeRecorder:
    """Record manual Slicer session into a recipe file."""

    def __init__(self):
        self.actions = []
        self.recording = False
        self.sample_data = None
        self.segment_name = None

    def start(self, sample_data: str, segment_name: str):
        """Start recording."""
        self.sample_data = sample_data
        self.segment_name = segment_name
        self.actions = []
        self.recording = True
        self._install_hooks()

    def stop(self) -> Recipe:
        """Stop recording and return Recipe."""
        self.recording = False
        self._remove_hooks()

        return Recipe(
            name=f"recorded_{datetime.now():%Y%m%d_%H%M%S}",
            description="Recorded from manual session",
            sample_data=self.sample_data,
            segment_name=self.segment_name,
            actions=self.actions,
        )

    def save(self, output_path: Path):
        """Save recipe to Python file."""
        recipe = self.stop()
        self._write_python_file(recipe, output_path)

    def _install_hooks(self):
        """Hook into Slicer effect events."""
        # Subscribe to effect parameter changes
        # Subscribe to stroke completion events
        pass

    def _on_adaptive_brush_stroke(self, effect, xy, view_widget, erase_mode):
        """Called when Adaptive Brush completes a stroke."""
        if not self.recording:
            return

        # Capture all current parameters
        ras = self._xy_to_ras(xy, view_widget)

        action = Action.adaptive_brush(
            ras=ras,
            algorithm=effect.algorithmCombo.currentData,
            brush_radius_mm=effect.radiusSlider.value,
            edge_sensitivity=effect.sensitivitySlider.value,
            mode="erase" if erase_mode else "add",
            # ... other algorithm-specific params
        )
        self.actions.append(action)

    def _write_python_file(self, recipe: Recipe, output_path: Path):
        """Write recipe as Python file."""
        code = f'''"""
Segmentation recipe: {recipe.name}

Created: {datetime.now():%Y-%m-%d %H:%M}
Sample Data: {recipe.sample_data}
"""
from segmentation_recipes import Recipe, Action

recipe = Recipe(
    name="{recipe.name}",
    description="{recipe.description}",
    sample_data="{recipe.sample_data}",
    segment_name="{recipe.segment_name}",

    actions=[
'''
        for action in recipe.actions:
            code += self._format_action(action)

        code += '''    ]
)

optimization_hints = {
    "vary_globally": ["edge_sensitivity", "threshold_zone"],
    "vary_per_action": ["brush_radius_mm"],
}
'''
        output_path.write_text(code)
```

### Directory Structure

```
SegmentEditorAdaptiveBrushTester/
├── SegmentEditorAdaptiveBrushTesterLib/
│   ├── Recipe.py                    # Recipe and Action classes
│   ├── RecipeRunner.py              # Execute recipes in Slicer
│   ├── RecipeRecorder.py            # Record manual sessions
│   └── ...
├── recipes/                          # Segmentation recipes
│   ├── __init__.py
│   ├── brain_tumor_1.py             # 5-click watershed
│   ├── brain_tumor_detailed.py      # 10-click multi-algorithm
│   └── template.py                  # Template for new recipes
└── ...
```

### Workflow

```
Manual Session → Record → Recipe.py → Edit/Optimize → Gold Standard
                           ↑                              ↓
                           └──────────────────────────────┘
                                  (iterate)
```

1. **Record**: User performs segmentation manually, recorder captures actions
2. **Save**: Recipe saved as Python file
3. **Edit**: User or AI adjusts parameters, adds/removes actions
4. **Replay**: RecipeRunner executes recipe exactly
5. **Optimize**: Optuna varies parameters to find optimal settings
6. **Promote**: Best result becomes new gold standard

### Benefits

| Benefit | Description |
|---------|-------------|
| **Per-Click Parameters** | Each action has its own brush size, algorithm, sensitivity |
| **Multi-Tool Support** | Mix Adaptive Brush, Paint, Threshold, Grow from Seeds, etc. |
| **Recordable** | Capture manual sessions automatically |
| **Editable** | Python files - humans and AI can modify easily |
| **Version Control** | Recipes are code - track changes with git |
| **Optimization** | Vary specific parameters while keeping structure |
| **Reproducible** | Exact replay of segmentation process |
| **Testable** | Compare recipe output to gold standard |

## Consequences

### Positive

- **Exact reproduction**: Recipes capture complete workflows
- **Optimization ready**: Parameter overrides enable systematic tuning
- **Multi-effect support**: Not limited to Adaptive Brush
- **Human readable**: Python is familiar to most users
- **AI friendly**: Claude can read and modify recipes

### Negative

- **Coordinate fragility**: RAS coordinates tied to specific sample data
- **Effect API changes**: Recipes may break if Slicer effects change
- **Recording complexity**: Hooking into all effects requires effort

### Trade-offs

| Aspect | Alternative | Chosen | Reason |
|--------|-------------|--------|--------|
| Format | YAML | Python | More flexible, type hints |
| Coordinates | IJK | RAS | Consistent across orientations |
| Recording | Manual only | Auto hooks | More accurate, less effort |

## Alternatives Considered

### YAML Recipe Format

**Rejected**: Less flexible, no type hints, requires custom parser. Python files can be imported directly.

### IJK Coordinates

**Rejected**: Dependent on volume orientation. RAS is consistent.

### Record All Mouse Events

**Rejected**: Too verbose, captures noise. Record effect-level events instead.

### Single Effect Only

**Rejected**: Real workflows use multiple effects. Support them all.

## Implementation Notes

### Slicer Event Hooks

The recorder hooks into:
- `SegmentEditorEffect.modifySelectedSegmentByLabelmap()` calls
- Parameter widget value changes
- Effect activation/deactivation

### Sample Data Names

Use exact Slicer SampleData names:
- `MRHead`
- `MRBrainTumor1`
- `MRBrainTumor2`
- `CTChest`
- etc.

### Algorithm Parameter Mapping

Map UI widget names to recipe parameter names:
```python
PARAM_MAP = {
    "radiusSlider": "brush_radius_mm",
    "sensitivitySlider": "edge_sensitivity",
    "thresholdZoneSlider": "threshold_zone",
    "watershedGradientScaleSlider": "watershedGradientScale",
}
```

## References

- [ADR-010](ADR-010-testing-framework.md): Slicer Testing Framework
- [ADR-011](ADR-011-optimization-framework.md): Smart Optimization Framework
- [Slicer SampleData](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#download-sample-data)
- [Segment Editor Effects](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
