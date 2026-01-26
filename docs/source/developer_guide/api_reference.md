# API Reference

This page documents the Python API for SlicerAdaptiveBrush testing and optimization.

## Core Classes

### Recipe

```{py:class} Recipe(name, sample_data, gold_standard=None, description=None)

A segmentation recipe defining a reproducible workflow.

**Parameters:**
- `name` (str): Recipe name
- `sample_data` (str): Slicer SampleData name to load
- `gold_standard` (str, optional): Name of gold standard for comparison
- `description` (str, optional): Recipe description

**Methods:**

```{py:method} add_action(action)
Add an action to the recipe.

:param action: Action to add
:type action: Action
```

```{py:method} save(path)
Save recipe to file.

:param path: Output path (.py or .json)
:type path: str or Path
```

```{py:method} load(path)
Load recipe from file.

:param path: Recipe file path
:type path: str or Path
:returns: Loaded recipe
:rtype: Recipe
```
```

### Action

```{py:class} Action(action_type, ras, params, description=None)

A single segmentation action.

**Parameters:**
- `action_type` (str): Type of action ("adaptive_brush", "paint", "erase")
- `ras` (list): RAS coordinates [R, A, S]
- `params` (dict): Action parameters
- `description` (str, optional): Action description

**Example:**

```python
action = Action(
    action_type="adaptive_brush",
    ras=[-5.31, 34.77, 20.83],
    params={
        "algorithm": "watershed",
        "brush_radius_mm": 25.0,
    },
    description="Initial click at tumor center"
)
```
```

### RecipeRunner

```{py:class} RecipeRunner()

Executes recipes in Slicer.

**Methods:**

```{py:method} run(recipe)
Execute a recipe.

:param recipe: Recipe to execute
:type recipe: Recipe
:returns: Execution result with voxel counts
:rtype: RecipeResult
```
```

### RecipeTestRunner

```{py:class} RecipeTestRunner(threshold=0.85)

Runs recipes and compares against gold standards.

**Parameters:**
- `threshold` (float): Minimum Dice coefficient for passing (default: 0.85)

**Methods:**

```{py:method} run_recipe(path)
Run recipe and compare to gold standard.

:param path: Path to recipe file
:type path: str or Path
:returns: Test result with metrics
:rtype: TestResult
```
```

## Optimization Classes

### OptunaOptimizer

```{py:class} OptunaOptimizer(config, output_dir=None)

Optuna-based parameter optimizer.

**Parameters:**
- `config` (OptimizationConfig): Optimization configuration
- `output_dir` (Path, optional): Output directory

**Methods:**

```{py:method} create_study(study_name=None)
Create Optuna study with configured sampler and pruner.

:param study_name: Optional study name
:type study_name: str
:returns: Created study
:rtype: optuna.Study
```

```{py:method} optimize(objective, n_trials)
Run optimization.

:param objective: Objective function(trial, params) -> float
:param n_trials: Number of trials
:type n_trials: int
:returns: Optimization results
:rtype: OptimizationResults
```

```{py:method} resume()
Resume optimization from previous study.
```
```

### OptimizationConfig

```{py:class} OptimizationConfig

Configuration for optimization runs.

**Attributes:**
- `name` (str): Configuration name
- `n_trials` (int): Number of trials
- `timeout_minutes` (int): Timeout in minutes
- `pruning` (bool): Enable pruning
- `sampler` (str): Sampler type ("tpe", "random")
- `pruner` (str): Pruner type ("hyperband", "median")
- `recipes` (list): Recipe specifications
- `parameter_space` (dict): Parameter search space

**Methods:**

```{py:method} load(path)
Load configuration from YAML file.

:param path: YAML file path
:type path: str or Path
:returns: Loaded configuration
:rtype: OptimizationConfig
:classmethod:
```
```

## Gold Standard Management

### GoldStandardManager

```{py:class} GoldStandardManager(base_path=None)

Manages gold standard segmentations.

**Parameters:**
- `base_path` (Path, optional): Base path for gold standards

**Methods:**

```{py:method} save_as_gold(segmentation_node, volume_node, segment_id, name, click_locations, description, algorithm, parameters)
Save segmentation as gold standard.

:param segmentation_node: Slicer segmentation node
:param volume_node: Slicer volume node
:param segment_id: Segment ID to save
:param name: Gold standard name
:param click_locations: List of click dicts with "ras" key
:param description: Description
:param algorithm: Algorithm used
:param parameters: Parameter dict
```

```{py:method} load(name)
Load gold standard.

:param name: Gold standard name
:returns: Tuple of (segmentation_node, metadata)
```

```{py:method} list_standards()
List available gold standards.

:returns: List of gold standard names
:rtype: list[str]
```
```

## Screenshot Capture

### ScreenshotCapture

```{py:class} ScreenshotCapture(base_folder, flat_mode=False)

Captures screenshots during testing/optimization.

**Parameters:**
- `base_folder` (Path): Output folder for screenshots
- `flat_mode` (bool): If True, save all to same folder

**Methods:**

```{py:method} screenshot(description)
Capture current view.

:param description: Screenshot description
:returns: Path to saved screenshot
:rtype: Path
```

```{py:method} set_group(name)
Set current group for organizing screenshots.

:param name: Group name (e.g., "trial_001")
```

```{py:method} save_manifest()
Save screenshot manifest JSON.
```
```

## Metrics

### SegmentationMetrics

```{py:class} SegmentationMetrics

Static methods for computing segmentation metrics.

**Methods:**

```{py:method} dice(test_array, gold_array)
Compute Dice coefficient.

:param test_array: Test segmentation array
:param gold_array: Gold standard array
:returns: Dice coefficient (0-1)
:rtype: float
:staticmethod:
```

```{py:method} hausdorff_95(test_array, gold_array, spacing)
Compute 95th percentile Hausdorff distance.

:param test_array: Test segmentation array
:param gold_array: Gold standard array
:param spacing: Voxel spacing [x, y, z]
:returns: Hausdorff distance in mm
:rtype: float
:staticmethod:
```
```

## Usage Examples

### Running a Simple Recipe

```python
from SegmentEditorAdaptiveBrushTesterLib import Recipe, Action, RecipeRunner

# Create recipe
recipe = Recipe(name="Test", sample_data="MRBrainTumor1")
recipe.add_action(Action(
    action_type="adaptive_brush",
    ras=[-5.31, 34.77, 20.83],
    params={"algorithm": "watershed", "brush_radius_mm": 25.0}
))

# Run
runner = RecipeRunner()
result = runner.run(recipe)
print(f"Segmented {result.voxel_count} voxels")
```

### Running Optimization

```python
from SegmentEditorAdaptiveBrushTesterLib import OptimizationConfig, OptunaOptimizer

# Load config
config = OptimizationConfig.load("configs/quick_test.yaml")

# Create optimizer
optimizer = OptunaOptimizer(config)
optimizer.create_study()

# Run
def objective(trial, params):
    # ... run segmentation ...
    return dice_score

results = optimizer.optimize(objective, n_trials=50)
print(f"Best Dice: {results.best_trial.value:.4f}")
```
