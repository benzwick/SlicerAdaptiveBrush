# Creating Recipes

This guide explains how to create segmentation recipes for reproducible workflows,
testing, and optimization.

## Recipe Formats

SlicerAdaptiveBrush supports two recipe formats:

| Format | Extension | Use Case |
|--------|-----------|----------|
| Python function | `.py` | Programmatic recipes with logic |
| JSON action list | `.json` | Simple recorded workflows |

## Python Function Recipes

### Basic Structure

```python
# recipes/brain_tumor.py
"""Brain tumor segmentation recipe."""

from SegmentEditorAdaptiveBrushTesterLib import Recipe, Action


def create_recipe() -> Recipe:
    """Create the brain tumor segmentation recipe.

    Returns:
        Recipe configured for MRBrainTumor1 tumor segmentation.
    """
    recipe = Recipe(
        name="Brain Tumor Segmentation",
        sample_data="MRBrainTumor1",
        gold_standard="MRBrainTumor1_tumor",
        description="5-click tumor segmentation using watershed algorithm",
    )

    # Define click locations (RAS coordinates)
    clicks = [
        {"ras": [-5.31, 34.77, 20.83], "description": "Tumor center"},
        {"ras": [-5.31, 25.12, 35.97], "description": "Superior edge"},
        {"ras": [-12.45, 30.21, 25.67], "description": "Lateral edge"},
        {"ras": [2.15, 32.45, 22.34], "description": "Medial edge"},
        {"ras": [-5.31, 38.92, 18.45], "description": "Inferior edge"},
    ]

    # Default parameters
    params = {
        "algorithm": "watershed",
        "brush_radius_mm": 25.0,
        "edge_sensitivity": 65,
        "threshold_zone": 60,
    }

    # Add actions
    for click in clicks:
        recipe.add_action(
            Action(
                action_type="adaptive_brush",
                ras=click["ras"],
                params=params,
                description=click.get("description"),
            )
        )

    return recipe
```

### With Parameter Variations

```python
def create_recipe() -> Recipe:
    """Recipe with different parameters per click."""
    recipe = Recipe(
        name="Multi-parameter Segmentation",
        sample_data="MRBrainTumor1",
    )

    # First click: large brush for rough segmentation
    recipe.add_action(Action(
        action_type="adaptive_brush",
        ras=[-5.31, 34.77, 20.83],
        params={
            "algorithm": "watershed",
            "brush_radius_mm": 30.0,
            "edge_sensitivity": 50,
        },
        description="Initial rough segmentation",
    ))

    # Subsequent clicks: smaller brush for refinement
    refinement_clicks = [
        [-5.31, 25.12, 35.97],
        [-12.45, 30.21, 25.67],
    ]

    for ras in refinement_clicks:
        recipe.add_action(Action(
            action_type="adaptive_brush",
            ras=ras,
            params={
                "algorithm": "watershed",
                "brush_radius_mm": 15.0,
                "edge_sensitivity": 70,
            },
            description="Boundary refinement",
        ))

    return recipe
```

### With Conditional Logic

```python
def create_recipe(aggressive: bool = False) -> Recipe:
    """Recipe with conditional parameters.

    Args:
        aggressive: If True, use more aggressive segmentation.
    """
    recipe = Recipe(
        name="Conditional Segmentation",
        sample_data="MRBrainTumor1",
    )

    if aggressive:
        params = {
            "algorithm": "connected_threshold",
            "brush_radius_mm": 35.0,
            "edge_sensitivity": 30,
        }
    else:
        params = {
            "algorithm": "watershed",
            "brush_radius_mm": 25.0,
            "edge_sensitivity": 65,
        }

    clicks = [[-5.31, 34.77, 20.83], [-5.31, 25.12, 35.97]]

    for ras in clicks:
        recipe.add_action(Action(
            action_type="adaptive_brush",
            ras=ras,
            params=params,
        ))

    return recipe
```

## JSON Action Recipes

### Basic Structure

```json
{
    "version": "1.0",
    "name": "Brain Tumor Segmentation",
    "description": "5-click tumor segmentation",
    "sample_data": "MRBrainTumor1",
    "gold_standard": "MRBrainTumor1_tumor",
    "actions": [
        {
            "type": "adaptive_brush",
            "ras": [-5.31, 34.77, 20.83],
            "params": {
                "algorithm": "watershed",
                "brush_radius_mm": 25.0,
                "edge_sensitivity": 65
            },
            "description": "Tumor center"
        },
        {
            "type": "adaptive_brush",
            "ras": [-5.31, 25.12, 35.97],
            "params": {
                "algorithm": "watershed",
                "brush_radius_mm": 25.0,
                "edge_sensitivity": 65
            },
            "description": "Superior edge"
        }
    ],
    "metadata": {
        "author": "user",
        "created": "2026-01-26T10:00:00",
        "notes": "Created from manual segmentation"
    }
}
```

### Action Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `adaptive_brush` | Paint with adaptive brush | `ras`, `params` |
| `paint` | Standard paint brush | `ras`, `radius` |
| `erase` | Erase voxels | `ras`, `radius` |
| `threshold` | Apply threshold | `lower`, `upper` |

### Adaptive Brush Parameters

```json
{
    "type": "adaptive_brush",
    "ras": [x, y, z],
    "params": {
        "algorithm": "watershed|geodesic_distance|connected_threshold|...",
        "brush_radius_mm": 25.0,
        "edge_sensitivity": 50,
        "threshold_zone": 50,
        "watershed_gradient_scale": 1.2,
        "watershed_smoothing": 0.7
    }
}
```

## Recording Recipes

### Using the Recorder

1. Open **SegmentEditorAdaptiveBrushReviewer** module
2. Click **Start Recording**
3. Switch to **Segment Editor**
4. Perform segmentation with Adaptive Brush
5. Return to Reviewer and click **Stop Recording**
6. Click **Save Recipe**

### What Gets Recorded

- Click locations (RAS coordinates)
- Active algorithm
- Parameter values at each click
- Timestamps

### Cleaning Up Recorded Recipes

Recorded recipes may need cleanup:

```python
import json

# Load recorded recipe
with open("recorded_recipe.json") as f:
    recipe = json.load(f)

# Remove unnecessary clicks
recipe["actions"] = [a for a in recipe["actions"] if a["ras"][2] > 10]

# Standardize parameters
for action in recipe["actions"]:
    action["params"]["brush_radius_mm"] = 25.0

# Save cleaned recipe
with open("cleaned_recipe.json", "w") as f:
    json.dump(recipe, f, indent=2)
```

## Converting Between Formats

### Python to JSON

```python
from SegmentEditorAdaptiveBrushTesterLib import Recipe

# Load Python recipe
import recipes.brain_tumor as module
recipe = module.create_recipe()

# Convert to JSON
recipe.save_json("brain_tumor.json")
```

### JSON to Python

```python
# Load JSON recipe
with open("brain_tumor.json") as f:
    data = json.load(f)

# Generate Python code
code = f'''
from SegmentEditorAdaptiveBrushTesterLib import Recipe, Action

def create_recipe() -> Recipe:
    recipe = Recipe(
        name="{data['name']}",
        sample_data="{data['sample_data']}",
    )
'''

for action in data["actions"]:
    code += f'''
    recipe.add_action(Action(
        action_type="{action['type']}",
        ras={action['ras']},
        params={action['params']},
    ))
'''

code += '''
    return recipe
'''

print(code)
```

## Testing Recipes

### Quick Test

```python
from SegmentEditorAdaptiveBrushTesterLib import RecipeRunner

# Load and run
import recipes.brain_tumor as module
recipe = module.create_recipe()

runner = RecipeRunner()
result = runner.run(recipe)

print(f"Voxels segmented: {result.voxel_count}")
```

### With Gold Standard Comparison

```python
from SegmentEditorAdaptiveBrushTesterLib import RecipeTestRunner

runner = RecipeTestRunner()
result = runner.run_recipe("recipes/brain_tumor.py")

print(f"Dice: {result.dice:.4f}")
print(f"Hausdorff95: {result.hausdorff_95:.2f}mm")
print(f"Passed: {result.passed}")
```

## Best Practices

### Click Location Selection

1. **Start at center**: First click should be in the middle of the structure
2. **Cover edges**: Add clicks at boundary regions that need attention
3. **Use landmarks**: Place clicks at reproducible anatomical landmarks
4. **Minimize count**: Use fewest clicks needed for good segmentation

### Parameter Selection

1. **Be consistent**: Use same parameters across clicks when possible
2. **Document rationale**: Explain why specific values were chosen
3. **Test variations**: Verify recipe works with small parameter changes

### Recipe Organization

```
recipes/
├── brain/
│   ├── tumor.py
│   ├── white_matter.py
│   └── ventricles.py
├── chest/
│   ├── lung.py
│   └── heart.py
└── template.py
```

### Version Control

1. **Commit recipes**: Store in version control
2. **Track changes**: Note parameter updates
3. **Link to gold standards**: Reference associated gold standards

### Documentation

```python
def create_recipe() -> Recipe:
    """Brain tumor segmentation using watershed algorithm.

    This recipe segments the tumor in MRBrainTumor1 sample data.
    It uses 5 clicks starting from the tumor center and moving
    outward to capture the full extent.

    Optimized parameters from 50-trial Optuna optimization:
    - Best Dice: 0.9991
    - Algorithm: watershed
    - Edge sensitivity: 65

    Returns:
        Recipe configured for tumor segmentation.

    Notes:
        - Works best with T1-weighted MRI
        - May need adjustment for different tumor sizes
        - See optimization results: optimization_results/2026-01-26_*/
    """
```
