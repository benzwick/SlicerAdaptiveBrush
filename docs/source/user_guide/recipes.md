# Segmentation Recipes

Recipes capture complete segmentation workflows that can be replayed, tested, and optimized.
They record click locations, parameters, and expected results for reproducible segmentation.

## What is a Recipe?

A recipe is a saved sequence of segmentation actions:

```python
# Example recipe structure
{
    "name": "Brain Tumor Segmentation",
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
            }
        },
        # ... more actions
    ]
}
```

## Using Recipes

### Loading a Recipe

1. Open the **SegmentEditorAdaptiveBrushReviewer** module
2. Click **Load Recipe**
3. Select a `.json` recipe file

### Replaying a Recipe

The Reviewer module provides step-by-step replay:

| Button | Action |
|--------|--------|
| **Play** | Run all steps automatically |
| **Step** | Execute one action at a time |
| **Reset** | Return to initial state |

### Comparing to Gold Standard

When a recipe has an associated gold standard:

1. The gold standard loads automatically (shown in green)
2. Your segmentation shows in red
3. Overlap shows in yellow
4. Dice coefficient displays in real-time

## Creating Recipes

### From the Recorder

1. Open **SegmentEditorAdaptiveBrushReviewer**
2. Click **Start Recording**
3. Perform your segmentation in Segment Editor
4. Click **Stop Recording**
5. Save the recipe

### From Python

```python
from SegmentEditorAdaptiveBrushTesterLib import Recipe, Action

recipe = Recipe(
    name="My Segmentation",
    sample_data="MRBrainTumor1",
    gold_standard="MRBrainTumor1_tumor"
)

# Add actions
recipe.add_action(Action(
    action_type="adaptive_brush",
    ras=[-5.31, 34.77, 20.83],
    params={
        "algorithm": "watershed",
        "brush_radius_mm": 25.0,
    }
))

# Save
recipe.save("my_recipe.json")
```

### From Optimization Results

After running optimization, the best trial can be converted to a recipe:

```python
# In Slicer Python console
from SegmentEditorAdaptiveBrushTesterLib import Recipe
import json

# Load optimization results
with open("optimization_results/2026-01-26_.../results.json") as f:
    results = json.load(f)

best = results["best_trial"]

# Create recipe from best trial
recipe = Recipe(
    name="Optimized Brain Tumor",
    sample_data="MRBrainTumor1",
    gold_standard="MRBrainTumor1_tumor"
)

for click in best["user_attrs"]["click_locations"]:
    recipe.add_action(Action(
        action_type="adaptive_brush",
        ras=click["ras"],
        params=click["params"]
    ))

recipe.save("optimized_recipe.json")
```

## Recipe File Format

Recipes are stored as JSON:

```json
{
    "version": "1.0",
    "name": "Brain Tumor Segmentation",
    "description": "5-click tumor segmentation using watershed",
    "sample_data": "MRBrainTumor1",
    "gold_standard": "MRBrainTumor1_tumor",
    "created": "2026-01-26T06:20:00",
    "actions": [
        {
            "type": "adaptive_brush",
            "ras": [-5.31, 34.77, 20.83],
            "params": {
                "algorithm": "watershed",
                "brush_radius_mm": 25.0,
                "edge_sensitivity": 65,
                "threshold_zone": 60
            }
        }
    ],
    "metadata": {
        "author": "optimization",
        "dice": 0.9991,
        "notes": "Best result from 50-trial optimization"
    }
}
```

## Gold Standards

### What is a Gold Standard?

A gold standard is a reference segmentation used to:
- Measure recipe quality (Dice coefficient)
- Test for regressions
- Optimize parameters

### Gold Standard Structure

```
GoldStandards/
└── MRBrainTumor1_tumor/
    ├── gold.seg.nrrd    # The segmentation
    ├── metadata.json    # Parameters, clicks, etc.
    └── reference_screenshots/
```

### Creating a Gold Standard

From the Reviewer module:

1. Create a high-quality segmentation
2. Click **Save as Gold Standard**
3. Enter a name and description
4. The gold standard is saved with metadata

From Python:

```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()
manager.save_as_gold(
    segmentation_node=seg_node,
    volume_node=vol_node,
    segment_id="Segment_1",
    name="MRBrainTumor1_tumor",
    click_locations=clicks,
    description="Reference tumor segmentation",
    algorithm="watershed",
    parameters={"brush_radius_mm": 25.0}
)
```

## Testing Recipes

### Running Regression Tests

```bash
# From command line
Slicer --python-script scripts/run_regression.py recipes/brain_tumor_1.py
```

### In Python

```python
from SegmentEditorAdaptiveBrushTesterLib import RecipeTestRunner

runner = RecipeTestRunner()
result = runner.run_recipe("recipes/brain_tumor_1.py")

print(f"Dice: {result.dice:.4f}")
print(f"Passed: {result.passed}")
```

## Best Practices

### Recipe Design

1. **Use consistent click locations** that capture the structure well
2. **Start from the center** of the structure
3. **Add boundary clicks** for edges that need attention
4. **Document the purpose** in the recipe description

### Gold Standard Quality

1. **Manual refinement** - Correct any errors before saving
2. **Include edge cases** - Test challenging boundaries
3. **Document parameters** - Record what worked

### Version Control

- Store recipes in version control
- Gold standards can be regenerated from recipes
- Track parameter changes over time
