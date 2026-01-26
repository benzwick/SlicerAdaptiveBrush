# Testing and Optimization Workflow

This guide describes the complete workflow for testing, optimizing, and validating
Adaptive Brush segmentation algorithms.

## Workflow Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Create    │────▶│    Run      │────▶│   Review    │────▶│    Save     │
│   Recipe    │     │ Optimization│     │   Results   │     │ Gold Std    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                                                            │
       │                                                            │
       └────────────────────────────────────────────────────────────┘
                              Iterate
```

## Phase 1: Create Initial Recipe

### 1.1 Load Sample Data

```python
# In Slicer Python console
import SampleData
volume = SampleData.downloadSample("MRBrainTumor1")
```

### 1.2 Manual Segmentation

1. Open **Segment Editor**
2. Create a new segment
3. Select **Adaptive Brush** effect
4. Segment the structure with multiple clicks
5. Note the click locations (displayed in status bar)

### 1.3 Record Click Locations

```python
# Get click locations from effect
effect = slicer.modules.segmenteditor.widgetRepresentation().self().editor.activeEffect().self()
clicks = effect.clickHistory  # If available

# Or manually record RAS coordinates
clicks = [
    {"ras": [-5.31, 34.77, 20.83]},
    {"ras": [-5.31, 25.12, 35.97]},
    # ...
]
```

### 1.4 Create Recipe File

```python
# recipes/my_segmentation.py
from SegmentEditorAdaptiveBrushTesterLib import Recipe, Action

def create_recipe():
    recipe = Recipe(
        name="My Segmentation",
        sample_data="MRBrainTumor1",
        gold_standard="MRBrainTumor1_my_structure"
    )

    clicks = [
        {"ras": [-5.31, 34.77, 20.83]},
        {"ras": [-5.31, 25.12, 35.97]},
    ]

    for click in clicks:
        recipe.add_action(Action(
            action_type="adaptive_brush",
            ras=click["ras"],
            params={
                "algorithm": "watershed",
                "brush_radius_mm": 25.0,
            }
        ))

    return recipe
```

## Phase 2: Create Initial Gold Standard

### 2.1 Refine Segmentation

Before saving as gold standard:
1. Review the segmentation carefully
2. Fix any over/under-segmentation
3. Ensure boundaries are correct

### 2.2 Save Gold Standard

```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()

# Get current nodes
seg_node = slicer.util.getNode("Segmentation")
vol_node = slicer.util.getNode("MRBrainTumor1")

manager.save_as_gold(
    segmentation_node=seg_node,
    volume_node=vol_node,
    segment_id="Segment_1",
    name="MRBrainTumor1_my_structure",
    click_locations=clicks,
    description="Initial segmentation for optimization",
    algorithm="watershed",
    parameters={
        "brush_radius_mm": 25.0,
        "edge_sensitivity": 50,
    }
)
```

## Phase 3: Run Optimization

### 3.1 Create Configuration

```yaml
# configs/my_optimization.yaml
version: "1.0"
name: "My Structure Optimization"
description: "Find optimal parameters"

settings:
  n_trials: 50
  pruning: true

recipes:
  - path: "recipes/my_segmentation.py"
    gold_standard: "GoldStandards/MRBrainTumor1_my_structure/gold.seg.nrrd"

parameter_space:
  global:
    edge_sensitivity: {type: int, range: [20, 80], step: 10}
    brush_radius_mm: {type: float, range: [15.0, 35.0]}

  algorithm_substitution:
    enabled: true
    candidates: ["watershed", "geodesic_distance", "level_set"]
```

### 3.2 Run Optimization

```bash
source .env
$SLICER_PATH --python-script scripts/run_optimization.py \
    configs/my_optimization.yaml --no-exit
```

### 3.3 Monitor Progress

The terminal shows real-time progress:

```
Trial 0: {'algorithm': 'watershed', 'edge_sensitivity': 45, ...}
  Click 1/5: Dice=0.3421
  Click 2/5: Dice=0.5234
  Click 3/5: Dice=0.7123
  Click 4/5: Dice=0.8456
  Click 5/5: Dice=0.9234
  Final Dice: 0.9234 (4200ms)

Trial 1: {'algorithm': 'geodesic_distance', ...}
  Click 1/5: Dice=0.1234
  Trial pruned at click 1
```

## Phase 4: Review Results

### 4.1 Read Lab Notebook

```bash
cat optimization_results/*/lab_notebook.md
```

Key metrics to check:
- **Best Dice**: Should be > 0.85 for good segmentation
- **Pruned trials**: 30-70% is healthy
- **Parameter importance**: Shows which parameters matter

### 4.2 Visual Review in Slicer

1. With `--no-exit`, Slicer stays open
2. Switch to **SegmentEditorAdaptiveBrushReviewer** module
3. Click **Load Optimization Results**
4. Browse trials with the selector

### 4.3 Compare Screenshots

```bash
# View best trial screenshots
ls optimization_results/*/screenshots/trial_011/
```

Each trial has:
- `001.png`: Before painting
- `002.png`: After click 1
- `003.png`: After click 2
- etc.

## Phase 5: Update Gold Standard

If optimization found better parameters:

### 5.1 Via Reviewer UI

1. Select the best trial
2. Click **Save as Gold Standard**
3. Enter name (e.g., `MRBrainTumor1_my_structure_v2`)

### 5.2 Via Python

```python
# Load best trial segmentation
import json
from pathlib import Path

results_dir = Path("optimization_results/2026-01-26_...")
with open(results_dir / "results.json") as f:
    results = json.load(f)

best = results["best_trial"]
trial_num = best["trial_number"]

# Load segmentation
seg_path = results_dir / f"segmentations/trial_{trial_num:03d}.seg.nrrd"
seg_node = slicer.util.loadSegmentation(str(seg_path))

# Save as new gold standard
manager.save_as_gold(
    segmentation_node=seg_node,
    volume_node=vol_node,
    segment_id=seg_node.GetSegmentation().GetNthSegmentID(0),
    name="MRBrainTumor1_my_structure_v2",
    click_locations=best["user_attrs"]["click_locations"],
    description=f"Optimized, Dice={best['value']:.4f}",
    algorithm=best["params"]["algorithm"],
    parameters=best["params"],
)
```

## Phase 6: Regression Testing

### 6.1 Update Recipe

Update the recipe with optimized parameters:

```python
# recipes/my_segmentation.py (updated)
def create_recipe():
    recipe = Recipe(
        name="My Segmentation (Optimized)",
        sample_data="MRBrainTumor1",
        gold_standard="MRBrainTumor1_my_structure_v2"  # New gold standard
    )

    # Use optimized parameters
    for click in clicks:
        recipe.add_action(Action(
            action_type="adaptive_brush",
            ras=click["ras"],
            params={
                "algorithm": "watershed",  # From optimization
                "brush_radius_mm": 25.0,
                "edge_sensitivity": 70,    # From optimization
            }
        ))

    return recipe
```

### 6.2 Run Regression Test

```bash
$SLICER_PATH --python-script scripts/run_regression.py \
    recipes/my_segmentation.py
```

Expected output:
```
Running: My Segmentation (Optimized)
  Dice: 0.9991
  Status: PASSED (threshold: 0.85)
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/regression.yml
name: Regression Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Slicer
        run: |
          wget https://download.slicer.org/...
          tar xzf Slicer-*.tar.gz

      - name: Run regression tests
        run: |
          ./Slicer --python-script scripts/run_regression.py recipes/
```

## Best Practices

### Recipe Design

1. **Minimal clicks**: Use fewest clicks needed
2. **Consistent locations**: Place clicks at reproducible landmarks
3. **Document rationale**: Explain why each click is needed

### Gold Standard Quality

1. **Expert review**: Have domain expert validate
2. **Multiple reviewers**: Cross-check between reviewers
3. **Version tracking**: Keep history of gold standard updates

### Optimization Strategy

1. **Start broad**: Wide parameter ranges initially
2. **Narrow down**: Reduce ranges based on importance
3. **Cross-validate**: Test on multiple datasets
4. **Document findings**: Record what works and why

### Regression Testing

1. **Run on every change**: Catch regressions early
2. **Multiple datasets**: Don't overfit to one case
3. **Track metrics over time**: Monitor for degradation
