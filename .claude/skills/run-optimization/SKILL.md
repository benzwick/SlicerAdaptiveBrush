# run-optimization

Run parameter optimization for segmentation algorithms.

## Usage

```
/run-optimization <algorithm> --gold <gold_name> [--trials <n>]
```

Where:
- `algorithm` - Algorithm to optimize (watershed, level_set_cpu, connected_threshold, etc.)
- `--gold` - Gold standard to compare against
- `--trials` - Number of trials (default: 20)

## What This Skill Does

1. Loads the specified gold standard
2. Runs N trials with different parameter combinations
3. Uses exactly 5 clicks per trial (from gold standard metadata)
4. Tracks Dice/Hausdorff after each stroke
5. Identifies best parameters and diminishing returns
6. Saves results and generates lab notebook

## Prerequisites

1. Gold standard exists in `GoldStandards/<gold_name>/`
2. Slicer is available (set in .env file)

## Execution Steps

### Step 1: Launch Slicer with Optimization Script

First, read the .env file to get SLICER_PATH, then launch:

```bash
<SLICER_PATH> --python-script scripts/run_optimization.py <algorithm> <gold_name> <trials>
```

### Step 2: Monitor Progress

The script outputs progress to stdout:

```
Trial 1/20: edge_sensitivity=30, threshold_zone=50, brush_radius=20
  Stroke 1: Dice=0.42
  Stroke 2: Dice=0.67
  Stroke 3: Dice=0.81
  Stroke 4: Dice=0.85
  Stroke 5: Dice=0.87
  Final: Dice=0.87, HD95=5.2mm
...
```

### Step 3: Review Results

Results are saved to:

```
test_runs/<timestamp>_optimization/
├── optimization_results.json    # All trial data
├── best_params.json             # Best parameters found
├── screenshots/                 # Best trial visualization
└── lab_notebook.md              # Human-readable summary
```

## Output Format

### optimization_results.json

```json
{
  "algorithm": "watershed",
  "gold_standard": "MRBrainTumor1_tumor",
  "trials": [
    {
      "trial_id": 1,
      "params": {
        "edge_sensitivity": 30,
        "threshold_zone": 50,
        "brush_radius_mm": 20
      },
      "dice": 0.87,
      "hausdorff_95": 5.2,
      "strokes": 5,
      "duration_ms": 1234
    }
  ],
  "summary": {
    "best_dice": 0.92,
    "best_params": {...}
  }
}
```

### lab_notebook.md

Human-readable markdown document:

```markdown
# Watershed Parameter Optimization

**Date:** 2026-01-24
**Gold Standard:** MRBrainTumor1_tumor

## Summary
- Best Dice: 0.92
- Best Hausdorff 95%: 3.4mm
- Trials: 20

## Best Parameters
- edge_sensitivity: 40
- threshold_zone: 60
- brush_radius_mm: 25

## Parameter Sensitivity
- edge_sensitivity: Strong correlation (r=0.67)
- threshold_zone: Moderate correlation (r=0.34)
...
```

## Search Strategy

By default, uses random search over parameter space:

| Parameter | Range | Step |
|-----------|-------|------|
| edge_sensitivity | 10-90 | 10 |
| threshold_zone | 30-70 | 10 |
| brush_radius_mm | 10-40 | 5 |
| (algorithm-specific) | ... | ... |

## Tips

- Start with 20 trials to explore the space
- If promising region found, run more trials in that region
- Look for diminishing returns - when Dice improvement < 0.01
- Check the lab notebook for parameter sensitivity analysis
