# Optimization Tutorial

This tutorial explains how to use the Optuna-based optimization framework to find
optimal parameters for Adaptive Brush algorithms.

## Overview

The optimization framework uses:
- **Optuna** with TPE sampler for intelligent parameter search
- **HyperbandPruner** for early stopping of poor trials
- **FAnova** for parameter importance analysis

## Prerequisites

1. **Slicer** installed with SlicerAdaptiveBrush extension
2. **Gold standard** segmentation for comparison
3. **Configuration file** defining the search space

## Quick Start

```bash
# Set Slicer path
export SLICER_PATH=/path/to/Slicer

# Run quick test (10 trials)
$SLICER_PATH --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/quick_test.yaml

# Run full optimization (50 trials)
$SLICER_PATH --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/tumor_optimization.yaml --trials 50
```

## Configuration Files

### Basic Structure

```yaml
version: "1.0"
name: "My Optimization"
description: "Find optimal parameters for tumor segmentation"

settings:
  n_trials: 50
  timeout_minutes: 60
  pruning: true
  pruner: "hyperband"
  sampler: "tpe"
  primary_metric: "dice"

recipes:
  - path: "recipes/brain_tumor_1.py"
    gold_standard: "GoldStandards/MRBrainTumor1_tumor/gold.seg.nrrd"

parameter_space:
  global:
    edge_sensitivity: {type: int, range: [20, 80], step: 10}
    threshold_zone: {type: int, range: [30, 70], step: 10}
    brush_radius_mm: {type: float, range: [15.0, 35.0]}

  algorithm_substitution:
    enabled: true
    candidates: ["watershed", "geodesic_distance", "connected_threshold"]

  algorithms:
    watershed:
      watershed_gradient_scale: {type: float, range: [0.5, 2.5]}
      watershed_smoothing: {type: float, range: [0.2, 1.0]}
```

### Available Configs

| Config | Trials | Purpose |
|--------|--------|---------|
| `quick_test.yaml` | 10 | Verify setup works |
| `default.yaml` | 50 | General optimization |
| `tumor_optimization.yaml` | 100 | Thorough tumor optimization |

## Running Optimization

### Command Line Options

```bash
$SLICER_PATH --python-script scripts/run_optimization.py CONFIG [OPTIONS]

Options:
  --trials N      Override number of trials
  --timeout M     Timeout in minutes
  --output DIR    Output directory
  --resume        Resume from previous study
  --no-exit       Keep Slicer open after completion
```

### Examples

```bash
# Quick test
$SLICER_PATH --python-script scripts/run_optimization.py \
    configs/quick_test.yaml

# Override trials
$SLICER_PATH --python-script scripts/run_optimization.py \
    configs/default.yaml --trials 100

# Resume interrupted study
$SLICER_PATH --python-script scripts/run_optimization.py \
    configs/tumor_optimization.yaml --resume

# Keep Slicer open for inspection
$SLICER_PATH --python-script scripts/run_optimization.py \
    configs/quick_test.yaml --no-exit
```

## Understanding Output

### Directory Structure

```
optimization_results/2026-01-26_062020_MRBrainTumor1_Tumor_Optimization/
├── config.yaml              # Copy of configuration
├── optuna_study.db          # SQLite database (resumable)
├── results.json             # Complete results
├── parameter_importance.json # FAnova importance scores
├── lab_notebook.md          # Human-readable summary
├── logs/
│   └── slicer_session.log   # Full Slicer log
├── screenshots/
│   ├── trial_000/
│   │   ├── 001.png          # Before painting
│   │   ├── 002.png          # After click 1
│   │   └── manifest.json
│   └── trial_001/
│       └── ...
└── segmentations/
    ├── trial_000.seg.nrrd
    └── trial_001.seg.nrrd
```

### Lab Notebook

The `lab_notebook.md` provides a human-readable summary:

```markdown
# MRBrainTumor1 Tumor Optimization - Results

## Summary
- **Best Dice:** 0.9991
- **Total Trials:** 50
- **Pruned Trials:** 38
- **Duration:** 226.2s

## Best Parameters
{
  "algorithm": "watershed",
  "edge_sensitivity": 70,
  "brush_radius_mm": 25.0
}

## Parameter Importance
| Parameter | Importance |
|-----------|------------|
| algorithm | 0.731 |
| brush_radius_mm | 0.186 |
```

### Parameter Importance

FAnova analysis reveals which parameters matter most:

```json
{
  "algorithm": 0.731,
  "brush_radius_mm": 0.186,
  "threshold_zone": 0.051,
  "edge_sensitivity": 0.032
}
```

**Interpretation:**
- `algorithm` (73%): By far the most important - choose carefully
- `brush_radius_mm` (19%): Second most important - tune to structure size
- Others (<10%): Less critical, default values usually work

## Pruning Behavior

HyperbandPruner stops trials early if intermediate Dice scores are poor:

```
Trial 5: Click 1/5, Dice=0.12 → PRUNED (too low)
Trial 6: Click 1/5, Dice=0.45 → Continue
Trial 6: Click 2/5, Dice=0.52 → Continue
Trial 6: Click 3/5, Dice=0.48 → PRUNED (not improving)
```

**Benefits:**
- ~4x speedup over full evaluation
- More trials in same time budget
- Focus computation on promising regions

## Reviewing Results

### In Slicer

1. Run optimization with `--no-exit`
2. Switch to **SegmentEditorAdaptiveBrushReviewer** module
3. Click **Load Optimization Results**
4. Select the output directory
5. Use trial selector to browse results

### Comparing Trials

The Reviewer shows:
- **Gold standard** (green)
- **Trial segmentation** (red)
- **Overlap** (yellow)
- **Dice coefficient**

### Saving Best Trial as Gold Standard

If a trial outperforms the current gold standard:

1. Select the best trial in Reviewer
2. Click **Save as Gold Standard**
3. Enter a new name (e.g., `MRBrainTumor1_tumor_v2`)

## Advanced Usage

### Custom Objective Functions

The default objective maximizes Dice coefficient. For custom objectives:

```python
def custom_objective(trial, params):
    # Run segmentation
    dice = compute_dice(...)
    hausdorff = compute_hausdorff(...)

    # Custom composite score
    return 0.7 * dice - 0.3 * hausdorff / 10.0
```

### Multi-Recipe Optimization

Optimize across multiple datasets:

```yaml
recipes:
  - path: "recipes/brain_tumor_1.py"
    gold_standard: "GoldStandards/MRBrainTumor1_tumor/gold.seg.nrrd"
    weight: 1.0
  - path: "recipes/brain_tumor_2.py"
    gold_standard: "GoldStandards/MRBrainTumor2_tumor/gold.seg.nrrd"
    weight: 1.0
```

### Algorithm-Specific Spaces

Define different search spaces per algorithm:

```yaml
algorithms:
  watershed:
    watershed_gradient_scale: {type: float, range: [0.5, 2.5]}
  level_set:
    level_set_iterations: {type: int, range: [30, 150], step: 20}
    level_set_propagation: {type: float, range: [0.5, 2.0]}
```

## Troubleshooting

### Common Issues

**"Optuna not available"**
```bash
# Install Optuna in Slicer's Python
$SLICER_PATH --python-script -c "import pip; pip.main(['install', 'optuna'])"
```

**Segmentation save errors**
- Ensure output directory has write permissions
- Check disk space

**All trials pruned**
- Lower the pruning threshold
- Check gold standard quality
- Verify click locations are correct

### Performance Tips

1. **Start with quick_test.yaml** to verify setup
2. **Use --no-exit** to inspect first few trials
3. **Resume interrupted runs** with --resume
4. **Focus on important parameters** based on FAnova results
