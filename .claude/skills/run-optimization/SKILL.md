# run-optimization

Run Optuna parameter optimization for Adaptive Brush segmentation algorithms.

## Usage

```
/run-optimization [config.yaml] [--trials N] [--no-exit]
```

Where:
- `config.yaml` - Path to YAML config (default: configs/quick_test.yaml)
- `--trials N` - Override number of trials
- `--no-exit` - Keep Slicer open after completion for inspection

## What This Skill Does

1. Loads Slicer with the optimization script
2. Loads sample data and gold standard segmentation
3. Runs N trials with Optuna TPE sampler and HyperbandPruner
4. For each trial:
   - Applies suggested parameters to the effect
   - Paints all clicks from the gold standard
   - Computes Dice coefficient after each click (for pruning)
   - Reports final Dice as objective value
5. Saves results and generates lab notebook

## Prerequisites

1. Slicer path set in `.env` file: `SLICER_PATH=/path/to/Slicer`
2. Gold standard exists in `SegmentEditorAdaptiveBrushTester/GoldStandards/`
3. Config YAML exists in `SegmentEditorAdaptiveBrushTester/configs/`

## Execution Steps

### Step 1: Read .env and Launch Slicer

```bash
source .env && "$SLICER_PATH" --python-script scripts/run_optimization.py configs/quick_test.yaml
```

### Step 2: Monitor Progress

The script outputs progress to terminal:

```
14:30:01 [INFO] optimization: Loaded config: Quick Test
14:30:05 [INFO] optimization: Loaded gold standard: MRBrainTumor1_tumor with 5 clicks
14:30:06 [INFO] optimization: Starting optimization: 10 trials
14:30:08 [INFO] optimization: Trial 0: {'algorithm': 'geodesic_distance', 'edge_sensitivity': 50, ...}
14:30:08 [INFO] optimization:   Click 1/5: Dice=0.3421
14:30:09 [INFO] optimization:   Click 2/5: Dice=0.5234
...
14:30:12 [INFO] optimization:   Final Dice: 0.8734 (4200ms)
```

### Step 3: Review Results

Results are saved to:

```
optimization_results/<timestamp>_<config_name>/
├── config.yaml              # Copy of config used
├── optuna_study.db          # SQLite database (resumable)
├── results.json             # Full results with all trials
├── parameter_importance.json # FAnova importance scores
└── lab_notebook.md          # Human-readable summary
```

## Available Configs

| Config | Trials | Purpose |
|--------|--------|---------|
| `quick_test.yaml` | 10 | Fast testing (random sampler) |
| `default.yaml` | 50 | General optimization |
| `tumor_optimization.yaml` | 100 | Thorough tumor optimization |

## Output Format

### results.json

```json
{
  "config_name": "Quick Test",
  "n_trials": 10,
  "best_trial": {
    "trial_number": 7,
    "params": {
      "algorithm": "geodesic_distance",
      "edge_sensitivity": 50,
      "threshold_zone": 50
    },
    "value": 0.8734,
    "duration_ms": 4200
  },
  "parameter_importance": {
    "algorithm": 0.45,
    "edge_sensitivity": 0.30,
    "threshold_zone": 0.15
  }
}
```

### lab_notebook.md

Human-readable markdown with:
- Summary statistics
- Best parameters
- Parameter importance table
- Top 5 trials
- Complete trial history

## Examples

### Quick test run
```bash
source .env && "$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/quick_test.yaml
```

### Full optimization with 100 trials
```bash
source .env && "$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/tumor_optimization.yaml
```

### Override trials and keep Slicer open
```bash
source .env && "$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/default.yaml --trials 20 --no-exit
```

### Resume previous study
```bash
source .env && "$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/tumor_optimization.yaml --resume
```

## Tips

- Start with `quick_test.yaml` to verify setup
- Use `--no-exit` to inspect results visually in Slicer
- The SQLite database allows resuming interrupted runs
- Parameter importance helps focus future optimization
- Look for diminishing returns in trial history
