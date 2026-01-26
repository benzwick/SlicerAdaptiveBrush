# optimize-workflow

End-to-end workflow for optimizing Adaptive Brush parameters and updating gold standards.

## Usage

```
/optimize-workflow [config.yaml] [--review-only] [--save-gold]
```

Options:
- `config.yaml` - YAML config (default: configs/quick_test.yaml)
- `--review-only` - Skip optimization, review most recent results
- `--save-gold` - Automatically save best trial as gold standard

## What This Workflow Does

This skill orchestrates the complete optimization → review → gold standard workflow:

1. **Optimize**: Run Optuna optimization with screenshots and click recording
2. **Review**: Analyze results, view trial screenshots, compare to gold standard
3. **Save**: Optionally promote best trial segmentation to gold standard

## Complete Workflow

### Phase 1: Run Optimization

```bash
source .env && "$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/quick_test.yaml --no-exit
```

Outputs saved to `optimization_results/<timestamp>/`:
- `results.json` - All trial results
- `lab_notebook.md` - Human-readable summary
- `screenshots/trial_XXX/` - Screenshots per trial (before painting, after each click)
- `segmentations/trial_XXX.seg.nrrd` - Segmentation from each trial

### Phase 2: Review Results

1. **Read lab notebook** for quick overview:
   ```bash
   ls -td optimization_results/*/ | head -1 | xargs -I{} cat {}lab_notebook.md
   ```

2. **Read detailed results**:
   ```bash
   ls -td optimization_results/*/ | head -1 | xargs -I{} cat {}results.json | python -m json.tool
   ```

3. **View best trial screenshots** using Read tool on the PNG files

4. **Analyze parameter importance**:
   ```bash
   ls -td optimization_results/*/ | head -1 | xargs -I{} cat {}parameter_importance.json
   ```

### Phase 3: Visual Review in Slicer

If optimization was run with `--no-exit`, Slicer is still open. Use the Reviewer module:

1. Switch to `SegmentEditorAdaptiveBrushReviewer` module
2. Click "Load Optimization Results"
3. Select the optimization folder
4. Use trial selector to navigate between trials
5. Compare test segmentation (red) vs gold standard (green)
6. Review screenshots in the thumbnail viewer

### Phase 4: Save as Gold Standard

If a trial achieved good results, promote it to gold standard:

**Option A: Via Reviewer UI**
1. In Reviewer module, select the best trial
2. Click "Save as Gold Standard"
3. Enter name (e.g., "MRBrainTumor1_improved")
4. Click Save

**Option B: Via Python console**
```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()

# Load optimization results
import json
from pathlib import Path

results_dir = Path("optimization_results")
latest = sorted(results_dir.iterdir())[-1]

with open(latest / "results.json") as f:
    results = json.load(f)

best_trial = results["best_trial"]
trial_num = best_trial["trial_number"]

# Load the trial segmentation
seg_path = latest / f"segmentations/trial_{trial_num:03d}.seg.nrrd"
seg_node = slicer.util.loadSegmentation(str(seg_path))

# Get volume node
vol_node = slicer.util.getNode(results["sample_data"])

# Save as gold standard
manager.save_as_gold(
    segmentation_node=seg_node,
    volume_node=vol_node,
    segment_id=seg_node.GetSegmentation().GetNthSegmentID(0),
    name=f"{results['gold_standard']}_optimized",
    click_locations=best_trial.get("user_attrs", {}).get("click_locations", []),
    description=f"Optimized with Dice={best_trial['value']:.4f}",
    algorithm=best_trial["params"].get("algorithm", "unknown"),
    parameters=best_trial["params"],
)
```

## Results Analysis

### Key Metrics to Check

| Metric | Good | Action if Poor |
|--------|------|----------------|
| Best Dice | > 0.85 | Increase trials, adjust parameter space |
| Pruned trials | 30-70% | Working well; < 30% = pruner too lenient |
| Parameter importance | algorithm > 0.3 | Algorithm selection is key |
| Convergence | Stable after trial 20 | Can reduce trial count |

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| All trials pruned | Lower pruner threshold or check gold standard |
| Best Dice < 0.5 | Check click locations, increase brush radius |
| High variance | Narrow parameter ranges, more trials |
| Slow convergence | Use TPE sampler (default), not random |

## Integration with Other Skills

This workflow integrates:

- `/run-optimization` - Phase 1 execution
- `/review-test-results` - Can analyze optimization like test results
- `/create-gold-standard` - Phase 4 manual approach
- `/run-regression` - Verify new gold standard reproduces

## Full Automation Example

For a fully automated workflow:

```bash
# 1. Run optimization
source .env && "$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/tumor_optimization.yaml

# 2. Find and review results
LATEST=$(ls -td optimization_results/*/ | head -1)
echo "Results: $LATEST"
cat "${LATEST}lab_notebook.md"

# 3. Check if Dice is acceptable (> 0.9)
DICE=$(cat "${LATEST}results.json" | python3 -c "import json,sys; print(json.load(sys.stdin)['best_trial']['value'])")
echo "Best Dice: $DICE"

# 4. If good, can be saved as gold standard via Slicer
```

## Output Locations

```
optimization_results/
└── 2026-01-26_143025_Quick_Test/
    ├── config.yaml              # Copy of config
    ├── optuna_study.db          # SQLite (resumable)
    ├── results.json             # All trials
    ├── parameter_importance.json
    ├── lab_notebook.md          # Summary
    ├── screenshots/
    │   ├── trial_000/
    │   │   ├── 001_before_painting.png
    │   │   ├── 002_after_click_1_dice_0.3421.png
    │   │   └── ...
    │   └── trial_001/
    │       └── ...
    └── segmentations/
        ├── trial_000.seg.nrrd
        ├── trial_001.seg.nrrd
        └── ...
```

## Tips

- Start with `quick_test.yaml` to verify the workflow
- Use `--no-exit` to visually inspect results in Slicer
- Compare screenshots between best and worst trials to understand parameters
- Run regression tests after saving new gold standard
- Parameter importance guides future optimization focus
