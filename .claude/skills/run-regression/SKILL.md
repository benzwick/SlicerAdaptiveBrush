# run-regression

Run regression tests against gold standard segmentations.

## Usage

```
/run-regression [--gold <name>] [--all]
```

Where:
- `--gold <name>` - Test a specific gold standard
- `--all` - Test all gold standards (default)

## What This Skill Does

1. Loads gold standard(s) from GoldStandards/
2. Runs segmentation with recorded parameters
3. Computes Dice/Hausdorff vs gold
4. Flags regressions if metrics fall below thresholds
5. Generates regression report

## Regression Thresholds

Default thresholds (can be configured):
- **Dice coefficient:** >= 0.80
- **Hausdorff 95%:** <= 10.0mm

Regressions are flagged when:
- Dice drops below threshold
- Hausdorff exceeds threshold

## Execution Steps

### Step 1: Read .env file

Get SLICER_PATH from .env

### Step 2: Launch Regression Tests

```bash
<SLICER_PATH> --python-script scripts/run_tests.py --exit regression
```

Or for interactive mode (stays open):

```bash
<SLICER_PATH> --python-script scripts/run_tests.py regression
```

### Step 3: Check Results

Results are saved to:

```
test_runs/<timestamp>_regression/
├── results.json        # Pass/fail per gold standard
├── metrics.json        # Detailed metrics
└── screenshots/        # Visual comparison
```

## Output Format

### results.json

```json
{
  "summary": {
    "total_tests": 2,
    "passed": 1,
    "failed": 1
  },
  "tests": [
    {
      "name": "regression_gold",
      "gold_standards": [
        {
          "name": "MRBrainTumor1_tumor",
          "dice": 0.89,
          "hausdorff_95": 4.2,
          "passed": true
        },
        {
          "name": "MRHead_ventricle",
          "dice": 0.75,
          "hausdorff_95": 12.3,
          "passed": false,
          "regression": true
        }
      ]
    }
  ]
}
```

## When to Run

- **Before commits:** Ensure no regressions
- **After algorithm changes:** Verify improvements don't break existing cases
- **After parameter tuning:** Confirm tuning is effective
- **CI/CD pipeline:** Automated regression detection

## Interpreting Results

### PASS

```
MRBrainTumor1_tumor:
  Dice: 0.89
  Hausdorff 95%: 4.2mm
  PASS
```

The algorithm reproduces the gold standard within thresholds.

### REGRESSION

```
MRHead_ventricle:
  Dice: 0.75
  Hausdorff 95%: 12.3mm
  ** REGRESSION **
```

Investigate:
1. Was an algorithm changed recently?
2. Were parameters modified?
3. Is the gold standard still appropriate?

## Tips

- Run regression tests after any algorithm change
- If a regression is valid (gold standard was wrong), update the gold standard
- Keep track of why regressions occur in git commit messages
- Consider adding more gold standards for different tissue types
