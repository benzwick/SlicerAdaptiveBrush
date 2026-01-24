# run-slicer-tests

Run the Slicer-based test suite for AdaptiveBrush.

## Usage

```
/run-slicer-tests [suite]
```

Where `suite` is one of:
- `all` (default) - Run all tests
- `algorithm` - Run algorithm tests only
- `ui` - Run UI tests only
- `workflow` - Run workflow tests only

## What This Skill Does

1. Reads `SLICER_PATH` from `.env` file
2. Launches Slicer with the test runner script
3. Executes registered test cases
4. Captures screenshots and metrics
5. Leaves Slicer open with interactive testing panel

## Prerequisites

1. Copy `.env.example` to `.env` and set `SLICER_PATH`
2. Ensure the extension is installed in Slicer

## Execution Steps

### Step 1: Read Slicer Path from .env

```bash
source .env && echo "SLICER_PATH=$SLICER_PATH"
```

### Step 2: Launch Slicer with Test Runner

```bash
"$SLICER_PATH" --python-script scripts/run_tests.py $ARGUMENTS
```

Where `$ARGUMENTS` is the suite name (or "all" if not specified).

### Step 3: Review Output

After tests complete, Slicer stays open. The test output is saved to:
- `test_runs/<timestamp>_<suite>/results.json` - Test results
- `test_runs/<timestamp>_<suite>/screenshots/` - Captured screenshots
- `test_runs/<timestamp>_<suite>/logs/` - Test and Slicer logs

Use `/review-test-results` to analyze the output with the test-reviewer agent.

## Output Location

Test runs are saved to `test_runs/` in the extension directory. Each run creates a timestamped folder:

```
test_runs/2026-01-24_143025_all/
├── metadata.json
├── results.json
├── metrics.json
├── screenshots/
│   ├── manifest.json
│   └── *.png
└── logs/
    ├── test_run.log
    └── slicer_session.log
```

## Interactive Testing

After automated tests complete, Slicer stays open with the Adaptive Brush Tester module visible. You can:

1. Start recording manual actions
2. Take screenshots
3. Add notes
4. Mark pass/fail

Actions are logged to `manual_actions.jsonl` for later review.
