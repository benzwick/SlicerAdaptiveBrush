# run-slicer-tests

Run the Slicer-based test suite for AdaptiveBrush.

## Usage

```
/run-slicer-tests [--exit] [suite]
```

Where:
- `--exit` - Exit Slicer after tests complete (for automated iteration)
- `suite` is one of:
  - `all` (default) - Run all tests
  - `algorithm` - Run algorithm tests only
  - `ui` - Run UI tests only
  - `workflow` - Run workflow tests only

## What This Skill Does

1. Reads `SLICER_PATH` from `.env` file
2. Launches Slicer with the test runner script
3. Executes registered test cases
4. Captures screenshots and metrics
5. Either exits (with `--exit`) or leaves Slicer open for manual testing

## Prerequisites

1. Copy `.env.example` to `.env` and set `SLICER_PATH`
2. Ensure the extension is installed in Slicer (symlinked to Slicer's module path)

## Execution Steps

### Step 1: Read Slicer Path from .env

Use the Read tool to read the `.env` file and extract the SLICER_PATH value.

### Step 2: Launch Slicer with Test Runner

Use Bash to run Slicer with the path from .env:

For automated runs (exits when done):

```bash
<SLICER_PATH> --python-script scripts/run_tests.py --exit all
```

For interactive follow-up (stays open):

```bash
<SLICER_PATH> --python-script scripts/run_tests.py all
```

Replace `<SLICER_PATH>` with the actual path from the .env file.

### Step 3: Check Exit Code

When using `--exit`, the return code indicates:
- `0` - All tests passed
- `1` - One or more tests failed

### Step 4: Review Output

Test output is saved to:
- `test_runs/<timestamp>_<suite>/results.json` - Test results
- `test_runs/<timestamp>_<suite>/metrics.json` - Performance metrics
- `test_runs/<timestamp>_<suite>/screenshots/` - Screenshots per test
- `test_runs/<timestamp>_<suite>/logs/test_run.log` - Test execution log
- `test_runs/<timestamp>_<suite>/logs/slicer_session.log` - Slicer application log

Use `/review-test-results` to analyze the output (errors, warnings, failures, performance).

## Output Location

Test runs are saved to `test_runs/` in the extension directory. Each run creates a timestamped folder:

```
test_runs/2026-01-24_143025_all/
├── metadata.json
├── results.json
├── metrics.json
├── screenshots/
│   ├── manifest.json
│   ├── workflow_basic/
│   │   ├── 001.png
│   │   ├── 002.png
│   │   └── ...
│   ├── algorithm_watershed/
│   │   ├── 001.png
│   │   └── ...
│   └── ui_options_panel/
│       └── ...
└── logs/
    ├── test_run.log
    └── slicer_session.log
```

## Interactive Testing (without --exit)

After automated tests complete, Slicer stays open with the Adaptive Brush Tester module visible. You can:

1. Start recording manual actions
2. Create new screenshot groups
3. Take screenshots (auto-numbered within groups)
4. Add notes
5. Mark pass/fail

Actions are logged to `manual_actions.jsonl` for later review.

## Automated Iteration (with --exit)

Use `--exit` when iterating on code:

1. Run tests: `/run-slicer-tests --exit`
2. Review results: Read `test_runs/.../results.json`
3. Fix issues in code
4. Run tests again

This is faster than staying open for manual testing.
