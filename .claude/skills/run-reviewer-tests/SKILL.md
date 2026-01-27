# run-reviewer-tests

Run UI tests specifically for the SegmentEditorAdaptiveBrushReviewer module.

## Usage

```
/run-reviewer-tests [--exit] [suite]
```

Where:
- `--exit` - Exit Slicer after tests complete (for automated iteration)
- `suite` is one of:
  - `reviewer` (default) - Run all reviewer tests
  - `reviewer_unit` - Run unit tests only (SequenceRecorder, ViewGroupManager, SceneViewBookmarks)
  - `reviewer_ui` - Run UI tests only (slice navigation, bookmarks, playback, visualization, rating, keyboard)
  - `reviewer_integration` - Run integration tests only (full workflow)

## What This Skill Does

1. Reads `SLICER_PATH` from `.env` file
2. Launches Slicer with the test runner script
3. Executes Reviewer test cases (10 tests total)
4. Captures screenshots at each UI interaction
5. Reports pass/fail with detailed assertions

## Test Categories

### Unit Tests (`reviewer_unit`) - 3 tests
| Test | Description |
|------|-------------|
| `unit_sequence_recorder` | SequenceRecorder start/stop/step/navigation |
| `unit_view_group_manager` | ViewGroupManager linking, slice offsets |
| `unit_scene_view_bookmarks` | SceneViewBookmarks add/restore/delete |

### UI Tests (`reviewer_ui`) - 6 tests
| Test | Description |
|------|-------------|
| `ui_reviewer_slice_navigation` | Slider, nav buttons, boundaries, linking |
| `ui_reviewer_bookmarks` | Add/restore/delete buttons, combo box |
| `ui_reviewer_workflow_playback` | Recording start/stop, playback navigation |
| `ui_reviewer_visualization` | Layout buttons, view modes, toggles |
| `ui_reviewer_rating` | Rating buttons 1-4, notes, save, trial nav |
| `ui_reviewer_keyboard` | All keyboard shortcuts |

### Integration Tests (`reviewer_integration`) - 1 test
| Test | Description |
|------|-------------|
| `integration_reviewer_workflow` | Full 9-phase workflow test |

## Execution Steps

### Step 1: Read Slicer Path from .env

```bash
grep SLICER_PATH .env | cut -d= -f2
```

### Step 2: Launch Slicer with Test Runner

For automated iteration (faster):
```bash
"$SLICER_PATH" --python-script scripts/run_tests.py --exit reviewer_unit
```

For interactive follow-up:
```bash
"$SLICER_PATH" --python-script scripts/run_tests.py reviewer
```

### Step 3: Check Results

Read the output folder for results:
```
test_runs/<timestamp>_reviewer/
├── results.json          # Pass/fail for each test
├── metadata.json         # Summary statistics
├── screenshots/
│   ├── manifest.json     # Screenshot descriptions
│   └── *.png             # Captured UI states
└── logs/
    ├── test_run.log      # Test execution log
    └── slicer_session.log
```

### Step 4: Analyze Failures

For each failed test:
1. Read the error message in `results.json`
2. Check screenshots for visual context
3. Search logs for tracebacks

## Expected Bugs

The plan identified these likely bugs to find:

1. **`'list' object has no attribute 'get'`** in `_on_run_selected` (line ~1509)
   - Cause: ResultsLoader returns list for test format runs
   - Fix: Add type check before `.get()` call

2. **Bidirectional sync race conditions**
   - Cause: Slider updates triggering callbacks during programmatic changes
   - Fix: Use guard flag or disconnect signals during update

3. **Bookmark restoration with changed scene**
   - Cause: SceneView references deleted nodes
   - Fix: Validate nodes exist before restore

4. **Metrics with empty segments**
   - Cause: Division by zero or empty labelmap
   - Fix: Check segment voxel count > 0

5. **Recording without loaded segmentation**
   - Cause: Start recording clicked before loading data
   - Fix: Disable button until segmentation loaded

6. **Keyboard shortcuts when dialogs open**
   - Cause: Shortcuts fire even during modal dialogs
   - Fix: Check for modal state in event filter

## Recommended Workflow

1. **Start with unit tests** (fastest feedback):
   ```
   /run-reviewer-tests --exit reviewer_unit
   ```

2. **Fix any unit test failures** before moving to UI tests

3. **Run UI tests**:
   ```
   /run-reviewer-tests --exit reviewer_ui
   ```

4. **Review screenshots** for UI issues

5. **Run full integration**:
   ```
   /run-reviewer-tests --exit reviewer_integration
   ```

## Follow-up Skills

After running tests:
- `/review-test-results` - Analyze failures and screenshots
- Use `reviewer-bug-fixer` agent - Fix identified bugs
- Use `ui-improver` agent - Address UI issues from screenshots
