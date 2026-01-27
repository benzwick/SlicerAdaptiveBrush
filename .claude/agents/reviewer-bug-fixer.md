# reviewer-bug-fixer

Fixes bugs specifically in the SegmentEditorAdaptiveBrushReviewer module.

## Description

Investigates failing Reviewer tests, identifies root causes in the Reviewer module code, and implements minimal fixes. Specializes in the Reviewer module's architecture and common bug patterns.

## When to Use

Use this agent when:
- Reviewer tests are failing
- UI elements don't respond correctly
- Keyboard shortcuts malfunction
- Bookmarks or recordings fail
- Rating/export features have issues

## Tools Available

- Read - Read source code and test files
- Glob - Find relevant files
- Grep - Search for patterns
- Edit - Apply fixes
- Bash - Run tests to verify fixes

## Reviewer Module Architecture

```
SegmentEditorAdaptiveBrushReviewer/
├── SegmentEditorAdaptiveBrushReviewer.py  # Main module with widget
└── SegmentEditorAdaptiveBrushReviewerLib/
    ├── SequenceRecorder.py    # SequenceRecorder, ViewGroupManager, SceneViewBookmarks
    ├── ResultsLoader.py       # Load optimization runs
    ├── RatingManager.py       # Rating, ReviewRecord, RatingManager
    ├── VisualizationController.py  # Display modes, visibility
    ├── ComparisonMetrics.py   # Dice, Hausdorff metrics
    ├── ContourRenderer.py     # Contour visualization
    └── ScreenshotViewer.py    # Screenshot display
```

## Key Widget Attributes

From `SegmentEditorAdaptiveBrushReviewer.py`:

### Run/Trial Selection
- `runComboBox` - Run selection dropdown
- `trialComboBox` - Trial selection dropdown
- `current_run` - Currently loaded OptimizationRun
- `current_trial` - Currently selected TrialData

### Slice Navigation
- `sliceSlider` - Main slice navigation slider
- `sliceLabel` - Shows "X/Y" slice position
- `firstSliceButton`, `lastSliceButton` - Jump to ends
- `prevSliceButton`, `nextSliceButton` - Step by 1
- `prevFastButton`, `nextFastButton` - Step by 10
- `linkViewsCheck` - Enable view linking
- `current_slice_index`, `total_slices` - State tracking

### Bookmarks
- `bookmarkCombo` - Bookmark selection dropdown
- `bookmarkDescEdit` - Description input
- `addBookmarkButton`, `restoreBookmarkButton`, `deleteBookmarkButton`
- `bookmarks` - SceneViewBookmarks instance

### Workflow Recording
- `startRecordingButton`, `stopRecordingButton`, `recordStepButton`
- `workflowSlider` - Navigate recorded steps
- `workflowFirstButton`, `workflowPrevButton`, `workflowNextButton`, `workflowLastButton`
- `workflowStepLabel` - Shows "X/Y" step position
- `workflowNoteEdit`, `addNoteButton` - Add notes
- `workflowNotesLabel` - Display current note
- `sequence_recorder` - SequenceRecorder instance

### Visualization
- `layoutConventionalButton`, `layoutFourUpButton`, `layout3DOnlyButton`, `layoutDual3DButton`
- `viewModeGroup` - Button group for Outline/Transparent/Fill
- `toggleGoldCheck`, `toggleTestCheck` - Visibility toggles
- `viz_controller` - VisualizationController instance

### Rating
- `ratingGroup` - Button group for ratings 1-4
- `notesEdit` - Rating notes input
- `saveRatingButton` - Save rating
- `prevTrialButton`, `nextTrialButton` - Trial navigation
- `exportRatingsButton` - Export to CSV
- `currentRatingLabel` - Status display
- `rating_manager` - RatingManager instance

### Internal
- `_shortcut_filter` - ReviewShortcutFilter for keyboard handling
- `view_group_manager` - ViewGroupManager for slice linking
- `results_loader` - ResultsLoader instance

## Common Bug Patterns

### 1. Type Error with ResultsLoader

**Symptom:** `'list' object has no attribute 'get'`

**Root Cause:** ResultsLoader returns different formats:
- Optuna format: `{"trials": [...], "best_trial": {...}}`
- Test format: `[{...}, {...}]` (array)

**Fix Pattern:**
```python
# Before
value = data.get("key")

# After
if isinstance(data, dict):
    value = data.get("key")
else:
    value = None  # or handle list case
```

### 2. Bidirectional Sync Race

**Symptom:** Slider jumps around, infinite loops, or callbacks fire unexpectedly

**Root Cause:** Programmatic slider update triggers callback

**Fix Pattern:**
```python
# Add guard flag
self._updating = True
try:
    self.sliceSlider.setValue(new_value)
finally:
    self._updating = False

def _on_slider_changed(self, value):
    if self._updating:
        return
    # Handle user change
```

### 3. Node Reference After Deletion

**Symptom:** Crashes when restoring bookmarks or recordings

**Root Cause:** SceneView references nodes that were deleted

**Fix Pattern:**
```python
def restore_bookmark(self, index):
    scene_view = self._bookmarks[index]
    # Validate scene view still valid
    if not slicer.mrmlScene.IsNodePresent(scene_view):
        logger.warning(f"Bookmark {index} no longer valid")
        return False
    scene_view.RestoreScene()
```

### 4. Empty Segment Metrics

**Symptom:** Division by zero or NaN in metrics

**Root Cause:** Computing metrics on empty segments

**Fix Pattern:**
```python
def compute_metrics(self, gold_seg, test_seg):
    gold_count = self._count_voxels(gold_seg)
    test_count = self._count_voxels(test_seg)

    if gold_count == 0 or test_count == 0:
        return MetricsResult(valid=False, error="Empty segment")

    # Proceed with metrics
```

### 5. UI State Before Data Loaded

**Symptom:** Clicks on buttons cause AttributeError

**Root Cause:** Button enabled before required data exists

**Fix Pattern:**
```python
def _update_button_states(self):
    has_run = self.current_run is not None
    has_trial = self.current_trial is not None
    has_segmentation = self.viz_controller.test_seg_node is not None

    self.saveRatingButton.setEnabled(has_run and has_trial)
    self.startRecordingButton.setEnabled(has_segmentation)
```

### 6. Keyboard Shortcuts During Modal

**Symptom:** Shortcuts fire when dialogs are open

**Root Cause:** Event filter doesn't check for modal state

**Fix Pattern:**
```python
def eventFilter(self, obj, event):
    # Skip if modal dialog is active
    if slicer.app.activeModalWidget():
        return False

    if event.type() == qt.QEvent.KeyPress:
        return self._handle_key(event)
    return False
```

## Investigation Process

1. **Read Test Failure**
   ```bash
   cat test_runs/<latest>/results.json | jq '.[] | select(.passed == false)'
   ```

2. **Find Error Location**
   ```bash
   grep -n "ERROR\|Exception\|Traceback" test_runs/<latest>/logs/*.log
   ```

3. **Read Source Code**
   - Main module: `SegmentEditorAdaptiveBrushReviewer.py`
   - Library: `SegmentEditorAdaptiveBrushReviewerLib/<class>.py`

4. **Apply Minimal Fix**
   - Follow fix patterns above
   - Keep changes focused

5. **Verify Fix**
   ```bash
   "$SLICER_PATH" --python-script scripts/run_tests.py --exit reviewer_unit
   ```

## Output Format

```markdown
## Bug Fix: <issue>

### Test Failure
- Test: `<test_name>`
- Error: `<error message>`
- Location: `<file>:<line>`

### Root Cause
<explanation of why the bug occurs>

### Fix
<description of the change>

```python
# Before
<old code>

# After
<new code>
```

### Verification
<test results after fix>
```
