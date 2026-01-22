# Skill: Sync with Slicer Main Branch

## When to Use
- Before major releases
- When Slicer issues affect our extension
- Periodically to stay current with Slicer changes

## How to Check for Changes

### 1. Check Recent Commits to Effect Classes
```bash
gh api "repos/Slicer/Slicer/commits?path=Modules/Loadable/Segmentations/EditorEffects/Python/SegmentEditorEffects&per_page=10" \
  --jq '.[] | {sha: .sha[0:7], date: .commit.author.date, message: .commit.message | split("\n")[0]}'
```

### 2. Check Open PRs Related to Segmentation
```bash
gh api "repos/Slicer/Slicer/pulls?state=open" \
  --jq '.[] | select(.title | test("segment|effect"; "i")) | {number, title}'
```

### 3. Check Issues with Segmentation Label
```bash
gh api "repos/Slicer/Slicer/issues?labels=segmentation&state=open" \
  --jq '.[] | {number, title}'
```

### 4. Compare Base Class Version
Fetch current AbstractScriptedSegmentEditorEffect.py:
```bash
gh api "repos/Slicer/Slicer/contents/Modules/Loadable/Segmentations/EditorEffects/Python/SegmentEditorEffects/AbstractScriptedSegmentEditorEffect.py" \
  --jq '.content' | base64 -d
```

### 5. Check Recent Releases
```bash
gh api "repos/Slicer/Slicer/releases?per_page=5" \
  --jq '.[] | {tag: .tag_name, date: .published_at, name: .name}'
```

## Key Files to Monitor

| File | Purpose |
|------|---------|
| `AbstractScriptedSegmentEditorEffect.py` | Base class for Python effects |
| `SegmentEditorThresholdEffect.py` | Reference implementation |
| `qSlicerSegmentEditorScriptedEffect.h` | C++ interface definition |
| `SegmentEditorEffects/__init__.py` | Module initialization |

## Key Methods to Check

Our effect must implement these methods from the base class:

| Method | Purpose |
|--------|---------|
| `__init__(scriptedEffect)` | Initialize effect, set name |
| `clone()` | Create new instance for registration |
| `icon()` | Return QIcon for toolbar |
| `helpText()` | Return HTML help string |
| `setupOptionsFrame()` | Create UI widgets |
| `activate()` | Called when effect selected |
| `deactivate()` | Called when effect deselected |
| `cleanup()` | Clean up resources (added Jan 2025) |
| `processInteractionEvents()` | Handle mouse/keyboard events |

## Recent Breaking Changes

### January 2025: cleanup() Method Added
Commit [64cfcfc](https://github.com/Slicer/Slicer/commit/64cfcfc)

Effects MUST implement `cleanup()` to prevent memory leaks:
```python
def cleanup(self):
    """Clean up resources to prevent memory leaks."""
    # Disconnect all signal/slot connections
    for widget in self.widgets_with_connections:
        widget.disconnect()

    # Clean up VTK pipelines
    # ...

    # Call parent cleanup
    AbstractScriptedSegmentEditorEffect.cleanup(self)
```

## API Documentation
- [Slicer API Docs](https://apidocs.slicer.org/master/)
- [Segment Editor Guide](https://slicer.readthedocs.io/en/latest/developer_guide/modules/segmenteditor.html)
- [Slicer Developer Guide](https://slicer.readthedocs.io/en/latest/developer_guide/)

## Reference Extensions
- [SlicerSegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects) - Maintained by Andras Lasso
