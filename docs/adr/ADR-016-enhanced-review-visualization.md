# ADR-016: Enhanced Review Visualization

## Status

Implemented

## Context

ADR-012 established the basic Results Review module for comparing trial segmentations against gold standards. However, during use, several limitations were identified:

1. **Contour Quality**: Morphological outline (dilation - erosion) produces jagged, pixelated boundaries that are difficult to distinguish at segment edges.

2. **Navigation**: Reviewing slice-by-slice requires manual interaction with Slicer's slice controls, breaking workflow.

3. **View Linking**: Users must manually synchronize navigation across multiple slice views.

4. **Rating System**: No structured way to record review decisions for audit trails.

5. **Keyboard Efficiency**: Mouse-heavy interaction slows down review of many trials.

### Research Findings

Review of existing Slicer extensions and NAMIC Project Week discussions revealed:

- **SegmentationReview** (MIT): Uses Likert-scale ratings, achieves 3x faster review
- **mpReview**: Linked views using View Groups for synchronized navigation
- **SlicerCaseIterator**: Ctrl+N/P keyboard shortcuts for case navigation
- **skimage.measure.find_contours**: Marching squares produces smooth sub-pixel contours

## Decision

Enhance the SegmentEditorAdaptiveBrushReviewer module with:

### 1. Dual Contour Rendering Modes (ContourRenderer)

Support both smooth and pixel-level contour visualization:

**Mode 1: Smooth Contours (marching squares)**
```python
from skimage import measure

def find_smooth_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Find smooth sub-pixel contours using marching squares."""
    binary = mask.astype(bool).astype(np.float64)
    contours = measure.find_contours(binary, level=0.5)
    return contours
```

**Mode 2: Pixel Outlines (morphological)**
```python
from scipy import ndimage

def get_morphological_outline(mask: np.ndarray) -> np.ndarray:
    """Get pixel-level outline using dilation - erosion."""
    dilated = ndimage.binary_dilation(mask, iterations=1)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    return dilated & ~eroded
```

**Comparison**:

| Mode | Best For | Characteristics |
|------|----------|-----------------|
| Smooth | Comparing boundary shape | Sub-pixel interpolated, smooth curves |
| Pixel | Seeing actual voxels | Shows exact pixel boundaries, jagged |

Both modes have value:
- **Smooth**: Better for assessing overall segmentation quality and boundary accuracy
- **Pixel**: Essential for understanding exactly what the algorithm segmented at voxel level

### 2. Linked Slice Navigation

Add slice navigation controls that synchronize all slice views:

```
Slice Navigation:
[|<] [<] [<<]  62/92  [>>] [>] [>|]
[─────────────●──────────────────]
☑ Link all views
```

**Implementation**:
- Slider maps to slice offset in all views
- Link checkbox enables/disables synchronized navigation
- View linking uses `sliceLogic.SetSliceOffset()` for each view

### 3. Keyboard Shortcuts (ReviewShortcutFilter)

Install event filter on main window for review-specific shortcuts:

| Key | Action |
|-----|--------|
| ← / → | Previous/Next slice |
| Ctrl+← / Ctrl+→ | Jump 10 slices |
| Home / End | First/Last slice |
| P / N | Previous/Next trial |
| 1-4 | Set rating (Accept/Minor/Major/Reject) |
| S | Save rating |
| Space | Toggle view mode |
| Ctrl+B | Add bookmark |
| Ctrl+R | Restore last bookmark |

**Implementation**: Qt QObject event filter pattern.

### 4. Comparison Metrics (ComparisonMetrics)

Real-time metric computation between loaded segmentations:

```
Comparison Metrics:
┌─────────────────┬─────────────────┐
│ Dice: 0.9876    │ Sensitivity: 0.9912
│ Hausdorff: 1.2mm│ Precision: 0.9840
│ Volume diff: +2%│ TP/FP/FN: 1234/12/8
└─────────────────┴─────────────────┘
[Compute]
```

**Metrics**:
- Dice coefficient (primary)
- Hausdorff distance (surface accuracy)
- Volume difference (percent)
- Sensitivity/Precision (clinical interpretation)
- TP/FP/FN counts (confusion matrix)

### 5. Rating System (RatingManager)

Structured review ratings with persistence and export:

```python
class Rating(IntEnum):
    UNRATED = 0
    ACCEPT = 1   # Ready for use
    MINOR = 2    # Small corrections needed
    MAJOR = 3    # Significant rework needed
    REJECT = 4   # Unsuitable for use

@dataclass
class ReviewRecord:
    trial_id: str
    run_name: str
    rating: Rating
    notes: str
    reviewer: str
    timestamp: str
    metrics: dict
```

**Features**:
- Session-based persistence (JSON)
- CSV export for analysis
- Automatic loading of existing ratings
- Notes field for reviewer comments

### Module Structure Update

```
SegmentEditorAdaptiveBrushReviewerLib/
├── __init__.py
├── ResultsLoader.py           # (existing)
├── VisualizationController.py # (existing)
├── ScreenshotViewer.py        # (existing)
├── ContourRenderer.py         # smooth contour visualization
├── ComparisonMetrics.py       # Dice, Hausdorff, etc.
├── RatingManager.py           # rating persistence
└── SequenceRecorder.py        # NEW: workflow recording, view groups, bookmarks
    ├── SequenceRecorder      # Workflow recording using Slicer Sequences
    ├── ViewGroupManager      # Native Slicer view linking
    └── SceneViewBookmarks    # Scene View bookmark management
```

### 6. Native View Linking (ViewGroupManager)

Replaced manual slice iteration with Slicer's native View Groups:

**Manual approach (removed):**
```python
# Old: Manual iteration over hardcoded views
for name in ["Red", "Yellow", "Green"]:
    slice_widget = layout_manager.sliceWidget(name)
    slice_logic.SetSliceOffset(offset)
```

**Native approach (implemented):**
```python
class ViewGroupManager:
    def enable_linking(self, view_group: int = 0):
        for slice_node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
            slice_node.SetViewGroup(view_group)
        for composite_node in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite_node.SetLinkedControl(True)
            composite_node.SetHotLinkedControl(True)
```

**Benefits:**
- Bidirectional sync (user drags in view → other views follow)
- Zoom/pan sync included automatically
- Layout-aware (works with any view configuration)
- Less code, more robust

### 7. Sequence Recording (SequenceRecorder)

Full workflow recording using Slicer Sequences for audit trails:

```python
class SequenceRecorder:
    def start_recording(self, segmentation_node, reference_volume):
        # Create browser for synchronized playback
        self._browser_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceBrowserNode"
        )

        # Sequences for: segmentation, all slice views, camera, notes
        self._sequences["segmentation"] = self._create_sequence(...)
        for slice_node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
            self._sequences[f"slice_{name}"] = self._create_sequence(...)

    def record_step(self, action_description: str):
        # Record all synchronized sequences at current index
        for seq in self._sequences.values():
            seq.SetDataNodeAtValue(proxy_node, index_value)
```

**What Sequences capture:**
- Segmentation state at each brush stroke
- All slice positions (Red/Yellow/Green)
- 3D camera position
- Text notes for reviewer annotations

### 8. Scene View Bookmarks (SceneViewBookmarks)

Quick bookmarks for interesting slices during review:

```python
class SceneViewBookmarks:
    def add_bookmark(self, description: str, name: str | None = None) -> int:
        scene_view = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSceneViewNode")
        scene_view.SetSceneViewDescription(description)
        scene_view.StoreScene()
        return index

    def restore_bookmark(self, index: int) -> bool:
        self._bookmarks[index].RestoreScene()
```

**What Scene Views capture:**
- All slice positions
- Camera positions (3D)
- Node visibility
- Display properties
- Window/level

## Consequences

### Positive

- **Faster Review**: Keyboard shortcuts enable rapid trial navigation
- **Clearer Visualization**: Smooth contours show true boundaries
- **Audit Trail**: Rating records provide documentation for validation
- **Synchronized Navigation**: Linked views reduce cognitive load (native bidirectional sync)
- **Quantitative Context**: Real-time metrics inform rating decisions
- **Workflow Recording**: Full segmentation workflow captured for playback
- **Bookmarking**: Save and restore interesting view states
- **Bidirectional Sync**: Slider updates when user navigates in slice views

### Negative

- **Additional Dependencies**: Requires skimage for contour extraction
- **Keyboard Conflicts**: Shortcuts may conflict with other modules (mitigated by focus-based activation)
- **Memory Usage**: Caching contours and sequence recordings uses additional RAM
- **Sequence Overhead**: Recording creates multiple MRML nodes

### Trade-offs

| Aspect | Simple Approach | Chosen Approach | Reason |
|--------|-----------------|-----------------|--------|
| Contours | Morphological | Marching squares | Sub-pixel accuracy |
| Navigation | Mouse only | Keyboard + Mouse | Speed for bulk review |
| Ratings | External spreadsheet | Integrated | Same environment |
| Metrics | Post-hoc | Real-time | Immediate feedback |
| View Sync | Manual iteration | Native View Groups | Bidirectional, robust |
| Workflow Recording | Manual screenshots | Sequences | Full state capture |
| Bookmarks | External notes | Scene Views | In-slicer restore |

## Alternatives Considered

### Web-based Review (OHIF)

**Rejected**: Requires additional infrastructure. Users already have Slicer open.

### External Rating CSV

**Rejected**: Breaks workflow. Integrated rating is faster.

### VTK Contour Extraction

**Rejected**: More complex than skimage. Less Python-friendly API.

## Implementation Notes

### Screenshot Output Structure

The review screenshot generator creates three visualization directories:

```
review_output/
├── compare_smooth/     # Marching squares contours (sub-pixel)
│   ├── 0045.png
│   ├── 0046.png
│   └── ...
├── compare_pixel/      # Morphological outlines (actual pixels)
│   ├── 0045.png
│   ├── 0046.png
│   └── ...
├── error/              # TP/FP/FN region coloring
│   ├── 0045.png
│   ├── 0046.png
│   └── ...
└── manifest.json       # Metadata and slice index
```

### Event Filter Pattern

```python
class ReviewShortcutFilter(qt.QObject):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def eventFilter(self, obj, event):
        if event.type() != qt.QEvent.KeyPress:
            return False
        # Handle key events
        key = event.key()
        if key == qt.Qt.Key_Left:
            self.widget._on_prev_slice()
            return True
        return False
```

### Session Persistence

Ratings are stored per-session in `reviews/sessions/`:

```
reviews/
├── sessions/
│   └── 20260127_143025_tumor_opt.json
└── exports/
    └── review_ratings.csv
```

### Cleanup

Event filter is removed on module cleanup to prevent memory leaks:

```python
def cleanup(self):
    if self._shortcut_filter:
        slicer.util.mainWindow().removeEventFilter(self._shortcut_filter)
        self._shortcut_filter = None
```

## References

- [ADR-012](ADR-012-results-review-module.md): Results Review Module
- [SegmentationReview Extension](https://github.com/zapaishchykova/SegmentationReview)
- [skimage.measure.find_contours](https://scikit-image.org/docs/stable/api/skimage.measure.html#find-contours)
- [Slicer View Linking](https://discourse.slicer.org/t/how-can-i-make-all-6-views-link-together/14874)
- [NAMIC PW42 - Segmentation Review](https://projectweek.na-mic.org/PW42_2025_GranCanaria/Projects/ReviewOfSegmentationResultsQualityAcrossVariousMultiOrganSegmentationModels/)
