# ADR-012: Results Review Module

## Status

Accepted

## Context

The optimization framework generates rich output:
- Multiple trial segmentations
- Screenshots at each stroke
- Quality metrics (Dice, Hausdorff)
- Parameter configurations

Users need to:
1. Visually compare trial segmentations against gold standards
2. Browse screenshots to understand algorithm behavior
3. Promote good trial results to new gold standards
4. Copy paths for use in documentation and reports

### Challenges

- Slicer's built-in tools don't support dual segmentation overlay
- Screenshot galleries need organization and metadata
- Comparing metrics requires context (parameter values)
- Gold standard creation should be streamlined

## Decision

Create a dedicated **SegmentEditorAdaptiveBrushReviewer** Slicer module for reviewing optimization results and managing gold standards.

### Module Structure

```
SegmentEditorAdaptiveBrushReviewer/
├── CMakeLists.txt
├── SegmentEditorAdaptiveBrushReviewer.py   # Module entry point
├── SegmentEditorAdaptiveBrushReviewerLib/
│   ├── __init__.py
│   ├── ReviewerWidget.py             # Main UI panel
│   ├── VisualizationController.py    # Dual segmentation display
│   ├── ScreenshotViewer.py           # Thumbnail grid viewer
│   └── ResultsLoader.py              # Load optimization runs
└── Resources/
    └── Icons/
        └── SegmentEditorAdaptiveBrushReviewer.png
```

### User Interface

```
┌─────────────────────────────────────────────────────────────────┐
│                 Adaptive Brush Results Review                    │
├─────────────────────────────────────────────────────────────────┤
│ Optimization Run: [2026-01-25_tumor_opt ▼] [Open Folder] [📋]   │
│ Trial: [#42 - Dice: 0.923 ▼] (Best: #42)                        │
├─────────────────────────────────────────────────────────────────┤
│ View Mode: (•) Outline  ( ) Transparent  ( ) Fill               │
│ Legend: ████ Gold  ████ Test  ████ Overlap                      │
│ [Load Gold] [Load Test] [Toggle Gold] [Toggle Test]             │
├─────────────────────────────────────────────────────────────────┤
│ Parameters:                          │ Metrics:                 │
│ ├─ algorithm: watershed              │ ├─ Dice: 0.923          │
│ ├─ edge_sensitivity: 45              │ ├─ HD95: 2.3mm          │
│ └─ gradient_scale: 1.5               │ └─ Time: 156ms          │
├─────────────────────────────────────────────────────────────────┤
│ Screenshots: [thumb] [thumb] [thumb] [◄ Prev] [Next ►]          │
│ Selected: 003_after.png  [View Full] [Copy Path]                │
├─────────────────────────────────────────────────────────────────┤
│ [Save as Gold Standard] [Export Report] [Compare Algorithms]    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### ReviewerWidget

Main panel managing all review functionality:

```python
class ReviewerWidget(qt.QWidget):
    """Main review UI panel."""

    def __init__(self, parent=None):
        self.results_loader = ResultsLoader()
        self.viz_controller = VisualizationController()
        self.screenshot_viewer = ScreenshotViewer()

    def load_run(self, run_path: Path):
        """Load optimization run for review."""
        self.current_run = self.results_loader.load(run_path)
        self._populate_trial_dropdown()
        self._load_gold_standard()

    def select_trial(self, trial_id: int):
        """Load and display a specific trial."""
        trial = self.current_run.get_trial(trial_id)
        self._display_parameters(trial.params)
        self._display_metrics(trial.metrics)
        self.viz_controller.load_test_segmentation(trial.segmentation_path)
        self.screenshot_viewer.set_screenshots(trial.screenshots)

    def save_as_gold(self):
        """Promote current trial to gold standard."""
        # Copy segmentation to gold_standards/
        # Create metadata.json with parameters
        # Update gold standard registry
```

#### VisualizationController

Manages dual segmentation display:

```python
class VisualizationController:
    """Control dual segmentation visualization."""

    # Color scheme
    GOLD_COLOR = (1.0, 0.84, 0.0)  # Gold
    TEST_COLOR = (0.0, 1.0, 1.0)   # Cyan
    OVERLAP_COLOR = (0.0, 1.0, 0.5)  # Green

    def __init__(self):
        self.gold_seg_node = None
        self.test_seg_node = None
        self.view_mode = "outline"  # outline, transparent, fill

    def load_gold_segmentation(self, path: Path):
        """Load gold standard segmentation."""
        self.gold_seg_node = slicer.util.loadSegmentation(str(path))
        self._apply_color(self.gold_seg_node, self.GOLD_COLOR)
        self._set_display_mode(self.gold_seg_node, self.view_mode)

    def load_test_segmentation(self, path: Path):
        """Load trial segmentation for comparison."""
        # Remove previous test if exists
        if self.test_seg_node:
            slicer.mrmlScene.RemoveNode(self.test_seg_node)

        self.test_seg_node = slicer.util.loadSegmentation(str(path))
        self._apply_color(self.test_seg_node, self.TEST_COLOR)
        self._set_display_mode(self.test_seg_node, self.view_mode)

    def set_view_mode(self, mode: str):
        """Change display mode: outline, transparent, fill."""
        self.view_mode = mode
        for node in [self.gold_seg_node, self.test_seg_node]:
            if node:
                self._set_display_mode(node, mode)

    def toggle_gold(self, visible: bool):
        """Toggle gold standard visibility."""
        if self.gold_seg_node:
            self.gold_seg_node.GetDisplayNode().SetVisibility(visible)

    def toggle_test(self, visible: bool):
        """Toggle test segmentation visibility."""
        if self.test_seg_node:
            self.test_seg_node.GetDisplayNode().SetVisibility(visible)
```

#### ScreenshotViewer

Thumbnail grid with selection and actions:

```python
class ScreenshotViewer(qt.QWidget):
    """Screenshot gallery with thumbnails."""

    def __init__(self):
        self.current_screenshots = []
        self.selected_index = 0

    def set_screenshots(self, screenshots: list[ScreenshotInfo]):
        """Update gallery with new screenshots."""
        self.current_screenshots = screenshots
        self._rebuild_thumbnails()
        self.select(0)

    def select(self, index: int):
        """Select a screenshot."""
        self.selected_index = index
        info = self.current_screenshots[index]
        self._update_preview(info)
        self._update_info_label(info)

    def view_full_size(self):
        """Open selected screenshot in system viewer."""
        info = self.current_screenshots[self.selected_index]
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(info.path)))

    def copy_path(self):
        """Copy screenshot path to clipboard."""
        info = self.current_screenshots[self.selected_index]
        clipboard = qt.QApplication.clipboard()
        clipboard.setText(str(info.path))
```

#### ResultsLoader

Load and parse optimization run output:

```python
class ResultsLoader:
    """Load optimization run results."""

    def load(self, run_path: Path) -> OptimizationRun:
        """Load complete optimization run."""
        config = self._load_config(run_path / "config.yaml")
        results = self._load_results(run_path / "results.json")
        importance = self._load_importance(run_path / "parameter_importance.json")

        return OptimizationRun(
            path=run_path,
            config=config,
            trials=results["trials"],
            best_trial_id=results["best_trial_id"],
            importance=importance,
        )

    def list_runs(self) -> list[Path]:
        """List available optimization runs."""
        runs_dir = Path("optimization_results")
        return sorted(runs_dir.iterdir(), reverse=True)
```

### View Modes

| Mode | Gold Display | Test Display | Use Case |
|------|--------------|--------------|----------|
| Outline | Yellow outline | Cyan outline | Compare boundaries |
| Transparent | Yellow 30% | Cyan 30% | See overlap |
| Fill | Yellow solid | Cyan solid | Toggle comparison |

### Actions

| Action | Description |
|--------|-------------|
| Load Gold | Load gold standard segmentation |
| Load Test | Load trial segmentation |
| Toggle Gold/Test | Show/hide each segmentation |
| Save as Gold | Promote trial to gold standard |
| Export Report | Generate markdown comparison report |
| Compare Algorithms | Side-by-side algorithm comparison view |
| Copy Path | Copy screenshot path to clipboard |
| Open Folder | Open run folder in file manager |

### Output Organization

```
optimization_results/
└── 2026-01-25_143025_tumor_opt/
    ├── config.yaml               # Input configuration
    ├── optuna_study.db           # Optuna SQLite storage
    ├── results.json              # Trial results
    ├── parameter_importance.json # FAnova analysis
    ├── segmentations/
    │   ├── manifest.json         # Trial -> file mapping
    │   ├── trial_001.seg.nrrd
    │   ├── trial_042.seg.nrrd    # Best trial
    │   └── ...
    └── screenshots/
        ├── manifest.json
        ├── trial_001/
        │   ├── stroke_01_1_before.png
        │   ├── stroke_01_2_after.png
        │   └── ...
        └── trial_042/
            └── ...
```

## Consequences

### Positive

- **Visual comparison**: See gold vs test side-by-side in Slicer
- **Workflow integration**: Review happens in same environment as testing
- **Gold standard management**: Easy promotion of good results
- **Documentation support**: Copy paths and export reports

### Negative

- **New module**: Additional code to maintain
- **Slicer dependency**: Review only works inside Slicer
- **Memory usage**: Loading multiple segmentations uses RAM

### Trade-offs

| Aspect | Alternative | Chosen | Reason |
|--------|-------------|--------|--------|
| Location | External viewer | Slicer module | Same environment as testing |
| Display | Single overlay | Dual with toggle | Clearer comparison |
| Screenshots | File browser | Embedded gallery | Integrated workflow |

## Alternatives Considered

### External Comparison Tool

**Rejected**: Requires exporting data, loses Slicer context. Users already have Slicer open.

### ITK-SNAP Comparison

**Rejected**: Different tool, different workflow. Keep everything in Slicer.

### Command-Line Only

**Rejected**: Visual review needs GUI. Screenshots need thumbnails.

### Single Segmentation Display

**Rejected**: Can't easily see differences. Dual display with toggle is clearer.

## Implementation Notes

### Module Registration

The reviewer module is a standard Slicer scripted module:

```python
# SegmentEditorAdaptiveBrushReviewer.py
class SegmentEditorAdaptiveBrushReviewer(ScriptedLoadableModule):
    def __init__(self, parent):
        parent.title = "Adaptive Brush Reviewer"
        parent.categories = ["Testing"]
        parent.dependencies = []
```

### CMakeLists.txt

```cmake
set(MODULE_NAME SegmentEditorAdaptiveBrushReviewer)
set(MODULE_PYTHON_SCRIPTS ${MODULE_NAME}.py)
set(MODULE_PYTHON_RESOURCES Resources)
```

### Integration with Tester Module

The Tester module can launch the Reviewer:

```python
def on_review_results_clicked(self):
    slicer.util.selectModule("SegmentEditorAdaptiveBrushReviewer")
    reviewer = slicer.modules.SegmentEditorAdaptiveBrushReviewerWidget
    reviewer.load_run(self.last_run_path)
```

## References

- [ADR-010](ADR-010-testing-framework.md): Slicer Testing Framework
- [ADR-011](ADR-011-optimization-framework.md): Smart Optimization Framework
- [Slicer Module Tutorial](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html)
- [Qt Designer for Slicer](https://slicer.readthedocs.io/en/latest/developer_guide/widgets.html)
