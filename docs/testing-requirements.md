# Testing Framework Requirements

This document captures all requirements for the SegmentEditorAdaptiveBrushTester testing framework, derived from iterative development and user feedback.

## Purpose

Create a comprehensive testing and documentation system that:
1. Runs automated tests inside 3D Slicer
2. Captures screenshots usable for documentation and tutorials
3. Tests all algorithms with various parameters
4. Can regenerate documentation automatically by re-running tests

## Screenshot Requirements

### Visual Elements That Must Be Visible

1. **Brush Circle (Outline)**
   - Yellow outer circle showing maximum brush extent
   - Cyan inner circle showing threshold sampling zone
   - Must be visible at the paint location before each operation
   - Color changes to red/orange in erase mode

2. **Options Panel**
   - Adaptive Brush effect must be selected and visible in left panel
   - All relevant options for the current algorithm should be visible
   - Collapsible sections should be expanded to show settings

3. **Slice Views**
   - Red (axial), Yellow (sagittal), Green (coronal) views
   - Crosshair at the paint location
   - Any existing segmentation overlay visible

4. **Segmentation Results**
   - Green overlay showing segmented regions
   - Clear visibility of boundaries

### Screenshot Organization

- **Flat folder structure**: All screenshots in a single `screenshots/` folder
- **Sequential numbering**: `001.png`, `002.png`, etc. across all tests
- **Descriptive names in manifest**: `[tag] Description` format
  - Tags indicate context: `[setup]`, `[algorithm_name]`, `[verify]`
- **Manifest file** (`manifest.json`):
  - Lists all screenshots with descriptions
  - Includes `by_test` section mapping tests to screenshot numbers
  - Contains timestamp, dimensions, and view type

### Screenshot Timing

Take screenshots at these key moments:
1. **Setup complete**: After data loaded, effect activated
2. **Before each operation**: Showing brush position and current settings
3. **After each operation**: Showing results
4. **Verification**: Final state with metrics

## Brush Circle Visibility

The brush outline is hidden by default and only appears during mouse interaction.

### Making Brush Visible in Tests

**IMPORTANT**: The default brush radius (5mm) may be too small to see clearly in screenshots.
Set the radius to 25mm or larger for documentation screenshots.

```python
# Get the scripted effect
scripted_effect = effect.self()

# Set a visible brush radius (default 5mm is too small for screenshots)
scripted_effect.radiusSlider.value = 25.0  # 25mm radius
slicer.app.processEvents()

# Get slice widget
layoutManager = slicer.app.layoutManager()
red_widget = layoutManager.sliceWidget("Red")

# Navigate to a specific slice (ensures view is properly initialized)
red_logic = red_widget.sliceLogic()
red_logic.SetSliceOffset(0)  # Or specific RAS z-coordinate
slicer.app.processEvents()

# Convert RAS to XY for brush position (recommended approach)
import vtk
slice_node = red_logic.GetSliceNode()
slice_to_ras = slice_node.GetSliceToRAS()
center_ras = [slice_to_ras.GetElement(0, 3),
              slice_to_ras.GetElement(1, 3),
              slice_to_ras.GetElement(2, 3)]

xy_to_ras = slice_node.GetXYToRAS()
ras_to_xy = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)
ras_point = [center_ras[0], center_ras[1], center_ras[2], 1]
xy_point = [0, 0, 0, 1]
ras_to_xy.MultiplyPoint(ras_point, xy_point)
center_xy = (int(xy_point[0]), int(xy_point[1]))

# Trigger brush preview (makes outline visible)
scripted_effect._updateBrushPreview(center_xy, red_widget, eraseMode=False)
red_widget.sliceView().forceRender()
slicer.app.processEvents()

# Now take screenshot
ctx.screenshot("[description] Brush visible at location")
```

### Brush Outline Colors

| Mode | Outer Circle | Inner Circle |
|------|-------------|--------------|
| Add (default) | Yellow (1.0, 0.9, 0.1) | Cyan (0.2, 0.9, 1.0) |
| Erase | Red/Orange (1.0, 0.3, 0.1) | Light Orange (1.0, 0.5, 0.3) |

## Widget Interaction Requirements

Tests must interact with widgets like a real user would:

### Selecting Algorithms
```python
combo = scripted_effect.algorithmCombo
idx = combo.findData("watershed")  # Use data value, not display text
if idx >= 0:
    combo.setCurrentIndex(idx)  # Triggers onAlgorithmChanged signal
slicer.app.processEvents()
```

### Setting Parameters
```python
# Use the slider widget directly
scripted_effect.sensitivitySlider.value = 50
scripted_effect.radiusSlider.value = 25.0

# For checkboxes
scripted_effect.sphereModeCheckbox.checked = True
```

### Simulating Paint Operations
```python
# Set up for painting
scripted_effect.scriptedEffect.saveStateForUndo()
scripted_effect.isDrawing = True
scripted_effect._currentStrokeEraseMode = False

# Paint at screen coordinates
scripted_effect.processPoint(xy, slice_widget)

# Clean up
scripted_effect.isDrawing = False
slicer.app.processEvents()
```

## GUI Visibility Requirements

### Module Panel
- Switch to Segment Editor module: `slicer.util.selectModule("SegmentEditor")`
- Use module's editor widget (not hidden widget):
  ```python
  segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
  segment_editor_widget = segment_editor_module.editor
  ```

### Effect Activation
```python
segment_editor_widget.setActiveEffectByName("Adaptive Brush")
effect = segment_editor_widget.activeEffect()
scripted_effect = effect.self()
```

## Test Structure Requirements

Each test case must have:

1. **setup()**: Load data, create segmentation, activate effect
2. **run()**: Execute test operations with screenshots
3. **verify()**: Assert expected conditions, capture final state
4. **teardown()**: Clean up, but preserve scene for inspection

### Test Registration
```python
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

@register_test(category="algorithm")
class TestAlgorithmWatershed(TestCase):
    name = "algorithm_watershed"
    description = "Test watershed algorithm"
```

### Categories
- `algorithm`: Individual algorithm tests
- `optimization`: Parameter optimization tests
- `ui`: User interface tests
- `workflow`: End-to-end workflow tests

## Output Structure

```
test_runs/2026-01-24_143025_all/
├── metadata.json          # Run config, summary
├── results.json           # Test results with assertions
├── metrics.json           # Performance metrics
├── screenshots/
│   ├── manifest.json      # Screenshot descriptions
│   ├── 001.png           # Sequential screenshots
│   ├── 002.png
│   └── ...
└── logs/
    ├── test_run.log       # Test execution log
    └── slicer_session.log # Slicer log copy
```

## Algorithms to Test

All 8 algorithms must be tested:

| Algorithm | Data Value | Display Name |
|-----------|------------|--------------|
| Watershed | `watershed` | Watershed |
| Level Set CPU | `level_set_cpu` | Level Set (CPU) |
| Level Set GPU | `level_set_gpu` | Level Set (GPU) |
| Connected Threshold | `connected_threshold` | Connected Threshold |
| Region Growing | `region_growing` | Region Growing |
| Threshold Brush | `threshold_brush` | Threshold Brush |
| Geodesic Distance | `geodesic_distance` | Geodesic Distance |
| Random Walker | `random_walker` | Random Walker |

### Threshold Brush Sub-methods
- Otsu (`otsu`)
- Huang (`huang`)
- Triangle (`triangle`)
- Li (`li`)

## Sample Data

| Sample | Use Case |
|--------|----------|
| MRHead | General brain MRI testing |
| MRBrainTumor1 | Tumor segmentation, good contrast |

## Coordinate Systems

- **RAS**: Right-Anterior-Superior (Slicer world coordinates)
- **LPS**: Left-Posterior-Superior (DICOM standard)
- **IJK**: Volume array indices
- **XY**: Screen coordinates in slice views

### Conversion: LPS to RAS
```
R = -L
A = -P
S = S
```

### Conversion: RAS to XY (screen)
```python
def _rasToXy(ras, sliceWidget):
    sliceLogic = sliceWidget.sliceLogic()
    sliceNode = sliceLogic.GetSliceNode()

    xyToRas = sliceNode.GetXYToRAS()
    rasToXy = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(xyToRas, rasToXy)

    rasPoint = [ras[0], ras[1], ras[2], 1]
    xyPoint = [0, 0, 0, 1]
    rasToXy.MultiplyPoint(rasPoint, xyPoint)

    return (int(xyPoint[0]), int(xyPoint[1]))
```

## Performance Metrics

Track and report:
- Voxels segmented per algorithm
- Execution time (ms)
- Voxels per millisecond (efficiency)

### Baseline Performance (MRBrainTumor1, 5 clicks, 25mm brush, 40% sensitivity)

| Algorithm | Voxels | Time (ms) | Voxels/ms |
|-----------|--------|-----------|-----------|
| connected_threshold | 80,098 | 1,328 | 60.3 |
| threshold_brush | 16,116 | 1,475 | 10.9 |
| geodesic_distance | 15,508 | 1,538 | 10.1 |
| level_set_cpu | 14,034 | 1,474 | 9.5 |
| level_set_gpu | 14,034 | 1,498 | 9.4 |
| watershed | 13,383 | 2,831 | 4.7 |
| region_growing | 11,273 | 1,465 | 7.7 |
| random_walker | 3,843 | 9,069 | 0.4 |

**Notes:**
- connected_threshold produces most voxels but may over-segment
- watershed is slower but provides good boundary detection
- random_walker is slowest due to sparse linear system solving
- level_set_cpu/gpu produce identical results (no GPU implementation yet)

## Known Issues and Solutions

### Level Set Producing 0 Voxels
- **Cause**: `sitk.BinaryThreshold(levelSet, upperThreshold=0)` has default `lowerThreshold=0`, creating range [0,0]
- **Solution**: Use numpy: `(levelSetArray <= 0).astype(np.uint8)`

### Brush Circle Not Visible
- **Cause**: Outline only appears on mouse EnterEvent
- **Solution**: Call `_updateBrushPreview()` before screenshots
- **Additional**: Use `forceRender()` not `scheduleRender()` for immediate effect

### Algorithm Fallthrough Bug
- **Cause**: Missing return statements in algorithm switch
- **Solution**: Always return result or raise error for unknown algorithm

### Effect Panel Disappears After Segment Operations
- **Cause**: Calling `segmentation.RemoveAllSegments()` or similar segment operations can deactivate the effect or collapse the panel
- **Solution**: Re-activate the effect after segment operations:
  ```python
  segmentation.RemoveAllSegments()
  segment_id = segmentation.AddEmptySegment("NewSegment")
  segment_editor_widget.setCurrentSegmentID(segment_id)
  slicer.app.processEvents()

  # Re-activate effect (segment removal may have deactivated it)
  segment_editor_widget.setActiveEffectByName("Adaptive Brush")
  effect = segment_editor_widget.activeEffect()
  slicer.app.processEvents()
  ```

## Documentation Generation

These tests serve dual purpose:
1. **Automated testing**: Verify functionality works
2. **Documentation**: Screenshots used in tutorials/docs

Screenshots should be:
- Clear and readable
- Show all relevant UI elements
- Demonstrate typical user workflow
- Capture both "before" and "after" states

---

*Last updated: 2026-01-24*
*Generated from iterative testing and user feedback*
