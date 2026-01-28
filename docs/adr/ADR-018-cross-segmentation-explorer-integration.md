# ADR-018: CrossSegmentationExplorer Integration

## Status

**Proposed** (2026-01-28)

## Context

CrossSegmentationExplorer is a 3D Slicer extension for comparing multiple AI-generated segmentations side-by-side. It provides:

- Dynamic multi-view layouts (2D + 3D) for N segmentations
- View linking (synchronized navigation)
- Model grouping by keyword matching
- Segment-by-segment comparison across models
- DICOM SEG native support

Our optimization framework generates multiple trial segmentations that would benefit from this comparison capability. Currently, the Reviewer module (ADR-012, ADR-016) only supports comparing one trial against one gold standard at a time.

### Reference Implementation

CrossSegmentationExplorer is cloned at `__reference__/CrossSegmentationExplorer/` for reference. Key patterns:

```python
# Dynamic layout XML generation (SegmentationComparison.py:1693-1744)
def getLayoutXML(self, viewNumber, threedCheckbox, twodCheckbox, layout, viewNames):
    """Generate XML for custom Slicer layout with N view sets."""

# View assignment (SegmentationComparison.py:1424-1491)
def assignSegmentationsToViews(self, threed_enabled, twod_enabled, selectedVolume, nodeMapping, status_outline):
    """Assign segmentation nodes to their respective views."""

# View linking (SegmentationComparison.py:1561-1582)
def _set3DLink(self, status):
    """Link/unlink 3D view cameras."""

def _set2DLink(self, status):
    """Link/unlink 2D slice navigation."""
```

### Use Cases

1. **Compare top trials per algorithm** - See best watershed vs best geodesic vs best level_set
2. **Compare trial evolution** - View trial_001, trial_050, trial_100 side-by-side
3. **Compare against gold standard** - Gold + multiple trials simultaneously
4. **Robustness analysis** - Same algorithm with different click patterns

## Decision

Integrate CrossSegmentationExplorer-style comparison into the Reviewer module with two modes:

1. **Single Trial Mode** (existing) - Gold vs one trial, detailed metrics
2. **Cross-Comparison Mode** (new) - Multiple trials side-by-side

### Trial-to-Model Mapping

CrossSegmentationExplorer groups segmentations into "models" by keyword matching. For optimization trials, we map:

| Grouping Strategy | CSE Equivalent | Use Case |
|-------------------|----------------|----------|
| By algorithm | Model = algorithm name | Compare algorithm performance |
| By Dice range | Model = score bucket | Quality stratification |
| Top N per algorithm | Model = algorithm (filtered) | Best-of comparison |
| Manual selection | Custom model | Ad-hoc comparison |

```python
class TrialModelMapper:
    """Map optimization trials to CrossSegmentationExplorer models."""

    def group_by_algorithm(self, trials: list[TrialData]) -> dict[str, list[TrialData]]:
        """
        Group trials by algorithm parameter.

        Returns: {"watershed": [trial1, trial2], "geodesic": [trial3, trial4], ...}
        """
        groups = {}
        for trial in trials:
            algo = trial.params.get("algorithm", "unknown")
            groups.setdefault(algo, []).append(trial)
        return groups

    def get_top_n_per_algorithm(
        self, trials: list[TrialData], n: int = 1
    ) -> dict[str, list[TrialData]]:
        """
        Get top N trials by Dice score for each algorithm.

        Returns: {"watershed": [best_watershed], "geodesic": [best_geodesic], ...}
        """
        by_algo = self.group_by_algorithm(trials)
        return {
            algo: sorted(group, key=lambda t: t.value, reverse=True)[:n]
            for algo, group in by_algo.items()
        }

    def group_by_dice_range(
        self, trials: list[TrialData], ranges: list[tuple[float, float, str]]
    ) -> dict[str, list[TrialData]]:
        """
        Group trials by Dice score ranges.

        ranges: [(0.95, 1.0, "excellent"), (0.90, 0.95, "good"), ...]
        Returns: {"excellent": [...], "good": [...], ...}
        """
```

### UI Integration

Add "Cross-Comparison" collapsible section to Reviewer:

```
┌─────────────────────────────────────────────────────────────────┐
│ ▼ Cross-Comparison Mode                                         │
├─────────────────────────────────────────────────────────────────┤
│ Grouping: [By Algorithm ▼]  [Top N: 1 ▼]                        │
│                                                                 │
│ Select Models:                                                  │
│ ☑ watershed (best: 0.956)    ☑ geodesic (best: 0.948)          │
│ ☑ level_set (best: 0.942)    ☐ random_walker (best: 0.921)     │
│ ☑ Gold Standard                                                 │
│                                                                 │
│ Layout: (•) 2D only  ( ) 3D only  ( ) Both                      │
│ Arrangement: (•) Horizontal  ( ) Vertical                       │
│ ☑ Link views  ☑ Show outlines                                  │
│                                                                 │
│ [Apply Layout]                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layout Generation

Port CrossSegmentationExplorer's dynamic layout XML generation:

```python
def generate_comparison_layout(
    self,
    model_names: list[str],
    show_2d: bool = True,
    show_3d: bool = False,
    vertical: bool = False
) -> str:
    """
    Generate Slicer layout XML for multi-model comparison.

    Creates view sets for each model:
    - 3D view (optional)
    - Axial, Sagittal, Coronal slice views

    Based on CrossSegmentationExplorer.getLayoutXML()
    """
    layout_type_outer = "horizontal" if not vertical else "vertical"
    layout_type_inner = "vertical" if not vertical else "horizontal"

    xml = f'<layout type="{layout_type_outer}" split="true">\n'

    for model_name in model_names:
        xml += '  <item>\n'
        xml += f'    <layout type="{layout_type_inner}" split="true">\n'

        if show_3d:
            xml += f'''
      <item><view class="vtkMRMLViewNode" singletontag="View{model_name}">
        <property name="viewlabel" action="default">{model_name}</property>
      </view></item>
'''

        if show_2d:
            for orientation, color, prefix in [
                ("Axial", "#F34A33", "R"),
                ("Sagittal", "#4AF333", "G"),
                ("Coronal", "#F3E833", "Y")
            ]:
                xml += f'''
      <item><view class="vtkMRMLSliceNode" singletontag="{prefix} {model_name}">
        <property name="orientation" action="default">{orientation}</property>
        <property name="viewlabel" action="default">{prefix} {model_name}</property>
        <property name="viewcolor" action="default">{color}</property>
      </view></item>
'''

        xml += '    </layout>\n'
        xml += '  </item>\n'

    xml += '</layout>'
    return xml
```

### View Assignment

After creating the layout, assign segmentations to their views:

```python
def assign_trials_to_views(
    self,
    volume_node: vtkMRMLVolumeNode,
    model_mapping: dict[str, list[TrialData]],
    show_2d: bool,
    show_3d: bool,
    outline_mode: bool
):
    """
    Load and assign trial segmentations to comparison views.

    Based on CrossSegmentationExplorer.assignSegmentationsToViews()
    """
    layout_manager = slicer.app.layoutManager()

    for model_name, trials in model_mapping.items():
        # Load segmentation(s) for this model
        seg_nodes = [self._load_trial_segmentation(t) for t in trials]

        if show_3d:
            view_node = self._get_3d_view_node(f"View{model_name}")
            for seg_node in seg_nodes:
                display = seg_node.GetDisplayNode()
                display.AddViewNodeID(view_node.GetID())
                display.SetVisibility3D(True)

        if show_2d:
            for prefix in ["R", "G", "Y"]:
                slice_name = f"{prefix} {model_name}"
                slice_widget = layout_manager.sliceWidget(slice_name)
                composite = slice_widget.sliceLogic().GetSliceCompositeNode()
                composite.SetBackgroundVolumeID(volume_node.GetID())

                view_node = slice_widget.mrmlSliceNode()
                for seg_node in seg_nodes:
                    display = seg_node.GetDisplayNode()
                    display.AddViewNodeID(view_node.GetID())
                    display.SetVisibility2DFill(not outline_mode)
                    display.SetVisibility2DOutline(True)
```

### View Linking

Use native Slicer view groups for synchronized navigation:

```python
def link_comparison_views(self, linked: bool):
    """
    Enable/disable synchronized navigation across all comparison views.

    Uses ViewGroupManager from ADR-016 for native Slicer linking.
    """
    # 2D linking via SliceCompositeNode
    for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
        composite.SetLinkedControl(linked)
        composite.SetHotLinkedControl(linked)

    # 3D linking via ViewNode
    for view_node in slicer.util.getNodesByClass("vtkMRMLViewNode"):
        view_node.SetLinkedControl(linked)
```

## Integration with DICOM

This integration depends on ADR-017 (DICOM SEG Data Format):

1. **Load trials from DICOM database** - Query by ReferencedSeriesSequence
2. **Use DICOM metadata** - SeriesDescription for model naming
3. **CrossSegmentationExplorer compatibility** - Same data loads in both modules

## Consequences

### Positive

- **Multi-trial comparison** - See multiple algorithms/trials simultaneously
- **Leverages proven patterns** - CrossSegmentationExplorer's layout/view code
- **Enhanced review workflow** - Quick visual comparison before detailed analysis
- **Algorithm selection insight** - See which algorithm works best visually

### Negative

- **UI complexity** - More options in Reviewer module
- **Memory usage** - Multiple segmentations loaded simultaneously
- **Layout limitations** - Screen space limits practical comparison to ~4-5 models

### Trade-offs

| Aspect | Single Trial Mode | Cross-Comparison Mode |
|--------|-------------------|----------------------|
| Detail | High (full metrics) | Lower (visual only) |
| Throughput | One at a time | Many at once |
| Use case | Detailed review | Quick comparison |
| Memory | Low | Higher |

## Module Relationship

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reviewer Module                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │ Single Trial Mode   │    │ Cross-Comparison Mode           │ │
│  │ (ADR-012, ADR-016)  │    │ (ADR-018)                       │ │
│  │                     │    │                                 │ │
│  │ - Gold vs Trial     │    │ - Multi-trial layout            │ │
│  │ - Full metrics      │    │ - Trial-to-model mapping        │ │
│  │ - Ratings           │    │ - View linking                  │ │
│  │ - Workflow recording│    │ - Based on CSE patterns         │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    DICOM Data Layer (ADR-017)                    │
│  - DICOM SEG storage                                             │
│  - Synthetic DICOM for SampleData                                │
│  - DICOM database queries                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Alternatives Considered

### Use CrossSegmentationExplorer Directly

**Rejected**: CSE assumes clinical DICOM workflow (patient studies, DICOM database queries). Our optimization results need trial-to-model mapping that CSE doesn't support.

### Fork CrossSegmentationExplorer

**Rejected**: Creates maintenance burden. Better to port specific patterns into our module.

### External Comparison Tool

**Rejected**: Breaks workflow. Users want comparison in same environment as review.

## References

- [ADR-012: Results Review Module](ADR-012-results-review-module.md)
- [ADR-016: Enhanced Review Visualization](ADR-016-enhanced-review-visualization.md)
- [ADR-017: DICOM SEG Data Format Standard](ADR-017-dicom-seg-data-format.md)
- [CrossSegmentationExplorer](https://github.com/ImagingDataCommons/CrossSegmentationExplorer)
- [Slicer Layout Documentation](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#customize-view-layout)
