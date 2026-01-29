# ADR-018: CrossSegmentationExplorer Integration

## Status

**Accepted** (2026-01-28)

## Context

CrossSegmentationExplorer (CSE) is a 3D Slicer extension for comparing multiple AI-generated segmentations side-by-side. It provides:

- Dynamic multi-view layouts (2D + 3D) for N segmentations
- View linking (synchronized navigation)
- Model grouping by keyword matching
- Segment-by-segment comparison across models
- DICOM SEG native support

Our optimization framework generates multiple trial segmentations that benefit from this comparison capability. The Reviewer module (ADR-012, ADR-016) handles single gold-vs-trial comparison with detailed metrics and ratings.

### Use Cases

1. **Compare top trials per algorithm** - See best watershed vs best geodesic vs best level_set
2. **Compare trial evolution** - View trial_001, trial_050, trial_100 side-by-side
3. **Compare against gold standard** - Gold + multiple trials simultaneously
4. **Robustness analysis** - Same algorithm with different click patterns

### Reference Implementation

CrossSegmentationExplorer is cloned at `__reference__/CrossSegmentationExplorer/` for reference. Key patterns we may need to extend or fix:

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

### Trial-to-Model Mapping

CSE groups segmentations into "models" by keyword matching. For optimization trials, we may need custom mapping logic:

| Grouping Strategy | CSE Equivalent | Use Case |
|-------------------|----------------|----------|
| By algorithm | Model = algorithm name | Compare algorithm performance |
| By Dice range | Model = score bucket | Quality stratification |
| Top N per algorithm | Model = algorithm (filtered) | Best-of comparison |
| Manual selection | Custom model | Ad-hoc comparison |

```python
class TrialModelMapper:
    """Map optimization trials to CrossSegmentationExplorer models.

    This class can be used to extend CSE with optimization-aware grouping.
    """

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
        groups = {name: [] for _, _, name in ranges}
        for trial in trials:
            for low, high, name in ranges:
                if low <= trial.value < high:
                    groups[name].append(trial)
                    break
        return groups
```

### Layout Generation Pattern

CSE's dynamic layout XML generation pattern (for custom extensions):

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

### View Assignment Pattern

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

### View Linking Pattern

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

## Decision

**Use CrossSegmentationExplorer directly** for multi-trial comparison instead of reimplementing comparison features.

Our DICOM SEG output is fully CSE-compatible (verified 2026-01-28):

| Requirement | Status | Details |
|-------------|--------|---------|
| Same StudyInstanceUID | ✓ | Volume and all trial SEGs share StudyInstanceUID |
| ReferencedSeriesSequence | ✓ | Correctly points to volume SeriesInstanceUID |
| Modality | ✓ | SEG |
| SeriesDescription | ✓ | Contains trial info for grouping |

### Module Division

| Module | Purpose | Format |
|--------|---------|--------|
| **Reviewer** | Single trial review (gold vs trial) | .seg.nrrd + DICOM |
| **CSE** | Multi-trial comparison | DICOM SEG |

### Why Not Reimplement?

1. **CSE is actively maintained** - No maintenance burden for us
2. **Better features** - Multi-pane layout, view linking, outline modes
3. **No code duplication** - CSE already does this well
4. **Future compatibility** - CSE developer considering Slicer format support

## Workflow: Multi-Trial Comparison with CSE

### Step 1: Run Optimization

```bash
Slicer --python-script scripts/run_optimization.py configs/tumor_optimization.yaml
```

Output structure:
```
optimization_results/<timestamp>/
├── dicom/
│   ├── volume/              # Synthetic DICOM of SampleData
│   │   └── *.dcm
│   └── segmentations/       # DICOM SEG per trial
│       ├── trial_000/seg.dcm
│       ├── trial_001/seg.dcm
│       └── ...
├── results.json
└── ...
```

### Step 2: Import to DICOM Browser

1. Open 3D Slicer
2. **File → Import DICOM Files...**
3. Select folder: `optimization_results/<run>/dicom/`
4. Wait for import to complete

All trials appear under the same Patient/Study because they share StudyInstanceUID.

### Step 3: Open CrossSegmentationExplorer

1. **Modules → Informatics → CrossSegmentationExplorer**
2. Or search "Cross" in module finder

### Step 4: Select Reference Volume

1. In CSE, select the reference volume (the imported SampleData)
2. CSE auto-discovers all DICOM SEGs that reference this volume
3. Trials appear grouped by SeriesDescription

### Step 5: Compare

CSE provides:
- **Multi-pane layout** - Up to 6 segmentations side-by-side
- **Synchronized navigation** - All views move together
- **Outline mode** - See boundaries clearly
- **3D view** - Volume rendering comparison

### Example Session

```
┌─────────────────────────────────────────────────────────────────┐
│ CSE: Multi-Pane Layout (4 trials)                               │
├────────────────┬────────────────┬────────────────┬──────────────┤
│ Trial 000      │ Trial 025      │ Trial 050      │ Trial 099    │
│ (watershed)    │ (geodesic)     │ (level_set)    │ (best)       │
│ Dice: 0.823    │ Dice: 0.891    │ Dice: 0.934    │ Dice: 0.956  │
├────────────────┼────────────────┼────────────────┼──────────────┤
│                │                │                │              │
│  [Axial view]  │  [Axial view]  │  [Axial view]  │ [Axial view] │
│                │                │                │              │
│  Outlines show │  segmentation  │  boundaries    │  for visual  │
│                │                │                │  comparison  │
└────────────────┴────────────────┴────────────────┴──────────────┘
       ↑ Views are synchronized - scroll one, all scroll ↑
```

## Gold Standard in CSE

To include gold standard in CSE comparison:

1. Generate DICOM cache: The GoldStandardManager creates `.dicom_cache/` on demand
2. Import gold's DICOM: `GoldStandards/<name>/.dicom_cache/`
3. Gold appears alongside trials in CSE

Or use the **Reviewer module** for dedicated gold-vs-trial comparison with metrics.

## Integration with Reviewer Module

The modules are complementary:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Review Workflow                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Quick Comparison (CSE)                                      │
│     └── Multi-trial visual comparison                           │
│     └── Identify promising trials                               │
│                                                                 │
│  2. Detailed Review (Reviewer Module)                           │
│     └── Gold vs trial overlay                                   │
│     └── Dice, Hausdorff, Surface Dice metrics                   │
│     └── Ratings and notes                                       │
│     └── Workflow recording                                      │
│                                                                 │
│  3. Export Results                                              │
│     └── Ratings CSV                                             │
│     └── Best parameters                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Potential Custom UI Integration

If we need to add optimization-specific widgets to CSE or create a launcher:

```
┌─────────────────────────────────────────────────────────────────┐
│ ▼ Optimization Run Comparison                                    │
├─────────────────────────────────────────────────────────────────┤
│ Run: [2026-01-28_tumor_opt ▼]                                   │
│                                                                 │
│ Grouping: [By Algorithm ▼]  [Top N: 1 ▼]                        │
│                                                                 │
│ Select Models:                                                  │
│ ☑ watershed (best: 0.956)    ☑ geodesic (best: 0.948)          │
│ ☑ level_set (best: 0.942)    ☐ random_walker (best: 0.921)     │
│ ☑ Gold Standard                                                 │
│                                                                 │
│ [Open in CSE]  [Import to DICOM Browser]                        │
└─────────────────────────────────────────────────────────────────┘
```

This could be added as a collapsible section in the Reviewer module or as a separate launcher widget.

## Module Relationship

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reviewer Module                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │ Single Trial Mode   │    │ CSE Launcher (future)           │ │
│  │ (ADR-012, ADR-016)  │    │                                 │ │
│  │                     │    │ - Select optimization run        │ │
│  │ - Gold vs Trial     │    │ - Group trials by algorithm      │ │
│  │ - Full metrics      │    │ - Import to DICOM browser        │ │
│  │ - Ratings           │    │ - Launch CSE                     │ │
│  │ - Workflow recording│    │                                 │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    CrossSegmentationExplorer                     │
│  - Multi-trial layout                                            │
│  - View linking                                                  │
│  - Outline/fill modes                                            │
│  - (May extend with custom widgets)                              │
├─────────────────────────────────────────────────────────────────┤
│                    DICOM Data Layer (ADR-017)                    │
│  - DICOM SEG storage                                             │
│  - Synthetic DICOM for SampleData                                │
│  - DICOM database queries                                        │
└─────────────────────────────────────────────────────────────────┘
```

## DICOM Requirements for CSE Compatibility

Our optimization pipeline (run_optimization.py) generates DICOM with these tags:

```python
# Volume and all SEGs share same StudyInstanceUID
StudyInstanceUID = "1.2.826.0.1.3680043.8.498.{hash}"

# SEG references volume via:
ReferencedSeriesSequence[0].SeriesInstanceUID = volume_series_uid

# SEG identification:
Modality = "SEG"
SeriesDescription = f"Trial {trial_num:03d} - {algorithm}"
```

This allows CSE to:
1. Group all SEGs under one Study
2. Link SEGs to the correct volume
3. Display meaningful labels for each segmentation

## Integration with DICOM Ecosystem

This integration depends on ADR-017 (DICOM SEG Data Format):

1. **LABELMAP encoding (Supplement 243)** - Efficient multi-segment storage
2. **Compression (RLELossless/JPEG2000)** - Compact files for many trials
3. **Load trials from DICOM database** - Query by ReferencedSeriesSequence
4. **Use DICOM metadata** - SeriesDescription for model naming
5. **OHIF v3.11 compatibility** - LABELMAP optimized for OHIF viewer
6. **CrossSegmentationExplorer compatibility** - Same data loads in both modules

### Why LABELMAP Matters for Cross-Comparison

When comparing N algorithms × M trials, storage efficiency is critical:

| Format | 50 trials × 5 segments each |
|--------|----------------------------|
| .seg.nrrd | ~25MB (gzip) |
| DICOM SEG BINARY | ~4GB uncompressed |
| DICOM SEG LABELMAP | ~25MB (JPEG2000) |

LABELMAP encoding makes multi-trial comparison practical.

### OHIF Viewer Support

Our DICOM SEG output is compatible with OHIF v3.11+:
- LABELMAP encoding supported since OHIF 3.11
- Can view optimization results in web browser
- Useful for remote review or collaboration

```
Workflow: Local optimization → DICOM SEG → Upload to OHIF server → Web review
```

## Consequences

### Positive

- **Leverage existing tool** - Use CSE directly for core comparison
- **Better features** - CSE has more comparison options than we could build alone
- **Clear separation** - Reviewer for detailed review, CSE for multi-trial comparison
- **Extensible** - Can add custom widgets/logic as needed
- **Reference code available** - CSE patterns documented for bug fixes or extensions

### Negative

- **Requires CSE installation** - Additional extension dependency
- **DICOM workflow** - Must import to DICOM browser first
- **Learning curve** - Users must learn two modules
- **May need CSE modifications** - Some optimization-specific features may require CSE changes

### Trade-offs

| Aspect | Reviewer Module | CrossSegmentationExplorer |
|--------|-----------------|---------------------------|
| Detail | High (full metrics, ratings) | Visual comparison only |
| Trials | One at a time | Many simultaneously |
| Use case | Quality assessment | Quick comparison |
| Customization | Full control | May need upstream PRs |

## Alternatives Considered

### Reimplement Cross-Comparison from Scratch

**Rejected**: Would duplicate CSE functionality. CSE is better maintained and has more features. Reference patterns kept for potential extensions.

### Fork CrossSegmentationExplorer

**Deferred**: Creates maintenance burden. Prefer upstream contributions. May revisit if CSE doesn't accept needed features.

### External Comparison Tool

**Rejected**: Breaks workflow. Users want comparison within Slicer environment.

## References

- [ADR-012: Results Review Module](ADR-012-results-review-module.md)
- [ADR-016: Enhanced Review Visualization](ADR-016-enhanced-review-visualization.md)
- [ADR-017: DICOM SEG Data Format Standard](ADR-017-dicom-seg-data-format.md)
- [CrossSegmentationExplorer GitHub](https://github.com/ImagingDataCommons/CrossSegmentationExplorer)
- [CrossSegmentationExplorer Documentation](https://github.com/ImagingDataCommons/CrossSegmentationExplorer#readme)
- [OHIF v3.11 LABELMAP Support](https://ohif.org/release-notes/3p11/)
- [DICOM Supplement 243: Label Map Segmentation](https://www.dicomstandard.org/news-dir/current/docs/sups/sup243.pdf)
- [Slicer Layout Documentation](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#customize-view-layout)
