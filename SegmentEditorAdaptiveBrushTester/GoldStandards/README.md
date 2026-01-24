# Gold Standards

This directory contains gold standard segmentations for regression testing and optimization.

## Directory Structure

Each gold standard is stored in a subdirectory:

```
GoldStandards/
├── README.md                     # This file
├── MRBrainTumor1_tumor/
│   ├── gold.seg.nrrd            # The gold standard segmentation
│   ├── metadata.json            # Creation info, parameters, clicks
│   └── reference_screenshots/   # Visual references
│       ├── manifest.json
│       └── *.png
└── MRHead_ventricle/
    └── ...
```

## Metadata Format

Each gold standard includes `metadata.json`:

```json
{
  "created": "2026-01-24T15:30:00",
  "volume": {
    "name": "MRBrainTumor1",
    "spacing": [0.9375, 0.9375, 1.5],
    "dimensions": [256, 256, 130]
  },
  "segment_id": "Tumor",
  "description": "Tumor segmentation using watershed, 5 clicks, optimized parameters",
  "algorithm": "watershed",
  "parameters": {
    "brush_radius_mm": 25.0,
    "edge_sensitivity": 40,
    "watershed_gradient_scale": 1.0
  },
  "clicks": [
    {
      "ras": [-5.31, 34.77, 20.83],
      "ijk": [128, 100, 45],
      "params": {"algorithm": "watershed", "radius_mm": 25}
    }
  ],
  "voxel_count": 45230
}
```

## Usage

### Loading a Gold Standard

```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()
gold_seg, metadata = manager.load_gold("MRBrainTumor1_tumor")
```

### Creating a New Gold Standard

```python
manager.save_as_gold(
    segmentation_node=seg_node,
    volume_node=vol_node,
    segment_id="Tumor",
    name="MRBrainTumor1_tumor",
    click_locations=clicks,
    description="Optimized tumor segmentation",
    algorithm="watershed",
    parameters={"brush_radius_mm": 25.0, "edge_sensitivity": 40}
)
```

### Listing Available Gold Standards

```python
standards = manager.list_gold_standards()
for std in standards:
    print(f"{std['name']}: {std['voxel_count']} voxels")
```

## Guidelines

1. **One gold standard per anatomy/sample**: e.g., `MRBrainTumor1_tumor`, `MRHead_ventricle`
2. **Document the creation process**: Include click locations and parameters in metadata
3. **Add reference screenshots**: Visual references help verify the gold standard quality
4. **Version control**: Gold standards are git-tracked for reproducibility
