# External Gold Standards

This directory contains **metadata only** for gold standards sourced from
public datasets. The actual image/label data is stored in `../PublicDatasets/`.

## Structure

Each gold standard has its own subdirectory:

```
ExternalGoldStandards/
├── README.md                         # This file
├── MSD_BrainTumor_001/
│   ├── metadata.json                 # Points to PublicDatasets/MSD/...
│   └── reference_screenshots/        # Visual references
│       └── *.png
├── MSD_Lung_001/
│   └── metadata.json
└── CT_ORG_Bone_001/
    └── metadata.json
```

## Metadata Format

```json
{
  "name": "MSD_BrainTumor_001",
  "source": {
    "dataset": "Medical Segmentation Decathlon - Brain Tumor (BraTS)",
    "case_id": "BRATS_001",
    "url": "http://medicaldecathlon.com/",
    "license": "CC-BY-SA 4.0"
  },
  "files": {
    "image": "SegmentEditorAdaptiveBrushTester/PublicDatasets/MSD/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
    "label": "SegmentEditorAdaptiveBrushTester/PublicDatasets/MSD/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz"
  },
  "label_mapping": {
    "1": "necrotic_core",
    "2": "edema",
    "3": "enhancing_tumor"
  },
  "segment_of_interest": "whole_tumor",
  "segment_labels": [1, 2, 3],
  "maps_to_recipe": "MRBrainTumor2_tumor",
  "modality": "MRI",
  "created": "2026-01-26T12:00:00"
}
```

## Creating Metadata

After downloading public datasets:

```bash
python scripts/download_public_datasets.py --create-metadata
```

## Relationship to Internal Gold Standards

| Type | Location | Data Storage | Git Tracked |
|------|----------|--------------|-------------|
| Internal | `GoldStandards/` | In directory | Yes (small) |
| External | `ExternalGoldStandards/` | `PublicDatasets/` | Metadata only |

**Internal** gold standards (like `MRBrainTumor1_tumor`) are created manually
from Slicer SampleData and stored directly in the repository.

**External** gold standards reference larger public datasets that users
download on-demand.

## Usage

```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()

# Load internal gold standard
vol, seg = manager.load_gold("MRBrainTumor1_tumor")

# Load external gold standard (requires PublicDatasets download)
vol, seg = manager.load_gold("MSD_BrainTumor_001")
```

## Mapping to Recipes

External gold standards can substitute for missing internal ones:

| Recipe Needs | External Source |
|--------------|-----------------|
| MRBrainTumor2_tumor | MSD_BrainTumor_001 |
| CTChest_lung | MSD_Lung_001 |
| CTChest_bone | CT_ORG_Bone_001 |
| CTACardio_left_ventricle | MSD_Heart_001 |
