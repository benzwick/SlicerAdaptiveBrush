# Public Datasets

This directory contains downloaded public medical image segmentation datasets
with expert-annotated ground truth for validation and optimization.

**Note:** Data files are git-ignored due to size. Only this README is tracked.

## Downloading Datasets

Use the download script to fetch datasets:

```bash
# List available datasets
python scripts/download_public_datasets.py --list

# Download Medical Segmentation Decathlon (brain tumor)
python scripts/download_public_datasets.py --msd brain

# Download all MSD tasks
python scripts/download_public_datasets.py --msd brain lung heart

# Download CT-ORG (includes bone segmentation)
python scripts/download_public_datasets.py --ct-org

# Download everything
python scripts/download_public_datasets.py --all
```

## Alternative: Download in Slicer

After installing the extension, datasets appear in:
**File > Download Sample Data > Adaptive Brush Gold Standards**

## Directory Structure

After download:

```
PublicDatasets/
├── README.md                     # This file (git-tracked)
├── MSD/                          # Medical Segmentation Decathlon
│   ├── Task01_BrainTumour/
│   │   ├── imagesTr/             # Training images (NIfTI)
│   │   ├── labelsTr/             # Training labels (NIfTI)
│   │   └── dataset.json          # Dataset metadata
│   ├── Task02_Heart/
│   └── Task06_Lung/
└── CT-ORG/                       # CT-ORG Multi-Organ
    ├── volume-*.nii.gz           # CT volumes
    └── labels/
        └── volume-*.nii.gz       # Segmentation labels
```

## Datasets

### Medical Segmentation Decathlon (MSD)

**License:** CC-BY-SA 4.0
**URL:** http://medicaldecathlon.com/
**Paper:** https://www.nature.com/articles/s41467-022-30695-9

| Task | Modality | Anatomy | Size |
|------|----------|---------|------|
| Task01_BrainTumour | MRI | Glioma (necrotic, edema, enhancing) | 7.5 GB |
| Task02_Heart | MRI | Left atrium | 1.1 GB |
| Task06_Lung | CT | Lung tumors | 0.3 GB |

### CT-ORG

**License:** CC-BY 4.0
**URL:** https://zenodo.org/record/7860267
**Paper:** https://www.nature.com/articles/s41597-020-00715-8

140 CT scans with segmentations for:
- Liver
- Lungs
- Bladder
- Kidneys
- **Bones** (unique to this dataset)
- Brain

## Usage in Tests

After downloading, create metadata files:

```bash
python scripts/download_public_datasets.py --create-metadata
```

Then use in recipes:

```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()
volume, segmentation = manager.load_external("MSD_BrainTumor_001")
```

## Storage Requirements

| Dataset | Size |
|---------|------|
| MSD Brain Tumor | ~7.5 GB |
| MSD Heart | ~1.1 GB |
| MSD Lung | ~0.3 GB |
| CT-ORG | ~10 GB |
| **Total (all)** | **~19 GB** |

Download only what you need for testing.
