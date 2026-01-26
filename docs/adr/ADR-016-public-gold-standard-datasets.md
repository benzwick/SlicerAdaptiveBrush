# ADR-016: Public Gold Standard Datasets

## Status

Proposed

## Context

The Adaptive Brush extension uses gold standard segmentations for:

1. **Regression testing**: Ensure algorithm changes don't degrade quality
2. **Parameter optimization**: Tune parameters to maximize Dice score
3. **Validation**: Verify algorithms work correctly on diverse data

Currently, we have only one gold standard (`MRBrainTumor1_tumor`) created manually. The test warnings indicate missing gold standards for:

- `MRBrainTumor2_tumor`
- `CTChest_bone`
- `CTChest_lung`
- `MRHead_ventricles`
- `CTACardio_left_ventricle`
- `MRHead_white_matter`

Creating these manually is time-consuming and subjective. Professional medical image segmentation datasets with expert-annotated ground truth exist and can provide high-quality gold standards.

### Requirements

1. **License compatibility**: Must be permissible for use in open-source testing
2. **Format compatibility**: Must be loadable in 3D Slicer (NIfTI, NRRD, DICOM)
3. **Anatomical coverage**: Should cover brain, lung, bone, cardiac anatomy
4. **Expert annotation**: Must have professionally created ground truth
5. **Reproducibility**: Must be publicly downloadable

## Decision

Use the following public datasets as sources for gold standard segmentations:

### Primary Datasets

| Dataset | License | Anatomies | Format | Size |
|---------|---------|-----------|--------|------|
| **Medical Segmentation Decathlon** | CC-BY-SA 4.0 | Brain tumor, Lung, Liver, Heart, etc. | NIfTI | ~30GB total |
| **CT-ORG** | CC-BY 4.0 | Lung, Bone, Liver, Kidney, Brain | NIfTI | ~10GB |
| **BraTS 2021** | CC-BY 4.0 | Brain tumor (multi-region) | NIfTI | ~15GB |

### Dataset-to-Recipe Mapping

| Missing Gold Standard | Source Dataset | Task/Subset |
|-----------------------|----------------|-------------|
| `MRBrainTumor2_tumor` | MSD Task01 (Brain Tumor) | Use case BRATS_001 |
| `CTChest_lung` | MSD Task06 (Lung) | Use case Lung_001 |
| `CTChest_bone` | CT-ORG | Bone segmentation subset |
| `MRHead_ventricles` | MSD Task04 (Hippocampus) or custom | Ventricle region |
| `CTACardio_left_ventricle` | MSD Task02 (Heart) | Left atrium → adapt for LV |
| `MRHead_white_matter` | Custom from MRHead | White matter threshold |

### Storage Strategy

Public dataset cases are stored in a separate git-ignored directory to avoid bloating the repository:

```
SegmentEditorAdaptiveBrushTester/
├── GoldStandards/                    # Git-tracked (small, curated)
│   ├── MRBrainTumor1_tumor/          # Manually created from Slicer SampleData
│   └── README.md
├── PublicDatasets/                   # Git-ignored (large, downloaded)
│   ├── README.md                     # Instructions (git-tracked)
│   ├── .gitkeep
│   ├── MSD/                          # Medical Segmentation Decathlon
│   │   ├── Task01_BrainTumour/
│   │   ├── Task02_Heart/
│   │   └── Task06_Lung/
│   └── CT-ORG/
│       └── ...
└── ExternalGoldStandards/            # Git-tracked (metadata only)
    ├── MSD_BrainTumor_001/
    │   ├── metadata.json             # Points to PublicDatasets/MSD/...
    │   └── reference_screenshots/
    └── CT-ORG_Bone_001/
        └── metadata.json
```

### Metadata for External Gold Standards

```json
{
  "name": "MSD_BrainTumor_001",
  "source": {
    "dataset": "Medical Segmentation Decathlon",
    "task": "Task01_BrainTumour",
    "case_id": "BRATS_001",
    "url": "http://medicaldecathlon.com/",
    "license": "CC-BY-SA 4.0"
  },
  "files": {
    "image": "PublicDatasets/MSD/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
    "label": "PublicDatasets/MSD/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz"
  },
  "label_mapping": {
    "1": "necrotic_core",
    "2": "edema",
    "3": "enhancing_tumor"
  },
  "segment_of_interest": "whole_tumor",
  "segment_labels": [1, 2, 3],
  "volume_info": {
    "modality": "MRI",
    "sequences": ["FLAIR", "T1w", "T1gd", "T2w"],
    "spacing": [1.0, 1.0, 1.0],
    "dimensions": [240, 240, 155]
  },
  "created": "2026-01-26",
  "description": "Brain tumor from BraTS challenge, whole tumor region"
}
```

## Implementation

### Download Script

Create `scripts/download_public_datasets.py`:

```python
#!/usr/bin/env python3
"""Download public datasets for gold standard testing.

Usage:
    python scripts/download_public_datasets.py --dataset msd --tasks brain lung
    python scripts/download_public_datasets.py --dataset ct-org
    python scripts/download_public_datasets.py --all
"""

import argparse
import hashlib
import json
import logging
import os
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Dataset URLs and checksums
MSD_TASKS = {
    "brain": {
        "name": "Task01_BrainTumour",
        "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        "size_gb": 7.5,
        "checksum": "...",  # Add actual checksum
    },
    "heart": {
        "name": "Task02_Heart",
        "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        "size_gb": 1.1,
    },
    "lung": {
        "name": "Task06_Lung",
        "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
        "size_gb": 0.3,
    },
}

CT_ORG_URL = "https://zenodo.org/record/7860267/files/CT-ORG.zip"

def download_msd_task(task_key: str, output_dir: Path):
    """Download a Medical Segmentation Decathlon task."""
    task = MSD_TASKS[task_key]
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / f"{task['name']}.tar"

    if (output_dir / task["name"]).exists():
        logger.info(f"Task {task['name']} already downloaded")
        return

    logger.info(f"Downloading {task['name']} ({task['size_gb']:.1f} GB)...")
    urllib.request.urlretrieve(task["url"], tar_path, reporthook=progress_hook)

    logger.info(f"Extracting {task['name']}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(output_dir)

    tar_path.unlink()  # Remove tar after extraction
    logger.info(f"Done: {task['name']}")

def create_gold_standard_metadata(
    dataset_dir: Path,
    external_gold_dir: Path,
    case_id: str,
    task_name: str,
    label_mapping: dict,
    segment_of_interest: str,
    segment_labels: list[int],
):
    """Create metadata.json for an external gold standard."""
    gold_dir = external_gold_dir / f"{task_name}_{case_id}"
    gold_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "name": f"{task_name}_{case_id}",
        "source": {
            "dataset": "Medical Segmentation Decathlon",
            "task": task_name,
            "case_id": case_id,
            "url": "http://medicaldecathlon.com/",
            "license": "CC-BY-SA 4.0",
        },
        "files": {
            "image": str(dataset_dir / task_name / "imagesTr" / f"{case_id}.nii.gz"),
            "label": str(dataset_dir / task_name / "labelsTr" / f"{case_id}.nii.gz"),
        },
        "label_mapping": label_mapping,
        "segment_of_interest": segment_of_interest,
        "segment_labels": segment_labels,
        "created": "2026-01-26",
    }

    (gold_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"Created metadata: {gold_dir / 'metadata.json'}")
```

### GoldStandardManager Extension

Extend `GoldStandardManager` to load external gold standards:

```python
class GoldStandardManager:
    def load_gold(self, name: str):
        """Load gold standard by name (internal or external)."""
        # Check internal first
        internal_path = self.gold_dir / name / "gold.seg.nrrd"
        if internal_path.exists():
            return self._load_internal(name)

        # Check external
        external_path = self.external_gold_dir / name / "metadata.json"
        if external_path.exists():
            return self._load_external(name)

        raise FileNotFoundError(f"Gold standard not found: {name}")

    def _load_external(self, name: str):
        """Load gold standard from public dataset."""
        metadata_path = self.external_gold_dir / name / "metadata.json"
        metadata = json.loads(metadata_path.read_text())

        # Load image and label from public dataset
        image_path = Path(metadata["files"]["image"])
        label_path = Path(metadata["files"]["label"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Public dataset not downloaded. Run:\n"
                f"  python scripts/download_public_datasets.py --dataset msd"
            )

        # Load in Slicer
        volume_node = slicer.util.loadVolume(str(image_path))
        label_node = slicer.util.loadLabelVolume(str(label_path))

        # Create segmentation from labels
        seg_node = self._label_to_segmentation(
            label_node,
            metadata["segment_labels"],
            metadata["segment_of_interest"]
        )

        return seg_node, metadata
```

### Recipe Integration

Update recipes to reference external gold standards:

```python
# recipes/msd_brain_tumor_001.py
recipe = Recipe(
    name="msd_brain_tumor_001",
    description="Brain tumor from MSD Task01",

    # Use external dataset instead of Slicer SampleData
    external_data={
        "gold_standard": "MSD_Task01_BRATS_001",
        "image": "PublicDatasets/MSD/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
    },

    segment_name="Tumor",
    actions=[...],
)
```

## Consequences

### Positive

- **Expert-annotated ground truth**: Professional radiologist annotations
- **Diverse anatomy**: Brain, lung, bone, cardiac coverage
- **Large sample size**: Hundreds of cases for statistical validation
- **Reproducible**: Public URLs, documented checksums
- **License-compatible**: CC-BY-SA allows derivative works

### Negative

- **Download size**: MSD is ~30GB total, requires disk space
- **Network dependency**: Initial download requires internet
- **Format conversion**: May need NIfTI → Slicer conversion
- **Subset selection**: Must choose representative cases from large datasets

### Mitigations

| Risk | Mitigation |
|------|------------|
| Large downloads | Download only needed tasks, not full dataset |
| Storage | Git-ignore data files, track only metadata |
| Conversion | Use Slicer's native NIfTI support |
| Case selection | Document selection criteria in metadata |

## Datasets Details

### Medical Segmentation Decathlon (MSD)

**URL**: http://medicaldecathlon.com/
**License**: CC-BY-SA 4.0
**Paper**: [Nature Communications](https://www.nature.com/articles/s41467-022-30695-9)

| Task | Modality | Target | Cases |
|------|----------|--------|-------|
| Task01_BrainTumour | MRI (4 sequences) | Glioma regions | 750 |
| Task02_Heart | MRI | Left atrium | 30 |
| Task06_Lung | CT | Lung tumors | 96 |

**Label structure (Brain Tumor)**:
- Label 1: Necrotic/non-enhancing tumor core
- Label 2: Peritumoral edema
- Label 3: GD-enhancing tumor

### CT-ORG

**URL**: https://www.nature.com/articles/s41597-020-00715-8
**License**: CC-BY 4.0
**Data**: https://zenodo.org/record/7860267

| Organ | Cases |
|-------|-------|
| Liver | 140 |
| Lungs | 140 |
| Bones | 140 |
| Bladder | 140 |
| Kidneys | 140 |
| Brain | 140 |

**Advantage**: Single dataset covers multiple anatomies including **bones** (missing from MSD).

### BraTS 2021

**URL**: https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/
**License**: CC-BY 4.0

More detailed brain tumor annotations than MSD Task01, with sub-regions:
- Enhancing tumor (ET)
- Tumor core (TC)
- Whole tumor (WT)

## Implementation Plan

### Phase 1: Infrastructure

1. Create download script with progress reporting
2. Add `PublicDatasets/` to `.gitignore`
3. Create `ExternalGoldStandards/` structure
4. Extend `GoldStandardManager` for external datasets

### Phase 2: Initial Dataset

1. Download MSD Task01 (Brain Tumor) - 7.5GB
2. Select 3 representative cases
3. Create metadata files
4. Verify loading in Slicer

### Phase 3: Full Coverage

1. Download remaining tasks (Lung, Heart)
2. Download CT-ORG for bone coverage
3. Create metadata for all missing gold standards
4. Update recipes to use external data

### Phase 4: Validation

1. Run regression tests against external gold standards
2. Compare optimization results
3. Document any quality differences from manual gold standards

## Alternatives Considered

### Create All Gold Standards Manually

**Rejected**: Time-consuming, subjective, limited expertise. Professional annotations are higher quality.

### Use Only Slicer SampleData

**Rejected**: No ground truth available for SampleData volumes.

### Download Full Datasets

**Rejected**: Unnecessary storage. Select representative cases instead.

### Store Data in Git LFS

**Rejected**: Too large even for LFS. Download on-demand is better.

## References

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- [CT-ORG Dataset](https://www.nature.com/articles/s41597-020-00715-8)
- [BraTS 2021](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/)
- [AWS Open Data - MSD](https://registry.opendata.aws/msd/)
- [ADR-013: Segmentation Recipes](ADR-013-segmentation-recipes.md)
- [ADR-010: Testing Framework](ADR-010-testing-framework.md)
