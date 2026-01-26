"""Sample data sources for gold standard datasets.

This module registers public segmentation datasets as Slicer sample data,
making them available via File > Download Sample Data menu.

Datasets are downloaded from public repositories:
- Medical Segmentation Decathlon (CC-BY-SA 4.0)
- CT-ORG (CC-BY 4.0)
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for sample data source dictionary
SampleDataSource = dict[str, Any]

# Sample data definitions
# Each entry creates a downloadable sample with image + segmentation
SAMPLE_DATA_SOURCES: list[SampleDataSource] = [
    {
        "category": "Adaptive Brush Gold Standards",
        "name": "MSD_BrainTumor_001",
        "display_name": "MSD Brain Tumor 001",
        "description": "Brain glioma from Medical Segmentation Decathlon (BraTS)",
        "image_url": "https://github.com/benzwick/SlicerAdaptiveBrushData/releases/download/v1.0/MSD_BrainTumor_001_image.nrrd",
        "label_url": "https://github.com/benzwick/SlicerAdaptiveBrushData/releases/download/v1.0/MSD_BrainTumor_001_label.nrrd",
        "image_checksum": None,  # Add after hosting
        "label_checksum": None,
        "license": "CC-BY-SA 4.0",
        "source": "Medical Segmentation Decathlon",
    },
    {
        "category": "Adaptive Brush Gold Standards",
        "name": "MSD_Lung_001",
        "display_name": "MSD Lung Tumor 001",
        "description": "Lung tumor from Medical Segmentation Decathlon",
        "image_url": "https://github.com/benzwick/SlicerAdaptiveBrushData/releases/download/v1.0/MSD_Lung_001_image.nrrd",
        "label_url": "https://github.com/benzwick/SlicerAdaptiveBrushData/releases/download/v1.0/MSD_Lung_001_label.nrrd",
        "license": "CC-BY-SA 4.0",
        "source": "Medical Segmentation Decathlon",
    },
    {
        "category": "Adaptive Brush Gold Standards",
        "name": "CT_ORG_Bone_001",
        "display_name": "CT-ORG Bone 001",
        "description": "CT with bone segmentation from CT-ORG",
        "image_url": "https://github.com/benzwick/SlicerAdaptiveBrushData/releases/download/v1.0/CT_ORG_Bone_001_image.nrrd",
        "label_url": "https://github.com/benzwick/SlicerAdaptiveBrushData/releases/download/v1.0/CT_ORG_Bone_001_label.nrrd",
        "license": "CC-BY 4.0",
        "source": "CT-ORG",
    },
]


def registerSampleDataSources():
    """Register gold standard datasets as Slicer sample data sources.

    Call this function when the SampleData module is discovered.
    After registration, datasets appear in File > Download Sample Data.
    """
    # Get the icon path
    icon_dir = Path(__file__).parent.parent / "Resources" / "Icons"
    default_icon = icon_dir / "SegmentEditorAdaptiveBrush.png"

    for source in SAMPLE_DATA_SOURCES:
        _registerSingleSource(source, default_icon)


def _registerSingleSource(source: dict, icon_path: Path):
    """Register a single sample data source."""
    import SampleData

    # Create custom loader that downloads both image and segmentation
    icon_str = str(icon_path) if icon_path.exists() else ""

    # Register the image volume
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category=source.get("category", "Adaptive Brush"),
        sampleName=source["name"],
        sampleDescription=source.get("description", ""),
        uris=source["image_url"],
        fileNames=f"{source['name']}_image.nrrd",
        checksums=source.get("image_checksum", ""),
        nodeNames=source["display_name"],
        thumbnailFileName=icon_str,
        loadFileType="VolumeFile",
    )

    # Register the segmentation label
    label_name = f"{source['name']}_Segmentation"
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category=source.get("category", "Adaptive Brush"),
        sampleName=label_name,
        sampleDescription=f"Ground truth segmentation for {source['display_name']}",
        uris=source["label_url"],
        fileNames=f"{source['name']}_label.nrrd",
        checksums=source.get("label_checksum", ""),
        nodeNames=f"{source['display_name']} Segmentation",
        thumbnailFileName=icon_str,
        loadFileType="SegmentationFile",
    )

    logger.debug(f"Registered sample data: {source['name']}")


def downloadGoldStandard(name: str):
    """Download a gold standard dataset by name.

    Args:
        name: The sample data name (e.g., "MSD_BrainTumor_001")

    Returns:
        Tuple of (volumeNode, segmentationNode).

    Raises:
        ImportError: If SampleData module not available.
        RuntimeError: If download fails.
    """
    import SampleData

    logic = SampleData.SampleDataLogic()

    # Download image
    volume_node = logic.downloadSample(name)
    if volume_node is None:
        raise RuntimeError(f"Failed to download image for {name}")

    # Download segmentation (optional - may not exist)
    seg_name = f"{name}_Segmentation"
    seg_node = logic.downloadSample(seg_name)

    return volume_node, seg_node


def listAvailableGoldStandards() -> list[dict]:
    """List all available gold standard sample data.

    Returns:
        List of dictionaries with sample data information
    """
    return [
        {
            "name": s["name"],
            "display_name": s["display_name"],
            "description": s.get("description", ""),
            "license": s.get("license", ""),
            "source": s.get("source", ""),
        }
        for s in SAMPLE_DATA_SOURCES
    ]
