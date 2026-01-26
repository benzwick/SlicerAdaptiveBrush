#!/usr/bin/env python3
"""Download public datasets for gold standard testing.

This script downloads publicly available medical image segmentation datasets
with expert-annotated ground truth. These can be used for:
- Regression testing
- Parameter optimization
- Algorithm validation

Datasets:
- Medical Segmentation Decathlon (MSD): Brain tumor, Lung, Heart
- CT-ORG: Multi-organ including bone

Usage:
    # Download specific MSD tasks
    python scripts/download_public_datasets.py --msd brain lung heart

    # Download CT-ORG dataset
    python scripts/download_public_datasets.py --ct-org

    # Download all datasets
    python scripts/download_public_datasets.py --all

    # List available datasets
    python scripts/download_public_datasets.py --list

    # Create metadata for gold standards after download
    python scripts/download_public_datasets.py --create-metadata
"""

import argparse
import json
import logging
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TESTER_DIR = PROJECT_DIR / "SegmentEditorAdaptiveBrushTester"
PUBLIC_DATASETS_DIR = TESTER_DIR / "PublicDatasets"
EXTERNAL_GOLD_DIR = TESTER_DIR / "ExternalGoldStandards"

# Medical Segmentation Decathlon tasks
# Source: https://registry.opendata.aws/msd/
MSD_BASE_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com"
MSD_TASKS: dict[str, dict[str, Any]] = {
    "brain": {
        "name": "Task01_BrainTumour",
        "display_name": "Brain Tumor (BraTS)",
        "url": f"{MSD_BASE_URL}/Task01_BrainTumour.tar",
        "size_gb": 7.5,
        "modality": "MRI",
        "sequences": ["FLAIR", "T1w", "T1gd", "T2w"],
        "labels": {
            "1": "necrotic_core",
            "2": "edema",
            "3": "enhancing_tumor",
        },
        "description": "Multimodal brain MRI with glioma segmentation from BraTS challenge",
    },
    "heart": {
        "name": "Task02_Heart",
        "display_name": "Heart (Left Atrium)",
        "url": f"{MSD_BASE_URL}/Task02_Heart.tar",
        "size_gb": 1.1,
        "modality": "MRI",
        "labels": {"1": "left_atrium"},
        "description": "Mono-modal MRI with left atrium segmentation",
    },
    "lung": {
        "name": "Task06_Lung",
        "display_name": "Lung Tumors",
        "url": f"{MSD_BASE_URL}/Task06_Lung.tar",
        "size_gb": 0.3,
        "modality": "CT",
        "labels": {"1": "lung_tumor"},
        "description": "CT scans with non-small cell lung cancer segmentation",
    },
}

# CT-ORG dataset
# Source: https://zenodo.org/record/7860267
CT_ORG: dict[str, Any] = {
    "name": "CT-ORG",
    "display_name": "CT-ORG Multi-Organ",
    "url": "https://zenodo.org/record/7860267/files/CT-ORG.zip",
    "size_gb": 10.0,
    "modality": "CT",
    "labels": {
        "1": "liver",
        "2": "bladder",
        "3": "lungs",
        "4": "kidneys",
        "5": "bones",
        "6": "brain",
    },
    "description": "CT scans with 6 organ segmentations including bones",
}

# Representative cases for gold standards
GOLD_STANDARD_CASES: dict[str, dict[str, Any]] = {
    "MSD_BrainTumor_001": {
        "dataset": "msd",
        "task": "brain",
        "case_id": "BRATS_001",
        "segment_of_interest": "whole_tumor",
        "segment_labels": [1, 2, 3],
        "maps_to": "MRBrainTumor2_tumor",
    },
    "MSD_BrainTumor_002": {
        "dataset": "msd",
        "task": "brain",
        "case_id": "BRATS_002",
        "segment_of_interest": "enhancing_tumor",
        "segment_labels": [3],
        "maps_to": None,
    },
    "MSD_Lung_001": {
        "dataset": "msd",
        "task": "lung",
        "case_id": "lung_001",
        "segment_of_interest": "lung_tumor",
        "segment_labels": [1],
        "maps_to": "CTChest_lung",
    },
    "MSD_Heart_001": {
        "dataset": "msd",
        "task": "heart",
        "case_id": "la_003",
        "segment_of_interest": "left_atrium",
        "segment_labels": [1],
        "maps_to": "CTACardio_left_ventricle",  # Close approximation
    },
    "CT_ORG_Bone_001": {
        "dataset": "ct-org",
        "case_id": "volume-0",
        "segment_of_interest": "bones",
        "segment_labels": [5],
        "maps_to": "CTChest_bone",
    },
    "CT_ORG_Lung_001": {
        "dataset": "ct-org",
        "case_id": "volume-0",
        "segment_of_interest": "lungs",
        "segment_labels": [3],
        "maps_to": None,
    },
}


def progress_hook(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()


def download_file(url: str, output_path: Path, expected_size_gb: Optional[float] = None) -> bool:
    """Download a file with progress reporting."""
    if output_path.exists():
        logger.info(f"File already exists: {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if expected_size_gb:
        logger.info(f"Downloading {output_path.name} (~{expected_size_gb:.1f} GB)...")
    else:
        logger.info(f"Downloading {output_path.name}...")

    try:
        urlretrieve(url, output_path, reporthook=progress_hook)
        return True
    except (URLError, HTTPError, OSError) as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_tar(tar_path: Path, output_dir: Path) -> bool:
    """Extract a tar archive."""
    logger.info(f"Extracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(output_dir)
        return True
    except (tarfile.TarError, OSError) as e:
        logger.error(f"Extraction failed: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract a zip archive."""
    logger.info(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        return True
    except (zipfile.BadZipFile, OSError) as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_msd_task(task_key: str) -> bool:
    """Download a Medical Segmentation Decathlon task."""
    if task_key not in MSD_TASKS:
        logger.error(f"Unknown MSD task: {task_key}")
        return False

    task = MSD_TASKS[task_key]
    output_dir = PUBLIC_DATASETS_DIR / "MSD"
    task_dir = output_dir / task["name"]

    # Check if already extracted
    if task_dir.exists() and (task_dir / "imagesTr").exists():
        logger.info(f"MSD {task['display_name']} already downloaded")
        return True

    # Download
    tar_path = output_dir / f"{task['name']}.tar"
    if not download_file(task["url"], tar_path, task["size_gb"]):
        return False

    # Extract
    if not extract_tar(tar_path, output_dir):
        return False

    # Clean up tar file
    tar_path.unlink()
    logger.info(f"MSD {task['display_name']} ready at {task_dir}")
    return True


def download_ct_org() -> bool:
    """Download the CT-ORG dataset."""
    output_dir = PUBLIC_DATASETS_DIR / "CT-ORG"

    # Check if already extracted
    if output_dir.exists() and any(output_dir.glob("volume-*.nii.gz")):
        logger.info("CT-ORG already downloaded")
        return True

    # Download
    zip_path = PUBLIC_DATASETS_DIR / "CT-ORG.zip"
    if not download_file(CT_ORG["url"], zip_path, CT_ORG["size_gb"]):
        return False

    # Extract
    if not extract_zip(zip_path, PUBLIC_DATASETS_DIR):
        return False

    # Clean up zip file
    zip_path.unlink()
    logger.info(f"CT-ORG ready at {output_dir}")
    return True


def create_metadata_file(gold_name: str, case_config: dict) -> bool:
    """Create metadata.json for an external gold standard."""
    gold_dir = EXTERNAL_GOLD_DIR / gold_name
    gold_dir.mkdir(parents=True, exist_ok=True)

    # Determine file paths based on dataset
    if case_config["dataset"] == "msd":
        task = MSD_TASKS[case_config["task"]]
        task_dir = PUBLIC_DATASETS_DIR / "MSD" / task["name"]
        case_id = case_config["case_id"]

        image_path = task_dir / "imagesTr" / f"{case_id}.nii.gz"
        label_path = task_dir / "labelsTr" / f"{case_id}.nii.gz"
        labels = task["labels"]
        modality = task["modality"]
        source_url = "http://medicaldecathlon.com/"
        license_info = "CC-BY-SA 4.0"
        dataset_name = f"Medical Segmentation Decathlon - {task['display_name']}"

    elif case_config["dataset"] == "ct-org":
        ct_org_dir = PUBLIC_DATASETS_DIR / "CT-ORG"
        case_id = case_config["case_id"]

        image_path = ct_org_dir / f"{case_id}.nii.gz"
        label_path = ct_org_dir / "labels" / f"{case_id}.nii.gz"
        labels = CT_ORG["labels"]
        modality = CT_ORG["modality"]
        source_url = "https://zenodo.org/record/7860267"
        license_info = "CC-BY 4.0"
        dataset_name = "CT-ORG Multi-Organ"

    else:
        logger.error(f"Unknown dataset: {case_config['dataset']}")
        return False

    metadata = {
        "name": gold_name,
        "source": {
            "dataset": dataset_name,
            "case_id": case_config["case_id"],
            "url": source_url,
            "license": license_info,
        },
        "files": {
            "image": str(image_path.relative_to(PROJECT_DIR)),
            "label": str(label_path.relative_to(PROJECT_DIR)),
        },
        "label_mapping": labels,
        "segment_of_interest": case_config["segment_of_interest"],
        "segment_labels": case_config["segment_labels"],
        "maps_to_recipe": case_config.get("maps_to"),
        "modality": modality,
        "created": datetime.now().isoformat(),
    }

    metadata_path = gold_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Create reference_screenshots directory
    (gold_dir / "reference_screenshots").mkdir(exist_ok=True)

    logger.info(f"Created metadata: {metadata_path}")
    return True


def create_all_metadata():
    """Create metadata files for all configured gold standards."""
    logger.info("Creating metadata files for gold standards...")

    for gold_name, config in GOLD_STANDARD_CASES.items():
        # Check if dataset is downloaded
        if config["dataset"] == "msd":
            task = MSD_TASKS[config["task"]]
            task_dir = PUBLIC_DATASETS_DIR / "MSD" / task["name"]
            if not task_dir.exists():
                logger.warning(f"Skipping {gold_name}: MSD {config['task']} not downloaded")
                continue

        elif config["dataset"] == "ct-org":
            ct_org_dir = PUBLIC_DATASETS_DIR / "CT-ORG"
            if not ct_org_dir.exists():
                logger.warning(f"Skipping {gold_name}: CT-ORG not downloaded")
                continue

        create_metadata_file(gold_name, config)


def list_datasets():
    """List available datasets and their status."""
    print("\n=== Medical Segmentation Decathlon (MSD) ===")
    print("License: CC-BY-SA 4.0")
    print("Source: http://medicaldecathlon.com/")
    print()

    for key, task in MSD_TASKS.items():
        task_dir = PUBLIC_DATASETS_DIR / "MSD" / task["name"]
        status = "Downloaded" if task_dir.exists() else "Not downloaded"
        print(f"  {key:8} - {task['display_name']:25} ({task['size_gb']:.1f} GB) [{status}]")
        print(f"            {task['description']}")
        print(f"            Labels: {', '.join(task['labels'].values())}")
        print()

    print("\n=== CT-ORG Multi-Organ ===")
    print("License: CC-BY 4.0")
    print("Source: https://zenodo.org/record/7860267")
    print()

    ct_org_dir = PUBLIC_DATASETS_DIR / "CT-ORG"
    status = "Downloaded" if ct_org_dir.exists() else "Not downloaded"
    print(f"  ct-org   - {CT_ORG['display_name']:25} ({CT_ORG['size_gb']:.1f} GB) [{status}]")
    print(f"            {CT_ORG['description']}")
    print(f"            Labels: {', '.join(CT_ORG['labels'].values())}")
    print()

    print("\n=== Gold Standard Mappings ===")
    print("These cases will be used as gold standards after download:\n")

    for gold_name, config in GOLD_STANDARD_CASES.items():
        maps_to = config.get("maps_to", "N/A")
        print(f"  {gold_name:25} -> {maps_to or '(standalone)'}")


def main():
    parser = argparse.ArgumentParser(
        description="Download public datasets for gold standard testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download brain tumor data only
    python scripts/download_public_datasets.py --msd brain

    # Download multiple MSD tasks
    python scripts/download_public_datasets.py --msd brain lung heart

    # Download CT-ORG for bone segmentation
    python scripts/download_public_datasets.py --ct-org

    # Download everything
    python scripts/download_public_datasets.py --all

    # List available datasets
    python scripts/download_public_datasets.py --list
        """,
    )

    parser.add_argument(
        "--msd",
        nargs="+",
        choices=list(MSD_TASKS.keys()),
        help="Download specific MSD tasks",
    )
    parser.add_argument(
        "--ct-org",
        action="store_true",
        help="Download CT-ORG dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and their status",
    )
    parser.add_argument(
        "--create-metadata",
        action="store_true",
        help="Create metadata files for gold standards",
    )

    args = parser.parse_args()

    # List datasets
    if args.list:
        list_datasets()
        return

    # Create directories
    PUBLIC_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_GOLD_DIR.mkdir(parents=True, exist_ok=True)

    success = True

    # Download MSD tasks
    if args.all or args.msd:
        tasks = list(MSD_TASKS.keys()) if args.all else args.msd
        for task in tasks:
            if not download_msd_task(task):
                success = False

    # Download CT-ORG
    if args.all or args.ct_org:
        if not download_ct_org():
            success = False

    # Create metadata
    if args.create_metadata or args.all:
        create_all_metadata()

    if success:
        logger.info("All downloads completed successfully")
        logger.info(f"Data location: {PUBLIC_DATASETS_DIR}")
        logger.info(f"Metadata location: {EXTERNAL_GOLD_DIR}")
    else:
        logger.error("Some downloads failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
