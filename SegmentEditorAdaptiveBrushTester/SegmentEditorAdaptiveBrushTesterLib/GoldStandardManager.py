"""Gold standard segmentation management.

Provides functionality to save, load, and compare gold standard segmentations
for regression testing and optimization workflows.

All gold standards are stored as DICOM SEG (no .seg.nrrd support).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import DicomManager from Reviewer module
_reviewer_path = Path(__file__).parent.parent.parent / "SegmentEditorAdaptiveBrushReviewer"
if str(_reviewer_path) not in sys.path:
    sys.path.insert(0, str(_reviewer_path))


class GoldStandardManager:
    """Manage gold standard segmentations for regression testing.

    Gold standards are saved in the GoldStandards/ directory with:
    - dicom/: DICOM SEG files
    - metadata.json: Creation info, parameters, click locations, DICOM UIDs
    - reference_screenshots/: Visual references

    Usage:
        manager = GoldStandardManager()

        # Save a gold standard
        manager.save_as_gold(
            segmentation_node=seg_node,
            volume_node=vol_node,
            segment_id="Tumor",
            name="MRBrainTumor1_tumor",
            click_locations=[{"ras": [5.6, -29.5, 28.4], ...}],
            description="Tumor segmentation using watershed"
        )

        # Load a gold standard
        gold_seg, metadata = manager.load_gold("MRBrainTumor1_tumor")

        # List available gold standards
        standards = manager.list_gold_standards()
    """

    # Gold standards directory relative to this file's package
    GOLD_DIR = Path(__file__).parent.parent / "GoldStandards"

    def __init__(self) -> None:
        """Initialize gold standard manager."""
        self.GOLD_DIR.mkdir(exist_ok=True)
        self._dicom_manager = None

    def _get_dicom_manager(self):
        """Get or create DicomManager instance."""
        if self._dicom_manager is None:
            from SegmentEditorAdaptiveBrushReviewerLib import DicomManager

            self._dicom_manager = DicomManager()
            if not self._dicom_manager.ensure_database_initialized():
                raise RuntimeError("DICOM database initialization failed")
        return self._dicom_manager

    def save_as_gold(
        self,
        segmentation_node,
        volume_node,
        segment_id: str,
        name: str,
        click_locations: list[dict],
        description: str = "",
        algorithm: str = "",
        parameters: dict | None = None,
    ) -> Path:
        """Save current segmentation as new gold standard (DICOM SEG format).

        Args:
            segmentation_node: Segmentation MRML node to save.
            volume_node: Volume node providing geometry.
            segment_id: Segment ID within the segmentation.
            name: Name for the gold standard (used as directory name).
            click_locations: List of click locations with parameters.
                Format: [{"ras": [R, A, S], "ijk": [I, J, K], "params": {...}}, ...]
            description: Human-readable description of this gold standard.
            algorithm: Algorithm used to create the segmentation.
            parameters: Algorithm parameters used.

        Returns:
            Path to the gold standard directory.
        """
        gold_path = self.GOLD_DIR / name
        gold_path.mkdir(parents=True, exist_ok=True)

        # Save segmentation as DICOM SEG
        dicom_manager = self._get_dicom_manager()
        dicom_dir = gold_path / "dicom"

        dicom_series_uid = dicom_manager.export_segmentation_as_dicom_seg(
            segmentation_node=segmentation_node,
            reference_volume_node=volume_node,
            series_description=f"GoldStandard_{name}",
            output_dir=dicom_dir,
            segment_metadata={
                "algorithm": algorithm,
                "description": description,
                "is_gold_standard": True,
            },
        )
        logger.info(f"Saved DICOM SEG to {dicom_dir}, UID: {dicom_series_uid}")

        # Count voxels
        voxel_count = self._count_voxels(segmentation_node, segment_id, volume_node)

        # Get volume info
        volume_info = {
            "name": volume_node.GetName(),
            "spacing": list(volume_node.GetSpacing()),
            "dimensions": list(volume_node.GetImageData().GetDimensions()),
        }

        # Save metadata with DICOM info
        metadata = {
            "created": datetime.now().isoformat(),
            "volume": volume_info,
            "segment_id": segment_id,
            "description": description,
            "algorithm": algorithm,
            "parameters": parameters or {},
            "clicks": click_locations,
            "voxel_count": voxel_count,
            "dicom": {
                "series_uid": dicom_series_uid,
                "seg_path": "dicom",
            },
        }

        metadata_file = gold_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        # Create screenshots directory
        (gold_path / "reference_screenshots").mkdir(exist_ok=True)

        logger.info(f"Gold standard saved: {name} ({voxel_count} voxels)")
        return gold_path

    def load_gold(self, name: str) -> tuple:
        """Load gold standard segmentation and metadata.

        Args:
            name: Name of the gold standard to load.

        Returns:
            Tuple of (segmentation_node, metadata_dict).

        Raises:
            FileNotFoundError: If gold standard does not exist.
        """
        import slicer
        from DICOMLib import DICOMUtils

        gold_path = self.GOLD_DIR / name

        if not gold_path.exists():
            raise FileNotFoundError(f"Gold standard not found: {name}")

        # Load metadata first to get DICOM path
        metadata_file = gold_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Load segmentation from DICOM SEG
        dicom_info = metadata.get("dicom", {})
        dicom_seg_path = gold_path / dicom_info.get("seg_path", "dicom")

        if not dicom_seg_path.exists():
            raise FileNotFoundError(f"DICOM SEG directory not found: {dicom_seg_path}")

        # Find DICOM file in directory
        dicom_files = list(dicom_seg_path.glob("*.dcm"))
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files in: {dicom_seg_path}")

        # Import to DICOM database
        indexer = DICOMUtils.importDicomToDatabase(str(dicom_seg_path))
        if indexer:
            indexer.waitForImportFinished()

        # Load the segmentation
        seg_node = slicer.util.loadNodeFromFile(
            str(dicom_files[0]),
            "DICOMSegmentationFile",
        )
        logger.info(f"Loaded DICOM segmentation from {dicom_seg_path}")

        return seg_node, metadata

    def list_gold_standards(self) -> list[dict]:
        """List all available gold standards with metadata.

        Returns:
            List of metadata dictionaries, each with added "name" field.
        """
        standards = []

        for path in sorted(self.GOLD_DIR.iterdir()):
            if not path.is_dir():
                continue

            metadata_file = path / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file) as f:
                    meta = json.load(f)
                    meta["name"] = path.name
                    meta["path"] = str(path)
                    standards.append(meta)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid metadata in {path}: {e}")

        return standards

    def gold_exists(self, name: str) -> bool:
        """Check if a gold standard exists.

        Args:
            name: Name of the gold standard.

        Returns:
            True if exists, False otherwise.
        """
        gold_path = self.GOLD_DIR / name
        dicom_dir = gold_path / "dicom"
        has_dicom = dicom_dir.exists() and any(dicom_dir.glob("*.dcm"))
        return has_dicom and (gold_path / "metadata.json").exists()

    def get_gold_path(self, name: str) -> Path:
        """Get path to a gold standard directory.

        Args:
            name: Name of the gold standard.

        Returns:
            Path to the gold standard directory.
        """
        return self.GOLD_DIR / name

    def update_metadata(self, name: str, updates: dict) -> None:
        """Update metadata for an existing gold standard.

        Args:
            name: Name of the gold standard.
            updates: Dictionary of metadata fields to update.
        """
        gold_path = self.GOLD_DIR / name
        metadata_file = gold_path / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Gold standard not found: {name}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        metadata.update(updates)
        metadata["last_modified"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Updated metadata for {name}")

    def delete_gold(self, name: str) -> None:
        """Delete a gold standard.

        Args:
            name: Name of the gold standard to delete.
        """
        import shutil

        gold_path = self.GOLD_DIR / name

        if not gold_path.exists():
            raise FileNotFoundError(f"Gold standard not found: {name}")

        shutil.rmtree(gold_path)
        logger.info(f"Deleted gold standard: {name}")

    def _count_voxels(self, segmentation_node, segment_id: str, volume_node) -> int:
        """Count the number of voxels in a segment.

        Args:
            segmentation_node: Segmentation MRML node.
            segment_id: Segment ID within the segmentation.
            volume_node: Volume node providing geometry.

        Returns:
            Number of foreground voxels.
        """
        import slicer

        try:
            arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentation_node, segment_id, volume_node
            )
            return int(np.sum(arr > 0))
        except Exception as e:
            logger.warning(f"Could not count voxels: {e}")
            return 0

    def save_reference_screenshot(
        self,
        name: str,
        screenshot_path: Path | str,
        description: str = "",
    ) -> Path:
        """Save a reference screenshot for a gold standard.

        Args:
            name: Name of the gold standard.
            screenshot_path: Path to the screenshot file.
            description: Description of what the screenshot shows.

        Returns:
            Path to the saved screenshot.
        """
        import shutil

        gold_path = self.GOLD_DIR / name
        screenshots_dir = gold_path / "reference_screenshots"
        screenshots_dir.mkdir(exist_ok=True)

        src = Path(screenshot_path)
        dst = screenshots_dir / src.name

        shutil.copy2(src, dst)

        # Update manifest
        manifest_file = screenshots_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
        else:
            manifest = {"screenshots": []}

        manifest["screenshots"].append(
            {
                "filename": src.name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
            }
        )

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved reference screenshot: {dst}")
        return dst
