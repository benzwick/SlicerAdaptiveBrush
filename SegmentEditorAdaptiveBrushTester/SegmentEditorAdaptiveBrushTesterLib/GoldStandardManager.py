"""Gold standard segmentation management.

Provides functionality to save, load, and compare gold standard segmentations
for regression testing and optimization workflows.

Storage format:
- gold.seg.nrrd: Canonical format (git-tracked, small)
- metadata.json: Statistics + creation info (git-tracked)
- .dicom_cache/: Generated on-demand for CSE compatibility (git-ignored)
"""

from __future__ import annotations

import hashlib
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

# Import DicomManager from Reviewer module (for on-demand DICOM generation)
_reviewer_path = Path(__file__).parent.parent.parent / "SegmentEditorAdaptiveBrushReviewer"
if str(_reviewer_path) not in sys.path:
    sys.path.insert(0, str(_reviewer_path))


class GoldStandardManager:
    """Manage gold standard segmentations for regression testing.

    Gold standards are saved in the GoldStandards/ directory with:
    - gold.seg.nrrd: Canonical segmentation format (git-tracked)
    - metadata.json: Creation info, parameters, statistics
    - reference_screenshots/: Visual references

    DICOM SEG files are generated on-demand in .dicom_cache/ for
    CrossSegmentationExplorer compatibility.

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
        """Get or create DicomManager instance (lazy load for on-demand DICOM)."""
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
        """Save current segmentation as new gold standard (.seg.nrrd format).

        Automatically computes and saves statistics for change detection.

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
        import slicer

        gold_path = self.GOLD_DIR / name
        gold_path.mkdir(parents=True, exist_ok=True)

        # Save segmentation as .seg.nrrd
        seg_file = gold_path / "gold.seg.nrrd"
        slicer.util.saveNode(segmentation_node, str(seg_file))
        logger.info(f"Saved segmentation to {seg_file}")

        # Compute statistics
        statistics = self._compute_statistics(segmentation_node, segment_id, volume_node)

        # Get volume info
        volume_info = {
            "name": volume_node.GetName(),
            "spacing": list(volume_node.GetSpacing()),
            "dimensions": list(volume_node.GetImageData().GetDimensions()),
        }

        # Build metadata
        metadata = {
            "created": datetime.now().isoformat(),
            "sample_data": volume_node.GetName(),
            "volume": volume_info,
            "segment_id": segment_id,
            "description": description,
            "algorithm": algorithm,
            "parameters": parameters or {},
            "clicks": click_locations,
            "statistics": statistics,
        }

        metadata_file = gold_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        # Create screenshots directory
        (gold_path / "reference_screenshots").mkdir(exist_ok=True)

        logger.info(
            f"Gold standard saved: {name} "
            f"({statistics['voxel_count']:,} voxels, "
            f"{statistics['volume_mm3']:.1f} mm³)"
        )
        return gold_path

    def load_gold(self, name: str, volume_node=None, verify: bool = True) -> tuple:
        """Load gold standard segmentation and metadata.

        Args:
            name: Name of the gold standard to load.
            volume_node: Reference volume for verification (optional but recommended).
            verify: If True (default), verify checksum matches metadata.

        Returns:
            Tuple of (segmentation_node, metadata_dict).

        Raises:
            FileNotFoundError: If gold standard does not exist.
            ValueError: If verification fails (checksum mismatch).
        """
        import slicer

        gold_path = self.GOLD_DIR / name

        if not gold_path.exists():
            raise FileNotFoundError(f"Gold standard not found: {name}")

        # Load metadata
        metadata_file = gold_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Load segmentation from .seg.nrrd
        seg_file = gold_path / "gold.seg.nrrd"
        if not seg_file.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_file}")

        seg_node = slicer.util.loadSegmentation(str(seg_file))
        logger.info(f"Loaded segmentation from {seg_file}")

        # Verify checksum if requested and volume provided
        if verify and volume_node is not None:
            stored_stats = metadata.get("statistics", {})
            stored_checksum = stored_stats.get("checksum_sha256")

            if stored_checksum:
                segment_id = metadata.get("segment_id", "Segment_1")
                current_stats = self._compute_statistics(seg_node, segment_id, volume_node)
                current_checksum = current_stats["checksum_sha256"]

                if current_checksum != stored_checksum:
                    logger.error(
                        f"CHECKSUM MISMATCH for {name}!\n"
                        f"  Stored:  {stored_checksum[:16]}...\n"
                        f"  Current: {current_checksum[:16]}...\n"
                        f"  Stored voxels:  {stored_stats.get('voxel_count', '?'):,}\n"
                        f"  Current voxels: {current_stats['voxel_count']:,}"
                    )
                    raise ValueError(
                        f"Gold standard '{name}' checksum mismatch! "
                        f"The segmentation file may have been modified or corrupted. "
                        f"Expected {stored_stats.get('voxel_count', '?'):,} voxels, "
                        f"found {current_stats['voxel_count']:,}."
                    )
                else:
                    logger.info(
                        f"Gold standard verified: {name} "
                        f"({stored_stats.get('voxel_count', 0):,} voxels, checksum OK)"
                    )
            else:
                logger.warning(
                    f"No checksum in metadata for {name}, skipping verification. "
                    f"Run update_statistics() to add checksum."
                )

        return seg_node, metadata

    def get_dicom_seg_path(self, name: str, volume_node=None) -> Path:
        """Get DICOM SEG path, generating if needed.

        Used for CrossSegmentationExplorer compatibility.

        Args:
            name: Name of the gold standard.
            volume_node: Reference volume (required if cache needs generation).

        Returns:
            Path to DICOM SEG file.

        Raises:
            FileNotFoundError: If gold standard does not exist.
            ValueError: If volume_node required but not provided.
        """
        gold_path = self.GOLD_DIR / name
        cache_dir = gold_path / ".dicom_cache"
        seg_path = cache_dir / "seg.dcm"

        if seg_path.exists():
            return seg_path

        # Need to generate DICOM cache
        if volume_node is None:
            raise ValueError(
                "volume_node required to generate DICOM cache. " "Load the reference volume first."
            )

        logger.info(f"Generating DICOM cache for {name}...")
        self._generate_dicom_cache(name, volume_node)
        return seg_path

    def _generate_dicom_cache(self, name: str, volume_node) -> None:
        """Generate DICOM cache for a gold standard.

        Args:
            name: Name of the gold standard.
            volume_node: Reference volume node.
        """
        import slicer

        gold_path = self.GOLD_DIR / name
        cache_dir = gold_path / ".dicom_cache"
        cache_dir.mkdir(exist_ok=True)

        # Load the segmentation
        seg_file = gold_path / "gold.seg.nrrd"
        seg_node = slicer.util.loadSegmentation(str(seg_file))

        try:
            dicom_manager = self._get_dicom_manager()

            # Create synthetic DICOM for volume
            volume_dicom_dir = cache_dir / "volume_dicom"
            volume_series_uid = dicom_manager.create_synthetic_dicom(
                volume_node=volume_node,
                patient_id=f"GoldStandard_{name}",
                study_description=f"Gold Standard: {name}",
                output_dir=volume_dicom_dir,
            )
            logger.info(f"Created synthetic DICOM, UID: {volume_series_uid}")

            # Export segmentation as DICOM SEG
            dicom_manager.export_segmentation_as_dicom_seg(
                segmentation_node=seg_node,
                reference_volume_node=volume_node,
                series_description=f"GoldStandard_{name}",
                output_dir=cache_dir,
                compression="RLELossless",
            )
            logger.info(f"Generated DICOM cache at {cache_dir}")

        finally:
            slicer.mrmlScene.RemoveNode(seg_node)

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
        seg_file = gold_path / "gold.seg.nrrd"
        metadata_file = gold_path / "metadata.json"
        return seg_file.exists() and metadata_file.exists()

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

    def update_statistics(self, name: str, segmentation_node, volume_node) -> dict:
        """Recompute and update statistics for a gold standard.

        Useful after modifying a gold standard segmentation.

        Args:
            name: Name of the gold standard.
            segmentation_node: Current segmentation node.
            volume_node: Reference volume node.

        Returns:
            The computed statistics dictionary.
        """
        gold_path = self.GOLD_DIR / name
        metadata_file = gold_path / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Gold standard not found: {name}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        segment_id = metadata.get("segment_id", "Segment_1")
        statistics = self._compute_statistics(segmentation_node, segment_id, volume_node)

        metadata["statistics"] = statistics
        metadata["statistics_updated"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Updated statistics for {name}")
        return statistics

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

    def _compute_statistics(self, segmentation_node, segment_id: str, volume_node) -> dict:
        """Compute statistics for a segmentation.

        Args:
            segmentation_node: Segmentation MRML node.
            segment_id: Segment ID within the segmentation.
            volume_node: Volume node providing geometry.

        Returns:
            Dictionary with statistics including voxel_count, volume_mm3,
            bounding_box_ijk, bounding_box_size_mm, centroid_ras, checksum_sha256.
        """
        import SimpleITK as sitk
        import sitkUtils
        import slicer

        # Get spacing for volume calculations
        spacing = volume_node.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

        # Export segmentation to labelmap
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        try:
            segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                segmentation_node,
                labelmap_node,
                slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
            )

            # Get numpy array
            labelmap_sitk = sitkUtils.PullVolumeFromSlicer(labelmap_node)
            labelmap_array = sitk.GetArrayFromImage(labelmap_sitk)

            # Find nonzero voxels
            nonzero_coords = np.argwhere(labelmap_array > 0)

            if len(nonzero_coords) == 0:
                return {
                    "voxel_count": 0,
                    "volume_mm3": 0.0,
                    "bounding_box_ijk": {"min": [0, 0, 0], "max": [0, 0, 0]},
                    "bounding_box_size_mm": [0.0, 0.0, 0.0],
                    "centroid_ras": [0.0, 0.0, 0.0],
                    "checksum_sha256": hashlib.sha256(labelmap_array.tobytes()).hexdigest(),
                }

            voxel_count = len(nonzero_coords)
            volume_mm3 = voxel_count * voxel_volume_mm3

            # Bounding box (sitk array is ZYX order)
            min_coords = nonzero_coords.min(axis=0)
            max_coords = nonzero_coords.max(axis=0)

            # Convert to IJK (X, Y, Z) order
            bbox_min_ijk = [int(min_coords[2]), int(min_coords[1]), int(min_coords[0])]
            bbox_max_ijk = [int(max_coords[2]), int(max_coords[1]), int(max_coords[0])]

            # Bounding box size in mm
            bbox_size_mm = [
                (bbox_max_ijk[0] - bbox_min_ijk[0] + 1) * spacing[0],
                (bbox_max_ijk[1] - bbox_min_ijk[1] + 1) * spacing[1],
                (bbox_max_ijk[2] - bbox_min_ijk[2] + 1) * spacing[2],
            ]

            # Centroid in IJK (mean of nonzero coordinates)
            centroid_zyx = nonzero_coords.mean(axis=0)
            centroid_ijk = [centroid_zyx[2], centroid_zyx[1], centroid_zyx[0]]

            # Convert centroid to RAS
            import vtk

            ijk_to_ras = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(ijk_to_ras)
            centroid_ras_homogeneous = ijk_to_ras.MultiplyPoint(
                [centroid_ijk[0], centroid_ijk[1], centroid_ijk[2], 1.0]
            )
            centroid_ras = [round(c, 2) for c in centroid_ras_homogeneous[:3]]

            # Checksum of labelmap data
            checksum = hashlib.sha256(labelmap_array.tobytes()).hexdigest()

            return {
                "voxel_count": int(voxel_count),
                "volume_mm3": round(volume_mm3, 2),
                "bounding_box_ijk": {"min": bbox_min_ijk, "max": bbox_max_ijk},
                "bounding_box_size_mm": [round(s, 2) for s in bbox_size_mm],
                "centroid_ras": centroid_ras,
                "checksum_sha256": checksum,
            }

        finally:
            slicer.mrmlScene.RemoveNode(labelmap_node)

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

    def compare_statistics(self, gold_stats: dict, trial_stats: dict) -> dict:
        """Compare trial statistics against gold standard.

        Args:
            gold_stats: Statistics from gold standard metadata.
            trial_stats: Statistics computed for a trial.

        Returns:
            Comparison metrics.
        """
        if gold_stats["voxel_count"] == 0:
            return {
                "voxel_count_diff": trial_stats["voxel_count"],
                "voxel_count_ratio": float("inf") if trial_stats["voxel_count"] > 0 else 1.0,
                "volume_diff_mm3": trial_stats["volume_mm3"],
                "centroid_distance_mm": 0.0,
                "bbox_iou": 0.0,
            }

        # Voxel count comparison
        voxel_diff = trial_stats["voxel_count"] - gold_stats["voxel_count"]
        voxel_ratio = trial_stats["voxel_count"] / gold_stats["voxel_count"]

        # Volume comparison
        volume_diff = trial_stats["volume_mm3"] - gold_stats["volume_mm3"]

        # Centroid distance
        gold_centroid = np.array(gold_stats["centroid_ras"])
        trial_centroid = np.array(trial_stats["centroid_ras"])
        centroid_dist = float(np.linalg.norm(trial_centroid - gold_centroid))

        # Bounding box IoU
        bbox_iou = self._compute_bbox_iou(
            gold_stats["bounding_box_ijk"],
            trial_stats["bounding_box_ijk"],
        )

        return {
            "voxel_count_diff": voxel_diff,
            "voxel_count_ratio": round(voxel_ratio, 3),
            "volume_diff_mm3": round(volume_diff, 2),
            "centroid_distance_mm": round(centroid_dist, 2),
            "bbox_iou": round(bbox_iou, 3),
        }

    def _compute_bbox_iou(self, bbox1: dict, bbox2: dict) -> float:
        """Compute intersection over union of two bounding boxes.

        Args:
            bbox1: First bounding box with "min" and "max" keys.
            bbox2: Second bounding box with "min" and "max" keys.

        Returns:
            IoU value between 0 and 1.
        """
        # Intersection
        inter_min = [max(bbox1["min"][i], bbox2["min"][i]) for i in range(3)]
        inter_max = [min(bbox1["max"][i], bbox2["max"][i]) for i in range(3)]

        # Check for no intersection
        if any(inter_min[i] > inter_max[i] for i in range(3)):
            return 0.0

        inter_vol = 1.0
        for i in range(3):
            inter_vol *= inter_max[i] - inter_min[i] + 1

        # Union = vol1 + vol2 - intersection
        vol1 = 1.0
        vol2 = 1.0
        for i in range(3):
            vol1 *= bbox1["max"][i] - bbox1["min"][i] + 1
            vol2 *= bbox2["max"][i] - bbox2["min"][i] + 1

        union_vol = vol1 + vol2 - inter_vol

        return inter_vol / union_vol if union_vol > 0 else 0.0

    def format_statistics(self, stats: dict) -> str:
        """Format statistics as human-readable string.

        Args:
            stats: Statistics dictionary.

        Returns:
            Formatted multi-line string.
        """
        lines = [
            f"Voxel count: {stats['voxel_count']:,}",
            f"Volume: {stats['volume_mm3']:.1f} mm³",
            f"Bounding box: {stats['bounding_box_size_mm']} mm",
            f"Centroid RAS: {stats['centroid_ras']}",
        ]
        return "\n".join(lines)

    def format_comparison(self, comparison: dict) -> str:
        """Format comparison as human-readable string.

        Args:
            comparison: Comparison dictionary from compare_statistics().

        Returns:
            Formatted multi-line string.
        """
        lines = [
            f"Voxel count: {comparison['voxel_count_ratio']:.1%} of gold ({comparison['voxel_count_diff']:+,})",
            f"Volume diff: {comparison['volume_diff_mm3']:+.1f} mm³",
            f"Centroid shift: {comparison['centroid_distance_mm']:.1f} mm",
            f"BBox IoU: {comparison['bbox_iou']:.1%}",
        ]
        return "\n".join(lines)
