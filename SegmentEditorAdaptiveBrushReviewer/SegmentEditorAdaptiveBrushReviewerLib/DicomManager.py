"""DICOM database management for optimization results.

Provides functionality to:
- Create synthetic DICOM series from non-DICOM volumes (SampleData)
- Export segmentations as DICOM SEG
- Query DICOM database for segmentation relationships
- Load segmentations from DICOM database

Requires QuantitativeReporting extension for DICOM SEG export.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DicomManagerError(Exception):
    """Base exception for DicomManager errors."""

    pass


class DicomDatabaseNotAvailable(DicomManagerError):
    """Raised when DICOM database is not initialized."""

    pass


class QuantitativeReportingNotAvailable(DicomManagerError):
    """Raised when QuantitativeReporting extension is not installed."""

    pass


class DicomManager:
    """Manage DICOM database operations for optimization results.

    This class provides a high-level interface for:
    - Converting SampleData volumes to synthetic DICOM
    - Exporting segmentations as DICOM SEG
    - Loading segmentations from DICOM database

    Usage:
        manager = DicomManager()

        # Create synthetic DICOM from SampleData volume
        volume_series_uid = manager.create_synthetic_dicom(
            volume_node=volume_node,
            patient_id="AdaptiveBrush_Test",
            study_description="Optimization Run",
            output_dir=Path("output/dicom/volume")
        )

        # Export segmentation as DICOM SEG
        seg_series_uid = manager.export_segmentation_as_dicom_seg(
            segmentation_node=seg_node,
            reference_volume_node=volume_node,
            series_description="trial_001_watershed",
            output_dir=Path("output/dicom/segmentations")
        )

        # Load segmentation from DICOM database
        seg_node = manager.load_segmentation_by_uid(seg_series_uid)
    """

    # DICOM UID prefix for generated UIDs
    # Using a fictional root (2.25) + UUID-based generation
    UID_PREFIX = "2.25"

    def __init__(self) -> None:
        """Initialize DICOM manager."""
        self._db = None
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        try:
            import slicer

            self._db = slicer.dicomDatabase
        except ImportError as err:
            raise DicomManagerError("Not running inside Slicer") from err

    @property
    def database(self):
        """Get Slicer DICOM database.

        Returns:
            ctkDICOMDatabase instance.

        Raises:
            DicomDatabaseNotAvailable: If database is not initialized.
        """
        import slicer

        db = slicer.dicomDatabase
        if db is None or not db.isOpen:
            raise DicomDatabaseNotAvailable(
                "DICOM database is not available. Open the DICOM module first."
            )
        return db

    def ensure_database_initialized(self) -> bool:
        """Ensure DICOM database is available and initialized.

        If database is not open, attempts to open/create a default one.

        Returns:
            True if database is ready, False otherwise.
        """
        import slicer

        db = slicer.dicomDatabase
        if db is not None and db.isOpen:
            return True

        # Try to initialize default database
        try:
            import DICOMLib

            if not slicer.dicomDatabase:
                # Create default database in Slicer settings directory
                settings_dir = Path(slicer.app.slicerHome) / "DICOM"
                settings_dir.mkdir(exist_ok=True)
                db_path = str(settings_dir / "ctkDICOM.sql")

                DICOMLib.DICOMUtils.openDatabase(db_path)
                logger.info(f"Initialized DICOM database at: {db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DICOM database: {e}")
            return False

    def generate_uid(self) -> str:
        """Generate a valid DICOM UID.

        Uses UUID-based generation with the 2.25 prefix for universally
        unique identifiers.

        Returns:
            A valid DICOM UID string.
        """
        # Convert UUID to integer and use as UID suffix
        uid_int = uuid.uuid4().int
        return f"{self.UID_PREFIX}.{uid_int}"

    def create_synthetic_dicom(
        self,
        volume_node,
        patient_id: str,
        study_description: str,
        output_dir: Path,
        patient_name: str | None = None,
        study_date: str | None = None,
        series_description: str | None = None,
    ) -> str:
        """Create synthetic DICOM series from a non-DICOM volume.

        Converts a volume node (e.g., from SampleData) to DICOM format,
        saves to disk, and imports into the DICOM database.

        Args:
            volume_node: vtkMRMLScalarVolumeNode to convert.
            patient_id: DICOM PatientID tag value.
            study_description: DICOM StudyDescription tag value.
            output_dir: Directory to save DICOM files.
            patient_name: DICOM PatientName (defaults to patient_id).
            study_date: DICOM StudyDate (defaults to today).
            series_description: DICOM SeriesDescription (defaults to volume name).

        Returns:
            SeriesInstanceUID of the created DICOM series.

        Raises:
            DicomManagerError: If export fails.
        """
        import slicer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate UIDs
        study_uid = self.generate_uid()
        series_uid = self.generate_uid()

        # Set defaults
        patient_name = patient_name or patient_id
        study_date = study_date or datetime.now().strftime("%Y%m%d")
        series_description = series_description or volume_node.GetName()

        logger.info(f"Creating synthetic DICOM for volume: {volume_node.GetName()}")
        logger.debug(f"  PatientID: {patient_id}")
        logger.debug(f"  StudyUID: {study_uid}")
        logger.debug(f"  SeriesUID: {series_uid}")

        try:
            # Get subject hierarchy for the volume
            subject_hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                slicer.mrmlScene
            )

            # Get subject hierarchy item for volume
            volume_item = subject_hierarchy.GetItemByDataNode(volume_node)
            if not volume_item:
                raise DicomManagerError("Volume not in subject hierarchy")

            # Export using DICOMScalarVolumePlugin
            from DICOMScalarVolumePlugin import DICOMScalarVolumePluginClass

            plugin = DICOMScalarVolumePluginClass()

            # Create exportable
            exportables = plugin.examineForExport(volume_item)
            if not exportables:
                raise DicomManagerError("Volume cannot be exported as DICOM")

            exportable = exportables[0]
            exportable.directory = str(output_dir)
            exportable.setTag("PatientID", patient_id)
            exportable.setTag("PatientName", patient_name)
            exportable.setTag("StudyDescription", study_description)
            exportable.setTag("StudyDate", study_date)
            exportable.setTag("SeriesDescription", series_description)
            exportable.setTag("StudyInstanceUID", study_uid)
            exportable.setTag("SeriesInstanceUID", series_uid)

            # Export
            plugin.export(exportable)

            logger.info(f"Exported DICOM to: {output_dir}")

            # Import into database
            self._import_dicom_folder(output_dir)

            # Store UIDs as attributes on the volume node
            volume_node.SetAttribute("DICOM.StudyInstanceUID", study_uid)
            volume_node.SetAttribute("DICOM.SeriesInstanceUID", series_uid)

            return series_uid

        except Exception as e:
            logger.exception(f"Failed to create synthetic DICOM: {e}")
            raise DicomManagerError(f"Failed to create synthetic DICOM: {e}") from e

    def export_segmentation_as_dicom_seg(
        self,
        segmentation_node,
        reference_volume_node,
        series_description: str,
        output_dir: Path,
        segment_metadata: dict | None = None,
    ) -> str:
        """Export segmentation as DICOM SEG.

        Requires QuantitativeReporting extension for proper DICOM SEG encoding.

        Args:
            segmentation_node: vtkMRMLSegmentationNode to export.
            reference_volume_node: Reference volume (must have DICOM attributes).
            series_description: DICOM SeriesDescription for the SEG.
            output_dir: Directory to save DICOM SEG file.
            segment_metadata: Optional metadata to include (algorithm params, etc.).

        Returns:
            SeriesInstanceUID of the created DICOM SEG.

        Raises:
            QuantitativeReportingNotAvailable: If extension not installed.
            DicomManagerError: If export fails.
        """
        import slicer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for QuantitativeReporting
        try:
            from DICOMSegmentationPlugin import DICOMSegmentationPluginClass
        except ImportError as err:
            raise QuantitativeReportingNotAvailable(
                "QuantitativeReporting extension is required for DICOM SEG export. "
                "Install it from the Extension Manager."
            ) from err

        # Get reference volume DICOM UIDs
        study_uid = reference_volume_node.GetAttribute("DICOM.StudyInstanceUID")
        ref_series_uid = reference_volume_node.GetAttribute("DICOM.SeriesInstanceUID")

        if not study_uid or not ref_series_uid:
            raise DicomManagerError(
                "Reference volume does not have DICOM attributes. "
                "Create synthetic DICOM first using create_synthetic_dicom()."
            )

        # Generate new series UID for segmentation
        seg_series_uid = self.generate_uid()

        logger.info(f"Exporting segmentation as DICOM SEG: {segmentation_node.GetName()}")
        logger.debug(f"  Reference SeriesUID: {ref_series_uid}")
        logger.debug(f"  SEG SeriesUID: {seg_series_uid}")

        try:
            # Get subject hierarchy
            subject_hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                slicer.mrmlScene
            )
            seg_item = subject_hierarchy.GetItemByDataNode(segmentation_node)

            if not seg_item:
                raise DicomManagerError("Segmentation not in subject hierarchy")

            # Use DICOMSegmentationPlugin for export
            plugin = DICOMSegmentationPluginClass()

            # Create exportable
            exportables = plugin.examineForExport(seg_item)
            if not exportables:
                raise DicomManagerError("Segmentation cannot be exported as DICOM SEG")

            exportable = exportables[0]
            exportable.directory = str(output_dir)
            exportable.setTag("SeriesDescription", series_description)
            exportable.setTag("StudyInstanceUID", study_uid)
            exportable.setTag("SeriesInstanceUID", seg_series_uid)

            # Add custom metadata as series description suffix if provided
            if segment_metadata:
                import json

                # Truncate to fit in DICOM field
                meta_str = json.dumps(segment_metadata)
                if len(meta_str) > 64:
                    meta_str = meta_str[:61] + "..."
                exportable.setTag("SeriesDescription", f"{series_description} | {meta_str}")

            # Export
            plugin.export(exportable)

            logger.info(f"Exported DICOM SEG to: {output_dir}")

            # Import into database
            self._import_dicom_folder(output_dir)

            # Store UID on segmentation node
            segmentation_node.SetAttribute("DICOM.SeriesInstanceUID", seg_series_uid)

            return seg_series_uid

        except QuantitativeReportingNotAvailable:
            raise
        except Exception as e:
            logger.exception(f"Failed to export DICOM SEG: {e}")
            raise DicomManagerError(f"Failed to export DICOM SEG: {e}") from e

    def _import_dicom_folder(self, folder: Path) -> list[str]:
        """Import DICOM files from folder into database.

        Args:
            folder: Directory containing DICOM files.

        Returns:
            List of imported series UIDs.
        """
        from DICOMLib import DICOMUtils

        folder = Path(folder)

        # Import to database
        DICOMUtils.importDicom(str(folder))

        # Get imported series
        series_uids = []
        for dcm_file in folder.glob("*.dcm"):
            try:
                uid = self.database.seriesForFile(str(dcm_file))
                if uid and uid not in series_uids:
                    series_uids.append(uid)
            except Exception:
                pass

        logger.debug(f"Imported {len(series_uids)} series from {folder}")
        return series_uids

    def load_segmentation_by_uid(self, series_uid: str):
        """Load DICOM SEG from database by SeriesInstanceUID.

        Args:
            series_uid: SeriesInstanceUID of the DICOM SEG.

        Returns:
            vtkMRMLSegmentationNode loaded from database.

        Raises:
            DicomManagerError: If loading fails.
        """
        from DICOMLib import DICOMUtils

        try:
            logger.info(f"Loading DICOM SEG: {series_uid}")
            loaded_nodes = DICOMUtils.loadSeriesByUID([series_uid])

            # Find the segmentation node
            for node in loaded_nodes if loaded_nodes else []:
                if node.IsA("vtkMRMLSegmentationNode"):
                    return node

            raise DicomManagerError(f"No segmentation found for series: {series_uid}")

        except Exception as e:
            logger.exception(f"Failed to load DICOM SEG: {e}")
            raise DicomManagerError(f"Failed to load DICOM SEG: {e}") from e

    def get_segmentations_for_volume(self, volume_series_uid: str) -> list[str]:
        """Find all DICOM SEG series that reference a volume.

        Args:
            volume_series_uid: SeriesInstanceUID of the reference volume.

        Returns:
            List of SeriesInstanceUIDs for DICOM SEG that reference this volume.
        """
        import importlib.util

        if importlib.util.find_spec("pydicom") is None:
            logger.warning("pydicom not available, cannot query referenced series")
            return []

        db = self.database
        study_uid = db.studyForSeries(volume_series_uid)

        if not study_uid:
            return []

        seg_series = []
        for series_uid in db.seriesForStudy(study_uid):
            # Skip the volume itself
            if series_uid == volume_series_uid:
                continue

            # Check if it's a SEG
            modality = db.fieldForSeries("Modality", series_uid)
            if modality != "SEG":
                continue

            # Check if it references our volume
            ref_uid = self._get_referenced_series_uid(series_uid)
            if ref_uid == volume_series_uid:
                seg_series.append(series_uid)

        return seg_series

    def _get_referenced_series_uid(self, seg_series_uid: str) -> str | None:
        """Get the referenced series UID from a DICOM SEG.

        Args:
            seg_series_uid: SeriesInstanceUID of the DICOM SEG.

        Returns:
            SeriesInstanceUID of the referenced volume, or None.
        """
        try:
            import pydicom
        except ImportError:
            return None

        db = self.database
        files = db.filesForSeries(seg_series_uid)

        if not files:
            return None

        try:
            ds = pydicom.dcmread(
                files[0], stop_before_pixels=True, specific_tags=["ReferencedSeriesSequence"]
            )
            return str(ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
        except Exception:
            return None

    def get_series_description(self, series_uid: str) -> str:
        """Get SeriesDescription for a series.

        Args:
            series_uid: SeriesInstanceUID.

        Returns:
            SeriesDescription string, or empty string if not found.
        """
        try:
            return self.database.fieldForSeries("SeriesDescription", series_uid) or ""
        except Exception:
            return ""
