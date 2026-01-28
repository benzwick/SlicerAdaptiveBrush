"""DICOM database management for optimization results.

Provides functionality to:
- Create synthetic DICOM series from non-DICOM volumes (SampleData)
- Export segmentations as DICOM SEG with LABELMAP encoding (Supplement 243)
- Query DICOM database for segmentation relationships
- Load segmentations from DICOM database

Uses highdicom for DICOM SEG creation with:
- LABELMAP encoding for efficient multi-segment storage
- Lossless compression (JPEG2000, JPEGLS, RLE)
- Full segment metadata (names, colors, terminology)

See ADR-017 for design rationale.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DicomManagerError(Exception):
    """Base exception for DicomManager errors."""

    pass


class DicomDatabaseNotAvailable(DicomManagerError):
    """Raised when DICOM database is not initialized."""

    pass


class HighdicomNotAvailable(DicomManagerError):
    """Raised when highdicom is not installed."""

    pass


class DicomManager:
    """Manage DICOM database operations for optimization results.

    This class provides a high-level interface for:
    - Converting SampleData volumes to synthetic DICOM
    - Exporting segmentations as DICOM SEG with LABELMAP encoding
    - Loading segmentations from DICOM database

    Uses highdicom for DICOM SEG creation (not QuantitativeReporting/dcmqi)
    to support LABELMAP encoding (DICOM Supplement 243) and compression.

    Usage:
        manager = DicomManager()

        # Create synthetic DICOM from SampleData volume
        volume_series_uid = manager.create_synthetic_dicom(
            volume_node=volume_node,
            patient_id="AdaptiveBrush_Test",
            study_description="Optimization Run",
            output_dir=Path("output/dicom/volume")
        )

        # Export segmentation as DICOM SEG with LABELMAP encoding
        seg_series_uid = manager.export_segmentation_as_dicom_seg(
            segmentation_node=seg_node,
            reference_volume_node=volume_node,
            series_description="trial_001_watershed",
            output_dir=Path("output/dicom/segmentations"),
            compression="JPEG2000Lossless"  # or "JPEGLSLossless", "RLELossless"
        )

        # Load segmentation from DICOM database
        seg_node = manager.load_segmentation_by_uid(seg_series_uid)
    """

    # DICOM UID prefix for generated UIDs
    # Using a fictional root (2.25) + UUID-based generation
    UID_PREFIX = "2.25"

    # Compression transfer syntax UIDs
    TRANSFER_SYNTAXES = {
        "RLELossless": "1.2.840.10008.1.2.5",
        "JPEG2000Lossless": "1.2.840.10008.1.2.4.90",
        "JPEGLSLossless": "1.2.840.10008.1.2.4.80",
        "ExplicitVRLittleEndian": "1.2.840.10008.1.2.1",  # No compression
    }

    # Default segment colors (RGB 0-255)
    DEFAULT_COLORS = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]

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

    def _ensure_highdicom(self) -> None:
        """Ensure highdicom is installed, install if not."""
        try:
            import highdicom  # noqa: F401
        except ImportError:
            logger.info("Installing highdicom...")
            import slicer

            slicer.util.pip_install("highdicom")
            import highdicom  # noqa: F401

            logger.info("highdicom installed successfully")

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

            if not slicer.dicomDatabase or not slicer.dicomDatabase.isOpen:
                # Create default database in Slicer settings directory
                settings_dir = Path(slicer.app.slicerHome) / "DICOM"
                settings_dir.mkdir(exist_ok=True)
                db_path = str(settings_dir / "ctkDICOM.sql")

                DICOMLib.DICOMUtils.openDatabase(db_path)
                logger.info(f"Initialized DICOM database at: {db_path}")

            # Verify database is actually open
            if slicer.dicomDatabase and slicer.dicomDatabase.isOpen:
                return True
            else:
                logger.error("DICOM database failed to open after initialization")
                return False
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

        Uses Slicer's DICOMScalarVolumePlugin for volume export.

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
            # Get subject hierarchy
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)

            # Create patient and study hierarchy for DICOM export
            # This is required by DICOMScalarVolumePlugin
            patientItemID = shNode.CreateSubjectItem(shNode.GetSceneItemID(), patient_name)
            studyItemID = shNode.CreateStudyItem(patientItemID, study_description)

            # Get volume's subject hierarchy item and move under study
            volumeShItemID = shNode.GetItemByDataNode(volume_node)
            if not volumeShItemID:
                raise DicomManagerError("Volume not in subject hierarchy")
            shNode.SetItemParent(volumeShItemID, studyItemID)

            # Export using DICOMScalarVolumePlugin
            import DICOMScalarVolumePlugin

            exporter = DICOMScalarVolumePlugin.DICOMScalarVolumePluginClass()

            # Get exportables for the volume
            exportables = exporter.examineForExport(volumeShItemID)
            if not exportables:
                raise DicomManagerError("Volume cannot be exported as DICOM")

            # Configure all exportables
            for exp in exportables:
                exp.directory = str(output_dir)
                exp.setTag("PatientID", patient_id)
                exp.setTag("PatientName", patient_name)
                exp.setTag("StudyID", study_description)
                exp.setTag("StudyDescription", study_description)
                exp.setTag("StudyDate", study_date)
                exp.setTag("SeriesDescription", series_description)
                exp.setTag("StudyInstanceUID", study_uid)
                exp.setTag("SeriesInstanceUID", series_uid)

            # Export (takes list of exportables)
            exporter.export(exportables)

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
        compression: str = "RLELossless",
        segment_metadata: dict | None = None,
    ) -> str:
        """Export segmentation as DICOM SEG with LABELMAP encoding.

        Uses highdicom for efficient multi-segment storage with:
        - LABELMAP encoding (DICOM Supplement 243)
        - Optional lossless compression (requires additional packages)
        - Full segment metadata

        Args:
            segmentation_node: vtkMRMLSegmentationNode to export.
            reference_volume_node: Reference volume (must have DICOM attributes).
            series_description: DICOM SeriesDescription for the SEG.
            output_dir: Directory to save DICOM SEG file.
            compression: Compression type - "ExplicitVRLittleEndian" (default, no compression),
                "JPEG2000Lossless" (requires pylibjpeg+pylibjpeg-openjpeg),
                "JPEGLSLossless" (requires pylibjpeg+pylibjpeg-libjpeg),
                "RLELossless" (requires pylibjpeg).
            segment_metadata: Optional metadata to include (algorithm params, etc.).

        Returns:
            SeriesInstanceUID of the created DICOM SEG.

        Raises:
            HighdicomNotAvailable: If highdicom cannot be installed.
            DicomManagerError: If export fails.
        """

        self._ensure_highdicom()

        import highdicom as hd
        from highdicom.seg import (
            SegmentationTypeValues,
        )
        from pydicom.uid import UID

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get reference volume DICOM UIDs
        study_uid = reference_volume_node.GetAttribute("DICOM.StudyInstanceUID")
        ref_series_uid = reference_volume_node.GetAttribute("DICOM.SeriesInstanceUID")

        if not study_uid or not ref_series_uid:
            raise DicomManagerError(
                "Reference volume does not have DICOM attributes. "
                "Create synthetic DICOM first using create_synthetic_dicom()."
            )

        logger.info(f"Exporting segmentation as DICOM SEG: {segmentation_node.GetName()}")
        logger.debug(f"  Reference SeriesUID: {ref_series_uid}")
        logger.debug(f"  Compression: {compression}")

        try:
            # Get source DICOM images
            source_images = self._load_source_dicom_images(ref_series_uid)
            if not source_images:
                raise DicomManagerError(f"No DICOM images found for series: {ref_series_uid}")

            # Extract labelmap from segmentation
            labelmap_array, segment_ids = self._get_labelmap_from_segmentation(
                segmentation_node, reference_volume_node
            )

            # Build segment descriptions
            segment_descriptions = self._build_segment_descriptions(
                segmentation_node, segment_ids, segment_metadata
            )

            # Get transfer syntax UID
            if compression not in self.TRANSFER_SYNTAXES:
                logger.warning(f"Unknown compression '{compression}', using JPEG2000Lossless")
                compression = "JPEG2000Lossless"
            transfer_syntax_uid = UID(self.TRANSFER_SYNTAXES[compression])

            # Create DICOM SEG with LABELMAP encoding
            seg = hd.seg.Segmentation(
                source_images=source_images,
                pixel_array=labelmap_array,
                segmentation_type=SegmentationTypeValues.LABELMAP,
                segment_descriptions=segment_descriptions,
                series_description=series_description,
                series_number=1,
                sop_instance_uid=hd.UID(),
                series_instance_uid=hd.UID(),
                instance_number=1,
                manufacturer="SlicerAdaptiveBrush",
                manufacturer_model_name="AdaptiveBrush Optimizer",
                software_versions="1.0",
                device_serial_number="0",
                transfer_syntax_uid=transfer_syntax_uid,
            )

            # Save to file
            output_path = output_dir / "seg.dcm"
            seg.save_as(str(output_path))

            seg_series_uid = str(seg.SeriesInstanceUID)
            logger.info(f"Exported DICOM SEG to: {output_path}")
            logger.debug(f"  SEG SeriesUID: {seg_series_uid}")

            # Import into database
            self._import_dicom_folder(output_dir)

            # Store UID on segmentation node
            segmentation_node.SetAttribute("DICOM.SeriesInstanceUID", seg_series_uid)

            return seg_series_uid

        except HighdicomNotAvailable:
            raise
        except Exception as e:
            logger.exception(f"Failed to export DICOM SEG: {e}")
            raise DicomManagerError(f"Failed to export DICOM SEG: {e}") from e

    def _load_source_dicom_images(self, series_uid: str) -> list:
        """Load source DICOM images for a series.

        Args:
            series_uid: SeriesInstanceUID to load.

        Returns:
            List of pydicom Dataset objects.
        """
        import pydicom

        db = self.database
        files = db.filesForSeries(series_uid)

        if not files:
            return []

        # Load and sort by instance number
        datasets = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to read DICOM file {f}: {e}")

        # Sort by InstanceNumber
        datasets.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))

        return datasets

    def _get_labelmap_from_segmentation(
        self, segmentation_node, reference_volume_node
    ) -> tuple[np.ndarray, list[str]]:
        """Extract labelmap array from Slicer segmentation node.

        Args:
            segmentation_node: vtkMRMLSegmentationNode.
            reference_volume_node: Reference volume for geometry.

        Returns:
            Tuple of (labelmap_array, segment_ids) where:
            - labelmap_array: numpy array with integer segment labels
            - segment_ids: list of segment IDs in label order
        """
        import sitkUtils
        import slicer

        # Create temporary labelmap volume
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        try:
            # Set reference geometry from volume
            segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                reference_volume_node
            )

            # Export segmentation to labelmap (3 arguments only)
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                segmentation_node,
                labelmap_node,
                slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
            )

            # Get segment IDs in order
            segmentation = segmentation_node.GetSegmentation()
            segment_ids = []
            for i in range(segmentation.GetNumberOfSegments()):
                segment_ids.append(segmentation.GetNthSegmentID(i))

            # Convert to numpy array
            labelmap_sitk = sitkUtils.PullVolumeFromSlicer(labelmap_node)
            import SimpleITK as sitk

            labelmap_array = sitk.GetArrayFromImage(labelmap_sitk)

            # highdicom expects (frames, rows, cols) with frames = Z slices
            # sitk array is already (Z, Y, X) which matches
            return labelmap_array.astype(np.uint8), segment_ids

        finally:
            slicer.mrmlScene.RemoveNode(labelmap_node)

    def _build_segment_descriptions(
        self,
        segmentation_node,
        segment_ids: list[str],
        segment_metadata: dict | None = None,
    ) -> list:
        """Build highdicom SegmentDescription objects from Slicer segments.

        Args:
            segmentation_node: vtkMRMLSegmentationNode.
            segment_ids: List of segment IDs in label order.
            segment_metadata: Optional metadata to include.

        Returns:
            List of highdicom SegmentDescription objects.
        """
        from highdicom.color import CIELabColor
        from highdicom.content import AlgorithmIdentificationSequence
        from highdicom.seg import (
            SegmentAlgorithmTypeValues,
            SegmentDescription,
        )
        from highdicom.sr.coding import CodedConcept
        from pydicom.sr.coding import Code

        segmentation = segmentation_node.GetSegmentation()
        descriptions = []

        for i, seg_id in enumerate(segment_ids):
            segment = segmentation.GetSegment(seg_id)
            if segment is None:
                continue

            # Get segment properties
            name = segment.GetName() or f"Segment_{i + 1}"

            # Get color (convert from 0-1 to 0-255) and convert to CIELab
            color_float = segment.GetColor()
            r, g, b = (int(c * 255) for c in color_float[:3])
            display_color = CIELabColor.from_rgb(r, g, b)

            # Create a generic tissue category and type
            # In production, would use proper SNOMED codes
            category = CodedConcept(
                value="85756007",
                meaning="Tissue",
                scheme_designator="SCT",
            )

            segmented_property = CodedConcept(
                value="85756007",
                meaning=name,
                scheme_designator="SCT",
            )

            # Build algorithm identification
            algo_name = "AdaptiveBrush"
            if segment_metadata and "algorithm" in segment_metadata:
                algo_name = segment_metadata["algorithm"]

            # Algorithm family code (using generic "Artificial Intelligence" category)
            algo_family = Code(
                value="129465004",
                scheme_designator="SCT",
                meaning="Artificial intelligence",
            )

            algo_identification = AlgorithmIdentificationSequence(
                name=algo_name,
                family=algo_family,
                version="1.0",
                source="SlicerAdaptiveBrush",
            )

            desc = SegmentDescription(
                segment_number=i + 1,
                segment_label=name,
                segmented_property_category=category,
                segmented_property_type=segmented_property,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algo_identification,
                display_color=display_color,
            )
            descriptions.append(desc)

        return descriptions

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
