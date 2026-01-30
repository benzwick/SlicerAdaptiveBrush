"""Custom DICOM plugin for loading Label Map Segmentation Storage files.

This plugin handles DICOM SEG files with LABELMAP encoding (DICOM Supplement 243),
which use SOP Class UID 1.2.840.10008.5.1.4.1.1.66.7 (Label Map Segmentation Storage).

The standard Slicer DICOMSegmentationPlugin (based on dcmqi) only supports
SOP Class UID 1.2.840.10008.5.1.4.1.1.66.4 (Segmentation Storage) used for
BINARY and FRACTIONAL encodings.

This plugin uses highdicom for reading LABELMAP files, which provides full
support for DICOM Supplement 243.

See ADR-019 for design rationale.

References:
- DICOM Supplement 243: Label Map Segmentation
- dcmqi issue #518: https://github.com/QIICR/dcmqi/issues/518
- OHIF v3.11+ supports LABELMAP natively
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# SOP Class UIDs
LABELMAP_SEG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.7"  # Label Map Segmentation Storage
STANDARD_SEG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.4"  # Segmentation Storage (BINARY/FRACTIONAL)


class DICOMLabelMapSegPlugin:
    """DICOM plugin to load Label Map Segmentation Storage files using highdicom.

    This plugin enables CrossSegmentationExplorer and other Slicer DICOM workflows
    to work with LABELMAP-encoded DICOM SEG files created by highdicom.

    Features:
    - Detects LABELMAP SOP Class (1.2.840.10008.5.1.4.1.1.66.7)
    - Uses highdicom.seg.Segmentation for parsing
    - Converts to vtkMRMLSegmentationNode with proper geometry
    - Preserves segment metadata (labels, colors, terminology)

    Usage:
        # Plugin is auto-registered when module loads
        # Then use standard DICOM import workflow:
        from DICOMLib import DICOMUtils
        DICOMUtils.loadSeriesByUID([series_uid])
    """

    # Plugin identification
    loadType = "DICOM Labelmap Segmentation (highdicom)"

    def __init__(self):
        """Initialize the plugin."""
        self.loadType = DICOMLabelMapSegPlugin.loadType
        self._highdicom_available: bool | None = None

    def _ensure_highdicom(self) -> bool:
        """Ensure highdicom is installed.

        Returns:
            True if highdicom is available, False otherwise.
        """
        if self._highdicom_available is not None:
            return self._highdicom_available

        try:
            import highdicom  # noqa: F401

            self._highdicom_available = True
        except ImportError:
            logger.info("Installing highdicom for LABELMAP DICOM SEG support...")
            try:
                import slicer

                slicer.util.pip_install("highdicom")
                import highdicom  # noqa: F401

                self._highdicom_available = True
                logger.info("highdicom installed successfully")
            except Exception as e:
                logger.error(f"Failed to install highdicom: {e}")
                self._highdicom_available = False

        return bool(self._highdicom_available)

    def examineForImport(self, fileLists: list[list[str]]) -> list:
        """Examine files to determine if they can be loaded.

        Called by the DICOM browser when examining files for import.

        Args:
            fileLists: List of file lists (each list is files for one series).

        Returns:
            List of DICOMLoadable objects for loadable files.
        """
        import pydicom
        from DICOMLib import DICOMLoadable

        loadables = []

        for files in fileLists:
            if not files:
                continue

            # Check first file for SOP Class
            try:
                ds = pydicom.dcmread(
                    files[0],
                    stop_before_pixels=True,
                    specific_tags=["SOPClassUID", "Modality", "SeriesDescription"],
                )

                # Check if it's a LABELMAP SEG
                sop_class = str(getattr(ds, "SOPClassUID", ""))
                modality = str(getattr(ds, "Modality", ""))

                if sop_class != LABELMAP_SEG_SOP_CLASS:
                    continue

                if modality != "SEG":
                    continue

                # This is a LABELMAP SEG file we can handle
                series_desc = str(getattr(ds, "SeriesDescription", "Segmentation"))

                loadable = DICOMLoadable()
                loadable.files = files
                loadable.name = series_desc
                loadable.tooltip = (
                    f"Label Map Segmentation (highdicom)\n"
                    f"SOP Class: Label Map Segmentation Storage\n"
                    f"Files: {len(files)}"
                )
                loadable.selected = True
                loadable.confidence = (
                    0.95  # High confidence - we specifically handle this SOP Class
                )

                # Store metadata for load()
                loadable.sop_class_uid = sop_class

                loadables.append(loadable)
                logger.debug(f"Found LABELMAP SEG: {series_desc} ({len(files)} files)")

            except Exception as e:
                logger.debug(f"Error examining {files[0]}: {e}")
                continue

        return loadables

    def load(self, loadable) -> bool:
        """Load the DICOM SEG file and create Slicer segmentation node.

        Args:
            loadable: DICOMLoadable object from examineForImport.

        Returns:
            True if loading succeeded, False otherwise.
        """
        if not self._ensure_highdicom():
            logger.error("highdicom not available, cannot load LABELMAP SEG")
            return False

        import numpy as np
        import pydicom
        import SimpleITK as sitk
        import sitkUtils
        import slicer
        from DICOMLib import DICOMUtils

        try:
            # Read the DICOM SEG file using highdicom
            import highdicom as hd

            # Get the file path (LABELMAP SEG is typically single file)
            seg_file = loadable.files[0]
            logger.info(f"Loading LABELMAP SEG: {seg_file}")

            # Read using highdicom
            seg = hd.seg.Segmentation.from_dcm(pydicom.dcmread(seg_file))  # type: ignore[attr-defined]

            # Get pixel array (already in labelmap format)
            # highdicom returns (frames, rows, cols) for LABELMAP
            pixel_array = seg.pixel_array.astype(np.uint8)

            logger.debug(f"Pixel array shape: {pixel_array.shape}")
            logger.debug(f"Unique labels: {np.unique(pixel_array)}")

            # Find the referenced volume
            ref_series_uid = self._get_referenced_series_uid(seg)
            volume_node = None

            if ref_series_uid:
                # Try to load referenced volume if not already in scene
                volume_node = self._find_volume_by_series_uid(ref_series_uid)
                if volume_node is None:
                    logger.info(f"Loading referenced volume: {ref_series_uid}")
                    try:
                        loaded = DICOMUtils.loadSeriesByUID([ref_series_uid])
                        for node in loaded or []:
                            if node.IsA("vtkMRMLScalarVolumeNode"):
                                volume_node = node
                                break
                    except Exception as e:
                        logger.warning(f"Could not load referenced volume: {e}")

            # Create segmentation node
            seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            seg_node.SetName(loadable.name)
            seg_node.CreateDefaultDisplayNodes()

            # Create labelmap volume node from pixel data
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

            try:
                # Create SimpleITK image from pixel array
                # pixel_array is (Z, Y, X) from DICOM - need to match volume orientation
                sitk_image = sitk.GetImageFromArray(pixel_array)

                # Get geometry from DICOM or reference volume
                if volume_node is not None:
                    # Use reference volume geometry
                    ref_sitk = sitkUtils.PullVolumeFromSlicer(volume_node)
                    sitk_image.SetSpacing(ref_sitk.GetSpacing())
                    sitk_image.SetOrigin(ref_sitk.GetOrigin())
                    sitk_image.SetDirection(ref_sitk.GetDirection())
                else:
                    # Extract geometry from DICOM SEG
                    spacing, origin, direction = self._get_geometry_from_seg(seg)
                    sitk_image.SetSpacing(spacing)
                    sitk_image.SetOrigin(origin)
                    sitk_image.SetDirection(direction)

                # Push to labelmap node
                sitkUtils.PushVolumeToSlicer(sitk_image, labelmap_node)

                # Set reference geometry if we have a volume
                if volume_node is not None:
                    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

                # Import labelmap to segmentation
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    labelmap_node, seg_node
                )

                # Apply segment metadata (names, colors)
                self._apply_segment_metadata(seg, seg_node)

                # Store DICOM UIDs as attributes
                seg_node.SetAttribute("DICOM.SOPClassUID", str(seg.SOPClassUID))
                seg_node.SetAttribute("DICOM.SeriesInstanceUID", str(seg.SeriesInstanceUID))
                if ref_series_uid:
                    seg_node.SetAttribute("DICOM.ReferencedSeriesInstanceUID", ref_series_uid)

                logger.info(f"Successfully loaded LABELMAP SEG: {seg_node.GetName()}")
                return True

            finally:
                # Clean up temporary labelmap node
                slicer.mrmlScene.RemoveNode(labelmap_node)

        except Exception as e:
            logger.exception(f"Failed to load LABELMAP SEG: {e}")
            return False

    def _get_referenced_series_uid(self, seg) -> str | None:
        """Get referenced series UID from DICOM SEG.

        Args:
            seg: highdicom Segmentation object.

        Returns:
            SeriesInstanceUID of referenced volume, or None.
        """
        try:
            if hasattr(seg, "ReferencedSeriesSequence") and seg.ReferencedSeriesSequence:
                return str(seg.ReferencedSeriesSequence[0].SeriesInstanceUID)
        except Exception as e:
            logger.debug(f"Could not get referenced series UID: {e}")
        return None

    def _find_volume_by_series_uid(self, series_uid: str):
        """Find a volume node in the scene by DICOM SeriesInstanceUID.

        Args:
            series_uid: SeriesInstanceUID to search for.

        Returns:
            vtkMRMLScalarVolumeNode if found, None otherwise.
        """
        import slicer

        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            node_uid = node.GetAttribute("DICOM.instanceUIDs")
            if node_uid and series_uid in node_uid:
                return node
            node_series_uid = node.GetAttribute("DICOM.SeriesInstanceUID")
            if node_series_uid == series_uid:
                return node
        return None

    def _get_geometry_from_seg(self, seg) -> tuple[tuple, tuple, tuple]:
        """Extract image geometry from DICOM SEG.

        Args:
            seg: highdicom Segmentation object.

        Returns:
            Tuple of (spacing, origin, direction).
        """
        import numpy as np

        # Default values
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        try:
            # Get pixel spacing from PixelMeasuresSequence or SharedFunctionalGroupsSequence
            if hasattr(seg, "SharedFunctionalGroupsSequence"):
                shared = seg.SharedFunctionalGroupsSequence[0]
                if hasattr(shared, "PixelMeasuresSequence"):
                    pm = shared.PixelMeasuresSequence[0]
                    pixel_spacing = pm.PixelSpacing
                    slice_thickness = getattr(pm, "SliceThickness", 1.0)
                    spacing = (
                        float(pixel_spacing[0]),
                        float(pixel_spacing[1]),
                        float(slice_thickness),
                    )

                # Get orientation from PlaneOrientationSequence
                if hasattr(shared, "PlaneOrientationSequence"):
                    po = shared.PlaneOrientationSequence[0]
                    if hasattr(po, "ImageOrientationPatient"):
                        iop = [float(x) for x in po.ImageOrientationPatient]
                        row_dir = np.array(iop[:3])
                        col_dir = np.array(iop[3:])
                        slice_dir = np.cross(row_dir, col_dir)
                        direction = tuple(row_dir) + tuple(col_dir) + tuple(slice_dir)

            # Get origin from first frame
            if hasattr(seg, "PerFrameFunctionalGroupsSequence"):
                first_frame = seg.PerFrameFunctionalGroupsSequence[0]
                if hasattr(first_frame, "PlanePositionSequence"):
                    pp = first_frame.PlanePositionSequence[0]
                    if hasattr(pp, "ImagePositionPatient"):
                        ipp = pp.ImagePositionPatient
                        origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))

        except Exception as e:
            logger.warning(f"Could not extract full geometry from SEG, using defaults: {e}")

        return spacing, origin, direction

    def _apply_segment_metadata(self, seg, seg_node) -> None:
        """Apply segment metadata from DICOM to Slicer segmentation.

        Args:
            seg: highdicom Segmentation object.
            seg_node: vtkMRMLSegmentationNode.
        """
        try:
            segmentation = seg_node.GetSegmentation()

            # Get segment descriptions from DICOM
            if not hasattr(seg, "SegmentSequence"):
                return

            for i, seg_desc in enumerate(seg.SegmentSequence):
                # Get segment number (1-indexed in DICOM)
                seg_num = int(getattr(seg_desc, "SegmentNumber", i + 1))

                # Find corresponding segment in Slicer (labelmap import creates Segment_1, Segment_2, etc.)
                # Or it might be numbered like the label value
                segment = None
                for seg_id_candidate in [f"Segment_{seg_num}", str(seg_num)]:
                    segment = segmentation.GetSegment(seg_id_candidate)
                    if segment:
                        break

                if segment is None:
                    # Try by index
                    if i < segmentation.GetNumberOfSegments():
                        seg_id = segmentation.GetNthSegmentID(i)
                        segment = segmentation.GetSegment(seg_id)

                if segment is None:
                    continue

                # Set segment label/name
                label = getattr(seg_desc, "SegmentLabel", None)
                if label:
                    segment.SetName(str(label))

                # Set segment color from RecommendedDisplayCIELabValue
                if hasattr(seg_desc, "RecommendedDisplayCIELabValue"):
                    cielab = seg_desc.RecommendedDisplayCIELabValue
                    rgb = self._cielab_to_rgb(cielab)
                    segment.SetColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

        except Exception as e:
            logger.warning(f"Could not apply segment metadata: {e}")

    def _cielab_to_rgb(self, cielab: list) -> tuple[int, int, int]:
        """Convert CIELab color to RGB.

        Args:
            cielab: DICOM CIELab values [L, a, b] (L: 0-65535, a/b: 0-65535 centered at 32768).

        Returns:
            RGB tuple (0-255 each).
        """
        try:
            # Convert DICOM CIELab to standard CIELab
            L = cielab[0] * 100.0 / 65535.0
            a = (cielab[1] - 32768.0) * 255.0 / 65535.0
            b = (cielab[2] - 32768.0) * 255.0 / 65535.0

            # CIELab to XYZ
            y = (L + 16.0) / 116.0
            x = a / 500.0 + y
            z = y - b / 200.0

            def f_inv(t):
                delta = 6.0 / 29.0
                if t > delta:
                    return t**3
                return 3 * delta**2 * (t - 4.0 / 29.0)

            # D65 illuminant reference
            X = 95.047 * f_inv(x)
            Y = 100.0 * f_inv(y)
            Z = 108.883 * f_inv(z)

            # XYZ to sRGB
            r = X * 3.2406 + Y * -1.5372 + Z * -0.4986
            g = X * -0.9689 + Y * 1.8758 + Z * 0.0415
            b = X * 0.0557 + Y * -0.2040 + Z * 1.0570

            # Gamma correction
            def gamma(u):
                if u > 0.0031308:
                    return 1.055 * (u ** (1 / 2.4)) - 0.055
                return 12.92 * u

            r = int(max(0, min(255, gamma(r / 100.0) * 255)))
            g = int(max(0, min(255, gamma(g / 100.0) * 255)))
            b = int(max(0, min(255, gamma(b / 100.0) * 255)))

            return (r, g, b)

        except Exception:
            return (255, 0, 0)  # Default to red on error


def register_plugin() -> None:
    """Register the DICOM plugin with Slicer.

    Call this function when the module loads to enable LABELMAP SEG loading.
    """
    try:
        from DICOMLib import DICOMPlugin

        # Check if already registered
        plugin_class_name = DICOMLabelMapSegPlugin.__name__
        for plugin in DICOMPlugin.DICOMPlugins:
            if plugin.__class__.__name__ == plugin_class_name:
                logger.debug("DICOMLabelMapSegPlugin already registered")
                return

        # Register our plugin
        DICOMPlugin.DICOMPlugins.append(DICOMLabelMapSegPlugin())
        logger.info(f"Registered {plugin_class_name} for LABELMAP DICOM SEG loading")

    except ImportError:
        logger.warning("DICOMLib not available, cannot register LABELMAP plugin")
    except Exception as e:
        logger.error(f"Failed to register LABELMAP plugin: {e}")


def is_labelmap_seg(filepath: str) -> bool:
    """Check if a DICOM file is a LABELMAP SEG.

    Utility function for quick SOP Class checking.

    Args:
        filepath: Path to DICOM file.

    Returns:
        True if file is a LABELMAP SEG, False otherwise.
    """
    try:
        import pydicom

        ds = pydicom.dcmread(filepath, stop_before_pixels=True, specific_tags=["SOPClassUID"])
        return str(getattr(ds, "SOPClassUID", "")) == LABELMAP_SEG_SOP_CLASS
    except Exception:
        return False
