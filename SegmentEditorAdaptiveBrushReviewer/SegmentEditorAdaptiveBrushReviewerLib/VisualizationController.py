"""Dual segmentation visualization controller.

Manages display of gold standard and test segmentations with
different view modes and color schemes.

Gold standards: Loaded from .seg.nrrd (primary) or DICOM cache (for CSE)
Test segmentations: Loaded from DICOM SEG (optimization output format)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _check_dicom_seg_plugin() -> bool:
    """Check if DICOMSegmentationPlugin is available.

    Returns:
        True if QuantitativeReporting extension is installed.
    """
    import slicer

    return "DICOMSegmentationPlugin" in slicer.modules.dicomPlugins


class VisualizationController:
    """Control dual segmentation visualization.

    Displays gold standard (yellow/gold) and test (cyan) segmentations
    with various view modes: outline, transparent, fill.

    Gold standards are loaded from .seg.nrrd files (git-tracked canonical format).
    Test segmentations are loaded from DICOM SEG (optimization output format).
    """

    # Color scheme (RGB 0-1)
    GOLD_COLOR = (1.0, 0.84, 0.0)  # Gold
    TEST_COLOR = (0.0, 1.0, 1.0)  # Cyan
    OVERLAP_COLOR = (0.0, 1.0, 0.5)  # Light green

    def __init__(self):
        """Initialize controller."""
        self.gold_seg_node: Any = None
        self.test_seg_node: Any = None
        self.view_mode: str = "outline"  # outline, transparent, fill
        self._gold_visible: bool = True
        self._test_visible: bool = True

    def load_gold_segmentation(self, path: Path | str) -> bool:
        """Load gold standard segmentation from .seg.nrrd file.

        This is the primary method for loading gold standards.
        Gold standards are stored as .seg.nrrd in git for version tracking.

        Args:
            path: Path to .seg.nrrd segmentation file.

        Returns:
            True if loaded successfully.
        """
        try:
            import slicer

            path = Path(path)
            if not path.exists():
                logger.error(f"Gold standard not found: {path}")
                return False

            # Remove previous gold if exists
            if self.gold_seg_node:
                slicer.mrmlScene.RemoveNode(self.gold_seg_node)

            self.gold_seg_node = slicer.util.loadSegmentation(str(path))

            if self.gold_seg_node:
                self._apply_color(self.gold_seg_node, self.GOLD_COLOR)
                self._set_display_mode(self.gold_seg_node, self.view_mode)
                self.gold_seg_node.SetName("Gold Standard")
                logger.info(f"Loaded gold standard: {path}")
                return True

            return False

        except Exception as e:
            logger.exception(f"Failed to load gold standard: {e}")
            return False

    def load_gold_from_dicom_cache(self, dicom_seg_path: Path | str) -> bool:
        """Load gold standard from DICOM cache.

        Used for CrossSegmentationExplorer compatibility when DICOM format
        is needed. The DICOM cache is generated on-demand from .seg.nrrd.

        Requires QuantitativeReporting extension for DICOMSegmentationPlugin.

        Args:
            dicom_seg_path: Path to DICOM SEG directory or file in .dicom_cache/.

        Returns:
            True if loaded successfully.
        """
        try:
            import pydicom
            import slicer
            from DICOMLib import DICOMUtils

            # Check for required extension
            if not _check_dicom_seg_plugin():
                logger.error(
                    "QuantitativeReporting extension required for DICOM SEG loading. "
                    "Install via: View > Extension Manager > QuantitativeReporting"
                )
                return False

            dicom_seg_path = Path(dicom_seg_path)
            if not dicom_seg_path.exists():
                logger.error(f"DICOM cache not found: {dicom_seg_path}")
                return False

            # Remove previous gold if exists
            if self.gold_seg_node:
                slicer.mrmlScene.RemoveNode(self.gold_seg_node)

            # Find the DICOM file to get SeriesInstanceUID
            if dicom_seg_path.is_dir():
                dicom_files = list(dicom_seg_path.glob("*.dcm"))
                if not dicom_files:
                    logger.error(f"No DICOM files in: {dicom_seg_path}")
                    return False
                dcm_file = dicom_files[0]
                import_path = str(dicom_seg_path)
            else:
                dcm_file = dicom_seg_path
                import_path = str(dicom_seg_path.parent)

            # Read SeriesInstanceUID from the DICOM file
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            series_uid = str(ds.SeriesInstanceUID)

            # Import to DICOM database (waits for import to finish)
            DICOMUtils.importDicom(import_path, slicer.dicomDatabase)

            # Load via DICOMUtils using the series UID
            loaded_node_ids = DICOMUtils.loadSeriesByUID([series_uid])

            if loaded_node_ids:
                self.gold_seg_node = slicer.mrmlScene.GetNodeByID(loaded_node_ids[0])
                if self.gold_seg_node:
                    self._apply_color(self.gold_seg_node, self.GOLD_COLOR)
                    self._set_display_mode(self.gold_seg_node, self.view_mode)
                    self.gold_seg_node.SetName("Gold Standard")
                    logger.info(f"Loaded gold standard from DICOM cache: {dicom_seg_path}")
                    return True

            logger.error(f"Failed to load DICOM series: {series_uid}")
            return False

        except Exception as e:
            logger.exception(f"Failed to load gold from DICOM cache: {e}")
            return False

    def load_test_segmentation_from_dicom(self, dicom_seg_path: Path | str) -> bool:
        """Load trial segmentation from DICOM SEG file.

        Test segmentations from optimization runs are stored as DICOM SEG.

        Requires QuantitativeReporting extension for DICOMSegmentationPlugin.

        Args:
            dicom_seg_path: Path to DICOM SEG directory or file.

        Returns:
            True if loaded successfully.
        """
        try:
            import pydicom
            import slicer
            from DICOMLib import DICOMUtils

            # Check for required extension
            if not _check_dicom_seg_plugin():
                logger.error(
                    "QuantitativeReporting extension required for DICOM SEG loading. "
                    "Install via: View > Extension Manager > QuantitativeReporting"
                )
                return False

            dicom_seg_path = Path(dicom_seg_path)
            if not dicom_seg_path.exists():
                logger.error(f"DICOM SEG path not found: {dicom_seg_path}")
                return False

            # Remove previous test if exists
            if self.test_seg_node:
                slicer.mrmlScene.RemoveNode(self.test_seg_node)

            # Find the DICOM file to get SeriesInstanceUID
            if dicom_seg_path.is_dir():
                dicom_files = list(dicom_seg_path.glob("*.dcm"))
                if not dicom_files:
                    logger.error(f"No DICOM files in: {dicom_seg_path}")
                    return False
                dcm_file = dicom_files[0]
                import_path = str(dicom_seg_path)
            else:
                dcm_file = dicom_seg_path
                import_path = str(dicom_seg_path.parent)

            # Read SeriesInstanceUID from the DICOM file
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            series_uid = str(ds.SeriesInstanceUID)

            # Import to DICOM database (waits for import to finish)
            DICOMUtils.importDicom(import_path, slicer.dicomDatabase)

            # Load via DICOMUtils using the series UID
            loaded_node_ids = DICOMUtils.loadSeriesByUID([series_uid])

            if loaded_node_ids:
                self.test_seg_node = slicer.mrmlScene.GetNodeByID(loaded_node_ids[0])
                if self.test_seg_node:
                    self._apply_color(self.test_seg_node, self.TEST_COLOR)
                    self._set_display_mode(self.test_seg_node, self.view_mode)
                    self.test_seg_node.SetName("Test Segmentation")
                    logger.info(f"Loaded test segmentation from DICOM: {dicom_seg_path}")
                    return True

            logger.error(f"Failed to load DICOM series: {series_uid}")
            return False

        except Exception as e:
            logger.exception(f"Failed to load DICOM segmentation: {e}")
            return False

    def set_view_mode(self, mode: str) -> None:
        """Change display mode for both segmentations.

        Args:
            mode: View mode - "outline", "transparent", or "fill".
        """
        if mode not in ("outline", "transparent", "fill"):
            logger.warning(f"Unknown view mode: {mode}")
            return

        self.view_mode = mode

        if self.gold_seg_node:
            self._set_display_mode(self.gold_seg_node, mode)

        if self.test_seg_node:
            self._set_display_mode(self.test_seg_node, mode)

        logger.debug(f"Set view mode to: {mode}")

    def toggle_gold(self, visible: bool) -> None:
        """Toggle gold standard visibility.

        Args:
            visible: Whether to show gold standard.
        """
        self._gold_visible = visible
        if self.gold_seg_node:
            display_node = self.gold_seg_node.GetDisplayNode()
            if display_node:
                display_node.SetVisibility(visible)

    def toggle_test(self, visible: bool) -> None:
        """Toggle test segmentation visibility.

        Args:
            visible: Whether to show test segmentation.
        """
        self._test_visible = visible
        if self.test_seg_node:
            display_node = self.test_seg_node.GetDisplayNode()
            if display_node:
                display_node.SetVisibility(visible)

    def _apply_color(self, seg_node: Any, color: tuple[float, float, float]) -> None:
        """Apply color to all segments in a segmentation.

        Args:
            seg_node: Segmentation node.
            color: RGB color tuple (0-1 range).
        """
        try:
            segmentation = seg_node.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segment = segmentation.GetNthSegment(i)
                segment.SetColor(color[0], color[1], color[2])
        except Exception as e:
            logger.warning(f"Could not apply color: {e}")

    def _set_display_mode(self, seg_node: Any, mode: str) -> None:
        """Set display mode for segmentation.

        Args:
            seg_node: Segmentation node.
            mode: View mode.
        """
        try:
            display_node = seg_node.GetDisplayNode()
            if not display_node:
                return

            if mode == "outline":
                # 2D outline only
                display_node.SetVisibility2DOutline(True)
                display_node.SetVisibility2DFill(False)
                display_node.SetOpacity2DOutline(1.0)
                display_node.SetOpacity3D(0.3)

            elif mode == "transparent":
                # Semi-transparent fill
                display_node.SetVisibility2DOutline(True)
                display_node.SetVisibility2DFill(True)
                display_node.SetOpacity2DOutline(1.0)
                display_node.SetOpacity2DFill(0.3)
                display_node.SetOpacity3D(0.3)

            elif mode == "fill":
                # Solid fill
                display_node.SetVisibility2DOutline(False)
                display_node.SetVisibility2DFill(True)
                display_node.SetOpacity2DFill(0.7)
                display_node.SetOpacity3D(0.7)

        except Exception as e:
            logger.warning(f"Could not set display mode: {e}")

    def cleanup(self) -> None:
        """Clean up loaded segmentations."""
        try:
            import slicer

            if self.gold_seg_node:
                slicer.mrmlScene.RemoveNode(self.gold_seg_node)
                self.gold_seg_node = None

            if self.test_seg_node:
                slicer.mrmlScene.RemoveNode(self.test_seg_node)
                self.test_seg_node = None

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def get_gold_node(self) -> Any:
        """Get gold standard segmentation node."""
        return self.gold_seg_node

    def get_test_node(self) -> Any:
        """Get test segmentation node."""
        return self.test_seg_node
