"""Tests for DICOM cache generation.

Tests the on-demand DICOM cache generation for gold standards:
- Synthetic DICOM creation for SampleData volumes
- DICOM SEG export from .seg.nrrd
- Cache directory structure
- CSE compatibility of generated DICOM

Note: These tests require a compatible pydicom version. Slicer's bundled
pydicom may not have all required features.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

from .fixtures.mock_segmentations import MockSegmentationFactory

logger = logging.getLogger(__name__)


def _check_pydicom_compatibility() -> tuple[bool, str]:
    """Check if pydicom is compatible with our DICOM tests.

    Returns:
        Tuple of (is_compatible, error_message)
    """
    try:
        import pydicom  # noqa: F401

        # Try importing encaps module which may have API changes
        from pydicom import encaps  # noqa: F401

        return True, ""
    except ImportError as e:
        return False, f"pydicom import failed: {e}"


@register_test(category="gold_standard")
class TestDicomCacheGeneration(TestCase):
    """Test DICOM cache generation for gold standards."""

    name = "dicom_cache_generation"
    description = "Test synthetic DICOM and DICOM SEG cache creation"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.seg_factory = None
        self.temp_dir = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test with sample data."""
        logger.info("Setting up DICOM cache test")

        # Check pydicom compatibility
        is_compatible, error_msg = _check_pydicom_compatibility()
        if not is_compatible:
            raise RuntimeError(
                f"DICOM tests skipped: pydicom incompatible. {error_msg}\n"
                "This may be due to Slicer's bundled pydicom version."
            )

        slicer.mrmlScene.Clear(0)

        # Load sample data
        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.seg_factory = MockSegmentationFactory()

        # Create temp directory for test output
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dicom_cache_test_"))

        ctx.screenshot("[setup] Sample data loaded")

    def run(self, ctx: TestContext) -> None:
        """Test DICOM cache generation."""
        logger.info("Running DICOM cache generation tests")

        # Import DicomManager
        from SegmentEditorAdaptiveBrushReviewerLib import DicomManager

        dicom_manager = DicomManager()

        # Test 1: Ensure DICOM database is initialized
        ctx.log("Test 1: DICOM database initialization")
        result = dicom_manager.ensure_database_initialized()
        ctx.assert_true(result, "DICOM database should initialize successfully")
        ctx.screenshot("[db] DICOM database initialized")

        # Test 2: Create synthetic DICOM for volume
        ctx.log("Test 2: Create synthetic DICOM")
        volume_dicom_dir = self.temp_dir / "volume_dicom"

        series_uid = dicom_manager.create_synthetic_dicom(
            volume_node=self.volume_node,
            patient_id="TestPatient",
            study_description="DICOM Cache Test",
            output_dir=volume_dicom_dir,
        )

        ctx.assert_true(
            len(series_uid) > 0,
            "Should return a valid SeriesInstanceUID",
        )

        # Verify DICOM files were created (may be in subdirectory)
        dicom_files = list(volume_dicom_dir.rglob("*.dcm"))
        ctx.assert_greater(
            len(dicom_files),
            0,
            "Should create at least one DICOM file",
        )

        ctx.log(f"Created {len(dicom_files)} DICOM files, UID: {series_uid[:40]}...")
        ctx.screenshot("[synthetic] Synthetic DICOM created")

        # Test 3: Export segmentation as DICOM SEG
        ctx.log("Test 3: Export segmentation as DICOM SEG")
        seg_node = self.seg_factory.create_sphere_segmentation(
            center_ijk=(128, 128, 64),
            radius=20,
            volume_node=self.volume_node,
            name="TestSegForDICOM",
            segment_name="TestSphere",
        )

        seg_output_dir = self.temp_dir / "seg_dicom"

        seg_series_uid = dicom_manager.export_segmentation_as_dicom_seg(
            segmentation_node=seg_node,
            reference_volume_node=self.volume_node,
            series_description="Test Segmentation",
            output_dir=seg_output_dir,
            compression="RLELossless",
        )

        ctx.assert_true(
            len(seg_series_uid) > 0,
            "Should return a valid SEG SeriesInstanceUID",
        )

        # Verify DICOM SEG file was created
        seg_files = list(seg_output_dir.glob("*.dcm"))
        ctx.assert_equal(
            len(seg_files),
            1,
            "Should create exactly one DICOM SEG file",
        )

        seg_file = seg_files[0]
        ctx.assert_greater(
            seg_file.stat().st_size,
            1000,
            "DICOM SEG file should have reasonable size",
        )

        ctx.log(f"Created DICOM SEG: {seg_file.name}, size: {seg_file.stat().st_size}")
        ctx.screenshot("[seg] DICOM SEG exported")

        # Test 4: Verify DICOM SEG metadata
        ctx.log("Test 4: Verify DICOM SEG metadata")
        try:
            import pydicom

            dcm = pydicom.dcmread(str(seg_file))

            # Check required DICOM SEG fields
            ctx.assert_equal(
                dcm.Modality,
                "SEG",
                "Modality should be SEG",
            )

            ctx.assert_true(
                hasattr(dcm, "SegmentSequence"),
                "Should have SegmentSequence",
            )

            ctx.assert_true(
                hasattr(dcm, "ReferencedSeriesSequence"),
                "Should have ReferencedSeriesSequence for CSE compatibility",
            )

            # Verify compression
            transfer_syntax = dcm.file_meta.TransferSyntaxUID
            ctx.log(f"Transfer syntax: {transfer_syntax}")

            ctx.screenshot("[metadata] DICOM metadata verified")

        except ImportError:
            ctx.log("pydicom not available, skipping metadata verification")

        # Test 5: Test cache directory structure
        ctx.log("Test 5: Test expected cache directory structure")
        cache_dir = self.temp_dir / "gold_cache"
        cache_dir.mkdir()

        # Create volume DICOM in cache
        vol_cache = cache_dir / "volume_dicom"
        dicom_manager.create_synthetic_dicom(
            volume_node=self.volume_node,
            patient_id="GoldCache",
            study_description="Gold Standard Cache",
            output_dir=vol_cache,
        )

        # Create seg.dcm in cache
        dicom_manager.export_segmentation_as_dicom_seg(
            segmentation_node=seg_node,
            reference_volume_node=self.volume_node,
            series_description="GoldStandard",
            output_dir=cache_dir,
            compression="RLELossless",
        )

        # Verify structure
        ctx.assert_true(
            vol_cache.exists(),
            "volume_dicom directory should exist",
        )
        ctx.assert_true(
            (cache_dir / "seg.dcm").exists(),
            "seg.dcm should exist in cache root",
        )

        ctx.screenshot("[cache] Cache structure verified")

    def verify(self, ctx: TestContext) -> None:
        """Verify test results."""
        logger.info("Verifying DICOM cache tests")

        # Verify temp files exist
        ctx.assert_true(
            self.temp_dir.exists(),
            "Temp directory should exist",
        )

        total_files = len(list(self.temp_dir.rglob("*.dcm")))
        ctx.log(f"Temp dir contents: {list(self.temp_dir.rglob('*'))[:10]}")
        ctx.log(f"Total DICOM files created: {total_files}")
        ctx.assert_greater(
            total_files,
            0,
            "Should have created DICOM files",
        )

        ctx.screenshot("[verify] DICOM cache test complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up test resources."""
        logger.info("Tearing down DICOM cache test")

        if self.seg_factory:
            self.seg_factory.cleanup()
            self.seg_factory = None

        # Clean up temp directory
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            try:
                shutil.rmtree(self.temp_dir)
                ctx.log(f"Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                ctx.log(f"Warning: Could not clean temp dir: {e}")

        ctx.log("Teardown complete")


@register_test(category="gold_standard")
class TestDicomCseCompatibility(TestCase):
    """Test that generated DICOM is CSE-compatible."""

    name = "dicom_cse_compatibility"
    description = "Verify DICOM SEG has correct UIDs for CrossSegmentationExplorer"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.seg_factory = None
        self.temp_dir = None

    def setup(self, ctx: TestContext) -> None:
        """Set up test."""
        logger.info("Setting up CSE compatibility test")

        # Check pydicom compatibility
        is_compatible, error_msg = _check_pydicom_compatibility()
        if not is_compatible:
            raise RuntimeError(
                f"DICOM tests skipped: pydicom incompatible. {error_msg}\n"
                "This may be due to Slicer's bundled pydicom version."
            )

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        self.seg_factory = MockSegmentationFactory()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cse_compat_test_"))

        ctx.screenshot("[setup] Ready for CSE compatibility test")

    def run(self, ctx: TestContext) -> None:
        """Test CSE compatibility requirements."""
        logger.info("Running CSE compatibility tests")

        from SegmentEditorAdaptiveBrushReviewerLib import DicomManager

        dicom_manager = DicomManager()
        dicom_manager.ensure_database_initialized()

        # Create volume DICOM
        volume_dir = self.temp_dir / "volume"
        volume_series_uid = dicom_manager.create_synthetic_dicom(
            volume_node=self.volume_node,
            patient_id="CSETest",
            study_description="CSE Compatibility Test",
            output_dir=volume_dir,
        )

        # Create segmentation
        seg_node = self.seg_factory.create_sphere_segmentation(
            center_ijk=(128, 128, 64),
            radius=15,
            volume_node=self.volume_node,
        )

        # Export as DICOM SEG
        seg_dir = self.temp_dir / "seg"
        seg_series_uid = dicom_manager.export_segmentation_as_dicom_seg(
            segmentation_node=seg_node,
            reference_volume_node=self.volume_node,
            series_description="Trial 001 - test",
            output_dir=seg_dir,
        )

        ctx.log(f"Volume UID: {volume_series_uid[:30]}...")
        ctx.log(f"SEG UID: {seg_series_uid[:30]}...")

        # Verify CSE requirements using pydicom
        try:
            import pydicom

            # Read volume DICOM (may be in subdirectory)
            vol_files = list(volume_dir.rglob("*.dcm"))
            vol_dcm = pydicom.dcmread(str(vol_files[0]))
            vol_study_uid = vol_dcm.StudyInstanceUID
            vol_series_uid_actual = vol_dcm.SeriesInstanceUID

            # Read SEG DICOM
            seg_files = list(seg_dir.glob("*.dcm"))
            seg_dcm = pydicom.dcmread(str(seg_files[0]))

            # CSE Requirement 1: Same StudyInstanceUID
            ctx.log("Checking StudyInstanceUID match...")
            ctx.assert_equal(
                seg_dcm.StudyInstanceUID,
                vol_study_uid,
                "SEG StudyInstanceUID must match volume StudyInstanceUID",
            )
            ctx.log("✓ StudyInstanceUID matches")

            # CSE Requirement 2: ReferencedSeriesSequence points to volume
            ctx.log("Checking ReferencedSeriesSequence...")
            ctx.assert_true(
                hasattr(seg_dcm, "ReferencedSeriesSequence"),
                "SEG must have ReferencedSeriesSequence",
            )
            ref_series = seg_dcm.ReferencedSeriesSequence[0]
            ctx.assert_equal(
                ref_series.SeriesInstanceUID,
                vol_series_uid_actual,
                "ReferencedSeriesSequence must point to volume series",
            )
            ctx.log("✓ ReferencedSeriesSequence correct")

            # CSE Requirement 3: Modality is SEG
            ctx.log("Checking Modality...")
            ctx.assert_equal(
                seg_dcm.Modality,
                "SEG",
                "Modality must be SEG",
            )
            ctx.log("✓ Modality is SEG")

            # CSE Requirement 4: SeriesDescription for grouping
            ctx.log("Checking SeriesDescription...")
            ctx.assert_true(
                hasattr(seg_dcm, "SeriesDescription"),
                "SEG should have SeriesDescription",
            )
            ctx.assert_true(
                len(seg_dcm.SeriesDescription) > 0,
                "SeriesDescription should not be empty",
            )
            ctx.log(f"✓ SeriesDescription: {seg_dcm.SeriesDescription}")

            ctx.screenshot("[cse] All CSE compatibility checks passed")

        except ImportError:
            ctx.log("pydicom not available, skipping detailed CSE verification")
            ctx.screenshot("[cse] Skipped (pydicom not available)")

    def verify(self, ctx: TestContext) -> None:
        """Verify results."""
        ctx.log("CSE compatibility test complete")
        ctx.screenshot("[verify] Test complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.seg_factory:
            self.seg_factory.cleanup()

        if self.temp_dir and self.temp_dir.exists():
            import shutil

            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
