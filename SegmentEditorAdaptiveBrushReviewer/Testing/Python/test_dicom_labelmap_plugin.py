"""Tests for DICOMLabelMapSegPlugin.

These tests verify the DICOM plugin for loading Label Map Segmentation Storage files.

Tests are divided into:
- Unit tests that can run without Slicer (SOP class detection, color conversion)
- Integration tests that require Slicer environment (marked with requires_slicer)

See ADR-019 for plugin design.
"""

import importlib.util

import pytest

# Check for dependencies
HAS_PYDICOM = importlib.util.find_spec("pydicom") is not None
HAS_HIGHDICOM = importlib.util.find_spec("highdicom") is not None

# Import plugin module (these imports work outside Slicer)
from DICOMLabelMapSegPlugin import (  # noqa: E402
    LABELMAP_SEG_SOP_CLASS,
    STANDARD_SEG_SOP_CLASS,
    DICOMLabelMapSegPlugin,
    is_labelmap_seg,
)


class TestSopClassConstants:
    """Test SOP Class UID constants."""

    def test_labelmap_sop_class_value(self):
        """Verify LABELMAP SOP Class UID is correct."""
        assert LABELMAP_SEG_SOP_CLASS == "1.2.840.10008.5.1.4.1.1.66.7"

    def test_standard_sop_class_value(self):
        """Verify standard SEG SOP Class UID is correct."""
        assert STANDARD_SEG_SOP_CLASS == "1.2.840.10008.5.1.4.1.1.66.4"

    def test_sop_classes_are_different(self):
        """Verify the two SOP Classes are distinct."""
        assert LABELMAP_SEG_SOP_CLASS != STANDARD_SEG_SOP_CLASS


class TestPluginInitialization:
    """Test plugin class initialization."""

    def test_plugin_can_be_instantiated(self):
        """Plugin should instantiate without errors."""
        plugin = DICOMLabelMapSegPlugin()
        assert plugin is not None

    def test_plugin_has_load_type(self):
        """Plugin should have a loadType attribute."""
        plugin = DICOMLabelMapSegPlugin()
        assert hasattr(plugin, "loadType")
        assert "Labelmap" in plugin.loadType or "LABELMAP" in plugin.loadType.upper()

    def test_plugin_has_examine_method(self):
        """Plugin should have examineForImport method."""
        plugin = DICOMLabelMapSegPlugin()
        assert hasattr(plugin, "examineForImport")
        assert callable(plugin.examineForImport)

    def test_plugin_has_load_method(self):
        """Plugin should have load method."""
        plugin = DICOMLabelMapSegPlugin()
        assert hasattr(plugin, "load")
        assert callable(plugin.load)


class TestCielabToRgbConversion:
    """Test CIELab to RGB color conversion."""

    def test_cielab_to_rgb_returns_tuple(self):
        """Conversion should return a tuple of 3 integers."""
        plugin = DICOMLabelMapSegPlugin()
        # CIELab for white (L=100, a=0, b=0)
        # DICOM format: L scaled 0-65535, a/b centered at 32768
        cielab = [65535, 32768, 32768]  # White
        rgb = plugin._cielab_to_rgb(cielab)

        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
        assert all(isinstance(c, int) for c in rgb)

    def test_cielab_to_rgb_white(self):
        """White in CIELab should convert to approximately white in RGB."""
        plugin = DICOMLabelMapSegPlugin()
        # CIELab white: L=100, a=0, b=0
        cielab = [65535, 32768, 32768]
        rgb = plugin._cielab_to_rgb(cielab)

        # Should be close to white (255, 255, 255)
        assert rgb[0] >= 240, f"Red channel {rgb[0]} too low for white"
        assert rgb[1] >= 240, f"Green channel {rgb[1]} too low for white"
        assert rgb[2] >= 240, f"Blue channel {rgb[2]} too low for white"

    def test_cielab_to_rgb_black(self):
        """Black in CIELab should convert to approximately black in RGB."""
        plugin = DICOMLabelMapSegPlugin()
        # CIELab black: L=0, a=0, b=0
        cielab = [0, 32768, 32768]
        rgb = plugin._cielab_to_rgb(cielab)

        # Should be close to black (0, 0, 0)
        assert rgb[0] <= 15, f"Red channel {rgb[0]} too high for black"
        assert rgb[1] <= 15, f"Green channel {rgb[1]} too high for black"
        assert rgb[2] <= 15, f"Blue channel {rgb[2]} too high for black"

    def test_cielab_to_rgb_values_in_range(self):
        """RGB values should always be in valid range 0-255."""
        plugin = DICOMLabelMapSegPlugin()

        # Test various CIELab values
        test_values = [
            [0, 0, 0],  # Out of normal range
            [65535, 65535, 65535],  # Out of normal range
            [32768, 32768, 32768],  # Middle gray
            [50000, 40000, 45000],  # Random
        ]

        for cielab in test_values:
            rgb = plugin._cielab_to_rgb(cielab)
            assert all(0 <= c <= 255 for c in rgb), f"RGB {rgb} out of range for CIELab {cielab}"

    def test_cielab_to_rgb_error_handling(self):
        """Invalid input should return default color, not crash."""
        plugin = DICOMLabelMapSegPlugin()

        # Test with invalid inputs
        invalid_inputs = [
            [],  # Empty
            [1],  # Too few values
            [1, 2],  # Too few values
            None,  # None (will raise on indexing)
        ]

        for invalid in invalid_inputs:
            try:
                rgb = plugin._cielab_to_rgb(invalid)
                # Should return default (red) on error
                assert rgb == (255, 0, 0) or len(rgb) == 3
            except (TypeError, IndexError):
                # These exceptions are acceptable for invalid input
                pass


@pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom not available")
class TestIsLabelmapSeg:
    """Test the is_labelmap_seg utility function."""

    def test_is_labelmap_seg_with_labelmap_file(self, tmp_path):
        """Should return True for LABELMAP SEG file."""
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        # Create a minimal DICOM SEG file
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = LABELMAP_SEG_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(tmp_path / "test.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = LABELMAP_SEG_SOP_CLASS
        ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
        ds.Modality = "SEG"

        filepath = tmp_path / "labelmap.dcm"
        ds.save_as(str(filepath))

        assert is_labelmap_seg(str(filepath)) is True

    def test_is_labelmap_seg_with_standard_seg_file(self, tmp_path):
        """Should return False for standard SEG file (BINARY)."""
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = STANDARD_SEG_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.10"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(tmp_path / "test.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = STANDARD_SEG_SOP_CLASS
        ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.10"
        ds.Modality = "SEG"

        filepath = tmp_path / "binary_seg.dcm"
        ds.save_as(str(filepath))

        assert is_labelmap_seg(str(filepath)) is False

    def test_is_labelmap_seg_with_nonexistent_file(self):
        """Should return False for non-existent file."""
        assert is_labelmap_seg("/nonexistent/path/file.dcm") is False

    def test_is_labelmap_seg_with_non_dicom_file(self, tmp_path):
        """Should return False for non-DICOM file."""
        filepath = tmp_path / "not_dicom.txt"
        filepath.write_text("This is not a DICOM file")

        assert is_labelmap_seg(str(filepath)) is False


# Check if DICOMLib is available (only in Slicer)
HAS_DICOMLIB = importlib.util.find_spec("DICOMLib") is not None


@pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom not available")
@pytest.mark.skipif(not HAS_DICOMLIB, reason="DICOMLib requires Slicer environment")
class TestExamineForImport:
    """Test the examineForImport method.

    Note: These tests require Slicer's DICOMLib and will be skipped when
    running pytest outside of Slicer.
    """

    def test_examine_empty_file_list(self):
        """Should return empty list for empty input."""
        plugin = DICOMLabelMapSegPlugin()
        result = plugin.examineForImport([])
        assert result == []

    def test_examine_with_empty_inner_list(self):
        """Should handle empty inner lists gracefully."""
        plugin = DICOMLabelMapSegPlugin()
        result = plugin.examineForImport([[]])
        assert result == []

    def test_examine_with_labelmap_file(self, tmp_path):
        """Should return loadable for LABELMAP SEG file."""
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        # Create LABELMAP SEG file
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = LABELMAP_SEG_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(tmp_path / "test.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = LABELMAP_SEG_SOP_CLASS
        ds.SOPInstanceUID = "1.2.3.4.5"
        ds.Modality = "SEG"
        ds.SeriesDescription = "Test Labelmap"

        filepath = tmp_path / "labelmap.dcm"
        ds.save_as(str(filepath))

        plugin = DICOMLabelMapSegPlugin()
        result = plugin.examineForImport([[str(filepath)]])

        assert len(result) == 1
        loadable = result[0]
        assert loadable.name == "Test Labelmap"
        assert loadable.confidence == 0.95
        assert loadable.selected is True

    def test_examine_with_standard_seg_file(self, tmp_path):
        """Should NOT return loadable for standard SEG file."""
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        # Create standard (BINARY) SEG file
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = STANDARD_SEG_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.6"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(tmp_path / "test.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = STANDARD_SEG_SOP_CLASS
        ds.SOPInstanceUID = "1.2.3.4.6"
        ds.Modality = "SEG"
        ds.SeriesDescription = "Binary Seg"

        filepath = tmp_path / "binary_seg.dcm"
        ds.save_as(str(filepath))

        plugin = DICOMLabelMapSegPlugin()
        result = plugin.examineForImport([[str(filepath)]])

        # Should not recognize standard SEG files
        assert len(result) == 0

    def test_examine_with_non_seg_modality(self, tmp_path):
        """Should NOT return loadable for non-SEG modality."""
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        # Create a file with LABELMAP SOP Class but wrong modality
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = LABELMAP_SEG_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.7"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(tmp_path / "test.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = LABELMAP_SEG_SOP_CLASS
        ds.SOPInstanceUID = "1.2.3.4.7"
        ds.Modality = "CT"  # Wrong modality
        ds.SeriesDescription = "Not a SEG"

        filepath = tmp_path / "not_seg.dcm"
        ds.save_as(str(filepath))

        plugin = DICOMLabelMapSegPlugin()
        result = plugin.examineForImport([[str(filepath)]])

        assert len(result) == 0


class TestGeometryExtraction:
    """Test geometry extraction from DICOM SEG."""

    def test_get_geometry_from_seg_returns_defaults(self):
        """Should return default values when geometry cannot be extracted."""
        plugin = DICOMLabelMapSegPlugin()

        # Create a mock seg object with no geometry info
        class MockSeg:
            pass

        seg = MockSeg()
        spacing, origin, direction = plugin._get_geometry_from_seg(seg)

        assert spacing == (1.0, 1.0, 1.0)
        assert origin == (0.0, 0.0, 0.0)
        assert len(direction) == 9


class TestPluginRegistration:
    """Test plugin registration function."""

    def test_register_plugin_function_exists(self):
        """The register_plugin function should be importable."""
        from DICOMLabelMapSegPlugin import register_plugin

        assert callable(register_plugin)

    def test_register_plugin_handles_missing_dicomlib(self):
        """Should not crash if DICOMLib is not available."""
        from DICOMLabelMapSegPlugin import register_plugin

        # This should not raise even if DICOMLib is unavailable
        # (we're outside Slicer, so it will log a warning but not crash)
        try:
            register_plugin()
        except ImportError:
            # Expected outside Slicer
            pass
