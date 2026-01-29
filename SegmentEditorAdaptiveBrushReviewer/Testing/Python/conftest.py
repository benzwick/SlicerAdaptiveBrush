"""Pytest configuration and fixtures for SegmentEditorAdaptiveBrushReviewer tests."""

import importlib.util
import os
import sys

import numpy as np
import pytest

# Add library path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEWER_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_LIB_DIR = os.path.join(_REVIEWER_DIR, "SegmentEditorAdaptiveBrushReviewerLib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# Check for optional dependencies
HAS_PYDICOM = importlib.util.find_spec("pydicom") is not None
HAS_HIGHDICOM = importlib.util.find_spec("highdicom") is not None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_pydicom: mark test as requiring pydicom")
    config.addinivalue_line("markers", "requires_highdicom: mark test as requiring highdicom")
    config.addinivalue_line("markers", "requires_slicer: mark test as requiring 3D Slicer")


# Skip decorators
requires_pydicom = pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom not available")
requires_highdicom = pytest.mark.skipif(not HAS_HIGHDICOM, reason="highdicom not available")
requires_slicer = pytest.mark.skip(reason="Requires 3D Slicer environment")


@pytest.fixture
def sample_labelmap_array():
    """Create a sample labelmap array with multiple segments.

    Shape: (10, 64, 64) - 10 slices, 64x64 each
    Labels: 0=background, 1=segment1 (sphere), 2=segment2 (cube)
    """
    size = 64
    slices = 10
    labelmap = np.zeros((slices, size, size), dtype=np.uint8)

    # Segment 1: sphere in center
    center = size // 2
    radius = 10
    for z in range(slices):
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - slices // 2) ** 2)
                if dist <= radius:
                    labelmap[z, y, x] = 1

    # Segment 2: cube in corner
    cube_start = 5
    cube_end = 15
    labelmap[2:8, cube_start:cube_end, cube_start:cube_end] = 2

    return labelmap


@pytest.fixture
def mock_dicom_dataset():
    """Create a mock pydicom dataset for testing SOP Class detection."""
    if not HAS_PYDICOM:
        pytest.skip("pydicom not available")

    from pydicom import Dataset

    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.7"  # Label Map Segmentation Storage
    ds.Modality = "SEG"
    ds.SeriesDescription = "Test Segmentation"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9"
    return ds


@pytest.fixture
def mock_standard_seg_dataset():
    """Create a mock pydicom dataset for standard SEG (not LABELMAP)."""
    if not HAS_PYDICOM:
        pytest.skip("pydicom not available")

    from pydicom import Dataset

    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"  # Segmentation Storage (BINARY)
    ds.Modality = "SEG"
    ds.SeriesDescription = "Binary Segmentation"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.10"
    return ds


@pytest.fixture
def cielab_red():
    """CIELab values for red color (DICOM format: L=0-65535, a/b centered at 32768)."""
    # Approximate CIELab for red (L=53, a=80, b=67)
    # DICOM format: L scaled to 65535, a/b offset by 32768
    L = int(53.2 * 65535 / 100)  # ~34898
    a = int(80 * 65535 / 255) + 32768  # ~53380
    b = int(67 * 65535 / 255) + 32768  # ~49997
    return [L, a, b]


@pytest.fixture
def cielab_green():
    """CIELab values for green color."""
    # Approximate CIELab for green (L=87, a=-86, b=83)
    L = int(87.7 * 65535 / 100)
    a = int(-86 * 65535 / 255) + 32768
    b = int(83 * 65535 / 255) + 32768
    return [L, a, b]
