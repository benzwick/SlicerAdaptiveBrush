"""Pytest configuration and fixtures for SlicerAdaptiveBrush tests."""

import importlib.util
import os
import sys

import numpy as np
import pytest

# Add library path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)), "SegmentEditorAdaptiveBrushLib"
)
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# Try to import SimpleITK (available in Slicer, may not be in standalone pytest)
try:
    import SimpleITK as sitk

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False

# Check sklearn availability for GMM tests
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_sitk: mark test as requiring SimpleITK")
    config.addinivalue_line("markers", "requires_sklearn: mark test as requiring scikit-learn")
    config.addinivalue_line("markers", "requires_slicer: mark test as requiring 3D Slicer")


# Skip decorators
requires_sitk = pytest.mark.skipif(not HAS_SIMPLEITK, reason="SimpleITK not available")
requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")


@pytest.fixture
def uniform_image_array():
    """Create a uniform intensity numpy array (100x100x10)."""
    return np.full((10, 100, 100), fill_value=100, dtype=np.float32)


@pytest.fixture
def bimodal_image_array():
    """
    Create a numpy array with two distinct intensity regions.

    Left half: mean=100, std=10
    Right half: mean=200, std=10
    Shape: (10, 100, 100) - 10 slices, 100x100 each
    """
    np.random.seed(42)  # Reproducible tests
    image = np.zeros((10, 100, 100), dtype=np.float32)

    # Left half: low intensity region
    image[:, :, :50] = np.random.normal(100, 10, (10, 100, 50)).astype(np.float32)

    # Right half: high intensity region
    image[:, :, 50:] = np.random.normal(200, 10, (10, 100, 50)).astype(np.float32)

    return image


@pytest.fixture
def gradient_image_array():
    """
    Create a numpy array with smooth intensity gradient.

    Intensity increases linearly from 0 to 255 along x-axis.
    Shape: (10, 100, 100)
    """
    image = np.zeros((10, 100, 100), dtype=np.float32)
    gradient = np.linspace(0, 255, 100, dtype=np.float32)
    image[:, :, :] = gradient[np.newaxis, np.newaxis, :]
    return image


@pytest.fixture
def noisy_bimodal_image_array():
    """
    Create a bimodal image with significant noise.

    Left half: mean=100, std=30 (high noise)
    Right half: mean=200, std=30 (high noise)
    """
    np.random.seed(42)
    image = np.zeros((10, 100, 100), dtype=np.float32)
    image[:, :, :50] = np.random.normal(100, 30, (10, 100, 50)).astype(np.float32)
    image[:, :, 50:] = np.random.normal(200, 30, (10, 100, 50)).astype(np.float32)
    return image


@pytest.fixture
def sphere_mask_array():
    """
    Create a binary sphere mask centered in a volume.

    Shape: (50, 50, 50), sphere radius: 15 voxels
    """
    size = 50
    radius = 15
    center = size // 2

    z, y, x = np.ogrid[:size, :size, :size]
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
    mask = (distance <= radius).astype(np.uint8)

    return mask


@pytest.fixture
@requires_sitk
def uniform_sitk_image(uniform_image_array):
    """Create a uniform SimpleITK image."""
    image = sitk.GetImageFromArray(uniform_image_array)
    image.SetSpacing([1.0, 1.0, 2.0])  # Anisotropic spacing
    image.SetOrigin([0.0, 0.0, 0.0])
    return image


@pytest.fixture
@requires_sitk
def bimodal_sitk_image(bimodal_image_array):
    """Create a bimodal SimpleITK image."""
    image = sitk.GetImageFromArray(bimodal_image_array)
    image.SetSpacing([1.0, 1.0, 2.0])
    image.SetOrigin([0.0, 0.0, 0.0])
    return image


@pytest.fixture
def seed_in_low_region():
    """Seed point in the low-intensity region (left half)."""
    return (25, 50, 5)  # (x, y, z) in image coordinates


@pytest.fixture
def seed_in_high_region():
    """Seed point in the high-intensity region (right half)."""
    return (75, 50, 5)  # (x, y, z) in image coordinates


@pytest.fixture
def brush_params_default():
    """Default brush parameters for testing."""
    return {
        "radius_mm": 10.0,
        "edge_sensitivity": 0.5,
        "adaptive_mode": True,
        "algorithm": "watershed",
        "backend": "cpu",
        "refinement_level": "balanced",
    }


@pytest.fixture
def brush_params_fast():
    """Fast brush parameters (lower quality, higher speed)."""
    return {
        "radius_mm": 10.0,
        "edge_sensitivity": 0.3,
        "adaptive_mode": True,
        "algorithm": "connected_threshold",
        "backend": "cpu",
        "refinement_level": "fast",
    }


@pytest.fixture
def brush_params_precise():
    """Precise brush parameters (higher quality, lower speed)."""
    return {
        "radius_mm": 10.0,
        "edge_sensitivity": 0.8,
        "adaptive_mode": True,
        "algorithm": "level_set",
        "backend": "cpu",
        "refinement_level": "precise",
    }
