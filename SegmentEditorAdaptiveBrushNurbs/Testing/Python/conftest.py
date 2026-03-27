"""Pytest configuration for NURBS module tests.

Provides fixtures and markers for testing NURBS generation
outside of Slicer.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

# Add library to path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)),
    "SegmentEditorAdaptiveBrushNurbsLib",
)
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# Also add the main library for DependencyManager
_MAIN_LIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))),
    "SegmentEditorAdaptiveBrush",
    "SegmentEditorAdaptiveBrushLib",
)
if _MAIN_LIB_DIR not in sys.path:
    sys.path.insert(0, _MAIN_LIB_DIR)

# Check for optional dependencies
HAS_GEOMDL = importlib.util.find_spec("geomdl") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_SIMPLEITK = importlib.util.find_spec("SimpleITK") is not None
HAS_VTK = importlib.util.find_spec("vtk") is not None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_geomdl: skip test if geomdl is not available",
    )
    config.addinivalue_line(
        "markers",
        "requires_scipy: skip test if scipy is not available",
    )
    config.addinivalue_line(
        "markers",
        "requires_sitk: skip test if SimpleITK is not available",
    )
    config.addinivalue_line(
        "markers",
        "requires_vtk: skip test if VTK is not available",
    )


# Skip decorators for optional dependencies
requires_geomdl = pytest.mark.skipif(not HAS_GEOMDL, reason="geomdl not available")
requires_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
requires_sitk = pytest.mark.skipif(not HAS_SIMPLEITK, reason="SimpleITK not available")
requires_vtk = pytest.mark.skipif(not HAS_VTK, reason="VTK not available")


# Fixtures


@pytest.fixture
def simple_control_points():
    """Create simple 2x2x2 control points for a unit cube."""
    points = np.zeros((2, 2, 2, 3))

    # Create unit cube
    for i in range(2):
        for j in range(2):
            for k in range(2):
                points[i, j, k, :] = [float(i), float(j), float(k)]

    return points


@pytest.fixture
def cubic_control_points():
    """Create 4x4x4 control points for cubic NURBS."""
    points = np.zeros((4, 4, 4, 3))

    # Create regular grid
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points[i, j, k, :] = [
                    float(i) / 3.0,
                    float(j) / 3.0,
                    float(k) / 3.0,
                ]

    return points


@pytest.fixture
def sphere_labelmap():
    """Create a binary sphere labelmap for testing."""
    size = 50
    center = size // 2
    radius = 15

    # Create 3D grid
    z, y, x = np.ogrid[:size, :size, :size]
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

    # Binary sphere
    labelmap = (distance <= radius).astype(np.uint8)

    return labelmap


@pytest.fixture
def tube_labelmap():
    """Create a binary tube (cylinder) labelmap for testing."""
    size = 50
    center = size // 2
    radius = 8

    # Create 3D grid
    z, y, x = np.ogrid[:size, :size, :size]

    # Cylinder along z-axis
    distance_xy = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    labelmap = (distance_xy <= radius).astype(np.uint8)

    return labelmap
