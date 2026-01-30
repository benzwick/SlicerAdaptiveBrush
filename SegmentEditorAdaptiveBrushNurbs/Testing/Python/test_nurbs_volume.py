"""Unit tests for NURBS volume generation.

Tests the core components:
- HexMesh creation
- NurbsVolumeBuilder
- Knot vector generation
- Quality metrics (basic)
"""

import numpy as np
import pytest
from conftest import requires_geomdl, requires_scipy


class TestHexMesh:
    """Tests for HexMesh data class."""

    def test_hex_mesh_creation(self, cubic_control_points):
        """Test creating a HexMesh with valid control points."""
        from HexMeshGenerator import HexMesh

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        assert hex_mesh.num_u == 4
        assert hex_mesh.num_v == 4
        assert hex_mesh.num_w == 4
        assert hex_mesh.num_control_points == 64

    def test_hex_mesh_weights_default(self, cubic_control_points):
        """Test that weights default to 1.0."""
        from HexMeshGenerator import HexMesh

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        assert hex_mesh.weights.shape == (4, 4, 4)
        assert np.allclose(hex_mesh.weights, 1.0)

    def test_hex_mesh_flat_arrays(self, cubic_control_points):
        """Test flat array properties."""
        from HexMeshGenerator import HexMesh

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        assert hex_mesh.flat_control_points.shape == (64, 3)
        assert hex_mesh.flat_weights.shape == (64,)


class TestNurbsVolumeBuilder:
    """Tests for NURBS volume construction."""

    def test_clamped_knot_vector(self):
        """Test clamped knot vector generation."""
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        builder = NurbsVolumeBuilder()

        # n=4 control points, degree=3 (cubic)
        knots = builder._compute_clamped_knot_vector(n=4, degree=3)

        # Should have n + degree + 1 = 8 knots
        assert len(knots) == 8

        # First 4 should be 0
        assert np.allclose(knots[:4], 0.0)

        # Last 4 should be 1
        assert np.allclose(knots[4:], 1.0)

    def test_clamped_knot_vector_more_control_points(self):
        """Test knot vector with more control points."""
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        builder = NurbsVolumeBuilder()

        # n=6 control points, degree=3
        knots = builder._compute_clamped_knot_vector(n=6, degree=3)

        # Should have n + degree + 1 = 10 knots
        assert len(knots) == 10

        # First 4 should be 0
        assert np.allclose(knots[:4], 0.0)

        # Last 4 should be 1
        assert np.allclose(knots[6:], 1.0)

        # Interior knots should be uniformly spaced
        # With 2 interior knots between 0 and 1
        # They should be at 1/3 and 2/3
        assert abs(knots[4] - 1 / 3) < 0.01 or abs(knots[4] - 0.5) < 0.01

    def test_build_nurbs_volume_minimal(self, cubic_control_points):
        """Test building a minimal NURBS volume."""
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        builder = NurbsVolumeBuilder()
        nurbs = builder.build(hex_mesh, degree=3)

        assert nurbs.num_control_points == 64
        assert nurbs.degrees == (3, 3, 3)
        assert nurbs.size == (4, 4, 4)

    def test_build_nurbs_volume_too_small(self, simple_control_points):
        """Test that building fails if control mesh is too small for degree."""
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=simple_control_points,
            num_u=2,
            num_v=2,
            num_w=2,
        )

        builder = NurbsVolumeBuilder()

        with pytest.raises(ValueError, match="too small"):
            builder.build(hex_mesh, degree=3)

    def test_build_nurbs_volume_degree_1(self, simple_control_points):
        """Test building linear (degree 1) NURBS volume."""
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=simple_control_points,
            num_u=2,
            num_v=2,
            num_w=2,
        )

        builder = NurbsVolumeBuilder()
        nurbs = builder.build(hex_mesh, degree=1)

        assert nurbs.num_control_points == 8
        assert nurbs.degrees == (1, 1, 1)

    @requires_geomdl
    def test_build_with_geomdl(self, cubic_control_points):
        """Test that geomdl volume is created when available."""
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        builder = NurbsVolumeBuilder()
        nurbs = builder.build(hex_mesh, degree=3)

        assert nurbs.geomdl_volume is not None

    @requires_geomdl
    def test_evaluate_nurbs(self, cubic_control_points):
        """Test evaluating NURBS at parametric points."""
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        builder = NurbsVolumeBuilder()
        nurbs = builder.build(hex_mesh, degree=3)

        # Evaluate at corners
        pt_000 = builder.evaluate(nurbs, 0, 0, 0)
        pt_111 = builder.evaluate(nurbs, 1, 1, 1)

        # Corners should be at (0,0,0) and (1,1,1) for unit cube
        assert np.allclose(pt_000, [0, 0, 0], atol=0.01)
        assert np.allclose(pt_111, [1, 1, 1], atol=0.01)

        # Evaluate at center
        pt_center = builder.evaluate(nurbs, 0.5, 0.5, 0.5)
        assert np.allclose(pt_center, [0.5, 0.5, 0.5], atol=0.1)


class TestMfemExport:
    """Tests for MFEM mesh export."""

    def test_mfem_export_format(self, cubic_control_points, tmp_path):
        """Test that MFEM export creates valid format."""
        from Exporters.MfemExporter import MfemExporter
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        builder = NurbsVolumeBuilder()
        nurbs = builder.build(hex_mesh, degree=3)

        exporter = MfemExporter()
        output_path = tmp_path / "test.mesh"
        exporter.export(nurbs, output_path)

        # Check file was created
        assert output_path.exists()

        # Check file content
        content = output_path.read_text()
        assert "MFEM NURBS mesh v1.0" in content
        assert "dimension\n3" in content
        assert "elements\n1" in content
        assert "boundary\n6" in content
        assert "knotvectors\n3" in content
        assert "weights\n64" in content
        assert "FiniteElementSpace" in content


class TestHexMeshGenerator:
    """Tests for hexahedral mesh generation."""

    def test_oriented_bounding_box(self, sphere_labelmap):
        """Test oriented bounding box computation."""
        from HexMeshGenerator import HexMeshGenerator

        generator = HexMeshGenerator()
        center, axes, extents = generator._compute_oriented_bounding_box(sphere_labelmap)

        # Center should be near volume center
        expected_center = np.array([25, 25, 25])
        assert np.allclose(center, expected_center, atol=5)

        # Axes should be orthonormal
        for i in range(3):
            assert np.allclose(np.linalg.norm(axes[i]), 1.0)

        # Extents should be approximately sphere radius * 1.1
        assert all(e > 10 for e in extents)
        assert all(e < 25 for e in extents)

    def test_create_grid_control_points(self):
        """Test creating regular grid of control points."""
        from HexMeshGenerator import HexMeshGenerator

        generator = HexMeshGenerator()

        center = np.array([0, 0, 0])
        axes = np.eye(3)  # Identity axes
        extents = np.array([1, 1, 1])

        points = generator._create_grid_control_points(center, axes, extents, resolution=4)

        assert points.shape == (4, 4, 4, 3)

        # Check corners
        assert np.allclose(points[0, 0, 0], [-1, -1, -1])
        assert np.allclose(points[3, 3, 3], [1, 1, 1])


class TestQualityMetrics:
    """Tests for quality metrics computation."""

    @requires_scipy
    def test_surface_distances(self):
        """Test surface distance computation."""
        from QualityMetrics import QualityMetrics

        metrics = QualityMetrics()

        # Create two overlapping point clouds
        np.random.seed(42)
        surface_a = np.random.randn(100, 3)
        surface_b = surface_a + np.random.randn(100, 3) * 0.1  # Small perturbation

        hausdorff, mean_dist, rms_dist = metrics._compute_surface_distances(surface_a, surface_b)

        # Distances should be small due to small perturbation
        assert hausdorff < 1.0
        assert mean_dist < 0.5
        assert rms_dist < 0.5
