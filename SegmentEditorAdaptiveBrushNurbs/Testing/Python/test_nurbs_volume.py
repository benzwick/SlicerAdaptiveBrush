"""Unit tests for NURBS volume generation.

Tests the core components:
- HexMesh creation
- NurbsVolumeBuilder
- Knot vector generation
- Quality metrics (basic)
"""

import numpy as np
import pytest
from conftest import requires_geomdl, requires_scipy, requires_vtk


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


class TestCenterline:
    """Tests for Centerline data class."""

    def test_centerline_creation(self):
        """Test creating a Centerline with valid data."""
        from SkeletonExtractor import Centerline

        # Simple straight centerline
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
                [0, 0, 20],
                [0, 0, 30],
            ]
        )
        radii = np.array([5.0, 5.0, 5.0, 5.0])

        centerline = Centerline(points=points, radii=radii)

        assert centerline.num_points == 4
        assert np.isclose(centerline.length, 30.0)

    def test_centerline_resample(self):
        """Test resampling a centerline to different point count."""
        from SkeletonExtractor import Centerline

        # Simple straight centerline
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
                [0, 0, 20],
                [0, 0, 30],
            ]
        )
        radii = np.array([5.0, 5.0, 5.0, 5.0])

        centerline = Centerline(points=points, radii=radii)

        # Resample to 7 points
        resampled = centerline.resample(7)

        assert resampled.num_points == 7
        assert np.isclose(resampled.length, 30.0)  # Length should be preserved
        assert resampled.tangents is not None
        assert resampled.tangents.shape == (7, 3)

    def test_centerline_resample_radii(self):
        """Test that radii are interpolated correctly during resampling."""
        from SkeletonExtractor import Centerline

        # Centerline with varying radii
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
            ]
        )
        radii = np.array([2.0, 8.0])

        centerline = Centerline(points=points, radii=radii)

        # Resample to 5 points
        resampled = centerline.resample(5)

        # Radii should interpolate from 2.0 to 8.0
        assert np.isclose(resampled.radii[0], 2.0)
        assert np.isclose(resampled.radii[-1], 8.0)
        assert np.isclose(resampled.radii[2], 5.0)  # Midpoint should be 5.0


class TestTubularHexMesh:
    """Tests for tubular hexahedral mesh generation."""

    def test_sweep_circular_template(self):
        """Test sweeping a circular template along a straight centerline."""
        from HexMeshGenerator import HexMeshGenerator

        generator = HexMeshGenerator()

        # Straight centerline along z-axis
        centerline_points = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
                [0, 0, 20],
                [0, 0, 30],
            ]
        )
        radii = np.array([5.0, 5.0, 5.0, 5.0])

        # Compute tangents
        tangents = generator._compute_tangents_from_points(centerline_points)

        # Sweep with 8 circumferential points
        control_points = generator._sweep_circular_template(
            centerline_points, radii, tangents, resolution=8
        )

        # Shape should be (n_circ, n_radial, n_axial, 3)
        assert control_points.shape == (8, 2, 4, 3)

        # Outer points should be at radius distance from centerline
        for k in range(4):
            center = centerline_points[k]
            for i in range(8):
                outer_pt = control_points[i, 1, k, :]
                dist = np.linalg.norm(outer_pt[:2] - center[:2])  # XY distance
                assert np.isclose(dist, 5.0, atol=0.1)

    def test_parallel_transport_frames(self):
        """Test that parallel transport produces orthonormal frames."""
        from HexMeshGenerator import HexMeshGenerator

        generator = HexMeshGenerator()

        # Curved centerline
        t = np.linspace(0, 2 * np.pi, 20)
        centerline_points = np.column_stack(
            [
                10 * np.cos(t),
                10 * np.sin(t),
                t * 5,
            ]
        )

        tangents = generator._compute_tangents_from_points(centerline_points)
        normals, binormals = generator._compute_parallel_transport_frames(
            centerline_points, tangents
        )

        # Check orthonormality at each point
        for i in range(len(centerline_points)):
            t = tangents[i]
            n = normals[i]
            b = binormals[i]

            # Unit vectors
            assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-6)
            assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-6)
            assert np.isclose(np.linalg.norm(b), 1.0, atol=1e-6)

            # Orthogonal
            assert np.isclose(np.dot(t, n), 0, atol=1e-6)
            assert np.isclose(np.dot(t, b), 0, atol=1e-6)
            assert np.isclose(np.dot(n, b), 0, atol=1e-6)

    def test_tubular_hex_mesh_shape(self):
        """Test that tubular hex mesh has correct shape."""
        from HexMeshGenerator import HexMeshGenerator

        generator = HexMeshGenerator()

        # Straight centerline
        centerline_points = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
                [0, 0, 20],
            ]
        )
        radii = np.array([5.0, 5.0, 5.0])
        tangents = generator._compute_tangents_from_points(centerline_points)

        # Generate control points
        control_points = generator._sweep_circular_template(
            centerline_points, radii, tangents, resolution=8
        )

        # Verify dimensions
        n_circ, n_radial, n_axial, _ = control_points.shape
        assert n_circ == 8
        assert n_radial == 2  # Inner and outer layers
        assert n_axial == 3


class TestSkeletonExtractor:
    """Tests for SkeletonExtractor class (without Slicer)."""

    def test_vmtk_not_available(self):
        """Test that is_vmtk_available returns False outside Slicer."""
        from SkeletonExtractor import SkeletonExtractor

        extractor = SkeletonExtractor()

        # Outside of Slicer, VMTK should not be available
        assert not extractor.is_vmtk_available()

    def test_extract_centerline_requires_vmtk(self):
        """Test that extract_centerline raises error without VMTK."""
        from SkeletonExtractor import SkeletonExtractor

        extractor = SkeletonExtractor()

        # Should raise RuntimeError since VMTK is not available
        with pytest.raises(RuntimeError, match="SlicerVMTK"):
            extractor.extract_centerline(
                segmentation_node=None,  # Would fail anyway
                segment_id="test",
            )

    def test_detect_branches_requires_vmtk(self):
        """Test that detect_branches raises error without VMTK."""
        from SkeletonExtractor import SkeletonExtractor

        extractor = SkeletonExtractor()

        # Should raise RuntimeError since VMTK is not available
        with pytest.raises(RuntimeError, match="SlicerVMTK"):
            extractor.detect_branches(
                segmentation_node=None,
                segment_id="test",
            )


class TestBranchTemplates:
    """Tests for BranchTemplates class."""

    def test_create_bifurcation_template(self):
        """Test creating a bifurcation template."""
        from BranchTemplates import BranchTemplates

        templates = BranchTemplates(resolution=4)

        # Create a symmetric bifurcation
        center = np.array([0.0, 0.0, 0.0])
        parent_dir = np.array([0.0, 0.0, -1.0])
        child1_dir = np.array([0.5, 0.0, 0.866])  # ~60 degrees from parent
        child2_dir = np.array([-0.5, 0.0, 0.866])

        bif = templates.create_bifurcation(
            center=center,
            parent_dir=parent_dir,
            child1_dir=child1_dir,
            child2_dir=child2_dir,
            parent_radius=5.0,
            child1_radius=3.5,
            child2_radius=3.5,
        )

        # Should have 3 patches
        assert bif.num_patches == 3

        # Bifurcation angle should be about 60 degrees
        angle_deg = np.degrees(bif.bifurcation_angle)
        assert 50 < angle_deg < 70

    def test_bifurcation_type_classification(self):
        """Test that bifurcation types are classified correctly."""
        from BranchTemplates import BifurcationType, BranchTemplates

        templates = BranchTemplates()

        # Symmetric bifurcation
        bif_sym = templates.create_bifurcation(
            center=np.array([0.0, 0.0, 0.0]),
            parent_dir=np.array([0.0, 0.0, -1.0]),
            child1_dir=np.array([0.5, 0.0, 0.866]),
            child2_dir=np.array([-0.5, 0.0, 0.866]),
            parent_radius=5.0,
            child1_radius=4.0,
            child2_radius=4.0,
        )
        assert bif_sym.bifurcation_type == BifurcationType.SYMMETRIC

        # Side branch (small radius)
        bif_side = templates.create_bifurcation(
            center=np.array([0.0, 0.0, 0.0]),
            parent_dir=np.array([0.0, 0.0, -1.0]),
            child1_dir=np.array([0.0, 0.0, 1.0]),  # Main continues
            child2_dir=np.array([1.0, 0.0, 0.0]),  # Side branch
            parent_radius=5.0,
            child1_radius=4.5,
            child2_radius=2.0,  # Small side branch
        )
        assert bif_side.bifurcation_type == BifurcationType.SIDE_BRANCH

    def test_trifurcation_template(self):
        """Test creating a trifurcation template."""
        from BranchTemplates import BranchTemplates

        templates = BranchTemplates(resolution=4)

        center = np.array([0.0, 0.0, 0.0])
        parent_dir = np.array([0.0, 0.0, -1.0])
        child_dirs = [
            np.array([1.0, 0.0, 0.5]),
            np.array([-0.5, 0.866, 0.5]),
            np.array([-0.5, -0.866, 0.5]),
        ]

        trif = templates.create_trifurcation(
            center=center,
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            parent_radius=5.0,
            child_radii=[3.0, 3.0, 3.0],
        )

        # Trifurcation should have 4-5 patches
        assert trif.num_patches >= 4

    def test_junction_patch_control_points(self):
        """Test that junction patches have valid control points."""
        from BranchTemplates import BranchTemplates

        templates = BranchTemplates(resolution=4)

        bif = templates.create_bifurcation(
            center=np.array([0.0, 0.0, 0.0]),
            parent_dir=np.array([0.0, 0.0, -1.0]),
            child1_dir=np.array([0.5, 0.0, 0.866]),
            child2_dir=np.array([-0.5, 0.0, 0.866]),
            parent_radius=5.0,
            child1_radius=3.5,
            child2_radius=3.5,
        )

        for patch in bif.patches:
            # Each patch should have shape (resolution, resolution, resolution, 3)
            assert patch.control_points.shape == (4, 4, 4, 3)

            # All points should be finite
            assert np.all(np.isfinite(patch.control_points))


class TestBranchPoint:
    """Tests for BranchPoint dataclass."""

    def test_branch_point_creation(self):
        """Test creating a BranchPoint."""
        from SkeletonExtractor import BranchPoint

        bp = BranchPoint(
            position=np.array([10.0, 20.0, 30.0]),
            radius=5.0,
            branch_ids=[0, 1, 2],
        )

        assert np.allclose(bp.position, [10.0, 20.0, 30.0])
        assert bp.radius == 5.0
        assert bp.branch_ids == [0, 1, 2]

    def test_branch_point_with_directions(self):
        """Test BranchPoint with direction vectors."""
        from SkeletonExtractor import BranchPoint

        bp = BranchPoint(
            position=np.array([0.0, 0.0, 0.0]),
            parent_direction=np.array([0.0, 0.0, -1.0]),
            child_directions=[
                np.array([0.5, 0.0, 0.866]),
                np.array([-0.5, 0.0, 0.866]),
            ],
            radius=5.0,
        )

        assert bp.parent_direction is not None
        assert bp.child_directions is not None
        assert len(bp.child_directions) == 2


class TestContainmentValidator:
    """Tests for ContainmentValidator class."""

    def test_containment_result_creation(self):
        """Test creating a ContainmentResult."""
        from ContainmentValidator import ContainmentResult

        result = ContainmentResult(
            is_contained=True,
            containment_ratio=0.95,
            outside_points=np.zeros((0, 3)),
            max_outside_distance=0.0,
        )

        assert result.is_contained is True
        assert result.containment_ratio == 0.95
        assert len(result.outside_points) == 0

    def test_containment_result_with_outside_points(self):
        """Test ContainmentResult with outside points."""
        from ContainmentValidator import ContainmentResult

        outside_pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = ContainmentResult(
            is_contained=False,
            containment_ratio=0.85,
            outside_points=outside_pts,
            max_outside_distance=2.5,
        )

        assert result.is_contained is False
        assert result.containment_ratio == 0.85
        assert len(result.outside_points) == 2
        assert result.max_outside_distance == 2.5

    def test_validator_initialization(self):
        """Test ContainmentValidator initialization."""
        from ContainmentValidator import ContainmentValidator

        validator = ContainmentValidator(tolerance=1.0, surface_resolution=20)

        assert validator.tolerance == 1.0
        assert validator.surface_resolution == 20

    def test_expansion_vectors_computation(self, cubic_control_points):
        """Test computing expansion vectors for outside points."""
        from ContainmentValidator import ContainmentValidator
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        # Create a simple hex mesh
        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )

        # Build NURBS volume
        builder = NurbsVolumeBuilder()
        nurbs_vol = builder.build(hex_mesh, degree=1)

        # Create validator
        validator = ContainmentValidator()

        # Points outside the bounding box
        outside_points = np.array(
            [
                [15.0, 5.0, 5.0],  # Beyond x max
                [5.0, 15.0, 5.0],  # Beyond y max
            ]
        )

        # Compute expansion vectors
        control_pts_3d = nurbs_vol.control_points.reshape(4, 4, 4, 3)
        vectors = validator._compute_expansion_vectors(
            control_pts_3d, outside_points, expansion_factor=1.1
        )

        # Should have computed some expansion vectors
        assert len(vectors) > 0

    def test_boundary_expansion_preserves_interior(self, cubic_control_points):
        """Test that boundary expansion doesn't distort interior."""
        from ContainmentValidator import ContainmentValidator

        # Create 4x4x4 control points
        control_pts_3d = cubic_control_points.copy()

        # Create expansion at one corner
        expansion_vectors = {
            (3, 3, 3): np.array([1.0, 1.0, 1.0]),  # Expand corner outward
        }

        validator = ContainmentValidator()
        expanded = validator._apply_boundary_expansion(control_pts_3d, expansion_vectors)

        # The expanded corner should be moved
        assert not np.allclose(expanded[3, 3, 3, :], control_pts_3d[3, 3, 3, :])

        # Interior points not directly adjacent should be unchanged
        # (Actually with smoothing, some neighbors might change slightly)
        # Check that the opposite corner is unchanged
        assert np.allclose(expanded[0, 0, 0, :], control_pts_3d[0, 0, 0, :])


class TestQualityMetricsJacobian:
    """Tests for QualityMetrics Jacobian computation."""

    def test_jacobian_from_control_points(self, cubic_control_points):
        """Test Jacobian computation from control points."""
        from NurbsVolumeBuilder import NurbsVolume
        from QualityMetrics import QualityMetrics

        # Create a simple NURBS volume (without geomdl)
        nurbs_vol = NurbsVolume(
            control_points=cubic_control_points.reshape(-1, 3),
            weights=np.ones(64),
            knot_vectors=(
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            ),
            degrees=(3, 3, 3),
            size=(4, 4, 4),
            geomdl_volume=None,
        )

        metrics = QualityMetrics()
        jacobian = metrics._compute_jacobian_from_control_points(nurbs_vol)

        # For a regular grid, Jacobian should be positive (no inversions)
        assert jacobian > 0

    def test_jacobian_inverted_mesh(self):
        """Test that inverted mesh gives negative Jacobian."""
        from NurbsVolumeBuilder import NurbsVolume
        from QualityMetrics import QualityMetrics

        # Create a 2x2x2 mesh with one inverted cell
        # (swap two vertices to create inversion)
        control_points = np.zeros((2, 2, 2, 3))

        # Normal hexahedron
        control_points[0, 0, 0, :] = [0, 0, 0]
        control_points[1, 0, 0, :] = [1, 0, 0]
        control_points[0, 1, 0, :] = [0, 1, 0]
        control_points[1, 1, 0, :] = [1, 1, 0]
        control_points[0, 0, 1, :] = [0, 0, 1]
        control_points[1, 0, 1, :] = [1, 0, 1]
        control_points[0, 1, 1, :] = [0, 1, 1]
        control_points[1, 1, 1, :] = [1, 1, 1]

        # Invert by swapping z-layers (makes negative volume)
        control_points[0, 0, 0, :], control_points[0, 0, 1, :] = (
            control_points[0, 0, 1, :].copy(),
            control_points[0, 0, 0, :].copy(),
        )
        control_points[1, 0, 0, :], control_points[1, 0, 1, :] = (
            control_points[1, 0, 1, :].copy(),
            control_points[1, 0, 0, :].copy(),
        )
        control_points[0, 1, 0, :], control_points[0, 1, 1, :] = (
            control_points[0, 1, 1, :].copy(),
            control_points[0, 1, 0, :].copy(),
        )
        control_points[1, 1, 0, :], control_points[1, 1, 1, :] = (
            control_points[1, 1, 1, :].copy(),
            control_points[1, 1, 0, :].copy(),
        )

        nurbs_vol = NurbsVolume(
            control_points=control_points.reshape(-1, 3),
            weights=np.ones(8),
            knot_vectors=(
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1, 1]),
            ),
            degrees=(1, 1, 1),
            size=(2, 2, 2),
            geomdl_volume=None,
        )

        metrics = QualityMetrics()
        jacobian = metrics._compute_jacobian_from_control_points(nurbs_vol)

        # Inverted mesh should have negative Jacobian
        assert jacobian < 0


class TestFittingQuality:
    """Tests for FittingQuality dataclass."""

    def test_fitting_quality_creation(self):
        """Test creating a FittingQuality result."""
        from QualityMetrics import FittingQuality

        quality = FittingQuality(
            hausdorff_distance=0.5,
            mean_surface_distance=0.2,
            rms_distance=0.3,
            volume_ratio=1.1,
            num_control_points=64,
            min_jacobian=0.8,
        )

        assert quality.hausdorff_distance == 0.5
        assert quality.mean_surface_distance == 0.2
        assert quality.rms_distance == 0.3
        assert quality.volume_ratio == 1.1
        assert quality.num_control_points == 64
        assert quality.min_jacobian == 0.8


class TestMfemExporter:
    """Tests for MFEM export functionality."""

    def test_mfem_single_patch_export(self, cubic_control_points, tmp_path):
        """Test exporting a single-patch NURBS to MFEM format."""
        from Exporters.MfemExporter import MfemExporter
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        # Create NURBS volume
        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )
        builder = NurbsVolumeBuilder()
        nurbs_vol = builder.build(hex_mesh, degree=1)

        # Export to MFEM format
        output_path = tmp_path / "test.mesh"
        exporter = MfemExporter()
        exporter.export(nurbs_vol, output_path)

        # Verify file was created
        assert output_path.exists()

        # Read and verify content
        content = output_path.read_text()
        assert "MFEM NURBS mesh v1.0" in content
        assert "dimension" in content
        assert "3" in content
        assert "elements" in content
        assert "boundary" in content
        assert "knotvectors" in content
        assert "weights" in content
        assert "FiniteElementSpace" in content

    def test_mfem_knot_vectors_format(self, cubic_control_points, tmp_path):
        """Test that knot vectors are correctly formatted."""
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
        nurbs_vol = builder.build(hex_mesh, degree=3)

        output_path = tmp_path / "test_knots.mesh"
        exporter = MfemExporter()
        exporter.export(nurbs_vol, output_path)

        content = output_path.read_text()

        # Should have 3 knot vectors (u, v, w)
        assert content.count("knotvectors") == 1

        # Parse knot vectors section
        lines = content.split("\n")
        knot_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "knotvectors":
                knot_idx = i
                break

        assert knot_idx is not None
        # Next line is count "3"
        assert lines[knot_idx + 1].strip() == "3"

    def test_mfem_boundary_faces(self, cubic_control_points, tmp_path):
        """Test that boundary faces are correctly defined."""
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
        nurbs_vol = builder.build(hex_mesh, degree=1)

        output_path = tmp_path / "test_boundary.mesh"
        exporter = MfemExporter()
        exporter.export(nurbs_vol, output_path)

        content = output_path.read_text()

        # Should have 6 boundary faces
        lines = content.split("\n")
        boundary_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "boundary":
                boundary_idx = i
                break

        assert boundary_idx is not None
        assert lines[boundary_idx + 1].strip() == "6"

    def test_mfem_multi_patch_export(self, cubic_control_points, tmp_path):
        """Test exporting a multi-patch NURBS to MFEM format."""
        from Exporters.MfemExporter import MfemExporter
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        # Create two patches
        hex_mesh1 = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )
        hex_mesh2 = HexMesh(
            control_points=cubic_control_points + np.array([10, 0, 0]),
            num_u=4,
            num_v=4,
            num_w=4,
        )

        builder = NurbsVolumeBuilder()
        multi_patch = builder.build_multi_patch([hex_mesh1, hex_mesh2], degree=1)

        output_path = tmp_path / "test_multi.mesh"
        exporter = MfemExporter()
        exporter.export_multi_patch(multi_patch, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        assert "MFEM NURBS mesh v1.0" in content

        # Should have 2 elements
        lines = content.split("\n")
        elem_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "elements":
                elem_idx = i
                break

        assert elem_idx is not None
        assert lines[elem_idx + 1].strip() == "2"


class TestMeshExporter:
    """Tests for triangulated mesh export functionality."""

    def test_mesh_exporter_initialization(self):
        """Test MeshExporter initialization."""
        from Exporters.MeshExporter import MeshExporter

        exporter = MeshExporter(resolution=20)
        assert exporter.resolution == 20

    @requires_vtk
    def test_hex_grid_creation(self, cubic_control_points):
        """Test creating hex grid from NURBS volume."""
        from Exporters.MeshExporter import MeshExporter
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )
        builder = NurbsVolumeBuilder()
        nurbs_vol = builder.build(hex_mesh, degree=1)

        exporter = MeshExporter()
        grid = exporter._create_hex_grid(nurbs_vol)

        # Should have 64 points (4x4x4)
        assert grid.GetNumberOfPoints() == 64

        # Should have 27 hex cells ((4-1)x(4-1)x(4-1))
        assert grid.GetNumberOfCells() == 27

    @requires_vtk
    def test_vtk_unstructured_export(self, cubic_control_points, tmp_path):
        """Test VTK unstructured grid export."""
        from Exporters.MeshExporter import MeshExporter
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        hex_mesh = HexMesh(
            control_points=cubic_control_points,
            num_u=4,
            num_v=4,
            num_w=4,
        )
        builder = NurbsVolumeBuilder()
        nurbs_vol = builder.build(hex_mesh, degree=1)

        output_path = tmp_path / "test.vtu"
        exporter = MeshExporter()
        exporter.export_vtk_unstructured(nurbs_vol, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestCadExporter:
    """Tests for CAD export functionality (IGES/STEP)."""

    def test_cad_exporter_initialization(self):
        """Test CadExporter initialization."""
        from Exporters.CadExporter import CadExporter

        exporter = CadExporter()
        # Should initialize without error
        assert exporter._occ_available is None  # Lazy evaluation

    def test_cad_exporter_is_available(self):
        """Test checking pythonocc availability."""
        from Exporters.CadExporter import CadExporter

        exporter = CadExporter()
        # After calling is_available, the cached value should be set
        result = exporter.is_available()
        assert isinstance(result, bool)
        assert exporter._occ_available == result

    def test_knots_to_occ_format(self):
        """Test knot vector conversion to OCC format."""
        from Exporters.CadExporter import CadExporter

        exporter = CadExporter()

        # Test clamped knot vector for degree 3 with 4 control points
        knots = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        unique_knots, multiplicities = exporter._knots_to_occ_format(knots)

        assert unique_knots == [0.0, 1.0]
        assert multiplicities == [4, 4]

    def test_knots_to_occ_format_with_interior_knots(self):
        """Test knot vector conversion with interior knots."""
        from Exporters.CadExporter import CadExporter

        exporter = CadExporter()

        # Knot vector with interior knots
        knots = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
        unique_knots, multiplicities = exporter._knots_to_occ_format(knots)

        assert len(unique_knots) == 5
        assert unique_knots == pytest.approx([0, 0.25, 0.5, 0.75, 1])
        assert multiplicities == [3, 1, 1, 1, 3]

    def test_iges_export_requires_pythonocc(self, cubic_control_points, tmp_path):
        """Test that IGES export raises error when pythonocc unavailable."""
        from Exporters.CadExporter import CadExporter
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        exporter = CadExporter()

        # If pythonocc is not available, should raise RuntimeError
        if not exporter.is_available():
            hex_mesh = HexMesh(
                control_points=cubic_control_points,
                num_u=4,
                num_v=4,
                num_w=4,
            )
            builder = NurbsVolumeBuilder()
            nurbs_vol = builder.build(hex_mesh, degree=1)

            with pytest.raises(RuntimeError, match="pythonocc-core is required"):
                exporter.export_iges(nurbs_vol, tmp_path / "test.igs")

    def test_step_export_requires_pythonocc(self, cubic_control_points, tmp_path):
        """Test that STEP export raises error when pythonocc unavailable."""
        from Exporters.CadExporter import CadExporter
        from HexMeshGenerator import HexMesh
        from NurbsVolumeBuilder import NurbsVolumeBuilder

        exporter = CadExporter()

        # If pythonocc is not available, should raise RuntimeError
        if not exporter.is_available():
            hex_mesh = HexMesh(
                control_points=cubic_control_points,
                num_u=4,
                num_v=4,
                num_w=4,
            )
            builder = NurbsVolumeBuilder()
            nurbs_vol = builder.build(hex_mesh, degree=1)

            with pytest.raises(RuntimeError, match="pythonocc-core is required"):
                exporter.export_step(nurbs_vol, tmp_path / "test.step")
