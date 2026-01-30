"""Hexahedral control mesh generation for NURBS volumes.

Generates hexahedral control meshes from segmented structures using
different strategies based on structure type:

- Simple shapes: Oriented bounding box with projected control points
- Tubular structures: Cylindrical sweeping along centerline
- Branching structures: Multi-patch with bifurcation templates

The hexahedral mesh serves as the control mesh for volumetric NURBS
construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vtkMRMLSegmentationNode import vtkMRMLSegmentationNode

logger = logging.getLogger(__name__)


@dataclass
class HexMesh:
    """Hexahedral control mesh for NURBS volume construction.

    Attributes:
        control_points: Control point positions, shape (nu, nv, nw, 3).
        weights: Control point weights, shape (nu, nv, nw). Default 1.0.
        num_u: Number of control points in u direction.
        num_v: Number of control points in v direction.
        num_w: Number of control points in w direction.
        ijk_to_ras: Transform from IJK to RAS coordinates (4x4 matrix).
    """

    control_points: np.ndarray
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    num_u: int = 0
    num_v: int = 0
    num_w: int = 0
    ijk_to_ras: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        if self.control_points.ndim == 4:
            self.num_u, self.num_v, self.num_w, _ = self.control_points.shape
        elif self.control_points.ndim == 2:
            # Flat array - infer from num_u, num_v, num_w
            pass

        # Initialize weights to 1.0 if not provided
        if self.weights.size == 0:
            self.weights = np.ones((self.num_u, self.num_v, self.num_w))

    @property
    def num_control_points(self) -> int:
        """Total number of control points."""
        return self.num_u * self.num_v * self.num_w

    @property
    def flat_control_points(self) -> np.ndarray:
        """Control points as flat array, shape (n, 3)."""
        return self.control_points.reshape(-1, 3)

    @property
    def flat_weights(self) -> np.ndarray:
        """Weights as flat array, shape (n,)."""
        return self.weights.flatten()


class HexMeshGenerator:
    """Generate hexahedral control meshes from segments.

    Provides different generation strategies based on structure type:
    - Simple: Oriented bounding box approach
    - Tubular: Centerline sweeping (requires SlicerVMTK)
    - Branching: Multi-patch with templates (requires SlicerVMTK)

    Example:
        generator = HexMeshGenerator()
        hex_mesh = generator.generate_simple(segmentation_node, segment_id)
    """

    # Default resolution for different structure types
    DEFAULT_RESOLUTION_SIMPLE = 4  # 4x4x4 = 64 control points
    DEFAULT_RESOLUTION_TUBULAR_CIRC = 8  # 8 points around circumference
    DEFAULT_RESOLUTION_TUBULAR_AXIAL = 10  # 10 points along length

    def __init__(self):
        """Initialize the hex mesh generator."""
        pass

    def generate_simple(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        resolution: int | None = None,
    ) -> HexMesh:
        """Generate hex mesh for a simple convex shape.

        Uses an oriented bounding box approach:
        1. Compute oriented bounding box of segment
        2. Create regular grid of control points within box
        3. Project boundary control points to segment surface
        4. Optimize interior points for smooth parameterization

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to mesh.
            resolution: Control points per direction (default 4).

        Returns:
            HexMesh with control points adapted to segment shape.

        Raises:
            ValueError: If segment is empty or not found.
        """
        if resolution is None:
            resolution = self.DEFAULT_RESOLUTION_SIMPLE

        logger.info(f"Generating simple hex mesh: resolution={resolution}")

        # Get segment as numpy array and its transform
        labelmap_array, ijk_to_ras = self._get_segment_with_transform(segmentation_node, segment_id)

        if labelmap_array is None or np.sum(labelmap_array) == 0:
            raise ValueError(f"Segment '{segment_id}' is empty or not found")

        # Compute oriented bounding box
        obb_center, obb_axes, obb_extents = self._compute_oriented_bounding_box(labelmap_array)

        # Generate initial control point grid
        control_points = self._create_grid_control_points(
            obb_center, obb_axes, obb_extents, resolution
        )

        # Project boundary control points to segment surface
        control_points = self._project_boundary_to_surface(
            control_points, labelmap_array, resolution
        )

        # Create hex mesh
        hex_mesh = HexMesh(
            control_points=control_points,
            num_u=resolution,
            num_v=resolution,
            num_w=resolution,
            ijk_to_ras=ijk_to_ras,
        )

        logger.info(f"Generated simple hex mesh: {hex_mesh.num_control_points} control points")

        return hex_mesh

    def generate_tubular(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        resolution: int | None = None,
    ) -> HexMesh:
        """Generate hex mesh for a tubular structure.

        Uses centerline sweeping approach:
        1. Extract centerline using VMTK
        2. Sample inscribed sphere radii along path
        3. Sweep circular cross-section template
        4. Project control points to segment boundary

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to mesh.
            resolution: Points around circumference (default 8).

        Returns:
            HexMesh adapted to tubular shape.

        Raises:
            ValueError: If segment is empty or not found.
            RuntimeError: If SlicerVMTK is not available.
        """
        if resolution is None:
            resolution = self.DEFAULT_RESOLUTION_TUBULAR_CIRC

        logger.info(f"Generating tubular hex mesh: resolution={resolution}")

        # Check for VMTK availability
        if not self._check_vmtk_available():
            raise RuntimeError(
                "SlicerVMTK extension is required for tubular structure meshing. "
                "Please install it from the Extension Manager."
            )

        # Get segment data
        labelmap_array, ijk_to_ras = self._get_segment_with_transform(segmentation_node, segment_id)

        if labelmap_array is None or np.sum(labelmap_array) == 0:
            raise ValueError(f"Segment '{segment_id}' is empty or not found")

        # Extract centerline using VMTK
        centerline_points, radii = self._extract_centerline_vmtk(segmentation_node, segment_id)

        # Generate cylindrical control mesh by sweeping
        control_points = self._sweep_circular_template(centerline_points, radii, resolution)

        # Project to segment boundary
        control_points = self._project_tubular_to_surface(
            control_points, labelmap_array, resolution
        )

        # Determine axial resolution from centerline
        axial_resolution = len(centerline_points)

        hex_mesh = HexMesh(
            control_points=control_points,
            num_u=resolution,
            num_v=resolution,
            num_w=axial_resolution,
            ijk_to_ras=ijk_to_ras,
        )

        logger.info(f"Generated tubular hex mesh: {hex_mesh.num_control_points} control points")

        return hex_mesh

    def generate_branching(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        resolution: int | None = None,
    ) -> HexMesh:
        """Generate hex mesh for a branching structure.

        Uses multi-patch approach with bifurcation templates:
        1. Extract centerline network using VMTK
        2. Detect bifurcation points
        3. Apply bifurcation templates for topology
        4. Sweep through branches with radius adaptation
        5. Ensure G1 continuity at branch junctions

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to mesh.
            resolution: Points around circumference (default 8).

        Returns:
            Multi-patch HexMesh for branching structure.

        Raises:
            ValueError: If segment is empty or not found.
            RuntimeError: If SlicerVMTK is not available.
        """
        if resolution is None:
            resolution = self.DEFAULT_RESOLUTION_TUBULAR_CIRC

        logger.info(f"Generating branching hex mesh: resolution={resolution}")

        # Check for VMTK availability
        if not self._check_vmtk_available():
            raise RuntimeError(
                "SlicerVMTK extension is required for branching structure meshing. "
                "Please install it from the Extension Manager."
            )

        # For now, fall back to simple generation
        # Full implementation requires VMTK branch detection
        logger.warning("Branching structure meshing not yet implemented, using simple approach")
        return self.generate_simple(segmentation_node, segment_id, resolution)

    # Helper methods

    def _get_segment_with_transform(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Extract segment as numpy array with its transform.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to extract.

        Returns:
            Tuple of (labelmap_array, ijk_to_ras_matrix).
        """
        import slicer
        import vtk

        segmentation = segmentation_node.GetSegmentation()
        if segmentation is None:
            return None, np.eye(4)

        segment = segmentation.GetSegment(segment_id)
        if segment is None:
            return None, np.eye(4)

        # Create temporary labelmap volume
        labelmap_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        try:
            # Export segment to labelmap
            segment_ids = vtk.vtkStringArray()
            segment_ids.InsertNextValue(segment_id)

            success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentation_node, segment_ids, labelmap_volume
            )

            if not success:
                return None, np.eye(4)

            # Get transform
            ijk_to_ras = vtk.vtkMatrix4x4()
            labelmap_volume.GetIJKToRASMatrix(ijk_to_ras)
            ijk_to_ras_array = np.array(
                [[ijk_to_ras.GetElement(i, j) for j in range(4)] for i in range(4)]
            )

            # Convert to numpy array
            labelmap_array = slicer.util.arrayFromVolume(labelmap_volume)

            return (labelmap_array > 0).astype(np.uint8), ijk_to_ras_array

        finally:
            slicer.mrmlScene.RemoveNode(labelmap_volume)

    def _compute_oriented_bounding_box(
        self, labelmap_array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute oriented bounding box of segment.

        Args:
            labelmap_array: Binary labelmap array.

        Returns:
            Tuple of (center, axes, extents):
            - center: Center point (3,)
            - axes: Principal axes (3, 3), each row is an axis
            - extents: Half-extents along each axis (3,)
        """
        # Get voxel coordinates
        coords = np.argwhere(labelmap_array > 0).astype(np.float64)

        if len(coords) == 0:
            return np.zeros(3), np.eye(3), np.ones(3)

        # Center of mass
        center = coords.mean(axis=0)

        # Center the coordinates
        centered = coords - center

        # PCA to get principal axes
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, idx].T  # Each row is an axis

        # Compute extents along each axis
        projected = centered @ axes.T
        extents = np.abs(projected).max(axis=0)

        # Add small margin
        extents = extents * 1.1

        return center, axes, extents

    def _create_grid_control_points(
        self,
        center: np.ndarray,
        axes: np.ndarray,
        extents: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        """Create regular grid of control points within bounding box.

        Args:
            center: Box center (3,).
            axes: Box axes (3, 3).
            extents: Half-extents (3,).
            resolution: Points per direction.

        Returns:
            Control points array, shape (resolution, resolution, resolution, 3).
        """
        # Create normalized grid [-1, 1] in each direction
        t = np.linspace(-1, 1, resolution)
        u, v, w = np.meshgrid(t, t, t, indexing="ij")

        # Scale by extents
        scaled_u = u * extents[0]
        scaled_v = v * extents[1]
        scaled_w = w * extents[2]

        # Transform to world coordinates
        control_points = np.zeros((resolution, resolution, resolution, 3))

        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    local_pos = np.array([scaled_u[i, j, k], scaled_v[i, j, k], scaled_w[i, j, k]])
                    world_pos = center + axes.T @ local_pos
                    control_points[i, j, k, :] = world_pos

        return control_points

    def _project_boundary_to_surface(
        self,
        control_points: np.ndarray,
        labelmap_array: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        """Project boundary control points to segment surface.

        Args:
            control_points: Initial control points (nu, nv, nw, 3).
            labelmap_array: Binary segment array.
            resolution: Control points per direction.

        Returns:
            Modified control points with boundary projected to surface.
        """
        import SimpleITK as sitk

        # Compute distance transform from segment boundary
        sitk_image = sitk.GetImageFromArray(labelmap_array.astype(np.uint8))
        distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
        distance_filter.SetUseImageSpacing(False)
        distance_image = distance_filter.Execute(sitk_image)
        distance_array = sitk.GetArrayFromImage(distance_image)

        # Compute gradient of distance field
        gradient_filter = sitk.GradientImageFilter()
        gradient_filter.SetUseImageSpacing(False)
        gradient_image = gradient_filter.Execute(distance_image)
        gradient_array = sitk.GetArrayFromImage(gradient_image)

        # Project boundary control points
        # Boundary = first/last index in any direction
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    is_boundary = (
                        i == 0
                        or i == resolution - 1
                        or j == 0
                        or j == resolution - 1
                        or k == 0
                        or k == resolution - 1
                    )

                    if is_boundary:
                        pos = control_points[i, j, k, :]
                        new_pos = self._project_point_to_surface(
                            pos, distance_array, gradient_array, labelmap_array.shape
                        )
                        control_points[i, j, k, :] = new_pos

        return control_points

    def _project_point_to_surface(
        self,
        point: np.ndarray,
        distance_array: np.ndarray,
        gradient_array: np.ndarray,
        shape: tuple,
    ) -> np.ndarray:
        """Project a point to the segment surface.

        Args:
            point: Point to project (3,) in IJK coordinates.
            distance_array: Signed distance field.
            gradient_array: Gradient of distance field.
            shape: Shape of the volume.

        Returns:
            Projected point on or near surface.
        """
        # Clip to volume bounds
        ijk = np.clip(point, [0, 0, 0], np.array(shape) - 1).astype(int)

        # Get distance at point
        distance = distance_array[ijk[0], ijk[1], ijk[2]]

        # If already on surface (distance near 0), return as-is
        if abs(distance) < 0.5:
            return point

        # Get gradient direction
        gradient = gradient_array[ijk[0], ijk[1], ijk[2], :]

        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < 1e-6:
            return point

        gradient = gradient / grad_norm

        # Move point along negative gradient by distance value
        # (negative gradient points toward surface from outside)
        new_point = point - distance * gradient

        # Clip to bounds
        new_point = np.clip(new_point, [0, 0, 0], np.array(shape) - 1)

        return new_point

    def _check_vmtk_available(self) -> bool:
        """Check if SlicerVMTK extension is available.

        Returns:
            True if VMTK is available.
        """
        try:
            import slicer

            return hasattr(slicer.modules, "extractcenterline")
        except Exception:
            return False

    def _extract_centerline_vmtk(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract centerline using SlicerVMTK.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to analyze.

        Returns:
            Tuple of (centerline_points, radii):
            - centerline_points: Points along centerline (n, 3)
            - radii: Inscribed sphere radius at each point (n,)
        """
        # TODO: Implement VMTK centerline extraction
        # This requires:
        # 1. Creating endpoint fiducials
        # 2. Running ExtractCenterline module
        # 3. Parsing the resulting curve

        raise NotImplementedError("VMTK centerline extraction not yet implemented")

    def _sweep_circular_template(
        self,
        centerline_points: np.ndarray,
        radii: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        """Sweep circular template along centerline.

        Args:
            centerline_points: Points along centerline (n, 3).
            radii: Radius at each point (n,).
            resolution: Points around circumference.

        Returns:
            Control points for cylindrical mesh.
        """
        n_axial = len(centerline_points)
        n_circ = resolution
        n_radial = 2  # Inner and outer layers

        control_points = np.zeros((n_circ, n_radial, n_axial, 3))

        for k, (center, radius) in enumerate(zip(centerline_points, radii)):
            # Compute local frame at this point
            if k == 0:
                tangent = centerline_points[1] - centerline_points[0]
            elif k == n_axial - 1:
                tangent = centerline_points[-1] - centerline_points[-2]
            else:
                tangent = centerline_points[k + 1] - centerline_points[k - 1]

            tangent = tangent / np.linalg.norm(tangent)

            # Find perpendicular vectors
            up = np.array([0, 0, 1])
            if abs(np.dot(tangent, up)) > 0.9:
                up = np.array([1, 0, 0])

            normal = np.cross(tangent, up)
            normal = normal / np.linalg.norm(normal)
            binormal = np.cross(tangent, normal)

            # Create circular cross-section
            for i, theta in enumerate(np.linspace(0, 2 * np.pi, n_circ, endpoint=False)):
                direction = np.cos(theta) * normal + np.sin(theta) * binormal

                # Outer point (on surface)
                control_points[i, 1, k, :] = center + radius * direction

                # Inner point (toward center)
                control_points[i, 0, k, :] = center + 0.5 * radius * direction

        return control_points

    def _project_tubular_to_surface(
        self,
        control_points: np.ndarray,
        labelmap_array: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        """Project tubular control points to segment surface.

        Args:
            control_points: Initial control points.
            labelmap_array: Binary segment array.
            resolution: Circumferential resolution.

        Returns:
            Modified control points projected to surface.
        """
        # TODO: Implement projection for tubular mesh
        # For now, return unchanged
        return control_points
