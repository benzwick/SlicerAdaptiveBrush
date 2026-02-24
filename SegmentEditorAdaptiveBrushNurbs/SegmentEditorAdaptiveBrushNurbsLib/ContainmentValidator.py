"""Containment validation for NURBS volumes.

Ensures that the original segment is fully contained within the
generated NURBS volume. This is important for IGA applications
where the NURBS domain must fully cover the physical region.

Methods:
- Sample random points from segment interior
- Check if points are inside NURBS volume boundary
- Adjust control points to ensure containment if needed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vtkMRMLSegmentationNode import vtkMRMLSegmentationNode

    from .NurbsVolumeBuilder import NurbsVolume

logger = logging.getLogger(__name__)


@dataclass
class ContainmentResult:
    """Result of containment validation.

    Attributes:
        is_contained: True if all sample points are inside NURBS volume.
        containment_ratio: Fraction of points inside (0.0 to 1.0).
        outside_points: Coordinates of points found outside NURBS.
        max_outside_distance: Maximum distance of outside points from surface.
    """

    is_contained: bool
    containment_ratio: float
    outside_points: np.ndarray
    max_outside_distance: float


class ContainmentValidator:
    """Validate segment containment within NURBS volume.

    Samples points from the segment interior and verifies they are
    all inside the NURBS volume boundary.

    Uses VTK's vtkSelectEnclosedPoints for accurate point-in-volume
    testing when a NURBS surface mesh is available.

    Example:
        validator = ContainmentValidator()
        result = validator.validate(nurbs_volume, segmentation_node, segment_id)
        if not result.is_contained:
            # Expand NURBS or adjust control points
            pass
    """

    def __init__(self, tolerance: float = 0.5, surface_resolution: int = 30):
        """Initialize the containment validator.

        Args:
            tolerance: Distance tolerance in mm for containment check.
            surface_resolution: Resolution for NURBS surface sampling.
        """
        self.tolerance = tolerance
        self.surface_resolution = surface_resolution
        self._cached_surface_polydata: Any | None = None
        self._cached_enclosure_filter: Any | None = None

    def validate(
        self,
        nurbs_volume: NurbsVolume,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        num_samples: int = 1000,
    ) -> ContainmentResult:
        """Validate that segment is contained within NURBS volume.

        Args:
            nurbs_volume: The NURBS volume to validate.
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to check containment.
            num_samples: Number of random sample points.

        Returns:
            ContainmentResult with validation details.
        """
        # Clear cache for new validation
        self.clear_cache()

        # Get sample points from segment interior
        sample_points = self._sample_segment_interior(segmentation_node, segment_id, num_samples)

        if len(sample_points) == 0:
            logger.warning("No sample points generated for containment check")
            return ContainmentResult(
                is_contained=True,
                containment_ratio=1.0,
                outside_points=np.zeros((0, 3)),
                max_outside_distance=0.0,
            )

        # Check each point
        inside_count = 0
        outside_points = []
        max_distance = 0.0

        for point in sample_points:
            is_inside, distance = self._check_point_inside(nurbs_volume, point)

            if is_inside:
                inside_count += 1
            else:
                outside_points.append(point)
                max_distance = max(max_distance, distance)

        containment_ratio = inside_count / len(sample_points)
        is_contained = len(outside_points) == 0

        result = ContainmentResult(
            is_contained=is_contained,
            containment_ratio=containment_ratio,
            outside_points=np.array(outside_points) if outside_points else np.zeros((0, 3)),
            max_outside_distance=max_distance,
        )

        logger.info(
            f"Containment validation: {containment_ratio * 100:.1f}% contained, "
            f"{len(outside_points)} outside points, max distance {max_distance:.2f}"
        )

        return result

    def _sample_segment_interior(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        num_samples: int,
    ) -> np.ndarray:
        """Sample random points from segment interior.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment.
            num_samples: Number of points to sample.

        Returns:
            Array of sample points in RAS coordinates, shape (n, 3).
        """
        import slicer
        import vtk

        segmentation = segmentation_node.GetSegmentation()
        if segmentation is None:
            return np.zeros((0, 3))

        # Create temporary labelmap
        labelmap_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        try:
            segment_ids = vtk.vtkStringArray()
            segment_ids.InsertNextValue(segment_id)

            success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentation_node, segment_ids, labelmap_volume
            )

            if not success:
                return np.zeros((0, 3))

            # Get array and transform
            labelmap_array = slicer.util.arrayFromVolume(labelmap_volume)
            ijk_to_ras = vtk.vtkMatrix4x4()
            labelmap_volume.GetIJKToRASMatrix(ijk_to_ras)

            # Find interior voxels
            interior_ijk = np.argwhere(labelmap_array > 0)

            if len(interior_ijk) == 0:
                return np.zeros((0, 3))

            # Sample random indices
            if len(interior_ijk) <= num_samples:
                sample_indices = np.arange(len(interior_ijk))
            else:
                rng = np.random.default_rng(42)  # Deterministic sampling
                sample_indices = rng.choice(len(interior_ijk), num_samples, replace=False)

            sample_ijk = interior_ijk[sample_indices]

            # Convert to RAS coordinates
            sample_ras = np.zeros((len(sample_ijk), 3))
            for i, ijk in enumerate(sample_ijk):
                # Note: numpy array is in [k, j, i] order (z, y, x)
                ijk_vtk = [float(ijk[2]), float(ijk[1]), float(ijk[0]), 1.0]
                ras = [0.0, 0.0, 0.0, 1.0]
                ijk_to_ras.MultiplyPoint(ijk_vtk, ras)
                sample_ras[i, :] = ras[:3]

            return sample_ras

        finally:
            slicer.mrmlScene.RemoveNode(labelmap_volume)

    def _build_nurbs_surface_mesh(
        self,
        nurbs_volume: NurbsVolume,
    ) -> object:
        """Build a closed surface mesh from NURBS volume boundary.

        Creates a vtkPolyData representing the NURBS volume boundary
        by sampling all six faces and creating a triangulated mesh.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            vtkPolyData with closed surface mesh.
        """
        import vtk

        if nurbs_volume.geomdl_volume is None:
            # No geomdl volume - use control point convex hull
            return self._build_control_point_hull(nurbs_volume)

        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        res = self.surface_resolution

        # Sample all six faces of the NURBS volume
        # Face definitions: (fixed_param, fixed_value, u_param, v_param)
        faces = [
            ("u", 0.0),  # u=0 face
            ("u", 1.0),  # u=1 face
            ("v", 0.0),  # v=0 face
            ("v", 1.0),  # v=1 face
            ("w", 0.0),  # w=0 face
            ("w", 1.0),  # w=1 face
        ]

        point_index = 0
        params_range = np.linspace(0, 1, res)

        for fixed_param, fixed_value in faces:
            face_indices = np.zeros((res, res), dtype=int)

            for i, p1 in enumerate(params_range):
                for j, p2 in enumerate(params_range):
                    # Build parameter tuple based on fixed parameter
                    if fixed_param == "u":
                        params = (fixed_value, p1, p2)
                    elif fixed_param == "v":
                        params = (p1, fixed_value, p2)
                    else:  # w
                        params = (p1, p2, fixed_value)

                    # Evaluate NURBS at this parameter
                    pt = nurbs_volume.geomdl_volume.evaluate_single(params)
                    points.InsertNextPoint(pt[0], pt[1], pt[2])
                    face_indices[i, j] = point_index
                    point_index += 1

            # Create triangles for this face
            for i in range(res - 1):
                for j in range(res - 1):
                    # Two triangles per quad
                    tri1 = vtk.vtkTriangle()
                    tri1.GetPointIds().SetId(0, face_indices[i, j])
                    tri1.GetPointIds().SetId(1, face_indices[i + 1, j])
                    tri1.GetPointIds().SetId(2, face_indices[i, j + 1])
                    cells.InsertNextCell(tri1)

                    tri2 = vtk.vtkTriangle()
                    tri2.GetPointIds().SetId(0, face_indices[i + 1, j])
                    tri2.GetPointIds().SetId(1, face_indices[i + 1, j + 1])
                    tri2.GetPointIds().SetId(2, face_indices[i, j + 1])
                    cells.InsertNextCell(tri2)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        # Clean up any duplicate points
        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputData(polydata)
        clean_filter.SetTolerance(0.0001)
        clean_filter.Update()

        return clean_filter.GetOutput()

    def _build_control_point_hull(
        self,
        nurbs_volume: NurbsVolume,
    ) -> object:
        """Build convex hull from control points as fallback.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            vtkPolyData with convex hull surface.
        """
        import vtk

        points = vtk.vtkPoints()
        for pt in nurbs_volume.control_points:
            points.InsertNextPoint(pt[0], pt[1], pt[2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Create convex hull
        hull_filter = vtk.vtkDelaunay3D()
        hull_filter.SetInputData(polydata)
        hull_filter.Update()

        # Extract surface
        surface_filter = vtk.vtkGeometryFilter()
        surface_filter.SetInputConnection(hull_filter.GetOutputPort())
        surface_filter.Update()

        return surface_filter.GetOutput()

    def _check_point_inside(
        self,
        nurbs_volume: NurbsVolume,
        point: np.ndarray,
    ) -> tuple[bool, float]:
        """Check if a point is inside the NURBS volume.

        Uses VTK's vtkSelectEnclosedPoints for accurate point-in-volume
        testing. Falls back to bounding box check if surface mesh
        cannot be created.

        Args:
            nurbs_volume: The NURBS volume.
            point: Point to check (3,) in RAS coordinates.

        Returns:
            Tuple of (is_inside, distance_to_surface).
        """
        import vtk

        # Build surface mesh if not cached
        if self._cached_surface_polydata is None:
            self._cached_surface_polydata = self._build_nurbs_surface_mesh(nurbs_volume)

            # Set up enclosure filter
            self._cached_enclosure_filter = vtk.vtkSelectEnclosedPoints()
            self._cached_enclosure_filter.SetSurfaceData(self._cached_surface_polydata)
            self._cached_enclosure_filter.SetTolerance(0.0)

        # Ensure filter is available (should always be after initialization)
        assert self._cached_enclosure_filter is not None

        # Create point data
        test_points = vtk.vtkPoints()
        test_points.InsertNextPoint(point[0], point[1], point[2])

        test_polydata = vtk.vtkPolyData()
        test_polydata.SetPoints(test_points)

        # Check enclosure
        self._cached_enclosure_filter.SetInputData(test_polydata)
        self._cached_enclosure_filter.Update()

        is_inside = bool(self._cached_enclosure_filter.IsInside(0))

        # Compute distance to surface
        distance = self._compute_distance_to_surface(point)

        # Apply tolerance - points within tolerance are considered inside
        if not is_inside and distance <= self.tolerance:
            is_inside = True

        return is_inside, distance

    def _compute_distance_to_surface(
        self,
        point: np.ndarray,
    ) -> float:
        """Compute distance from point to NURBS surface.

        Args:
            point: Point coordinates (3,).

        Returns:
            Distance to nearest surface point.
        """
        import vtk

        if self._cached_surface_polydata is None:
            return 0.0

        # Use cell locator for efficient distance computation
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(self._cached_surface_polydata)
        locator.BuildLocator()

        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        locator.FindClosestPoint(
            [point[0], point[1], point[2]], closest_point, cell_id, sub_id, dist2
        )

        return float(np.sqrt(dist2.get()))

    def clear_cache(self) -> None:
        """Clear cached surface mesh and filters.

        Call this when validating a different NURBS volume.
        """
        self._cached_surface_polydata = None
        self._cached_enclosure_filter = None

    def expand_to_contain(
        self,
        nurbs_volume: NurbsVolume,
        outside_points: np.ndarray,
        expansion_factor: float = 1.1,
    ) -> NurbsVolume:
        """Expand NURBS volume to contain outside points.

        Adjusts boundary control points outward to include all
        segment voxels by moving them along surface normals.

        Args:
            nurbs_volume: The NURBS volume to expand.
            outside_points: Points that are currently outside.
            expansion_factor: Factor to expand beyond required distance.

        Returns:
            New NurbsVolume with expanded boundaries.
        """
        from .NurbsVolumeBuilder import NurbsVolume

        if len(outside_points) == 0:
            return nurbs_volume

        logger.info(f"Expanding NURBS to contain {len(outside_points)} outside points")

        # Get current control points
        nu, nv, nw = nurbs_volume.size
        control_points = nurbs_volume.control_points.reshape(nu, nv, nw, 3).copy()

        # For each outside point, find the nearest boundary control point
        # and compute the required expansion
        expansion_vectors = self._compute_expansion_vectors(
            control_points, outside_points, expansion_factor
        )

        # Apply expansions to boundary control points
        expanded_control_points = self._apply_boundary_expansion(control_points, expansion_vectors)

        # Create new NurbsVolume with expanded control points
        expanded_nurbs = NurbsVolume(
            control_points=expanded_control_points.reshape(-1, 3),
            weights=nurbs_volume.weights.copy(),
            knot_vectors=nurbs_volume.knot_vectors,
            degrees=nurbs_volume.degrees,
            size=nurbs_volume.size,
            geomdl_volume=None,  # Will be rebuilt if needed
        )

        # Rebuild geomdl volume if it was available
        if nurbs_volume.geomdl_volume is not None:
            from .NurbsVolumeBuilder import NurbsVolumeBuilder

            builder = NurbsVolumeBuilder()
            expanded_nurbs = builder._rebuild_geomdl_volume(expanded_nurbs)

        logger.info("NURBS expansion complete")
        return expanded_nurbs

    def _compute_expansion_vectors(
        self,
        control_points: np.ndarray,
        outside_points: np.ndarray,
        expansion_factor: float,
    ) -> dict[tuple[int, int, int], np.ndarray]:
        """Compute expansion vectors for boundary control points.

        For each outside point, finds the nearest boundary face and
        computes the required expansion vector.

        Args:
            control_points: Control points array (nu, nv, nw, 3).
            outside_points: Points outside the volume (n, 3).
            expansion_factor: Factor to expand beyond required distance.

        Returns:
            Dictionary mapping boundary control point indices to expansion vectors.
        """
        nu, nv, nw, _ = control_points.shape
        expansion_vectors: dict[tuple[int, int, int], np.ndarray] = {}

        # Identify boundary control points (on any face of the grid)
        boundary_indices = []
        for i in range(nu):
            for j in range(nv):
                for k in range(nw):
                    if i == 0 or i == nu - 1 or j == 0 or j == nv - 1 or k == 0 or k == nw - 1:
                        boundary_indices.append((i, j, k))

        # Compute centroid of control points
        all_points = control_points.reshape(-1, 3)
        centroid = all_points.mean(axis=0)

        # For each outside point, find nearest boundary control point
        for outside_pt in outside_points:
            min_dist = float("inf")
            nearest_idx = None

            for idx in boundary_indices:
                i, j, k = idx
                boundary_pt = control_points[i, j, k, :]
                dist = float(np.linalg.norm(outside_pt - boundary_pt))

                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx

            if nearest_idx is not None:
                i, j, k = nearest_idx
                boundary_pt = control_points[i, j, k, :]

                # Compute outward normal (from centroid toward boundary point)
                normal = boundary_pt - centroid
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-6:
                    normal = normal / norm_len
                else:
                    normal = np.array([1.0, 0.0, 0.0])

                # Compute required expansion distance
                # Project outside point onto normal direction
                to_outside = outside_pt - boundary_pt
                expansion_dist = np.dot(to_outside, normal)

                # Only expand if point is truly outside in normal direction
                if expansion_dist > 0:
                    expansion_dist *= expansion_factor

                    # Accumulate expansion (take maximum if multiple outside points)
                    if nearest_idx in expansion_vectors:
                        existing = expansion_vectors[nearest_idx]
                        existing_mag = np.linalg.norm(existing)
                        new_mag = expansion_dist
                        if new_mag > existing_mag:
                            expansion_vectors[nearest_idx] = normal * expansion_dist
                    else:
                        expansion_vectors[nearest_idx] = normal * expansion_dist

        return expansion_vectors

    def _apply_boundary_expansion(
        self,
        control_points: np.ndarray,
        expansion_vectors: dict[tuple[int, int, int], np.ndarray],
    ) -> np.ndarray:
        """Apply expansion vectors to boundary control points.

        Also smoothly propagates expansion to neighboring points
        to maintain mesh quality.

        Args:
            control_points: Control points array (nu, nv, nw, 3).
            expansion_vectors: Dictionary of expansion vectors by index.

        Returns:
            Expanded control points array.
        """
        nu, nv, nw, _ = control_points.shape
        expanded = control_points.copy()

        # Apply expansion to boundary points
        for idx, expansion in expansion_vectors.items():
            i, j, k = idx
            expanded[i, j, k, :] += expansion

        # Smooth expansion to immediate neighbors (with decay)
        decay_factor = 0.5
        smoothed = expanded.copy()

        for idx, expansion in expansion_vectors.items():
            i, j, k = idx

            # Propagate to neighbors with decay
            neighbors = [
                (i - 1, j, k),
                (i + 1, j, k),
                (i, j - 1, k),
                (i, j + 1, k),
                (i, j, k - 1),
                (i, j, k + 1),
            ]

            for ni, nj, nk in neighbors:
                if 0 <= ni < nu and 0 <= nj < nv and 0 <= nk < nw:
                    # Only expand interior points (boundary points already handled)
                    is_boundary = (
                        ni == 0
                        or ni == nu - 1
                        or nj == 0
                        or nj == nv - 1
                        or nk == 0
                        or nk == nw - 1
                    )
                    if not is_boundary:
                        smoothed[ni, nj, nk, :] += expansion * decay_factor

        return smoothed

    def refine_until_contained(
        self,
        nurbs_volume: NurbsVolume,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        target_containment: float = 0.99,
        max_iterations: int = 5,
        num_samples: int = 1000,
    ) -> tuple[NurbsVolume, ContainmentResult]:
        """Iteratively refine NURBS until containment target is met.

        Repeatedly validates containment, expands if necessary, and
        re-validates until the target containment ratio is achieved
        or max iterations is reached.

        Args:
            nurbs_volume: Initial NURBS volume to refine.
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to check containment.
            target_containment: Target containment ratio (0.0 to 1.0).
            max_iterations: Maximum refinement iterations.
            num_samples: Number of sample points for validation.

        Returns:
            Tuple of (refined NurbsVolume, final ContainmentResult).
        """
        logger.info(
            f"Starting iterative refinement: target={target_containment * 100:.1f}%, "
            f"max_iter={max_iterations}"
        )

        current_volume = nurbs_volume
        result = None

        for iteration in range(max_iterations):
            # Validate containment
            result = self.validate(current_volume, segmentation_node, segment_id, num_samples)

            logger.info(
                f"Iteration {iteration + 1}: containment={result.containment_ratio * 100:.1f}%, "
                f"outside={len(result.outside_points)}"
            )

            # Check if target met
            if result.containment_ratio >= target_containment:
                logger.info(f"Target containment achieved at iteration {iteration + 1}")
                break

            # Expand to contain outside points
            if len(result.outside_points) > 0:
                current_volume = self.expand_to_contain(
                    current_volume,
                    result.outside_points,
                    expansion_factor=1.1,
                )
            else:
                # No outside points but still below target (shouldn't happen)
                logger.warning("No outside points but below target containment")
                break

        if result is None:
            # Should not happen, but handle gracefully
            result = self.validate(current_volume, segmentation_node, segment_id, num_samples)

        if result.containment_ratio < target_containment:
            logger.warning(
                f"Could not achieve target containment after {max_iterations} iterations. "
                f"Final: {result.containment_ratio * 100:.1f}%"
            )

        return current_volume, result
