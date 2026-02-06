"""Quality metrics for NURBS volume fitting.

Provides metrics to evaluate how well a NURBS volume fits
the original segmentation:
- Hausdorff distance (max deviation)
- Mean surface distance
- Volume coverage ratio
- Jacobian quality (mesh distortion)
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
class FittingQuality:
    """Quality metrics for NURBS fitting.

    Attributes:
        hausdorff_distance: Maximum distance between surfaces (mm).
        mean_surface_distance: Average surface distance (mm).
        rms_distance: Root mean square surface distance (mm).
        volume_ratio: NURBS volume / segment volume ratio.
        num_control_points: Total control points used.
        min_jacobian: Minimum Jacobian determinant (>0 for valid mesh).
    """

    hausdorff_distance: float
    mean_surface_distance: float
    rms_distance: float
    volume_ratio: float
    num_control_points: int
    min_jacobian: float


class QualityMetrics:
    """Compute quality metrics for NURBS volume fitting.

    Example:
        metrics = QualityMetrics()
        quality = metrics.compute(nurbs_volume, segmentation_node, segment_id)
        print(f"Hausdorff distance: {quality.hausdorff_distance:.2f} mm")
    """

    def __init__(self, surface_sample_resolution: int = 50):
        """Initialize quality metrics calculator.

        Args:
            surface_sample_resolution: Resolution for surface sampling.
        """
        self.surface_sample_resolution = surface_sample_resolution

    def compute(
        self,
        nurbs_volume: NurbsVolume,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> FittingQuality:
        """Compute all quality metrics.

        Args:
            nurbs_volume: The NURBS volume to evaluate.
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to compare.

        Returns:
            FittingQuality with computed metrics.
        """
        # Get segment surface points
        segment_surface = self._get_segment_surface_points(segmentation_node, segment_id)

        # Get NURBS surface points
        nurbs_surface = self._get_nurbs_surface_points(nurbs_volume)

        if len(segment_surface) == 0 or len(nurbs_surface) == 0:
            logger.warning("Could not compute quality metrics: empty surfaces")
            return FittingQuality(
                hausdorff_distance=float("inf"),
                mean_surface_distance=float("inf"),
                rms_distance=float("inf"),
                volume_ratio=0.0,
                num_control_points=nurbs_volume.num_control_points,
                min_jacobian=0.0,
            )

        # Compute surface distances
        hausdorff, mean_dist, rms_dist = self._compute_surface_distances(
            segment_surface, nurbs_surface
        )

        # Compute volume ratio
        volume_ratio = self._compute_volume_ratio(nurbs_volume, segmentation_node, segment_id)

        # Compute Jacobian quality
        min_jacobian = self._compute_jacobian_quality(nurbs_volume)

        quality = FittingQuality(
            hausdorff_distance=hausdorff,
            mean_surface_distance=mean_dist,
            rms_distance=rms_dist,
            volume_ratio=volume_ratio,
            num_control_points=nurbs_volume.num_control_points,
            min_jacobian=min_jacobian,
        )

        logger.info(
            f"Quality metrics: Hausdorff={hausdorff:.2f}mm, "
            f"mean={mean_dist:.2f}mm, volume_ratio={volume_ratio:.2f}"
        )

        return quality

    def _get_segment_surface_points(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> np.ndarray:
        """Extract surface points from segment.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment.

        Returns:
            Surface points array, shape (n, 3).
        """
        import vtk

        # Get closed surface representation
        segmentation = segmentation_node.GetSegmentation()
        if segmentation is None:
            return np.zeros((0, 3))

        # Create closed surface if not exists
        if not segmentation.ContainsRepresentation("Closed surface"):
            segmentation.CreateRepresentation("Closed surface")

        # Get the polydata
        polydata = vtk.vtkPolyData()
        segmentation_node.GetClosedSurfaceRepresentation(segment_id, polydata)

        if polydata.GetNumberOfPoints() == 0:
            return np.zeros((0, 3))

        # Extract points
        points = []
        for i in range(polydata.GetNumberOfPoints()):
            point = polydata.GetPoint(i)
            points.append(point)

        return np.array(points)

    def _get_nurbs_surface_points(
        self,
        nurbs_volume: NurbsVolume,
    ) -> np.ndarray:
        """Sample points from NURBS volume surface.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            Surface points array, shape (n, 3).
        """
        if nurbs_volume.geomdl_volume is None:
            # Fallback: use control point hull
            return nurbs_volume.control_points

        # Sample NURBS surface
        from .NurbsVolumeBuilder import NurbsVolumeBuilder

        builder = NurbsVolumeBuilder()
        return builder.sample_surface(nurbs_volume, self.surface_sample_resolution)

    def _compute_surface_distances(
        self,
        surface_a: np.ndarray,
        surface_b: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute surface distance metrics.

        Args:
            surface_a: Points from surface A, shape (n, 3).
            surface_b: Points from surface B, shape (m, 3).

        Returns:
            Tuple of (hausdorff_distance, mean_distance, rms_distance).
        """
        from scipy.spatial import cKDTree

        # Build KD-tree for fast nearest neighbor lookup
        tree_a = cKDTree(surface_a)
        tree_b = cKDTree(surface_b)

        # Distance from A to B (for each point in A, find closest in B)
        dist_a_to_b, _ = tree_b.query(surface_a)

        # Distance from B to A
        dist_b_to_a, _ = tree_a.query(surface_b)

        # Hausdorff: max of both directions
        hausdorff = max(dist_a_to_b.max(), dist_b_to_a.max())

        # Mean distance (symmetric)
        mean_dist = (dist_a_to_b.mean() + dist_b_to_a.mean()) / 2

        # RMS distance
        all_dists = np.concatenate([dist_a_to_b, dist_b_to_a])
        rms_dist = np.sqrt(np.mean(all_dists**2))

        return float(hausdorff), float(mean_dist), float(rms_dist)

    def _compute_volume_ratio(
        self,
        nurbs_volume: NurbsVolume,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> float:
        """Compute ratio of NURBS volume to segment volume.

        Args:
            nurbs_volume: The NURBS volume.
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment.

        Returns:
            Volume ratio (NURBS / segment).
        """
        import slicer
        import vtk

        # Get segment statistics
        segmentation = segmentation_node.GetSegmentation()
        if segmentation is None:
            return 0.0

        # Create labelmap to compute volume
        labelmap_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        try:
            segment_ids = vtk.vtkStringArray()
            segment_ids.InsertNextValue(segment_id)

            success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentation_node, segment_ids, labelmap_volume
            )

            if not success:
                return 0.0

            # Get voxel volume
            labelmap_array = slicer.util.arrayFromVolume(labelmap_volume)
            spacing = labelmap_volume.GetSpacing()
            voxel_volume = spacing[0] * spacing[1] * spacing[2]

            segment_volume = np.sum(labelmap_array > 0) * voxel_volume

            # Estimate NURBS volume from control point bounding box
            # (This is approximate; proper integration would be more accurate)
            control_points = nurbs_volume.control_points
            extents = control_points.max(axis=0) - control_points.min(axis=0)
            nurbs_volume_estimate = extents[0] * extents[1] * extents[2]

            if segment_volume == 0:
                return 0.0

            return nurbs_volume_estimate / segment_volume

        finally:
            slicer.mrmlScene.RemoveNode(labelmap_volume)

    def _compute_jacobian_quality(
        self,
        nurbs_volume: NurbsVolume,
        num_samples: int = 10,
    ) -> float:
        """Compute minimum Jacobian determinant of the NURBS mapping.

        A positive Jacobian everywhere indicates a valid (non-inverted)
        parameterization. The Jacobian determinant measures local volume
        change - negative values indicate inverted elements.

        Args:
            nurbs_volume: The NURBS volume.
            num_samples: Number of sample points per parametric direction.

        Returns:
            Minimum Jacobian determinant (>0 for valid mesh, <0 for invalid).
        """
        if nurbs_volume.geomdl_volume is None:
            # Without geomdl, use finite difference approximation on control points
            return self._compute_jacobian_from_control_points(nurbs_volume)

        # Sample Jacobian at interior points
        min_jacobian = float("inf")
        eps = 0.01  # Finite difference step

        # Sample at regular parametric intervals (avoiding boundaries)
        params = np.linspace(0.1, 0.9, num_samples)

        for u in params:
            for v in params:
                for w in params:
                    jacobian = self._compute_jacobian_at_point(
                        nurbs_volume.geomdl_volume, u, v, w, eps
                    )
                    min_jacobian = min(min_jacobian, jacobian)

        if min_jacobian == float("inf"):
            return 1.0  # Fallback

        return float(min_jacobian)

    def _compute_jacobian_at_point(
        self,
        geomdl_volume: Any,
        u: float,
        v: float,
        w: float,
        eps: float,
    ) -> float:
        """Compute Jacobian determinant at a parametric point.

        Uses finite differences to approximate partial derivatives
        of the NURBS mapping.

        Args:
            geomdl_volume: The geomdl Volume object.
            u, v, w: Parametric coordinates.
            eps: Finite difference step size.

        Returns:
            Jacobian determinant at (u, v, w).
        """
        # Evaluate partial derivatives using central differences
        # dX/du
        p_u_plus = np.array(geomdl_volume.evaluate_single((min(u + eps, 1.0), v, w)))
        p_u_minus = np.array(geomdl_volume.evaluate_single((max(u - eps, 0.0), v, w)))
        dx_du = (p_u_plus - p_u_minus) / (2 * eps)

        # dX/dv
        p_v_plus = np.array(geomdl_volume.evaluate_single((u, min(v + eps, 1.0), w)))
        p_v_minus = np.array(geomdl_volume.evaluate_single((u, max(v - eps, 0.0), w)))
        dx_dv = (p_v_plus - p_v_minus) / (2 * eps)

        # dX/dw
        p_w_plus = np.array(geomdl_volume.evaluate_single((u, v, min(w + eps, 1.0))))
        p_w_minus = np.array(geomdl_volume.evaluate_single((u, v, max(w - eps, 0.0))))
        dx_dw = (p_w_plus - p_w_minus) / (2 * eps)

        # Jacobian matrix
        jacobian_matrix = np.column_stack([dx_du, dx_dv, dx_dw])

        # Determinant
        return float(np.linalg.det(jacobian_matrix))

    def _compute_jacobian_from_control_points(
        self,
        nurbs_volume: NurbsVolume,
    ) -> float:
        """Compute approximate Jacobian quality from control points.

        Used as fallback when geomdl is not available. Checks for
        inverted hexahedral cells in the control mesh.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            Minimum Jacobian determinant estimate.
        """
        nu, nv, nw = nurbs_volume.size
        control_points = nurbs_volume.control_points.reshape(nu, nv, nw, 3)

        min_jacobian = float("inf")

        # Check each hexahedral cell in control mesh
        for i in range(nu - 1):
            for j in range(nv - 1):
                for k in range(nw - 1):
                    # Get 8 corner vertices of hex cell
                    p000 = control_points[i, j, k, :]
                    p100 = control_points[i + 1, j, k, :]
                    p010 = control_points[i, j + 1, k, :]
                    p001 = control_points[i, j, k + 1, :]

                    # Compute edge vectors from p000
                    e1 = p100 - p000  # du direction
                    e2 = p010 - p000  # dv direction
                    e3 = p001 - p000  # dw direction

                    # Jacobian determinant = scalar triple product
                    jacobian = np.dot(np.cross(e1, e2), e3)
                    min_jacobian = min(min_jacobian, jacobian)

        if min_jacobian == float("inf"):
            return 1.0

        return float(min_jacobian)
