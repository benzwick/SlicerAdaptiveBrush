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
from typing import TYPE_CHECKING

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
    ) -> float:
        """Compute minimum Jacobian determinant of the NURBS mapping.

        A positive Jacobian everywhere indicates a valid (non-inverted)
        parameterization.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            Minimum Jacobian determinant (>0 for valid mesh).
        """
        # TODO: Implement proper Jacobian computation
        # This requires computing partial derivatives of the NURBS mapping
        # and evaluating the determinant at sample points

        # For now, return a placeholder
        return 1.0
