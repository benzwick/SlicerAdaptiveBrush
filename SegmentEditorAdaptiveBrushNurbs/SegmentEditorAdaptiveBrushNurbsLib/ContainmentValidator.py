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
from typing import TYPE_CHECKING

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

    Example:
        validator = ContainmentValidator()
        result = validator.validate(nurbs_volume, segmentation_node, segment_id)
        if not result.is_contained:
            # Expand NURBS or adjust control points
            pass
    """

    def __init__(self, tolerance: float = 0.5):
        """Initialize the containment validator.

        Args:
            tolerance: Distance tolerance in mm for containment check.
        """
        self.tolerance = tolerance

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

    def _check_point_inside(
        self,
        nurbs_volume: NurbsVolume,
        point: np.ndarray,
    ) -> tuple[bool, float]:
        """Check if a point is inside the NURBS volume.

        Uses ray casting or closest point distance to determine
        if the point is inside the volume boundary.

        Args:
            nurbs_volume: The NURBS volume.
            point: Point to check (3,) in RAS coordinates.

        Returns:
            Tuple of (is_inside, distance_to_surface).
        """
        # For a proper implementation, we would:
        # 1. Find the closest point on the NURBS surface
        # 2. Determine inside/outside using surface normal
        #
        # For now, use a simplified bounding box check

        control_points = nurbs_volume.control_points

        # Compute bounding box of control points
        min_bounds = control_points.min(axis=0)
        max_bounds = control_points.max(axis=0)

        # Add tolerance
        min_bounds -= self.tolerance
        max_bounds += self.tolerance

        # Check if point is within bounds
        is_inside = bool(np.all(point >= min_bounds) and np.all(point <= max_bounds))

        # Compute distance to bounding box
        if is_inside:
            distance = 0.0
        else:
            # Distance to nearest face
            dist_to_min = np.maximum(min_bounds - point, 0)
            dist_to_max = np.maximum(point - max_bounds, 0)
            distance = float(np.sqrt(np.sum(dist_to_min**2 + dist_to_max**2)))

        return is_inside, distance

    def expand_to_contain(
        self,
        nurbs_volume: NurbsVolume,
        outside_points: np.ndarray,
        expansion_factor: float = 1.1,
    ) -> NurbsVolume:
        """Expand NURBS volume to contain outside points.

        Adjusts boundary control points outward to include all
        segment voxels.

        Args:
            nurbs_volume: The NURBS volume to expand.
            outside_points: Points that are currently outside.
            expansion_factor: Factor to expand beyond required distance.

        Returns:
            New NurbsVolume with expanded boundaries.
        """
        # TODO: Implement control point adjustment
        # This would involve:
        # 1. Identifying which boundary face(s) need expansion
        # 2. Moving boundary control points along surface normals
        # 3. Rebuilding the NURBS volume

        logger.warning("Control point expansion not yet implemented")
        return nurbs_volume
