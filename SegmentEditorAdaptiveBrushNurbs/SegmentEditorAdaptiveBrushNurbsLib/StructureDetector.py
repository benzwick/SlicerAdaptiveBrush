"""Structure type detection for segmented regions.

Classifies segmented structures into categories for appropriate
NURBS generation strategy:
- Simple: Convex shapes (tumors, nodules) - use bounding box approach
- Tubular: Single tubes (vessels, airways) - use centerline sweeping
- Branching: Branching trees (arterial, bronchial) - use branch templates

Detection uses topology analysis:
- Euler characteristic
- Skeleton length to volume ratio
- Branch point count
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from vtkMRMLSegmentationNode import vtkMRMLSegmentationNode

logger = logging.getLogger(__name__)

StructureType = Literal["simple", "tubular", "branching"]


@dataclass
class StructureMetrics:
    """Metrics computed during structure detection.

    Attributes:
        euler_characteristic: Topological invariant (1 for simple, <1 for holes)
        skeleton_volume_ratio: Length of skeleton / cube root of volume
        branch_count: Number of branch points in skeleton
        aspect_ratio: Max extent / min extent (high for tubular)
        sphericity: How spherical the shape is (1.0 = perfect sphere)
    """

    euler_characteristic: float
    skeleton_volume_ratio: float
    branch_count: int
    aspect_ratio: float
    sphericity: float


class StructureDetector:
    """Detect structure type from segment topology.

    Uses morphological analysis to classify segments into categories
    for appropriate NURBS generation strategy.

    Example:
        detector = StructureDetector()
        structure_type = detector.detect(segmentation_node, segment_id)
        # Returns "simple", "tubular", or "branching"
    """

    # Thresholds for classification
    ASPECT_RATIO_TUBULAR_THRESHOLD = 3.0  # Tubular if aspect ratio > 3
    SKELETON_RATIO_TUBULAR_THRESHOLD = 0.3  # Tubular if skeleton/volume^(1/3) > 0.3
    BRANCH_COUNT_THRESHOLD = 1  # Branching if branch count > 1

    def __init__(self):
        """Initialize the structure detector."""
        self._last_metrics: StructureMetrics | None = None

    @property
    def last_metrics(self) -> StructureMetrics | None:
        """Get metrics from the last detection.

        Returns:
            Metrics computed during last detect() call, or None.
        """
        return self._last_metrics

    def detect(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> StructureType:
        """Detect the structure type of a segment.

        Args:
            segmentation_node: MRML segmentation node containing the segment.
            segment_id: ID of the segment to analyze.

        Returns:
            Structure type: "simple", "tubular", or "branching".

        Raises:
            ValueError: If segment cannot be found or is empty.
        """
        # Get segment as numpy array
        labelmap_array = self._get_segment_array(segmentation_node, segment_id)

        if labelmap_array is None or np.sum(labelmap_array) == 0:
            raise ValueError(f"Segment '{segment_id}' is empty or not found")

        # Compute metrics
        self._last_metrics = self._compute_metrics(labelmap_array)

        # Classify based on metrics
        structure_type = self._classify(self._last_metrics)

        logger.info(
            f"Structure detection: type={structure_type}, "
            f"euler={self._last_metrics.euler_characteristic:.2f}, "
            f"skeleton_ratio={self._last_metrics.skeleton_volume_ratio:.2f}, "
            f"branches={self._last_metrics.branch_count}, "
            f"aspect={self._last_metrics.aspect_ratio:.2f}"
        )

        return structure_type

    def _get_segment_array(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> np.ndarray | None:
        """Extract segment as numpy array.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to extract.

        Returns:
            Binary numpy array of segment, or None if not found.
        """
        import slicer

        # Get labelmap representation
        segmentation = segmentation_node.GetSegmentation()
        if segmentation is None:
            return None

        segment = segmentation.GetSegment(segment_id)
        if segment is None:
            return None

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
                return None

            # Convert to numpy array
            labelmap_array = slicer.util.arrayFromVolume(labelmap_volume)

            # Return binary mask
            return (labelmap_array > 0).astype(np.uint8)

        finally:
            # Clean up temporary node
            slicer.mrmlScene.RemoveNode(labelmap_volume)

    def _compute_metrics(self, labelmap_array: np.ndarray) -> StructureMetrics:
        """Compute structure metrics from binary labelmap.

        Args:
            labelmap_array: Binary numpy array (1 = inside segment).

        Returns:
            StructureMetrics with computed values.
        """
        import SimpleITK as sitk

        # Convert to SimpleITK image
        sitk_image = sitk.GetImageFromArray(labelmap_array.astype(np.uint8))

        # Volume (number of voxels)
        volume = np.sum(labelmap_array)

        # Bounding box and aspect ratio
        nonzero = np.argwhere(labelmap_array)
        if len(nonzero) == 0:
            return StructureMetrics(
                euler_characteristic=1.0,
                skeleton_volume_ratio=0.0,
                branch_count=0,
                aspect_ratio=1.0,
                sphericity=1.0,
            )

        min_coords = nonzero.min(axis=0)
        max_coords = nonzero.max(axis=0)
        extents = max_coords - min_coords + 1

        # Aspect ratio: max extent / min extent
        aspect_ratio = float(extents.max()) / max(float(extents.min()), 1.0)

        # Sphericity: ratio of volume to bounding sphere volume
        # Simplified: use inscribed sphere radius
        inscribed_radius = extents.min() / 2.0
        inscribed_sphere_volume = (4 / 3) * np.pi * (inscribed_radius**3)
        sphericity = min(volume / max(inscribed_sphere_volume, 1.0), 1.0)

        # Skeleton analysis using SimpleITK
        try:
            # Skeletonize using binary thinning
            skeleton_filter = sitk.BinaryThinningImageFilter()
            skeleton = skeleton_filter.Execute(sitk_image)
            skeleton_array = sitk.GetArrayFromImage(skeleton)

            skeleton_length = np.sum(skeleton_array)
            skeleton_volume_ratio = skeleton_length / max(volume ** (1 / 3), 1.0)

            # Count branch points (voxels with > 2 neighbors in skeleton)
            branch_count = self._count_branch_points(skeleton_array)

        except RuntimeError:
            # Thinning might fail for some shapes
            logger.debug("Skeleton analysis failed, using defaults")
            skeleton_volume_ratio = 0.0
            branch_count = 0

        # Euler characteristic using SimpleITK
        try:
            label_shape = sitk.LabelShapeStatisticsImageFilter()
            label_shape.Execute(sitk_image)

            if label_shape.GetNumberOfLabels() > 0:
                # Euler number approximation: 1 - (holes)
                # SimpleITK doesn't directly provide this, estimate from roundness
                euler_characteristic = 1.0  # Assume simple for now
            else:
                euler_characteristic = 1.0

        except RuntimeError:
            euler_characteristic = 1.0

        return StructureMetrics(
            euler_characteristic=euler_characteristic,
            skeleton_volume_ratio=skeleton_volume_ratio,
            branch_count=branch_count,
            aspect_ratio=aspect_ratio,
            sphericity=sphericity,
        )

    def _count_branch_points(self, skeleton_array: np.ndarray) -> int:
        """Count branch points in a skeleton.

        A branch point is a skeleton voxel with more than 2 neighbors.

        Args:
            skeleton_array: Binary skeleton array.

        Returns:
            Number of branch points.
        """
        from scipy import ndimage

        if np.sum(skeleton_array) == 0:
            return 0

        # Count neighbors for each skeleton voxel using 26-connectivity
        kernel = np.ones((3, 3, 3), dtype=np.int32)
        kernel[1, 1, 1] = 0  # Don't count self

        neighbor_count = ndimage.convolve(skeleton_array.astype(np.int32), kernel, mode="constant")

        # Branch points: skeleton voxels with > 2 neighbors
        branch_points = (skeleton_array > 0) & (neighbor_count > 2)

        return int(np.sum(branch_points))

    def _classify(self, metrics: StructureMetrics) -> StructureType:
        """Classify structure type based on metrics.

        Args:
            metrics: Computed structure metrics.

        Returns:
            Structure type classification.
        """
        # Check for branching first (most specific)
        if metrics.branch_count > self.BRANCH_COUNT_THRESHOLD:
            return "branching"

        # Check for tubular
        is_elongated = metrics.aspect_ratio > self.ASPECT_RATIO_TUBULAR_THRESHOLD
        has_long_skeleton = metrics.skeleton_volume_ratio > self.SKELETON_RATIO_TUBULAR_THRESHOLD

        if is_elongated and has_long_skeleton:
            return "tubular"

        # Default to simple
        return "simple"


# Import vtk for type hints (may not be available outside Slicer)
try:
    import vtk
except ImportError:
    vtk = None  # type: ignore
