"""Mock segmentation factory for testing.

Creates test segmentation nodes with known properties for
testing metrics computation, visualization, and comparison.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MockSegmentationFactory:
    """Factory for creating mock segmentation nodes.

    Creates segmentation nodes with known geometric properties
    for testing metrics computation, visualization modes, etc.

    Usage:
        factory = MockSegmentationFactory()
        seg_node = factory.create_sphere_segmentation(
            center_ijk=(128, 128, 64),
            radius=20,
            volume_node=volume,
        )
        factory.cleanup()
    """

    def __init__(self) -> None:
        """Initialize the factory."""
        self._created_nodes: list[Any] = []

    def create_sphere_segmentation(
        self,
        center_ijk: tuple[int, int, int],
        radius: int,
        volume_node: Any,
        name: str = "TestSphere",
        segment_name: str = "Sphere",
    ) -> Any:
        """Create a segmentation with a spherical segment.

        Args:
            center_ijk: Center of sphere in IJK coordinates.
            radius: Radius of sphere in voxels.
            volume_node: Reference volume node for geometry.
            name: Name for segmentation node.
            segment_name: Name for the segment.

        Returns:
            vtkMRMLSegmentationNode with spherical segment.
        """
        import numpy as np
        import slicer

        # Create segmentation node
        seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        seg_node.SetName(name)
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        self._created_nodes.append(seg_node)

        # Add empty segment
        segment_id = seg_node.GetSegmentation().AddEmptySegment(segment_name)

        # Get labelmap representation
        seg_node.GetSegmentation().CreateRepresentation("Binary labelmap")

        # Create sphere in numpy
        dims = volume_node.GetImageData().GetDimensions()
        labelmap = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)

        # Create sphere mask
        z, y, x = np.ogrid[: dims[2], : dims[1], : dims[0]]
        cx, cy, cz = center_ijk
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        labelmap[dist <= radius] = 1

        # Update segment with labelmap
        self._update_segment_from_array(seg_node, segment_id, labelmap, volume_node)

        logger.info(
            f"Created sphere segmentation: center={center_ijk}, radius={radius}, "
            f"voxels={np.sum(labelmap)}"
        )
        return seg_node

    def create_cube_segmentation(
        self,
        corner_ijk: tuple[int, int, int],
        size: int,
        volume_node: Any,
        name: str = "TestCube",
        segment_name: str = "Cube",
    ) -> Any:
        """Create a segmentation with a cubic segment.

        Args:
            corner_ijk: Lower corner of cube in IJK coordinates.
            size: Size of cube in voxels.
            volume_node: Reference volume node for geometry.
            name: Name for segmentation node.
            segment_name: Name for the segment.

        Returns:
            vtkMRMLSegmentationNode with cubic segment.
        """
        import numpy as np
        import slicer

        # Create segmentation node
        seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        seg_node.SetName(name)
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        self._created_nodes.append(seg_node)

        # Add empty segment
        segment_id = seg_node.GetSegmentation().AddEmptySegment(segment_name)

        # Create cube in numpy
        dims = volume_node.GetImageData().GetDimensions()
        labelmap = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)

        x1, y1, z1 = corner_ijk
        x2, y2, z2 = x1 + size, y1 + size, z1 + size
        # Clamp to valid indices
        x2 = min(x2, dims[0])
        y2 = min(y2, dims[1])
        z2 = min(z2, dims[2])

        labelmap[z1:z2, y1:y2, x1:x2] = 1

        # Update segment with labelmap
        self._update_segment_from_array(seg_node, segment_id, labelmap, volume_node)

        logger.info(
            f"Created cube segmentation: corner={corner_ijk}, size={size}, "
            f"voxels={np.sum(labelmap)}"
        )
        return seg_node

    def create_gold_test_pair(
        self,
        volume_node: Any,
        overlap_ratio: float = 0.8,
        center_ijk: tuple[int, int, int] | None = None,
        radius: int = 20,
    ) -> tuple[Any, Any]:
        """Create overlapping gold/test segmentation pair for metrics testing.

        Args:
            volume_node: Reference volume node.
            overlap_ratio: Target overlap (0.0 - 1.0).
            center_ijk: Center of gold sphere. Defaults to volume center.
            radius: Radius in voxels.

        Returns:
            Tuple of (gold_node, test_node).
        """
        # Default to volume center
        if center_ijk is None:
            dims = volume_node.GetImageData().GetDimensions()
            center_ijk = (dims[0] // 2, dims[1] // 2, dims[2] // 2)

        # Create gold standard
        gold_node = self.create_sphere_segmentation(
            center_ijk=center_ijk,
            radius=radius,
            volume_node=volume_node,
            name="GoldStandard",
            segment_name="GoldSegment",
        )

        # Calculate offset for test segmentation to achieve target overlap
        # For spheres, overlap_ratio roughly corresponds to center offset
        # Full overlap (1.0) = 0 offset, no overlap (0.0) = 2*radius offset
        offset = int(radius * 2 * (1 - overlap_ratio) * 0.5)

        test_center = (center_ijk[0] + offset, center_ijk[1], center_ijk[2])

        # Create test segmentation
        test_node = self.create_sphere_segmentation(
            center_ijk=test_center,
            radius=radius,
            volume_node=volume_node,
            name="TestSegmentation",
            segment_name="TestSegment",
        )

        logger.info(
            f"Created gold/test pair: target overlap={overlap_ratio}, offset={offset} voxels"
        )
        return gold_node, test_node

    def _update_segment_from_array(
        self,
        seg_node: Any,
        segment_id: str,
        labelmap: Any,
        volume_node: Any,
    ) -> None:
        """Update a segment's labelmap from numpy array.

        Args:
            seg_node: Segmentation node.
            segment_id: ID of segment to update.
            labelmap: Numpy array with labelmap data.
            volume_node: Reference volume for geometry.
        """
        import slicer
        import vtk
        from vtk.util import numpy_support

        # Create vtkOrientedImageData from numpy array
        oriented_image = slicer.vtkOrientedImageData()

        # Get volume geometry
        image_data = volume_node.GetImageData()
        oriented_image.SetDimensions(image_data.GetDimensions())
        oriented_image.SetSpacing(volume_node.GetSpacing())
        oriented_image.SetOrigin(volume_node.GetOrigin())

        # Set directions from volume
        directions = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASDirectionMatrix(directions)
        oriented_image.SetDirectionMatrix(directions)

        # Set the labelmap data
        flat_array = labelmap.flatten(order="F")
        vtk_array = numpy_support.numpy_to_vtk(
            flat_array, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_array.SetNumberOfComponents(1)

        oriented_image.GetPointData().SetScalars(vtk_array)
        oriented_image.Modified()

        # Update segment using the Slicer segmentations logic
        slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
            oriented_image,
            seg_node,
            segment_id,
            slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE,
        )

    def cleanup(self) -> None:
        """Remove all created nodes from the scene."""
        import slicer

        for node in self._created_nodes:
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception as e:
                logger.debug(f"Error removing node: {e}")
        self._created_nodes.clear()
        logger.debug("MockSegmentationFactory cleaned up")
