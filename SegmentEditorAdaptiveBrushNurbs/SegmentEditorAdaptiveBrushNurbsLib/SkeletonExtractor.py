"""Skeleton and centerline extraction using SlicerVMTK.

Extracts centerlines from tubular segments for NURBS volume generation.
Uses VMTK's ExtractCenterline module which computes:
- Centerline path via weighted shortest path on Voronoi diagram
- Inscribed sphere radii along the centerline
- Branch detection for bifurcating structures

References:
- SlicerVMTK: https://github.com/vmtk/SlicerExtension-VMTK
- VMTK Documentation: http://www.vmtk.org/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vtkMRMLMarkupsFiducialNode import vtkMRMLMarkupsFiducialNode
    from vtkMRMLModelNode import vtkMRMLModelNode
    from vtkMRMLSegmentationNode import vtkMRMLSegmentationNode

logger = logging.getLogger(__name__)


@dataclass
class Centerline:
    """Centerline data extracted from a tubular segment.

    Attributes:
        points: Centerline points in RAS coordinates, shape (n, 3).
        radii: Inscribed sphere radius at each point, shape (n,).
        tangents: Tangent vectors at each point, shape (n, 3).
        curve_node: The MRML curve node (if created).
    """

    points: np.ndarray
    radii: np.ndarray
    tangents: np.ndarray | None = None
    curve_node: object | None = None

    @property
    def length(self) -> float:
        """Total arc length of the centerline."""
        if len(self.points) < 2:
            return 0.0
        segments = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(segments, axis=1)))

    @property
    def num_points(self) -> int:
        """Number of centerline points."""
        return len(self.points)

    def resample(self, num_points: int) -> Centerline:
        """Resample centerline to have a specific number of points.

        Uses linear interpolation along the centerline path.

        Args:
            num_points: Target number of points.

        Returns:
            New Centerline with resampled points.
        """
        if self.num_points < 2 or num_points < 2:
            return self

        # Compute cumulative arc length
        segments = np.diff(self.points, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        if total_length < 1e-6:
            return self

        # Normalize to [0, 1]
        normalized_length = cumulative_length / total_length

        # Target positions
        target_positions = np.linspace(0, 1, num_points)

        # Interpolate points
        new_points = np.zeros((num_points, 3))
        new_radii = np.zeros(num_points)

        for i, t in enumerate(target_positions):
            # Find segment containing t
            idx = np.searchsorted(normalized_length, t) - 1
            idx = max(0, min(idx, len(normalized_length) - 2))

            # Local parameter within segment
            t0 = normalized_length[idx]
            t1 = normalized_length[idx + 1]
            if t1 - t0 < 1e-10:
                local_t = 0
            else:
                local_t = (t - t0) / (t1 - t0)

            # Interpolate
            new_points[i] = (1 - local_t) * self.points[idx] + local_t * self.points[idx + 1]
            new_radii[i] = (1 - local_t) * self.radii[idx] + local_t * self.radii[idx + 1]

        # Compute tangents
        new_tangents = self._compute_tangents(new_points)

        return Centerline(
            points=new_points,
            radii=new_radii,
            tangents=new_tangents,
            curve_node=self.curve_node,
        )

    def _compute_tangents(self, points: np.ndarray) -> np.ndarray:
        """Compute tangent vectors at each point.

        Args:
            points: Centerline points (n, 3).

        Returns:
            Tangent vectors (n, 3), normalized.
        """
        n = len(points)
        tangents = np.zeros_like(points)

        for i in range(n):
            if i == 0:
                tangent = points[1] - points[0]
            elif i == n - 1:
                tangent = points[-1] - points[-2]
            else:
                tangent = points[i + 1] - points[i - 1]

            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                tangent /= norm
            tangents[i] = tangent

        return tangents


@dataclass
class BranchPoint:
    """Bifurcation point in a branching centerline.

    Attributes:
        position: Location of bifurcation in RAS coordinates.
        parent_direction: Direction of parent branch.
        child_directions: Directions of child branches.
        radius: Radius at bifurcation point.
        branch_ids: IDs of branches meeting at this point.
    """

    position: np.ndarray
    parent_direction: np.ndarray | None = None
    child_directions: list[np.ndarray] | None = None
    radius: float = 0.0
    branch_ids: list[int] | None = None


class SkeletonExtractor:
    """Extract centerlines from tubular segments using SlicerVMTK.

    This class wraps SlicerVMTK's ExtractCenterline module to compute
    centerlines from tubular segmentations (vessels, airways, etc.).

    Example:
        extractor = SkeletonExtractor()
        centerline = extractor.extract_centerline(
            segmentation_node, segment_id,
            start_point=(100, 50, 60),
            end_point=(100, 50, 120)
        )
        print(f"Centerline length: {centerline.length} mm")
    """

    # Default parameters for centerline extraction
    DEFAULT_RESAMPLE_STEP = 1.0  # mm between centerline points

    def __init__(self):
        """Initialize the skeleton extractor."""
        self._vmtk_available: bool | None = None

    def is_vmtk_available(self) -> bool:
        """Check if SlicerVMTK extension is installed.

        Returns:
            True if VMTK modules are available.
        """
        if self._vmtk_available is not None:
            return self._vmtk_available

        try:
            import slicer

            self._vmtk_available = hasattr(slicer.modules, "extractcenterline")
        except Exception:
            self._vmtk_available = False

        return self._vmtk_available

    def extract_centerline(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
        start_point: tuple[float, float, float] | np.ndarray | None = None,
        end_point: tuple[float, float, float] | np.ndarray | None = None,
        auto_detect_endpoints: bool = True,
        resample_step: float | None = None,
    ) -> Centerline:
        """Extract centerline from a tubular segment.

        Uses VMTK's ExtractCenterline which computes a weighted shortest
        path through the Voronoi diagram of the segment surface.

        Args:
            segmentation_node: MRML segmentation node containing the segment.
            segment_id: ID of the tubular segment.
            start_point: Start point for centerline (RAS coordinates).
            end_point: End point for centerline (RAS coordinates).
            auto_detect_endpoints: If True, automatically detect endpoints
                when not provided.
            resample_step: Distance between resampled centerline points (mm).
                If None, uses DEFAULT_RESAMPLE_STEP.

        Returns:
            Centerline object with points and radii.

        Raises:
            RuntimeError: If SlicerVMTK is not available.
            ValueError: If segment is empty or extraction fails.
        """
        if not self.is_vmtk_available():
            raise RuntimeError(
                "SlicerVMTK extension is required for centerline extraction. "
                "Please install it from the Extension Manager."
            )

        if resample_step is None:
            resample_step = self.DEFAULT_RESAMPLE_STEP

        logger.info(f"Extracting centerline from segment '{segment_id}'")

        import slicer

        # Get or create surface model from segment
        surface_model = self._create_surface_from_segment(segmentation_node, segment_id)

        try:
            # Auto-detect endpoints if needed
            if auto_detect_endpoints and (start_point is None or end_point is None):
                detected_start, detected_end = self._auto_detect_endpoints(surface_model)
                if start_point is None:
                    start_point = detected_start
                if end_point is None:
                    end_point = detected_end

            if start_point is None or end_point is None:
                raise ValueError("Could not determine centerline endpoints")

            # Create endpoint fiducials
            endpoint_node = self._create_endpoint_fiducials(start_point, end_point)

            try:
                # Run VMTK ExtractCenterline
                centerline_model, centerline_curve = self._run_vmtk_centerline(
                    surface_model, endpoint_node
                )

                try:
                    # Extract points and radii from result
                    points, radii = self._parse_centerline_result(centerline_model)

                    # Compute tangents
                    tangents = self._compute_tangents(points)

                    centerline = Centerline(
                        points=points,
                        radii=radii,
                        tangents=tangents,
                        curve_node=centerline_curve,
                    )

                    # Resample if requested
                    if resample_step > 0:
                        target_points = max(2, int(centerline.length / resample_step))
                        centerline = centerline.resample(target_points)

                    logger.info(
                        f"Extracted centerline: {centerline.num_points} points, "
                        f"length={centerline.length:.1f} mm"
                    )

                    return centerline

                finally:
                    # Clean up centerline model (keep curve if useful)
                    if centerline_model is not None:
                        slicer.mrmlScene.RemoveNode(centerline_model)

            finally:
                # Clean up endpoint fiducials
                slicer.mrmlScene.RemoveNode(endpoint_node)

        finally:
            # Clean up surface model
            slicer.mrmlScene.RemoveNode(surface_model)

    def detect_branches(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> tuple[list[Centerline], list[BranchPoint]]:
        """Detect branches in a branching tubular structure.

        Uses VMTK's branch detection to identify bifurcation points
        and extract centerlines for each branch.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of the branching segment.

        Returns:
            Tuple of (centerlines, branch_points):
            - centerlines: List of Centerline objects, one per branch
            - branch_points: List of BranchPoint objects at bifurcations

        Raises:
            RuntimeError: If SlicerVMTK is not available.
            NotImplementedError: Branch detection not yet implemented.
        """
        if not self.is_vmtk_available():
            raise RuntimeError(
                "SlicerVMTK extension is required for branch detection. "
                "Please install it from the Extension Manager."
            )

        # TODO: Implement using VMTK BranchClipper
        # This requires:
        # 1. Extract full centerline network
        # 2. Identify bifurcation points
        # 3. Split into individual branches
        # 4. Compute branch connectivity

        raise NotImplementedError("Branch detection not yet implemented")

    def _create_surface_from_segment(
        self,
        segmentation_node: vtkMRMLSegmentationNode,
        segment_id: str,
    ) -> vtkMRMLModelNode:
        """Create a surface model from a segment.

        Args:
            segmentation_node: MRML segmentation node.
            segment_id: ID of segment to convert.

        Returns:
            MRML model node with segment surface.

        Raises:
            ValueError: If segment not found or conversion fails.
        """
        import slicer
        import vtk

        segmentation = segmentation_node.GetSegmentation()
        if segmentation is None:
            raise ValueError("Segmentation node has no segmentation")

        segment = segmentation.GetSegment(segment_id)
        if segment is None:
            raise ValueError(f"Segment '{segment_id}' not found")

        # Create model node
        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        model_node.SetName(f"{segment_id}_surface")

        # Export segment as closed surface
        segment_ids = vtk.vtkStringArray()
        segment_ids.InsertNextValue(segment_id)

        # Get closed surface representation
        segmentation.CreateRepresentation("Closed surface")
        polydata = segmentation.GetSegmentClosedSurfaceRepresentation(segment_id)

        if polydata is None or polydata.GetNumberOfPoints() == 0:
            slicer.mrmlScene.RemoveNode(model_node)
            raise ValueError(f"Could not create surface for segment '{segment_id}'")

        # Copy polydata to model
        model_polydata = vtk.vtkPolyData()
        model_polydata.DeepCopy(polydata)

        # Apply segmentation transform
        parent_transform = segmentation_node.GetParentTransformNode()
        if parent_transform is not None:
            transform_matrix = vtk.vtkMatrix4x4()
            parent_transform.GetMatrixTransformToWorld(transform_matrix)

            transform = vtk.vtkTransform()
            transform.SetMatrix(transform_matrix)

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(model_polydata)
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            model_polydata = transform_filter.GetOutput()

        model_node.SetAndObservePolyData(model_polydata)

        return model_node

    def _auto_detect_endpoints(
        self,
        surface_model: vtkMRMLModelNode,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Automatically detect endpoints for centerline extraction.

        Uses the two most distant points along the principal axis
        of the surface as endpoints.

        Args:
            surface_model: MRML model node with segment surface.

        Returns:
            Tuple of (start_point, end_point) in RAS coordinates.

        Raises:
            ValueError: If surface is empty or too small.
        """

        polydata = surface_model.GetPolyData()
        if polydata is None or polydata.GetNumberOfPoints() < 10:
            raise ValueError("Surface model has insufficient points for endpoint detection")

        # Get all surface points
        points_list = []
        for i in range(polydata.GetNumberOfPoints()):
            pt = polydata.GetPoint(i)
            points_list.append(pt)
        points = np.array(points_list)

        # Compute principal axis via PCA
        center = points.mean(axis=0)
        centered = points - center

        # Covariance and eigenvectors
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal axis (largest eigenvalue)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Project points onto principal axis
        projections = centered @ principal_axis

        # Find extreme points
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)

        start_point = points[min_idx]
        end_point = points[max_idx]

        logger.debug(
            f"Auto-detected endpoints: start={start_point}, end={end_point}, "
            f"distance={np.linalg.norm(end_point - start_point):.1f} mm"
        )

        return start_point, end_point

    def _create_endpoint_fiducials(
        self,
        start_point: tuple[float, float, float] | np.ndarray,
        end_point: tuple[float, float, float] | np.ndarray,
    ) -> vtkMRMLMarkupsFiducialNode:
        """Create fiducial node with centerline endpoints.

        Args:
            start_point: Start point in RAS coordinates.
            end_point: End point in RAS coordinates.

        Returns:
            MRML markups fiducial node with two control points.
        """
        import slicer

        fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        fiducial_node.SetName("CenterlineEndpoints")

        # Add start point
        start = np.array(start_point)
        fiducial_node.AddControlPoint(start[0], start[1], start[2], "Start")

        # Add end point
        end = np.array(end_point)
        fiducial_node.AddControlPoint(end[0], end[1], end[2], "End")

        return fiducial_node

    def _run_vmtk_centerline(
        self,
        surface_model: vtkMRMLModelNode,
        endpoint_node: vtkMRMLMarkupsFiducialNode,
    ) -> tuple[vtkMRMLModelNode, object]:
        """Run VMTK ExtractCenterline module.

        Args:
            surface_model: Input surface model.
            endpoint_node: Fiducial node with start/end points.

        Returns:
            Tuple of (centerline_model, centerline_curve):
            - centerline_model: Model with centerline polydata
            - centerline_curve: Curve node (may be None)

        Raises:
            RuntimeError: If centerline extraction fails.
        """
        import slicer

        # Create output nodes
        centerline_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        centerline_model.SetName("CenterlineModel")

        centerline_curve = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        centerline_curve.SetName("CenterlineCurve")

        # Get the ExtractCenterline logic
        try:
            extract_logic = slicer.modules.extractcenterline.widgetRepresentation().self().logic
        except AttributeError:
            # Alternative way to get the logic
            extract_logic = slicer.modules.extractcenterline.logic()

        # Set parameters and run
        # Note: The exact API depends on SlicerVMTK version
        try:
            # SlicerVMTK approach: use the logic directly
            extract_logic.setInputSurfaceNode(surface_model)
            extract_logic.setInputEndpointsNode(endpoint_node)
            extract_logic.setOutputCenterlineModelNode(centerline_model)
            extract_logic.setOutputCenterlineCurveNode(centerline_curve)
            extract_logic.computeCenterlines()

        except AttributeError:
            # Fallback: try alternative API
            logger.warning("Using fallback VMTK API")

            # Create parameter node
            param_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScriptedModuleNode", "ExtractCenterlineParameters"
            )

            try:
                param_node.SetParameter("InputSurface", surface_model.GetID())
                param_node.SetParameter("EndPointsMarkups", endpoint_node.GetID())
                param_node.SetParameter("OutputCenterlineModel", centerline_model.GetID())
                param_node.SetParameter("OutputCenterlineCurve", centerline_curve.GetID())

                # Run extraction
                extract_logic.computeCenterlines(param_node)

            finally:
                slicer.mrmlScene.RemoveNode(param_node)

        # Verify output
        if centerline_model.GetPolyData() is None:
            slicer.mrmlScene.RemoveNode(centerline_model)
            slicer.mrmlScene.RemoveNode(centerline_curve)
            raise RuntimeError("Centerline extraction failed - no output produced")

        if centerline_model.GetPolyData().GetNumberOfPoints() == 0:
            slicer.mrmlScene.RemoveNode(centerline_model)
            slicer.mrmlScene.RemoveNode(centerline_curve)
            raise RuntimeError("Centerline extraction failed - empty output")

        return centerline_model, centerline_curve

    def _parse_centerline_result(
        self,
        centerline_model: vtkMRMLModelNode,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse centerline model to extract points and radii.

        VMTK stores the inscribed sphere radius as a point data array
        named "Radius" or "MaximumInscribedSphereRadius".

        Args:
            centerline_model: Model node with centerline polydata.

        Returns:
            Tuple of (points, radii):
            - points: Centerline points (n, 3)
            - radii: Inscribed sphere radii (n,)
        """

        polydata = centerline_model.GetPolyData()

        # Extract points
        num_points = polydata.GetNumberOfPoints()
        points = np.zeros((num_points, 3))
        for i in range(num_points):
            pt = polydata.GetPoint(i)
            points[i] = pt

        # Extract radii from point data
        radii = np.ones(num_points)  # Default to 1.0

        point_data = polydata.GetPointData()
        if point_data is not None:
            # Try different array names used by VMTK
            for name in ["Radius", "MaximumInscribedSphereRadius", "MISR"]:
                radius_array = point_data.GetArray(name)
                if radius_array is not None:
                    for i in range(num_points):
                        radii[i] = radius_array.GetValue(i)
                    logger.debug(f"Found radius array '{name}'")
                    break
            else:
                logger.warning("No radius array found in centerline, using default radii")

        return points, radii

    def _compute_tangents(self, points: np.ndarray) -> np.ndarray:
        """Compute tangent vectors at each centerline point.

        Args:
            points: Centerline points (n, 3).

        Returns:
            Tangent vectors (n, 3), normalized.
        """
        n = len(points)
        tangents = np.zeros_like(points)

        for i in range(n):
            if i == 0:
                tangent = points[1] - points[0] if n > 1 else np.array([0, 0, 1])
            elif i == n - 1:
                tangent = points[-1] - points[-2]
            else:
                tangent = points[i + 1] - points[i - 1]

            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                tangent /= norm
            else:
                tangent = np.array([0, 0, 1])

            tangents[i] = tangent

        return tangents
