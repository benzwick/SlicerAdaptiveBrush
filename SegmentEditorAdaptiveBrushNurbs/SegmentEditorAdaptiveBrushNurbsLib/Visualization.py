"""VTK visualization for NURBS volumes and control meshes.

Provides visualization components for:
- Hexahedral control mesh wireframe
- Control point markers
- NURBS volume surface
- Segment overlay for comparison

Visualization is displayed in Slicer's 3D view.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .HexMeshGenerator import HexMesh
    from .NurbsVolumeBuilder import NurbsVolume

logger = logging.getLogger(__name__)


class NurbsVisualization:
    """Manage NURBS visualization in Slicer 3D view.

    Creates and updates VTK actors for:
    - Control mesh (wireframe hexahedra)
    - Control points (spheres)
    - NURBS surface (polygonal mesh)

    Example:
        viz = NurbsVisualization()
        viz.show_control_mesh(hex_mesh)
        viz.show_nurbs_surface(nurbs_volume)
        # ... later
        viz.cleanup()
    """

    # Colors
    CONTROL_MESH_COLOR = (0.8, 0.8, 0.2)  # Yellow
    CONTROL_POINT_COLOR = (1.0, 0.3, 0.3)  # Red
    NURBS_SURFACE_COLOR = (0.3, 0.7, 1.0)  # Light blue

    def __init__(self):
        """Initialize visualization manager."""
        import vtk

        self._control_mesh_actor: vtk.vtkActor | None = None
        self._control_points_actor: vtk.vtkActor | None = None
        self._nurbs_surface_actor: vtk.vtkActor | None = None
        self._renderer: vtk.vtkRenderer | None = None

    def _get_3d_renderer(self):
        """Get the 3D view renderer.

        Returns:
            VTK renderer from Slicer's 3D view, or None.
        """
        try:
            import slicer

            layout_manager = slicer.app.layoutManager()
            three_d_widget = layout_manager.threeDWidget(0)
            if three_d_widget is None:
                return None
            three_d_view = three_d_widget.threeDView()
            if three_d_view is None:
                return None
            render_window = three_d_view.renderWindow()
            if render_window is None:
                return None
            renderers = render_window.GetRenderers()
            if renderers.GetNumberOfItems() == 0:
                return None
            return renderers.GetItemAsObject(0)
        except Exception as e:
            logger.debug(f"Could not get 3D renderer: {e}")
            return None

    def show_control_mesh(
        self,
        hex_mesh: HexMesh,
        color: tuple[float, float, float] | None = None,
        line_width: float = 2.0,
    ) -> None:
        """Show hexahedral control mesh as wireframe.

        Args:
            hex_mesh: The hexahedral control mesh.
            color: RGB color tuple (default yellow).
            line_width: Width of wireframe lines.
        """
        import vtk

        if color is None:
            color = self.CONTROL_MESH_COLOR

        # Remove existing actor
        self._remove_actor(self._control_mesh_actor)

        # Create polydata for wireframe
        polydata = self._create_hex_wireframe_polydata(hex_mesh)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetRepresentationToWireframe()

        # Add to renderer
        renderer = self._get_3d_renderer()
        if renderer is not None:
            renderer.AddActor(actor)
            self._control_mesh_actor = actor
            self._request_render()

        logger.debug("Control mesh visualization updated")

    def show_control_points(
        self,
        hex_mesh: HexMesh,
        color: tuple[float, float, float] | None = None,
        radius: float = 1.0,
    ) -> None:
        """Show control points as spheres.

        Args:
            hex_mesh: The hexahedral control mesh.
            color: RGB color tuple (default red).
            radius: Sphere radius in mm.
        """
        import vtk

        if color is None:
            color = self.CONTROL_POINT_COLOR

        # Remove existing actor
        self._remove_actor(self._control_points_actor)

        # Create glyph for spheres at control points
        points = vtk.vtkPoints()
        for pt in hex_mesh.flat_control_points:
            points.InsertNextPoint(pt[0], pt[1], pt[2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Create sphere source for glyphs
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetPhiResolution(8)
        sphere.SetThetaResolution(8)

        # Create glyph filter
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetInputData(polydata)
        glyph.SetScaleModeToDataScalingOff()

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)

        # Add to renderer
        renderer = self._get_3d_renderer()
        if renderer is not None:
            renderer.AddActor(actor)
            self._control_points_actor = actor
            self._request_render()

        logger.debug("Control points visualization updated")

    def show_nurbs_surface(
        self,
        nurbs_volume: NurbsVolume,
        color: tuple[float, float, float] | None = None,
        opacity: float = 0.8,
        resolution: int = 30,
    ) -> None:
        """Show NURBS volume boundary surface.

        Args:
            nurbs_volume: The NURBS volume.
            color: RGB color tuple (default light blue).
            opacity: Surface opacity (0-1).
            resolution: Surface sampling resolution.
        """
        import vtk

        if color is None:
            color = self.NURBS_SURFACE_COLOR

        # Remove existing actor
        self._remove_actor(self._nurbs_surface_actor)

        # Create surface mesh from NURBS
        polydata = self._create_nurbs_surface_polydata(nurbs_volume, resolution)

        if polydata.GetNumberOfPoints() == 0:
            logger.warning("Could not create NURBS surface visualization")
            return

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)

        # Add to renderer
        renderer = self._get_3d_renderer()
        if renderer is not None:
            renderer.AddActor(actor)
            self._nurbs_surface_actor = actor
            self._request_render()

        logger.debug("NURBS surface visualization updated")

    def set_opacity(self, opacity: float) -> None:
        """Set NURBS surface opacity.

        Args:
            opacity: Opacity value (0-1).
        """
        if self._nurbs_surface_actor is not None:
            self._nurbs_surface_actor.GetProperty().SetOpacity(opacity)
            self._request_render()

    def set_control_mesh_visible(self, visible: bool) -> None:
        """Set control mesh visibility.

        Args:
            visible: True to show, False to hide.
        """
        if self._control_mesh_actor is not None:
            self._control_mesh_actor.SetVisibility(visible)
            self._request_render()

    def set_control_points_visible(self, visible: bool) -> None:
        """Set control points visibility.

        Args:
            visible: True to show, False to hide.
        """
        if self._control_points_actor is not None:
            self._control_points_actor.SetVisibility(visible)
            self._request_render()

    def set_nurbs_surface_visible(self, visible: bool) -> None:
        """Set NURBS surface visibility.

        Args:
            visible: True to show, False to hide.
        """
        if self._nurbs_surface_actor is not None:
            self._nurbs_surface_actor.SetVisibility(visible)
            self._request_render()

    def cleanup(self) -> None:
        """Remove all visualization actors."""
        self._remove_actor(self._control_mesh_actor)
        self._remove_actor(self._control_points_actor)
        self._remove_actor(self._nurbs_surface_actor)

        self._control_mesh_actor = None
        self._control_points_actor = None
        self._nurbs_surface_actor = None

        self._request_render()
        logger.debug("Visualization cleaned up")

    def _remove_actor(self, actor) -> None:
        """Remove an actor from the renderer."""
        if actor is None:
            return

        renderer = self._get_3d_renderer()
        if renderer is not None:
            renderer.RemoveActor(actor)

    def _request_render(self) -> None:
        """Request a render update."""
        try:
            import slicer

            layout_manager = slicer.app.layoutManager()
            three_d_widget = layout_manager.threeDWidget(0)
            if three_d_widget is not None:
                three_d_view = three_d_widget.threeDView()
                if three_d_view is not None:
                    three_d_view.forceRender()
        except Exception:
            pass

    def _create_hex_wireframe_polydata(self, hex_mesh: HexMesh):
        """Create VTK polydata for hexahedral wireframe.

        Args:
            hex_mesh: The hexahedral control mesh.

        Returns:
            vtkPolyData with wireframe lines.
        """
        import vtk

        nu, nv, nw = hex_mesh.num_u, hex_mesh.num_v, hex_mesh.num_w
        control_points = hex_mesh.control_points

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        # Add all control points
        point_ids = {}
        for i in range(nu):
            for j in range(nv):
                for k in range(nw):
                    pt = control_points[i, j, k, :]
                    pid = points.InsertNextPoint(pt[0], pt[1], pt[2])
                    point_ids[(i, j, k)] = pid

        # Add lines along each direction
        # U direction
        for j in range(nv):
            for k in range(nw):
                for i in range(nu - 1):
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, point_ids[(i, j, k)])
                    line.GetPointIds().SetId(1, point_ids[(i + 1, j, k)])
                    lines.InsertNextCell(line)

        # V direction
        for i in range(nu):
            for k in range(nw):
                for j in range(nv - 1):
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, point_ids[(i, j, k)])
                    line.GetPointIds().SetId(1, point_ids[(i, j + 1, k)])
                    lines.InsertNextCell(line)

        # W direction
        for i in range(nu):
            for j in range(nv):
                for k in range(nw - 1):
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, point_ids[(i, j, k)])
                    line.GetPointIds().SetId(1, point_ids[(i, j, k + 1)])
                    lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        return polydata

    def _create_nurbs_surface_polydata(
        self,
        nurbs_volume: NurbsVolume,
        resolution: int,
    ):
        """Create VTK polydata for NURBS boundary surface.

        Args:
            nurbs_volume: The NURBS volume.
            resolution: Surface sampling resolution.

        Returns:
            vtkPolyData with triangulated surface.
        """
        import vtk

        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        if nurbs_volume.geomdl_volume is None:
            # Fallback: create surface from control point hull
            return self._create_control_hull_polydata(nurbs_volume)

        # Sample each face of the parametric cube
        params = np.linspace(0, 1, resolution)

        # Helper to add a quad face
        def add_face(sample_func):
            """Sample a face and add triangles."""
            face_points = []
            for u in params:
                row = []
                for v in params:
                    pt = sample_func(u, v)
                    row.append(pt)
                face_points.append(row)

            # Add points and create triangles
            start_idx = points.GetNumberOfPoints()
            for row in face_points:
                for pt in row:
                    points.InsertNextPoint(pt[0], pt[1], pt[2])

            # Create triangles
            n = resolution
            for i in range(n - 1):
                for j in range(n - 1):
                    idx = start_idx + i * n + j
                    # First triangle
                    tri1 = vtk.vtkTriangle()
                    tri1.GetPointIds().SetId(0, idx)
                    tri1.GetPointIds().SetId(1, idx + 1)
                    tri1.GetPointIds().SetId(2, idx + n)
                    polys.InsertNextCell(tri1)
                    # Second triangle
                    tri2 = vtk.vtkTriangle()
                    tri2.GetPointIds().SetId(0, idx + 1)
                    tri2.GetPointIds().SetId(1, idx + n + 1)
                    tri2.GetPointIds().SetId(2, idx + n)
                    polys.InsertNextCell(tri2)

        vol = nurbs_volume.geomdl_volume

        # Face u=0
        add_face(lambda u, v: np.array(vol.evaluate_single((0, u, v))))

        # Face u=1
        add_face(lambda u, v: np.array(vol.evaluate_single((1, u, v))))

        # Face v=0
        add_face(lambda u, v: np.array(vol.evaluate_single((u, 0, v))))

        # Face v=1
        add_face(lambda u, v: np.array(vol.evaluate_single((u, 1, v))))

        # Face w=0
        add_face(lambda u, v: np.array(vol.evaluate_single((u, v, 0))))

        # Face w=1
        add_face(lambda u, v: np.array(vol.evaluate_single((u, v, 1))))

        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        # Compute normals for better rendering
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ComputePointNormalsOn()
        normals_filter.Update()

        return normals_filter.GetOutput()

    def _create_control_hull_polydata(self, nurbs_volume: NurbsVolume):
        """Create surface from control point convex hull (fallback).

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

        # Compute convex hull
        hull = vtk.vtkDelaunay3D()
        hull.SetInputData(polydata)
        hull.Update()

        # Extract surface
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputConnection(hull.GetOutputPort())
        surface.Update()

        return surface.GetOutput()
