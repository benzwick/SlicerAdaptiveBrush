"""Triangulated mesh export for NURBS volumes.

Exports NURBS volume boundary surfaces to standard mesh formats:
- STL: Binary or ASCII stereolithography format
- OBJ: Wavefront OBJ format
- VTK: VTK unstructured grid format

The NURBS surface is sampled and triangulated before export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..NurbsVolumeBuilder import NurbsVolume

logger = logging.getLogger(__name__)


class MeshExporter:
    """Export NURBS volumes to triangulated mesh formats.

    Example:
        exporter = MeshExporter()
        exporter.export_stl(nurbs_volume, Path("output.stl"))
        exporter.export_obj(nurbs_volume, Path("output.obj"))
        exporter.export_vtk(nurbs_volume, Path("output.vtk"))
    """

    def __init__(self, resolution: int = 30):
        """Initialize mesh exporter.

        Args:
            resolution: Surface sampling resolution.
        """
        self.resolution = resolution

    def export_stl(
        self,
        nurbs_volume: NurbsVolume,
        output_path: Path,
        binary: bool = True,
    ) -> None:
        """Export NURBS surface to STL format.

        Args:
            nurbs_volume: The NURBS volume.
            output_path: Path to write .stl file.
            binary: True for binary STL, False for ASCII.
        """
        import vtk

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get triangulated surface
        polydata = self._get_triangulated_surface(nurbs_volume)

        # Write STL
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(polydata)

        if binary:
            writer.SetFileTypeToBinary()
        else:
            writer.SetFileTypeToASCII()

        writer.Write()

        logger.info(f"Exported STL: {output_path}")

    def export_obj(
        self,
        nurbs_volume: NurbsVolume,
        output_path: Path,
    ) -> None:
        """Export NURBS surface to Wavefront OBJ format.

        Args:
            nurbs_volume: The NURBS volume.
            output_path: Path to write .obj file.
        """
        import vtk

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get triangulated surface
        polydata = self._get_triangulated_surface(nurbs_volume)

        # Write OBJ
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(polydata)
        writer.Write()

        logger.info(f"Exported OBJ: {output_path}")

    def export_vtk(
        self,
        nurbs_volume: NurbsVolume,
        output_path: Path,
        ascii: bool = False,
    ) -> None:
        """Export NURBS surface to VTK polydata format.

        Args:
            nurbs_volume: The NURBS volume.
            output_path: Path to write .vtk file.
            ascii: True for ASCII VTK, False for binary.
        """
        import vtk

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get triangulated surface
        polydata = self._get_triangulated_surface(nurbs_volume)

        # Write VTK
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(polydata)

        if ascii:
            writer.SetFileTypeToASCII()
        else:
            writer.SetFileTypeToBinary()

        writer.Write()

        logger.info(f"Exported VTK: {output_path}")

    def export_vtk_unstructured(
        self,
        nurbs_volume: NurbsVolume,
        output_path: Path,
    ) -> None:
        """Export NURBS as VTK unstructured grid (hexahedral elements).

        This exports the control mesh as hexahedral elements,
        useful for visualization in ParaView.

        Args:
            nurbs_volume: The NURBS volume.
            output_path: Path to write .vtu file.
        """
        import vtk

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create unstructured grid from control mesh
        grid = self._create_hex_grid(nurbs_volume)

        # Write VTU
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(grid)
        writer.Write()

        logger.info(f"Exported VTK unstructured grid: {output_path}")

    def _get_triangulated_surface(self, nurbs_volume: NurbsVolume):
        """Get triangulated surface from NURBS volume.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            vtkPolyData with triangulated surface.
        """
        from ..Visualization import NurbsVisualization

        # Use visualization module to create surface polydata
        viz = NurbsVisualization()
        return viz._create_nurbs_surface_polydata(nurbs_volume, self.resolution)

    def _create_hex_grid(self, nurbs_volume: NurbsVolume):
        """Create VTK unstructured grid from control mesh.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            vtkUnstructuredGrid with hexahedral cells.
        """
        import vtk

        nu, nv, nw = nurbs_volume.size
        control_points = nurbs_volume.control_points.reshape(nu, nv, nw, 3)

        # Create points
        points = vtk.vtkPoints()
        point_ids = {}

        for i in range(nu):
            for j in range(nv):
                for k in range(nw):
                    pt = control_points[i, j, k, :]
                    pid = points.InsertNextPoint(pt[0], pt[1], pt[2])
                    point_ids[(i, j, k)] = pid

        # Create hexahedral cells
        # Each cell connects 8 adjacent control points
        cells = vtk.vtkCellArray()

        for i in range(nu - 1):
            for j in range(nv - 1):
                for k in range(nw - 1):
                    hex_cell = vtk.vtkHexahedron()
                    hex_cell.GetPointIds().SetId(0, point_ids[(i, j, k)])
                    hex_cell.GetPointIds().SetId(1, point_ids[(i + 1, j, k)])
                    hex_cell.GetPointIds().SetId(2, point_ids[(i + 1, j + 1, k)])
                    hex_cell.GetPointIds().SetId(3, point_ids[(i, j + 1, k)])
                    hex_cell.GetPointIds().SetId(4, point_ids[(i, j, k + 1)])
                    hex_cell.GetPointIds().SetId(5, point_ids[(i + 1, j, k + 1)])
                    hex_cell.GetPointIds().SetId(6, point_ids[(i + 1, j + 1, k + 1)])
                    hex_cell.GetPointIds().SetId(7, point_ids[(i, j + 1, k + 1)])
                    cells.InsertNextCell(hex_cell)

        # Create unstructured grid
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.SetCells(vtk.VTK_HEXAHEDRON, cells)

        return grid
