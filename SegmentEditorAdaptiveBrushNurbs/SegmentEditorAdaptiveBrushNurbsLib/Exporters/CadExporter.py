"""CAD format export for NURBS volumes.

Exports volumetric NURBS to CAD-compatible solid formats:
- IGES (Initial Graphics Exchange Specification)
- STEP AP214 (Standard for the Exchange of Product Data)

Requires optional pythonocc-core dependency (conda install).

References:
- pythonocc-core: https://github.com/tpaviot/pythonocc-core
- IGES standard: https://en.wikipedia.org/wiki/IGES
- STEP standard: https://en.wikipedia.org/wiki/ISO_10303
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..NurbsVolumeBuilder import MultiPatchNurbsVolume, NurbsVolume

logger = logging.getLogger(__name__)

# Check for optional pythonocc dependency
HAS_PYTHONOCC = importlib.util.find_spec("OCC") is not None


class CadExporter:
    """Export NURBS volumes to CAD formats (IGES, STEP).

    Converts volumetric NURBS representations to solid CAD formats
    compatible with CAD software like FreeCAD, SolidWorks, etc.

    Note: Requires pythonocc-core which must be installed via conda:
        conda install -c conda-forge pythonocc-core

    Example:
        exporter = CadExporter()
        if exporter.is_available():
            exporter.export_iges(nurbs_volume, "output.igs")
            exporter.export_step(nurbs_volume, "output.step")
    """

    def __init__(self):
        """Initialize the CAD exporter."""
        self._occ_available: bool | None = None

    def is_available(self) -> bool:
        """Check if pythonocc-core is available.

        Returns:
            True if pythonocc can be imported.
        """
        if self._occ_available is not None:
            return self._occ_available

        self._occ_available = HAS_PYTHONOCC

        if not self._occ_available:
            logger.warning(
                "pythonocc-core not available. Install with: "
                "conda install -c conda-forge pythonocc-core"
            )

        return self._occ_available

    def export_iges(
        self,
        nurbs_volume: NurbsVolume | MultiPatchNurbsVolume,
        output_path: str | Path,
    ) -> bool:
        """Export NURBS volume to IGES format.

        Args:
            nurbs_volume: The NURBS volume or multi-patch volume to export.
            output_path: Path for the output IGES file.

        Returns:
            True if export was successful.

        Raises:
            RuntimeError: If pythonocc-core is not available.
        """
        if not self.is_available():
            raise RuntimeError(
                "pythonocc-core is required for IGES export. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        output_path = Path(output_path)
        logger.info(f"Exporting NURBS volume to IGES: {output_path}")

        from OCC.Core.IGESControl import IGESControl_Writer

        writer = IGESControl_Writer()

        # Get shapes from NURBS volume
        shapes = self._nurbs_to_occ_shapes(nurbs_volume)

        for shape in shapes:
            writer.AddShape(shape)

        # Write to file
        success = writer.Write(str(output_path))

        if success:
            logger.info(f"Successfully exported to {output_path}")
        else:
            logger.error(f"Failed to export to {output_path}")

        return bool(success)

    def export_step(
        self,
        nurbs_volume: NurbsVolume | MultiPatchNurbsVolume,
        output_path: str | Path,
        application_protocol: str = "AP214",
    ) -> bool:
        """Export NURBS volume to STEP format.

        Args:
            nurbs_volume: The NURBS volume or multi-patch volume to export.
            output_path: Path for the output STEP file.
            application_protocol: STEP application protocol (default AP214).

        Returns:
            True if export was successful.

        Raises:
            RuntimeError: If pythonocc-core is not available.
        """
        if not self.is_available():
            raise RuntimeError(
                "pythonocc-core is required for STEP export. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        output_path = Path(output_path)
        logger.info(f"Exporting NURBS volume to STEP ({application_protocol}): {output_path}")

        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.Interface import Interface_Static_SetCVal
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

        # Set application protocol
        Interface_Static_SetCVal("write.step.schema", application_protocol)

        writer = STEPControl_Writer()

        # Get shapes from NURBS volume
        shapes = self._nurbs_to_occ_shapes(nurbs_volume)

        for shape in shapes:
            writer.Transfer(shape, STEPControl_AsIs)

        # Write to file
        status = writer.Write(str(output_path))

        success = status == IFSelect_RetDone

        if success:
            logger.info(f"Successfully exported to {output_path}")
        else:
            logger.error(f"Failed to export to {output_path}")

        return success

    def _nurbs_to_occ_shapes(
        self,
        nurbs_volume: NurbsVolume | MultiPatchNurbsVolume,
    ) -> list[Any]:
        """Convert NURBS volume to OpenCASCADE shapes.

        Creates a solid representation by building 6 boundary surfaces
        and creating a shell from them.

        Args:
            nurbs_volume: The NURBS volume to convert.

        Returns:
            List of OCC shapes.
        """
        from ..NurbsVolumeBuilder import MultiPatchNurbsVolume

        if isinstance(nurbs_volume, MultiPatchNurbsVolume):
            shapes = []
            for patch in nurbs_volume.patches:
                shapes.extend(self._single_nurbs_to_occ_shapes(patch))
            return shapes
        else:
            return self._single_nurbs_to_occ_shapes(nurbs_volume)

    def _single_nurbs_to_occ_shapes(
        self,
        nurbs_volume: NurbsVolume,
    ) -> list[Any]:
        """Convert a single NURBS volume to OpenCASCADE shapes.

        Since OpenCASCADE doesn't natively support volumetric NURBS,
        we create a solid from 6 boundary B-spline surfaces.

        Args:
            nurbs_volume: The NURBS volume to convert.

        Returns:
            List containing the solid shape.
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing

        # Create 6 boundary surfaces
        faces = self._create_boundary_faces(nurbs_volume)

        if not faces:
            logger.warning("No boundary faces created")
            return []

        # Sew faces into a shell
        sewing = BRepOffsetAPI_Sewing()
        for face in faces:
            sewing.Add(face)

        sewing.Perform()
        sewn_shape = sewing.SewedShape()

        # Try to create a solid from the shell
        try:
            solid_maker = BRepBuilderAPI_MakeSolid()
            solid_maker.Add(sewn_shape)

            if solid_maker.IsDone():
                return [solid_maker.Solid()]
            else:
                # Fall back to shell if solid creation fails
                logger.warning("Could not create solid, returning shell")
                return [sewn_shape]
        except Exception as e:
            logger.warning(f"Solid creation failed: {e}, returning shell")
            return [sewn_shape]

    def _create_boundary_faces(
        self,
        nurbs_volume: NurbsVolume,
    ) -> list[Any]:
        """Create 6 boundary B-spline faces from NURBS volume.

        Creates faces for u=0, u=1, v=0, v=1, w=0, w=1.

        Args:
            nurbs_volume: The NURBS volume.

        Returns:
            List of OCC Face objects.
        """
        faces = []
        nu, nv, nw = nurbs_volume.size
        control_points = nurbs_volume.control_points.reshape(nu, nv, nw, 3)
        weights = nurbs_volume.weights.reshape(nu, nv, nw)
        degree_u, degree_v, degree_w = nurbs_volume.degrees
        knot_u, knot_v, knot_w = nurbs_volume.knot_vectors

        # Create face for each parametric boundary
        face_configs = [
            # (fixed_axis, fixed_value, axis1_size, axis2_size, degree1, degree2, knots1, knots2)
            ("u", 0, nv, nw, degree_v, degree_w, knot_v, knot_w),  # u=0
            ("u", nu - 1, nv, nw, degree_v, degree_w, knot_v, knot_w),  # u=1
            ("v", 0, nu, nw, degree_u, degree_w, knot_u, knot_w),  # v=0
            ("v", nv - 1, nu, nw, degree_u, degree_w, knot_u, knot_w),  # v=1
            ("w", 0, nu, nv, degree_u, degree_v, knot_u, knot_v),  # w=0
            ("w", nw - 1, nu, nv, degree_u, degree_v, knot_u, knot_v),  # w=1
        ]

        for fixed_axis, fixed_idx, n1, n2, deg1, deg2, knots1, knots2 in face_configs:
            try:
                face = self._create_bspline_face(
                    control_points,
                    weights,
                    fixed_axis,
                    fixed_idx,
                    n1,
                    n2,
                    deg1,
                    deg2,
                    knots1,
                    knots2,
                )
                if face is not None:
                    faces.append(face)
            except Exception as e:
                logger.warning(f"Failed to create face {fixed_axis}={fixed_idx}: {e}")

        return faces

    def _create_bspline_face(
        self,
        control_points: np.ndarray,
        weights: np.ndarray,
        fixed_axis: str,
        fixed_idx: int,
        n1: int,
        n2: int,
        deg1: int,
        deg2: int,
        knots1: np.ndarray,
        knots2: np.ndarray,
    ) -> Any | None:
        """Create a B-spline surface for one boundary face.

        Args:
            control_points: 4D control points (nu, nv, nw, 3).
            weights: 3D weights (nu, nv, nw).
            fixed_axis: Which axis is fixed ("u", "v", or "w").
            fixed_idx: Index along fixed axis.
            n1: Number of control points in first direction.
            n2: Number of control points in second direction.
            deg1: Degree in first direction.
            deg2: Degree in second direction.
            knots1: Knot vector for first direction.
            knots2: Knot vector for second direction.

        Returns:
            OCC Face object or None if creation fails.
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.Geom import Geom_BSplineSurface
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TColgp import TColgp_Array2OfPnt
        from OCC.Core.TColStd import (
            TColStd_Array1OfInteger,
            TColStd_Array1OfReal,
            TColStd_Array2OfReal,
        )

        # Extract 2D slice of control points and weights
        if fixed_axis == "u":
            pts_2d = control_points[fixed_idx, :, :, :]  # (nv, nw, 3)
            wts_2d = weights[fixed_idx, :, :]  # (nv, nw)
        elif fixed_axis == "v":
            pts_2d = control_points[:, fixed_idx, :, :]  # (nu, nw, 3)
            wts_2d = weights[:, fixed_idx, :]  # (nu, nw)
        else:  # w
            pts_2d = control_points[:, :, fixed_idx, :]  # (nu, nv, 3)
            wts_2d = weights[:, :, fixed_idx]  # (nu, nv)

        # Create OCC arrays for control points (1-indexed)
        poles = TColgp_Array2OfPnt(1, n1, 1, n2)
        occ_weights = TColStd_Array2OfReal(1, n1, 1, n2)

        for i in range(n1):
            for j in range(n2):
                pt = pts_2d[i, j, :]
                poles.SetValue(i + 1, j + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))
                occ_weights.SetValue(i + 1, j + 1, float(wts_2d[i, j]))

        # Convert knot vectors to OCC format
        # Extract unique knots and multiplicities
        unique_knots1, multiplicities1 = self._knots_to_occ_format(knots1)
        unique_knots2, multiplicities2 = self._knots_to_occ_format(knots2)

        # Create OCC arrays for knots
        occ_knots1 = TColStd_Array1OfReal(1, len(unique_knots1))
        occ_mults1 = TColStd_Array1OfInteger(1, len(multiplicities1))
        for i, (k, m) in enumerate(zip(unique_knots1, multiplicities1)):
            occ_knots1.SetValue(i + 1, k)
            occ_mults1.SetValue(i + 1, m)

        occ_knots2 = TColStd_Array1OfReal(1, len(unique_knots2))
        occ_mults2 = TColStd_Array1OfInteger(1, len(multiplicities2))
        for i, (k, m) in enumerate(zip(unique_knots2, multiplicities2)):
            occ_knots2.SetValue(i + 1, k)
            occ_mults2.SetValue(i + 1, m)

        # Create B-spline surface
        try:
            bspline_surface = Geom_BSplineSurface(
                poles,
                occ_weights,
                occ_knots1,
                occ_knots2,
                occ_mults1,
                occ_mults2,
                deg1,
                deg2,
            )

            # Create face from surface
            face_maker = BRepBuilderAPI_MakeFace(bspline_surface, 1e-6)
            if face_maker.IsDone():
                return face_maker.Face()
            else:
                logger.warning(f"Face creation failed for {fixed_axis}={fixed_idx}")
                return None
        except Exception as e:
            logger.warning(f"B-spline surface creation failed: {e}")
            return None

    def _knots_to_occ_format(
        self,
        knots: np.ndarray,
    ) -> tuple[list[float], list[int]]:
        """Convert flat knot vector to OCC format with multiplicities.

        OCC expects unique knots and their multiplicities separately.

        Args:
            knots: Flat knot vector with repeated values.

        Returns:
            Tuple of (unique_knots, multiplicities).
        """
        unique_knots = []
        multiplicities = []

        prev_knot = None
        count = 0

        for knot in knots:
            if prev_knot is None or abs(knot - prev_knot) > 1e-10:
                if prev_knot is not None:
                    unique_knots.append(prev_knot)
                    multiplicities.append(count)
                prev_knot = knot
                count = 1
            else:
                count += 1

        # Don't forget the last knot
        if prev_knot is not None:
            unique_knots.append(prev_knot)
            multiplicities.append(count)

        return unique_knots, multiplicities
