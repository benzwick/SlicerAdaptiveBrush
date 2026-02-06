"""Bifurcation and trifurcation templates for branching NURBS volumes.

Provides hexahedral mesh templates for branch junctions in vascular and
airway structures. Based on the approach from:

"Patient-Specific Vascular NURBS Modeling for Isogeometric Analysis"
(Zhang et al., PMC2839408)

Key concepts:
- Bifurcation: Parent branch splits into 2 children -> 3 map-meshable regions
- Trifurcation: Parent branch splits into 3 children -> 5 template cases
- Templates define hex mesh topology at junctions
- Actual geometry is adapted to specific anatomy via control point placement

References:
- PMC2839408: https://pmc.ncbi.nlm.nih.gov/articles/PMC2839408/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BifurcationType(Enum):
    """Types of bifurcation based on angle and symmetry."""

    SYMMETRIC = "symmetric"  # Equal angles, similar diameters
    ASYMMETRIC = "asymmetric"  # Unequal angles or diameters
    SIDE_BRANCH = "side_branch"  # Small branch off main vessel


class TrifurcationType(Enum):
    """Types of trifurcation configurations."""

    TYPE_A = "type_a"  # Symmetric 3-way split
    TYPE_B = "type_b"  # Two branches close, one opposite
    TYPE_C = "type_c"  # Sequential bifurcations
    TYPE_D = "type_d"  # Star pattern
    TYPE_E = "type_e"  # Asymmetric star


@dataclass
class BranchConnection:
    """Connection information for a branch at a junction.

    Attributes:
        branch_id: Identifier for this branch.
        direction: Unit vector pointing away from junction center.
        radius: Radius of the branch at the junction.
        is_parent: True if this is the parent (inlet) branch.
        patch_face: Which face of the junction patch connects to this branch.
    """

    branch_id: int
    direction: np.ndarray
    radius: float
    is_parent: bool = False
    patch_face: str = ""  # "u0", "u1", "v0", "v1", "w0", "w1"


@dataclass
class JunctionPatch:
    """A hexahedral patch within a junction template.

    Attributes:
        patch_id: Identifier for this patch within the junction.
        control_points: Control points for this patch (nu, nv, nw, 3).
        connected_branches: Branch IDs connected to this patch.
        neighbor_patches: Adjacent patch IDs and their shared faces.
    """

    patch_id: int
    control_points: np.ndarray
    connected_branches: list[int] = field(default_factory=list)
    neighbor_patches: dict[str, int] = field(default_factory=dict)  # face -> patch_id


@dataclass
class BifurcationTemplate:
    """Template for a bifurcation junction.

    A bifurcation has one parent branch splitting into two child branches.
    The junction region is decomposed into 3 hexahedral patches that
    maintain G1 continuity.

    Attributes:
        center: Center point of the bifurcation.
        parent: Parent branch connection.
        child1: First child branch connection.
        child2: Second child branch connection.
        patches: List of JunctionPatch objects defining the hex mesh.
        bifurcation_type: Classification of the bifurcation geometry.
    """

    center: np.ndarray
    parent: BranchConnection
    child1: BranchConnection
    child2: BranchConnection
    patches: list[JunctionPatch] = field(default_factory=list)
    bifurcation_type: BifurcationType = BifurcationType.SYMMETRIC

    @property
    def num_patches(self) -> int:
        """Number of hex patches in this template."""
        return len(self.patches)

    @property
    def bifurcation_angle(self) -> float:
        """Angle between child branches in radians."""
        dot = np.dot(self.child1.direction, self.child2.direction)
        return np.arccos(np.clip(dot, -1.0, 1.0))


@dataclass
class TrifurcationTemplate:
    """Template for a trifurcation junction.

    A trifurcation has one parent branch splitting into three child branches.
    More complex than bifurcation, with 5 possible template configurations.

    Attributes:
        center: Center point of the trifurcation.
        parent: Parent branch connection.
        children: List of three child branch connections.
        patches: List of JunctionPatch objects.
        trifurcation_type: Classification of the configuration.
    """

    center: np.ndarray
    parent: BranchConnection
    children: list[BranchConnection] = field(default_factory=list)
    patches: list[JunctionPatch] = field(default_factory=list)
    trifurcation_type: TrifurcationType = TrifurcationType.TYPE_A

    @property
    def num_patches(self) -> int:
        """Number of hex patches in this template."""
        return len(self.patches)


class BranchTemplates:
    """Factory for creating branch junction templates.

    Creates hexahedral mesh templates for bifurcations and trifurcations
    that can be used to generate multi-patch NURBS volumes for branching
    structures.

    Example:
        templates = BranchTemplates()
        bif_template = templates.create_bifurcation(
            center=np.array([0, 0, 0]),
            parent_dir=np.array([0, 0, -1]),
            child1_dir=np.array([0.5, 0, 0.866]),
            child2_dir=np.array([-0.5, 0, 0.866]),
            parent_radius=5.0,
            child1_radius=3.5,
            child2_radius=3.5,
        )
    """

    # Default resolution for junction patches
    DEFAULT_RESOLUTION = 4  # 4x4x4 control points per patch

    def __init__(self, resolution: int | None = None):
        """Initialize branch templates factory.

        Args:
            resolution: Control points per direction in each patch.
        """
        self.resolution = resolution or self.DEFAULT_RESOLUTION

    def create_bifurcation(
        self,
        center: np.ndarray,
        parent_dir: np.ndarray,
        child1_dir: np.ndarray,
        child2_dir: np.ndarray,
        parent_radius: float,
        child1_radius: float,
        child2_radius: float,
        resolution: int | None = None,
    ) -> BifurcationTemplate:
        """Create a bifurcation junction template.

        The bifurcation is decomposed into 3 hexahedral patches:
        - Patch 0: Connects parent to junction center
        - Patch 1: Connects junction center to child 1
        - Patch 2: Connects junction center to child 2

        Args:
            center: Center point of bifurcation.
            parent_dir: Direction vector of parent branch (pointing away).
            child1_dir: Direction vector of first child branch.
            child2_dir: Direction vector of second child branch.
            parent_radius: Radius of parent branch.
            child1_radius: Radius of first child.
            child2_radius: Radius of second child.
            resolution: Override default resolution.

        Returns:
            BifurcationTemplate with 3 hex patches.
        """
        if resolution is None:
            resolution = self.resolution

        # Normalize directions
        parent_dir = parent_dir / np.linalg.norm(parent_dir)
        child1_dir = child1_dir / np.linalg.norm(child1_dir)
        child2_dir = child2_dir / np.linalg.norm(child2_dir)

        # Classify bifurcation type
        bif_type = self._classify_bifurcation(
            parent_dir, child1_dir, child2_dir, parent_radius, child1_radius, child2_radius
        )

        # Create branch connections
        parent = BranchConnection(
            branch_id=0,
            direction=parent_dir,
            radius=parent_radius,
            is_parent=True,
            patch_face="w0",
        )
        child1 = BranchConnection(
            branch_id=1,
            direction=child1_dir,
            radius=child1_radius,
            is_parent=False,
            patch_face="w1",
        )
        child2 = BranchConnection(
            branch_id=2,
            direction=child2_dir,
            radius=child2_radius,
            is_parent=False,
            patch_face="w1",
        )

        # Generate the 3 patches
        patches = self._generate_bifurcation_patches(center, parent, child1, child2, resolution)

        template = BifurcationTemplate(
            center=center,
            parent=parent,
            child1=child1,
            child2=child2,
            patches=patches,
            bifurcation_type=bif_type,
        )

        logger.info(
            f"Created bifurcation template: type={bif_type.value}, "
            f"angle={np.degrees(template.bifurcation_angle):.1f}°, "
            f"patches={len(patches)}"
        )

        return template

    def create_trifurcation(
        self,
        center: np.ndarray,
        parent_dir: np.ndarray,
        child_dirs: list[np.ndarray],
        parent_radius: float,
        child_radii: list[float],
        resolution: int | None = None,
    ) -> TrifurcationTemplate:
        """Create a trifurcation junction template.

        Trifurcations are decomposed into 5 patches based on the
        spatial arrangement of the three child branches.

        Args:
            center: Center point of trifurcation.
            parent_dir: Direction vector of parent branch.
            child_dirs: Direction vectors of three child branches.
            parent_radius: Radius of parent branch.
            child_radii: Radii of three child branches.
            resolution: Override default resolution.

        Returns:
            TrifurcationTemplate with appropriate patches.
        """
        if resolution is None:
            resolution = self.resolution

        if len(child_dirs) != 3 or len(child_radii) != 3:
            raise ValueError("Trifurcation requires exactly 3 child branches")

        # Normalize directions
        parent_dir = parent_dir / np.linalg.norm(parent_dir)
        child_dirs = [d / np.linalg.norm(d) for d in child_dirs]

        # Classify trifurcation type
        trif_type = self._classify_trifurcation(parent_dir, child_dirs)

        # Create branch connections
        parent = BranchConnection(
            branch_id=0,
            direction=parent_dir,
            radius=parent_radius,
            is_parent=True,
        )

        children = [
            BranchConnection(
                branch_id=i + 1,
                direction=child_dirs[i],
                radius=child_radii[i],
                is_parent=False,
            )
            for i in range(3)
        ]

        # Generate patches based on trifurcation type
        patches = self._generate_trifurcation_patches(
            center, parent, children, trif_type, resolution
        )

        template = TrifurcationTemplate(
            center=center,
            parent=parent,
            children=children,
            patches=patches,
            trifurcation_type=trif_type,
        )

        logger.info(
            f"Created trifurcation template: type={trif_type.value}, patches={len(patches)}"
        )

        return template

    def _classify_bifurcation(
        self,
        parent_dir: np.ndarray,
        child1_dir: np.ndarray,
        child2_dir: np.ndarray,
        parent_radius: float,
        child1_radius: float,
        child2_radius: float,
    ) -> BifurcationType:
        """Classify bifurcation type based on geometry.

        Args:
            parent_dir: Parent direction (normalized).
            child1_dir: First child direction.
            child2_dir: Second child direction.
            parent_radius: Parent radius.
            child1_radius: First child radius.
            child2_radius: Second child radius.

        Returns:
            BifurcationType classification.
        """
        # Compute angles
        angle1 = np.arccos(np.clip(-np.dot(parent_dir, child1_dir), -1.0, 1.0))
        angle2 = np.arccos(np.clip(-np.dot(parent_dir, child2_dir), -1.0, 1.0))

        # Compute radius ratios
        ratio1 = child1_radius / parent_radius
        ratio2 = child2_radius / parent_radius

        # Check for side branch (small branch off main vessel)
        if ratio1 < 0.5 or ratio2 < 0.5:
            return BifurcationType.SIDE_BRANCH

        # Check for symmetry
        angle_diff = abs(angle1 - angle2)
        radius_diff = abs(ratio1 - ratio2)

        if angle_diff < 0.2 and radius_diff < 0.2:  # ~11 degrees tolerance
            return BifurcationType.SYMMETRIC
        else:
            return BifurcationType.ASYMMETRIC

    def _classify_trifurcation(
        self,
        parent_dir: np.ndarray,
        child_dirs: list[np.ndarray],
    ) -> TrifurcationType:
        """Classify trifurcation type based on child branch arrangement.

        Args:
            parent_dir: Parent direction (normalized).
            child_dirs: List of 3 child directions.

        Returns:
            TrifurcationType classification.
        """
        # Compute angles between children
        angles = []
        for i in range(3):
            for j in range(i + 1, 3):
                dot = np.dot(child_dirs[i], child_dirs[j])
                angles.append(np.arccos(np.clip(dot, -1.0, 1.0)))

        # Sort angles
        angles.sort()

        # Classify based on angle distribution
        # Type A: All angles similar (symmetric star)
        if max(angles) - min(angles) < 0.3:  # ~17 degrees
            return TrifurcationType.TYPE_A

        # Type B: Two small angles, one large (two branches close)
        if angles[0] < 0.5 and angles[2] > 2.0:
            return TrifurcationType.TYPE_B

        # Type C: Sequential pattern
        if angles[0] < 0.8 and angles[1] < 1.2:
            return TrifurcationType.TYPE_C

        # Default to Type D (general star pattern)
        return TrifurcationType.TYPE_D

    def _generate_bifurcation_patches(
        self,
        center: np.ndarray,
        parent: BranchConnection,
        child1: BranchConnection,
        child2: BranchConnection,
        resolution: int,
    ) -> list[JunctionPatch]:
        """Generate 3 hex patches for a bifurcation.

        The junction is decomposed into:
        - Patch 0: Parent inlet region
        - Patch 1: Child 1 outlet region
        - Patch 2: Child 2 outlet region

        All three patches meet at a central saddle region.

        Args:
            center: Junction center.
            parent: Parent branch connection.
            child1: First child connection.
            child2: Second child connection.
            resolution: Control points per direction.

        Returns:
            List of 3 JunctionPatch objects.
        """
        patches = []

        # Compute local coordinate frames
        # Parent frame
        parent_frame = self._compute_branch_frame(parent.direction)

        # Child frames
        child1_frame = self._compute_branch_frame(child1.direction)
        child2_frame = self._compute_branch_frame(child2.direction)

        # Junction length (distance from center to branch inlet/outlet)
        junction_length = max(parent.radius, child1.radius, child2.radius) * 1.5

        # Generate patch 0: Parent region
        patch0_points = self._generate_transition_patch(
            start_center=center - parent.direction * junction_length,
            end_center=center,
            start_radius=parent.radius,
            end_radius=parent.radius * 0.8,  # Slight narrowing at junction
            start_frame=parent_frame,
            end_frame=parent_frame,
            resolution=resolution,
        )
        patches.append(
            JunctionPatch(
                patch_id=0,
                control_points=patch0_points,
                connected_branches=[parent.branch_id],
                neighbor_patches={"w1": 1, "v1": 2},  # Connects to both children
            )
        )

        # Generate patch 1: Child 1 region
        patch1_points = self._generate_transition_patch(
            start_center=center,
            end_center=center + child1.direction * junction_length,
            start_radius=child1.radius * 0.9,
            end_radius=child1.radius,
            start_frame=child1_frame,
            end_frame=child1_frame,
            resolution=resolution,
        )
        patches.append(
            JunctionPatch(
                patch_id=1,
                control_points=patch1_points,
                connected_branches=[child1.branch_id],
                neighbor_patches={"w0": 0},
            )
        )

        # Generate patch 2: Child 2 region
        patch2_points = self._generate_transition_patch(
            start_center=center,
            end_center=center + child2.direction * junction_length,
            start_radius=child2.radius * 0.9,
            end_radius=child2.radius,
            start_frame=child2_frame,
            end_frame=child2_frame,
            resolution=resolution,
        )
        patches.append(
            JunctionPatch(
                patch_id=2,
                control_points=patch2_points,
                connected_branches=[child2.branch_id],
                neighbor_patches={"w0": 0},
            )
        )

        return patches

    def _generate_trifurcation_patches(
        self,
        center: np.ndarray,
        parent: BranchConnection,
        children: list[BranchConnection],
        trif_type: TrifurcationType,
        resolution: int,
    ) -> list[JunctionPatch]:
        """Generate hex patches for a trifurcation.

        Based on the trifurcation type, generates 4-5 patches.

        Args:
            center: Junction center.
            parent: Parent branch connection.
            children: List of 3 child connections.
            trif_type: Trifurcation classification.
            resolution: Control points per direction.

        Returns:
            List of JunctionPatch objects.
        """
        patches = []

        # Compute local coordinate frames
        parent_frame = self._compute_branch_frame(parent.direction)
        child_frames = [self._compute_branch_frame(c.direction) for c in children]

        # Junction length
        max_radius = max(parent.radius, max(c.radius for c in children))
        junction_length = max_radius * 1.5

        # Patch 0: Parent region
        patch0_points = self._generate_transition_patch(
            start_center=center - parent.direction * junction_length,
            end_center=center,
            start_radius=parent.radius,
            end_radius=parent.radius * 0.7,
            start_frame=parent_frame,
            end_frame=parent_frame,
            resolution=resolution,
        )
        patches.append(
            JunctionPatch(
                patch_id=0,
                control_points=patch0_points,
                connected_branches=[parent.branch_id],
                neighbor_patches={},
            )
        )

        # Patches 1-3: Child regions
        for i, (child, frame) in enumerate(zip(children, child_frames)):
            patch_points = self._generate_transition_patch(
                start_center=center,
                end_center=center + child.direction * junction_length,
                start_radius=child.radius * 0.8,
                end_radius=child.radius,
                start_frame=frame,
                end_frame=frame,
                resolution=resolution,
            )
            patches.append(
                JunctionPatch(
                    patch_id=i + 1,
                    control_points=patch_points,
                    connected_branches=[child.branch_id],
                    neighbor_patches={"w0": 0},
                )
            )

        # Patch 4: Central connector (for complex trifurcations)
        if trif_type in [TrifurcationType.TYPE_A, TrifurcationType.TYPE_D]:
            # Add central octahedral patch
            central_points = self._generate_central_patch(center, parent, children, resolution)
            patches.append(
                JunctionPatch(
                    patch_id=4,
                    control_points=central_points,
                    connected_branches=[],
                    neighbor_patches={"w0": 0, "u0": 1, "u1": 2, "v1": 3},
                )
            )

        return patches

    def _compute_branch_frame(
        self,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute orthonormal frame for a branch direction.

        Args:
            direction: Branch direction (unit vector).

        Returns:
            Tuple of (tangent, normal, binormal) vectors.
        """
        tangent = direction

        # Find a vector not parallel to tangent
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tangent, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])

        # Compute normal and binormal
        normal = np.cross(tangent, up)
        normal = normal / np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)

        return tangent, normal, binormal

    def _generate_transition_patch(
        self,
        start_center: np.ndarray,
        end_center: np.ndarray,
        start_radius: float,
        end_radius: float,
        start_frame: tuple[np.ndarray, np.ndarray, np.ndarray],
        end_frame: tuple[np.ndarray, np.ndarray, np.ndarray],
        resolution: int,
    ) -> np.ndarray:
        """Generate control points for a transition patch.

        Creates a cylindrical-like patch that transitions from
        start to end with radius interpolation.

        Args:
            start_center: Center at start of patch.
            end_center: Center at end of patch.
            start_radius: Radius at start.
            end_radius: Radius at end.
            start_frame: (tangent, normal, binormal) at start.
            end_frame: (tangent, normal, binormal) at end.
            resolution: Control points per direction.

        Returns:
            Control points array (resolution, resolution, resolution, 3).
        """
        control_points = np.zeros((resolution, resolution, resolution, 3))

        # Parametric coordinates
        u_params = np.linspace(0, 1, resolution)  # Circumferential
        v_params = np.linspace(0, 1, resolution)  # Radial
        w_params = np.linspace(0, 1, resolution)  # Axial

        for k, w in enumerate(w_params):
            # Interpolate center, radius, and frame along axis
            center = (1 - w) * start_center + w * end_center
            radius = (1 - w) * start_radius + w * end_radius

            # Interpolate frame (simple linear for now, could use SLERP)
            _, n_start, b_start = start_frame
            _, n_end, b_end = end_frame
            normal = (1 - w) * n_start + w * n_end
            normal = normal / np.linalg.norm(normal)
            binormal = (1 - w) * b_start + w * b_end
            binormal = binormal / np.linalg.norm(binormal)

            for i, u in enumerate(u_params):
                # Angle around circumference
                theta = 2 * np.pi * u

                for j, v in enumerate(v_params):
                    # Radial position (v=0 is center, v=1 is surface)
                    r = radius * v

                    # Compute point position
                    direction = np.cos(theta) * normal + np.sin(theta) * binormal
                    point = center + r * direction
                    control_points[i, j, k, :] = point

        return control_points

    def _generate_central_patch(
        self,
        center: np.ndarray,
        parent: BranchConnection,
        children: list[BranchConnection],
        resolution: int,
    ) -> np.ndarray:
        """Generate central connector patch for complex junctions.

        Creates a small patch at the junction center that helps
        maintain smooth transitions between all branches.

        Args:
            center: Junction center.
            parent: Parent branch.
            children: List of child branches.
            resolution: Control points per direction.

        Returns:
            Control points array.
        """
        # Central patch size based on smallest radius
        min_radius = min(parent.radius, min(c.radius for c in children))
        patch_size = min_radius * 0.5

        control_points = np.zeros((resolution, resolution, resolution, 3))

        # Simple cube-like patch at center
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    u = (i / (resolution - 1) - 0.5) * 2  # -1 to 1
                    v = (j / (resolution - 1) - 0.5) * 2
                    w = (k / (resolution - 1) - 0.5) * 2

                    point = center + patch_size * np.array([u, v, w])
                    control_points[i, j, k, :] = point

        return control_points

    def ensure_g1_continuity(
        self,
        patch1: JunctionPatch,
        patch2: JunctionPatch,
        shared_face: str,
    ) -> None:
        """Adjust control points to ensure G1 continuity at shared boundary.

        G1 continuity requires:
        1. Shared control points on boundary (C0)
        2. Coplanar control points on either side of boundary
        3. Matching tangent directions across boundary

        Args:
            patch1: First patch.
            patch2: Second patch.
            shared_face: Face of patch1 shared with patch2.

        Modifies patch control points in-place.
        """
        # Get boundary control points from both patches
        face1_points = self._get_face_control_points(patch1.control_points, shared_face)

        # Determine corresponding face on patch2
        opposite_face = self._get_opposite_face(shared_face)
        face2_points = self._get_face_control_points(patch2.control_points, opposite_face)

        # Average boundary points for C0 continuity
        avg_points = (face1_points + face2_points) / 2

        # Set boundary points on both patches
        self._set_face_control_points(patch1.control_points, shared_face, avg_points)
        self._set_face_control_points(patch2.control_points, opposite_face, avg_points)

        # Adjust adjacent rows for G1 continuity
        self._adjust_for_g1(patch1.control_points, patch2.control_points, shared_face)

    def _get_face_control_points(
        self,
        control_points: np.ndarray,
        face: str,
    ) -> np.ndarray:
        """Extract control points on a face.

        Args:
            control_points: Full control point array (nu, nv, nw, 3).
            face: Face identifier ("u0", "u1", "v0", "v1", "w0", "w1").

        Returns:
            Face control points array.
        """
        if face == "u0":
            return control_points[0, :, :, :].copy()
        elif face == "u1":
            return control_points[-1, :, :, :].copy()
        elif face == "v0":
            return control_points[:, 0, :, :].copy()
        elif face == "v1":
            return control_points[:, -1, :, :].copy()
        elif face == "w0":
            return control_points[:, :, 0, :].copy()
        elif face == "w1":
            return control_points[:, :, -1, :].copy()
        else:
            raise ValueError(f"Unknown face: {face}")

    def _set_face_control_points(
        self,
        control_points: np.ndarray,
        face: str,
        face_points: np.ndarray,
    ) -> None:
        """Set control points on a face.

        Args:
            control_points: Full control point array (modified in-place).
            face: Face identifier.
            face_points: New face control points.
        """
        if face == "u0":
            control_points[0, :, :, :] = face_points
        elif face == "u1":
            control_points[-1, :, :, :] = face_points
        elif face == "v0":
            control_points[:, 0, :, :] = face_points
        elif face == "v1":
            control_points[:, -1, :, :] = face_points
        elif face == "w0":
            control_points[:, :, 0, :] = face_points
        elif face == "w1":
            control_points[:, :, -1, :] = face_points

    def _get_opposite_face(self, face: str) -> str:
        """Get the opposite face identifier."""
        opposites = {
            "u0": "u1",
            "u1": "u0",
            "v0": "v1",
            "v1": "v0",
            "w0": "w1",
            "w1": "w0",
        }
        return opposites[face]

    def _adjust_for_g1(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        shared_face: str,
    ) -> None:
        """Adjust control points adjacent to boundary for G1 continuity.

        Ensures tangent vectors are aligned across the boundary.

        Args:
            points1: Patch 1 control points (modified in-place).
            points2: Patch 2 control points (modified in-place).
            shared_face: Face of patch1 shared with patch2.
        """
        # Get boundary and adjacent layers
        if shared_face in ["u0", "u1"]:
            idx = 0 if shared_face == "u0" else -1
            adj_idx = 1 if shared_face == "u0" else -2

            # Compute average tangent direction
            tangent1 = points1[adj_idx, :, :, :] - points1[idx, :, :, :]
            opp_idx = -1 if shared_face == "u0" else 0
            opp_adj = -2 if shared_face == "u0" else 1
            tangent2 = points2[opp_idx, :, :, :] - points2[opp_adj, :, :, :]

            # Average tangent magnitude
            avg_mag = (np.linalg.norm(tangent1, axis=-1) + np.linalg.norm(tangent2, axis=-1)) / 2

            # Align adjacent points
            boundary = points1[idx, :, :, :]
            direction = tangent1 / (np.linalg.norm(tangent1, axis=-1, keepdims=True) + 1e-10)
            points1[adj_idx, :, :, :] = boundary + direction * avg_mag[..., np.newaxis]

        # Similar logic for v and w directions would go here
        # Simplified for now
