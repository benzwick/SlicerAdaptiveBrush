"""SegmentEditorAdaptiveBrushNurbs library.

Provides volumetric NURBS generation from painted segmentations.

Core Classes:
    StructureDetector: Detect structure type (simple/tubular/branching)
    SkeletonExtractor: Extract centerlines from tubular segments (VMTK)
    BranchTemplates: Bifurcation/trifurcation hex mesh templates
    HexMeshGenerator: Generate hexahedral control meshes
    NurbsVolumeBuilder: Build NURBS volumes from control meshes
    ContainmentValidator: Ensure segment is contained within NURBS
    QualityMetrics: Compute fitting quality metrics

Export Classes:
    MfemExporter: Export to MFEM volumetric NURBS format
    MeshExporter: Export to STL/OBJ/VTK triangulated mesh
"""

from .BranchTemplates import (
    BifurcationTemplate,
    BifurcationType,
    BranchConnection,
    BranchTemplates,
    JunctionPatch,
    TrifurcationTemplate,
    TrifurcationType,
)
from .ContainmentValidator import ContainmentValidator
from .HexMeshGenerator import HexMesh, HexMeshGenerator
from .NurbsVolumeBuilder import MultiPatchNurbsVolume, NurbsVolume, NurbsVolumeBuilder
from .QualityMetrics import QualityMetrics
from .SkeletonExtractor import BranchPoint, Centerline, SkeletonExtractor
from .StructureDetector import StructureDetector
from .Visualization import NurbsVisualization

__all__ = [
    # Structure detection
    "StructureDetector",
    # Skeleton extraction
    "SkeletonExtractor",
    "Centerline",
    "BranchPoint",
    # Branch templates
    "BranchTemplates",
    "BifurcationTemplate",
    "TrifurcationTemplate",
    "BranchConnection",
    "JunctionPatch",
    "BifurcationType",
    "TrifurcationType",
    # Mesh generation
    "HexMesh",
    "HexMeshGenerator",
    # NURBS construction
    "NurbsVolume",
    "MultiPatchNurbsVolume",
    "NurbsVolumeBuilder",
    # Validation
    "ContainmentValidator",
    "QualityMetrics",
    # Visualization
    "NurbsVisualization",
]
