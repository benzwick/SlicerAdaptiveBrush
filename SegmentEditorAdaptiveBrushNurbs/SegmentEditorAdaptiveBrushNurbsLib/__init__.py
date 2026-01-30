"""SegmentEditorAdaptiveBrushNurbs library.

Provides volumetric NURBS generation from painted segmentations.

Core Classes:
    StructureDetector: Detect structure type (simple/tubular/branching)
    HexMeshGenerator: Generate hexahedral control meshes
    NurbsVolumeBuilder: Build NURBS volumes from control meshes
    ContainmentValidator: Ensure segment is contained within NURBS
    QualityMetrics: Compute fitting quality metrics

Export Classes:
    MfemExporter: Export to MFEM volumetric NURBS format
    MeshExporter: Export to STL/OBJ/VTK triangulated mesh
"""

from .ContainmentValidator import ContainmentValidator
from .HexMeshGenerator import HexMesh, HexMeshGenerator
from .NurbsVolumeBuilder import NurbsVolumeBuilder
from .QualityMetrics import QualityMetrics
from .StructureDetector import StructureDetector
from .Visualization import NurbsVisualization

__all__ = [
    # Structure detection
    "StructureDetector",
    # Mesh generation
    "HexMesh",
    "HexMeshGenerator",
    # NURBS construction
    "NurbsVolumeBuilder",
    # Validation
    "ContainmentValidator",
    "QualityMetrics",
    # Visualization
    "NurbsVisualization",
]
