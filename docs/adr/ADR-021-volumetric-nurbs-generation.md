# ADR-021: Volumetric NURBS Generation

## Status

**Implemented** - Phases 1-6 complete

The SegmentEditorAdaptiveBrushNurbs module provides:
- Automatic structure type detection (simple/tubular/branching)
- Hexahedral control mesh generation
- Volumetric NURBS construction using geomdl
- MFEM mesh export for IGA simulations
- STL/OBJ/VTK triangulated mesh export
- IGES/STEP CAD export (requires pythonocc-core)
- Quality metrics (Hausdorff, containment, Jacobian)

## Context

SlicerAdaptiveBrush enables users to paint segmentations on medical images. For downstream applications in engineering simulation, users need to convert these segmentations into **volumetric NURBS** elements suitable for:

- **Isogeometric Analysis (IGA)**: Direct use of NURBS geometry in finite element simulations via MFEM
- **CAD Integration**: Export to IGES/STEP for engineering workflows
- **Mesh Generation**: Triangulated surface/volume meshes for traditional FEM

### Why Volumetric NURBS?

| Aspect | Surface NURBS | Volumetric NURBS |
|--------|---------------|------------------|
| **Representation** | 2D patches in 3D space | 3D solid hexahedral elements |
| **Control points** | 2D grid (u,v) | 3D grid (u,v,w) |
| **Use case** | Visualization, surface analysis | Solid mechanics FEM, IGA |
| **MFEM support** | Surface meshes | Full 3D NURBS elements |

### Structure Types

Medical image segmentations exhibit different topologies requiring specialized approaches:

| Structure Type | Examples | Approach |
|----------------|----------|----------|
| **Simple convex** | Tumors, nodules | Oriented bounding box → hexahedral NURBS |
| **Tubular** | Vessels, airways | Centerline + circular sweeping |
| **Branching** | Arterial trees, bronchi | VMTK + bifurcation templates |

## Decision

Create a dedicated **SegmentEditorAdaptiveBrushNurbs** Slicer module for converting painted segmentations into volumetric NURBS elements.

### Library Selection

| Library | Purpose | Reason for Selection |
|---------|---------|---------------------|
| **geomdl** | NURBS volume construction | Pure Python, supports 3D volumetric NURBS, well-documented |
| **SlicerVMTK** | Centerline extraction | Industry-standard for vascular modeling, integrated with Slicer |
| **pythonocc-core** | IGES/STEP export | OpenCASCADE wrapper, full CAD format support (optional) |
| **VTK** | Visualization, mesh export | Bundled with Slicer, robust geometry processing |

### Module Structure

```
SegmentEditorAdaptiveBrushNurbs/
├── CMakeLists.txt
├── SegmentEditorAdaptiveBrushNurbs.py       # Module entry point
├── SegmentEditorAdaptiveBrushNurbsLib/
│   ├── __init__.py
│   ├── StructureDetector.py                 # Detect structure type
│   ├── SkeletonExtractor.py                 # VMTK centerline wrapper
│   ├── BranchTemplates.py                   # Bifurcation templates
│   ├── HexMeshGenerator.py                  # Hexahedral control mesh
│   ├── NurbsVolumeBuilder.py                # NURBS construction
│   ├── ContainmentValidator.py              # Segment containment check
│   ├── QualityMetrics.py                    # Fitting quality metrics
│   ├── Visualization.py                     # VTK visualization
│   └── Exporters/
│       ├── __init__.py
│       ├── MfemExporter.py                  # MFEM mesh format
│       ├── MeshExporter.py                  # STL/OBJ/VTK export
│       └── CadExporter.py                   # IGES/STEP export
└── Testing/Python/
    ├── conftest.py                          # Test fixtures
    └── test_nurbs_volume.py                 # Unit tests
```

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Segmentation Input (from Adaptive Brush)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Structure Detection (StructureDetector)                     │
│     - Topology analysis (Euler characteristic, skeleton ratio)  │
│     - Returns: "simple" | "tubular" | "branching"               │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Simple:         │ │  Tubular:    │ │  Branching:      │
│  Bounding box    │ │  VMTK        │ │  VMTK + branch   │
│  hex mesh        │ │  centerline  │ │  templates       │
└────────┬─────────┘ └──────┬───────┘ └────────┬─────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Hexahedral Control Mesh (HexMeshGenerator)                  │
│     - Sweep templates along skeleton                            │
│     - Project to segment boundary                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. NURBS Volume Construction (NurbsVolumeBuilder)              │
│     - Compute clamped knot vectors                              │
│     - Build geomdl Volume object                                │
│     - Support multi-patch for branching                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Validation (ContainmentValidator, QualityMetrics)           │
│     - Verify segment points inside NURBS                        │
│     - Compute Hausdorff distance, Jacobian quality              │
│     - Iterative refinement if needed                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  MFEM Export     │ │  CAD Export  │ │  Mesh Export     │
│  (MfemExporter)  │ │ (CadExporter)│ │ (MeshExporter)   │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

### Key Algorithms

#### Simple Shape (Tumors, Nodules)

1. Compute oriented bounding box via PCA
2. Create hexahedral control mesh (e.g., 4×4×4)
3. Project boundary control points to segment surface
4. Build NURBS volume with uniform knot vectors

#### Tubular Structure (Vessels)

1. Extract centerline using VMTK (weighted shortest path)
2. Compute inscribed sphere radii along path
3. Sweep circular cross-section template along centerline
4. Adapt radius to match vessel diameter

#### Branching Structure (Arterial Trees)

Based on [Patient-Specific Vascular NURBS Modeling for IGA](https://pmc.ncbi.nlm.nih.gov/articles/PMC2839408/):

1. Extract full centerline network (VMTK)
2. Detect bifurcation points (VMTK BranchClipper)
3. Apply branch templates:
   - Bifurcation (n=2): 3 map-meshable regions
   - Trifurcation (n=3): 5 template cases
4. Ensure G¹ continuity at patch boundaries

### Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Hausdorff distance** | Maximum surface deviation | < tolerance |
| **Mean surface distance** | Average surface deviation | Minimize |
| **Volume ratio** | NURBS / segment volume | ≈ 1.0 |
| **Containment ratio** | % segment points inside NURBS | > 99% |
| **Minimum Jacobian** | Mesh quality (>0 = valid) | > 0 |

### Export Formats

| Format | File Extension | Use Case |
|--------|----------------|----------|
| **MFEM mesh** | `.mesh` | Isogeometric analysis |
| **IGES** | `.igs` | CAD interchange |
| **STEP AP214** | `.step` | CAD interchange |
| **VTK Unstructured** | `.vtu` | Visualization, FEM |
| **STL** | `.stl` | 3D printing, simple meshers |
| **OBJ** | `.obj` | General 3D applications |

## Consequences

### Positive

- **IGA-ready output**: Direct use in MFEM for isogeometric analysis
- **CAD compatibility**: IGES/STEP export for engineering workflows
- **Automatic detection**: No manual topology classification needed
- **Quality validation**: Containment and Jacobian checks ensure usable meshes
- **Multi-structure support**: Works with tumors, vessels, and branching trees

### Negative

- **Optional dependencies**: Full functionality requires geomdl, VMTK, pythonocc
- **Complexity**: Branching structures require sophisticated templates
- **Computational cost**: NURBS fitting can be slow for large segments

### Trade-offs

| Aspect | Alternative | Chosen | Reason |
|--------|-------------|--------|--------|
| NURBS library | OpenCASCADE | geomdl | Pure Python, easier integration |
| Centerline | Custom skeletonization | VMTK | Proven, handles bifurcations |
| Branching templates | Automatic meshing | Manual templates | Better control, G¹ continuity |

## Alternatives Considered

### SlicerSurfaceMarkup

**Rejected**: Only supports 2D NURBS patches (surfaces), not 3D volumetric elements needed for IGA.

### OpenCASCADE Directly

**Rejected**: C++ library requires complex wrapper. geomdl provides pure Python volumetric NURBS support.

### Tetrahedral Meshing (SlicerSegmentMesher)

**Rejected**: IGA specifically requires hexahedral NURBS elements. Tets can't represent NURBS geometry directly.

### Boundary-Only NURBS

**Rejected**: IGA requires volumetric parameterization, not just surface representation.

## Dependencies

| Dependency | Required | Installation |
|------------|----------|--------------|
| geomdl | Yes | `pip install geomdl` (auto-installed) |
| numpy | Yes | Bundled with Slicer |
| scipy | Yes | Bundled with Slicer |
| vtk | Yes | Bundled with Slicer |
| SlicerVMTK | For tubular/branching | Extension Manager |
| pythonocc-core | For IGES/STEP | `conda install -c conda-forge pythonocc-core` |

## References

### Papers

- [Patient-Specific Vascular NURBS Modeling for IGA](https://pmc.ncbi.nlm.nih.gov/articles/PMC2839408/) - Zhang et al., key reference for branching structures
- [Isogeometric Analysis](https://en.wikipedia.org/wiki/Isogeometric_analysis) - Background on IGA

### Libraries

- [NURBS-Python (geomdl)](https://nurbs-python.readthedocs.io/) - NURBS construction
- [MFEM](https://mfem.org/) - Finite element library with NURBS support
- [VMTK](http://www.vmtk.org/) - Vascular Modeling Toolkit
- [pythonocc-core](https://github.com/tpaviot/pythonocc-core) - OpenCASCADE Python wrapper

### Slicer Extensions

- [SlicerVMTK](https://github.com/vmtk/SlicerExtension-VMTK) - Centerline extraction
- [SlicerSegmentMesher](https://github.com/lassoan/SlicerSegmentMesher) - Tetrahedral meshing (comparison)

### Related ADRs

- [ADR-007](ADR-007-dependency-management.md): Dependency Management
- [ADR-010](ADR-010-testing-framework.md): Testing Framework
