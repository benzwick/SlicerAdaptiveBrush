# ADR-002: Implementation Strategy - CPU Foundation with GPU Roadmap

## Status

Implemented (Phase 1 Complete)

## Context

Performance is critical for interactive brush operations. Users expect smooth painting without noticeable lag. We need to support:

- **CPU execution**: Works everywhere, no special hardware needed
- **GPU execution**: Much faster for compute-intensive algorithms (future enhancement)
- **User choice**: Let users select their preferred backend

3D Slicer supports:
- Python scripted modules (easy development)
- SimpleITK/ITK for optimized CPU operations
- OpenCL/CUDA via VTK or custom code (GPU acceleration - Phase 2)

## Decision

Implement a **phased approach** starting with a complete CPU foundation, with GPU acceleration as a future enhancement.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Python Effect UI                      │
│              (SegmentEditorEffect.py)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Algorithm Methods                           │
│    (_watershed, _levelSet, _thresholdBrush, etc.)       │
└───────────┬─────────────────────────────────────────────┘
            │
┌───────────▼───────────────────┐   ┌─────────────────────────────┐
│       CPU Backend             │   │    GPU Backend (Phase 2)    │
│    (Python/SimpleITK)         │   │    (Planned - Not Started)  │
│         COMPLETE              │   │                             │
│                               │   │  - Level Set (sparse field) │
│  - Watershed                  │   │  - Gradient computation     │
│  - Level Set (GAC)            │   │  - Distance transforms      │
│  - Connected Threshold        │   │                             │
│  - Region Growing             │   │                             │
│  - Threshold Brush            │   │                             │
└───────────────────────────────┘   └─────────────────────────────┘
```

### Implementation Phases

#### Phase 1: CPU Foundation ✓ COMPLETE

All algorithms implemented in Python/SimpleITK:
- Clean effect class with all algorithm methods
- Full UI with algorithm and backend selectors
- Intensity analysis with GMM and fallback
- Performance cache structure (ready for optimization)
- Test scaffolding complete

| Component | Implementation | Status |
|-----------|---------------|--------|
| Effect UI | Python/Qt | Complete |
| ROI extraction | NumPy slicing | Complete |
| Intensity analysis | NumPy/sklearn GMM | Complete |
| Watershed | SimpleITK MorphologicalWatershedFromMarkers | Complete |
| Level Set | SimpleITK GeodesicActiveContourLevelSetImageFilter | Complete |
| Connected Threshold | SimpleITK ConnectedThreshold | Complete |
| Region Growing | SimpleITK ConfidenceConnected | Complete |
| Threshold Brush | NumPy + SimpleITK threshold filters | Complete |
| Brush visualization | VTK 2D pipeline | Complete |
| 3D sphere mode | NumPy distance masking | Complete |
| Mask application | NumPy array operations | Complete |

#### Phase 2: GPU Level Set (Planned)

- OpenCL sparse field level set solver
- Significant speedup for level set algorithm
- Fallback to CPU if GPU unavailable

#### Phase 3: GPU Acceleration (Future)

- GPU gradient computation
- GPU distance transforms
- GPU-accelerated watershed (if beneficial)

### Current Backend Selection

The Backend dropdown in the UI shows Auto/CPU/GPU options:

```python
# Current behavior (Phase 1)
def get_backend():
    # GPU options visible in UI but not implemented
    # All algorithms currently use CPU/SimpleITK
    return "CPU"
```

### Performance Targets

| Operation | CPU Current | CPU Target | GPU Target (Phase 2) |
|-----------|------------|-----------|---------------------|
| 2D brush (10mm) | 30-100ms | < 50ms | < 10ms |
| 3D brush (10mm) | 100-500ms | < 200ms | < 50ms |
| 3D brush (50mm) | 500-2000ms | < 2000ms | < 200ms |
| Drag operation | 30-100ms | < 30ms | < 10ms |

## Consequences

### Positive

- **Universal compatibility**: CPU works everywhere, no setup required
- **Rapid development**: Python/SimpleITK is productive
- **Clean architecture**: Algorithm methods are well-organized
- **Future-ready**: UI prepared for GPU backends
- **Reliable**: SimpleITK is well-tested and maintained

### Negative

- **Performance ceiling**: CPU has limits for large 3D operations
- **GPU not available**: Users with GPUs cannot leverage them yet
- **Deferred optimization**: Caching structure exists but not fully utilized

### Mitigation Strategies

1. **ROI extraction**: Limits computation to brush region only
2. **Algorithm choice**: Users can select faster algorithms (Connected Threshold)
3. **Cache structure**: Ready for optimization when needed
4. **GPU roadmap**: Clear path to GPU acceleration in Phase 2

## Phase 2 Design Notes

### GPU Technology Selection (Future)

```python
def get_gpu_backend():
    """Detect available GPU compute capability."""
    if cuda_available():
        return CUDABackend()
    if opencl_available():
        return OpenCLBackend()
    if metal_available():  # macOS
        return MetalBackend()
    return None  # Fall back to CPU
```

### Backend Selection Logic (Future)

```python
class BackendSelector:
    def select(self, algorithm: str, roi_size: tuple, user_pref: str) -> Backend:
        if user_pref == "CPU":
            return CPUBackend()
        if user_pref == "GPU":
            if not self.gpu_available():
                logging.warning("GPU requested but unavailable, using CPU")
                return CPUBackend()
            return GPUBackend()

        # Auto selection
        if algorithm == "Level Set" and self.gpu_available():
            return GPUBackend()  # Level set benefits most from GPU

        voxel_count = roi_size[0] * roi_size[1] * roi_size[2]
        if voxel_count > 1_000_000 and self.gpu_available():
            return GPUBackend()  # Large ROIs benefit from GPU

        return CPUBackend()
```

### OpenCL Sparse Field Level Set (Future)

```c
// Kernel for level set evolution
__kernel void evolve_level_set(
    __global float* phi,           // Level set function
    __global const float* speed,   // Speed image
    __global const int* active,    // Active voxel list
    const int num_active,
    const float dt
) {
    int idx = get_global_id(0);
    if (idx >= num_active) return;

    int voxel = active[idx];

    // Compute curvature and advection
    float curvature = compute_curvature(phi, voxel);
    float advection = compute_advection(phi, speed, voxel);

    // Update level set
    phi[voxel] += dt * (curvature + advection);
}
```

## References

- [VTK OpenCL](https://vtk.org/doc/nightly/html/classvtkOpenCLExtension.html)
- [GPU Level Set - Stanford](http://graphics.stanford.edu/papers/gpu_level_set/)
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
- [PyOpenCL](https://documen.tician.de/pyopencl/)
- [CuPy (CUDA)](https://cupy.dev/)
