# ADR-002: CPU and GPU Implementation Strategy

## Status

Accepted

## Context

Performance is critical for interactive brush operations. Users expect smooth painting without noticeable lag. We need to support:

- **CPU execution**: Works everywhere, no special hardware needed
- **GPU execution**: Much faster for compute-intensive algorithms
- **User choice**: Let users select their preferred backend

3D Slicer supports:
- Python scripted modules (easy development)
- C++ modules with VTK/ITK (fast CPU execution)
- OpenCL/CUDA via VTK or custom code (GPU acceleration)

## Decision

Implement a **dual-backend architecture** supporting both CPU and GPU from the start.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Python Effect UI                      │
│              (SegmentEditorEffect.py)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Algorithm Dispatcher                        │
│         (selects CPU or GPU backend)                     │
└───────────┬─────────────────────────────┬───────────────┘
            │                             │
┌───────────▼───────────┐   ┌─────────────▼───────────────┐
│     CPU Backend       │   │       GPU Backend           │
│  (Python/SimpleITK)   │   │   (OpenCL or CUDA)          │
│                       │   │                             │
│  - Watershed          │   │  - Level Set (sparse field) │
│  - Level Set          │   │  - Gradient computation     │
│  - Connected Thresh   │   │  - Distance transforms      │
│  - Region Growing     │   │                             │
└───────────────────────┘   └─────────────────────────────┘
```

### CPU Implementation (Python/SimpleITK)

| Component | Implementation | Notes |
|-----------|---------------|-------|
| Effect UI | Python/Qt | Standard Slicer pattern |
| ROI extraction | NumPy slicing | Fast, simple |
| Intensity analysis | NumPy/sklearn | GMM fitting |
| Watershed | SimpleITK | ITK-optimized, multi-threaded |
| Level Set | SimpleITK | Slower but accurate |
| Connected Threshold | SimpleITK | Very fast |
| Mask application | NumPy | Array operations |

### GPU Implementation

| Component | Technology | Notes |
|-----------|------------|-------|
| Level Set solver | OpenCL or CUDA | Sparse field method |
| Gradient magnitude | OpenCL/VTK | Parallel convolution |
| Distance transform | OpenCL | Fast marching on GPU |
| Threshold operations | OpenCL | Trivially parallel |

### GPU Technology Selection

```python
def get_gpu_backend():
    """Detect available GPU compute capability."""
    # Priority order
    if cuda_available():
        return CUDABackend()
    if opencl_available():
        return OpenCLBackend()
    if metal_available():  # macOS
        return MetalBackend()
    return None  # Fall back to CPU
```

### Performance Targets

| Operation | CPU Target | GPU Target |
|-----------|-----------|------------|
| 2D brush (10mm) | < 50ms | < 10ms |
| 3D brush (10mm) | < 200ms | < 50ms |
| 3D brush (50mm) | < 2000ms | < 200ms |
| Drag operation | < 30ms | < 10ms |

### Implementation Phases

#### Phase 1: CPU Foundation
- All algorithms in Python/SimpleITK
- Clean interfaces for future GPU backends
- Full test coverage

#### Phase 2: GPU Level Set
- OpenCL sparse field level set solver
- Significant speedup for level set algorithm
- Fallback to CPU if GPU unavailable

#### Phase 3: GPU Acceleration
- GPU gradient computation
- GPU distance transforms
- GPU-accelerated watershed (if beneficial)

### Backend Selection Logic

```python
class BackendSelector:
    def select(self, algorithm: str, roi_size: tuple, user_pref: str) -> Backend:
        # User explicitly chose
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

        return CPUBackend()  # Default to CPU
```

## Consequences

### Positive

- **Universal compatibility**: CPU works everywhere
- **Maximum performance**: GPU for users who have it
- **User control**: Can force CPU or GPU as needed
- **Clean architecture**: Backend abstraction enables testing
- **Future-proof**: Can add Metal, Vulkan, etc.

### Negative

- **Development complexity**: Two codepaths to maintain
- **Testing burden**: Must test CPU and GPU paths
- **GPU setup complexity**: Users may need drivers
- **Memory management**: GPU memory is limited

### Mitigation Strategies

1. **Shared test suite**: Same tests run against both backends
2. **Automatic fallback**: GPU failures fall back to CPU gracefully
3. **Clear error messages**: Help users with GPU setup
4. **Memory monitoring**: Check GPU memory before large operations

## GPU Implementation Notes

### OpenCL Sparse Field Level Set

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

### Memory Management

```python
class GPUMemoryManager:
    def __init__(self, max_gpu_memory_mb=512):
        self.max_memory = max_gpu_memory_mb * 1024 * 1024
        self.allocated = 0

    def can_allocate(self, size_bytes):
        return self.allocated + size_bytes <= self.max_memory

    def should_use_gpu(self, roi_size, dtype_size=4):
        required = np.prod(roi_size) * dtype_size * 3  # phi, speed, gradient
        return self.can_allocate(required)
```

## References

- [VTK OpenCL](https://vtk.org/doc/nightly/html/classvtkOpenCLExtension.html)
- [GPU Level Set - Stanford](http://graphics.stanford.edu/papers/gpu_level_set/)
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
- [PyOpenCL](https://documen.tician.de/pyopencl/)
- [CuPy (CUDA)](https://cupy.dev/)
