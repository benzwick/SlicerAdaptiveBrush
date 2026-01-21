# ADR-001: Algorithm Selection - Multiple User-Selectable Options

## Status

Accepted

## Context

We need algorithms for the adaptive brush that:
- Provide accurate boundary detection
- Run interactively (< 100ms for typical use)
- Adapt to different imaging modalities
- Support both CPU and GPU execution
- Allow users to choose their preferred approach

Different users have different needs:
- Some prefer speed over precision
- Some have powerful GPUs, others don't
- Some datasets work better with certain algorithms
- Power users want fine control, casual users want simplicity

## Decision

Implement **multiple algorithm backends** that users can select from the UI:

### Algorithm Options

| Algorithm | Backend | Speed | Precision | Best For |
|-----------|---------|-------|-----------|----------|
| Watershed | CPU (ITK) | Medium | High | General use, reliable |
| Level Set | GPU (OpenCL/CUDA) | Fast | Very High | Users with GPU |
| Level Set | CPU (ITK) | Slow | Very High | Precision without GPU |
| Connected Threshold | CPU | Very Fast | Low | Quick rough segmentation |
| Region Growing | CPU | Fast | Medium | Homogeneous regions |

### User Interface

```
Algorithm: [Dropdown]
├── Watershed (Recommended)
├── Level Set (GPU) - if GPU available
├── Level Set (CPU)
├── Connected Threshold (Fast)
└── Region Growing

Backend: [Auto / CPU / GPU]
```

### Algorithm Implementations

#### 1. Watershed (Default)
- **CPU**: SimpleITK `MorphologicalWatershedFromMarkers`
- Marker-based for controlled segmentation
- Good balance of speed and accuracy

#### 2. Level Set (GPU and CPU)
- **GPU**: Custom OpenCL/CUDA implementation using sparse field method
- **CPU**: SimpleITK `GeodesicActiveContourLevelSetImageFilter`
- Best boundary accuracy, adapts to weak edges
- GPU version for real-time performance

#### 3. Connected Threshold
- **CPU**: SimpleITK `ConnectedThreshold`
- Fastest option, good for uniform regions
- May leak across weak boundaries

#### 4. Region Growing
- **CPU**: SimpleITK region growing with similarity criteria
- Good for homogeneous tissues
- Configurable similarity threshold

### Shared Pipeline

All algorithms share common stages:

```
1. ROI Extraction (CPU/GPU)
   └── Extract region around cursor

2. Intensity Analysis (CPU)
   └── GMM or statistics-based threshold estimation

3. Algorithm-Specific Segmentation
   └── Watershed / Level Set / etc.

4. Post-processing (CPU/GPU)
   └── Mask to brush radius, smoothing

5. Apply to Segment (CPU)
   └── Modify labelmap
```

### Auto-Selection Logic

When "Auto" backend is selected:
```python
def select_backend(algorithm, roi_size):
    if algorithm == "Level Set" and gpu_available():
        return "GPU"
    if roi_size > LARGE_ROI_THRESHOLD and gpu_available():
        return "GPU"  # Large ROIs benefit from GPU
    return "CPU"
```

## Consequences

### Positive

- **User choice**: Different algorithms suit different use cases
- **Hardware flexibility**: Works on systems with/without GPU
- **Future extensibility**: Easy to add new algorithms
- **Best-in-class potential**: Can match or exceed any single tool
- **Graceful degradation**: Falls back to CPU if GPU unavailable

### Negative

- **More code to maintain**: Multiple implementations
- **Testing complexity**: Must test all combinations
- **User confusion**: Too many options can overwhelm
- **GPU dependency management**: OpenCL/CUDA setup complexity

### Mitigation

- Default to "Watershed" with "Auto" backend for simplicity
- Hide advanced options behind "Show Advanced" toggle
- Comprehensive tooltips explaining each option
- Auto-detect best algorithm based on image characteristics

## Algorithm Details

### Watershed Implementation

```python
def watershed_segment(roi, seed, gradient):
    # Create markers
    markers = create_markers(roi, seed)

    # Run watershed
    result = sitk.MorphologicalWatershedFromMarkers(
        gradient, markers,
        markWatershedLine=False,
        fullyConnected=True
    )
    return extract_seed_region(result, seed)
```

### GPU Level Set Implementation

```python
def gpu_level_set_segment(roi, seed, params):
    # Initialize level set from seed
    phi = initialize_level_set(roi, seed)

    # Sparse field GPU solver
    solver = GPUSparsefieldSolver(roi)
    for _ in range(params.iterations):
        phi = solver.evolve(phi, params)
        if solver.converged():
            break

    return threshold_level_set(phi)
```

## References

- ITK-SNAP source: [PaintbrushSettingsModel.cxx](https://github.com/pyushkevich/itksnap/blob/master/GUI/Model/PaintbrushSettingsModel.cxx)
- Smart Brush paper: [IEEE](https://ieeexplore.ieee.org/document/6703826)
- GPU Level Set: [Stanford](http://graphics.stanford.edu/papers/gpu_level_set/)
- ITK Watershed: [ITK Examples](https://examples.itk.org/src/segmentation/watersheds/segmentwithwatershedimagefilter/documentation)
