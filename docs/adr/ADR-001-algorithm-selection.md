# ADR-001: Algorithm Selection - Multiple User-Selectable Options

## Status

Implemented (Phase 1 - CPU Complete)

## Context

We need algorithms for the adaptive brush that:
- Provide accurate boundary detection
- Run interactively (< 100ms for typical use)
- Adapt to different imaging modalities
- Allow users to choose their preferred approach

Different users have different needs:
- Some prefer speed over precision
- Some datasets work better with certain algorithms
- Power users want fine control, casual users want simplicity
- Some need automatic threshold detection, others want manual control

## Decision

Implement **multiple algorithm backends** that users can select from the UI, with all algorithms initially implemented using CPU-based SimpleITK for reliability and portability.

### Algorithm Options

| Algorithm | Backend | Speed | Precision | Best For |
|-----------|---------|-------|-----------|----------|
| Watershed | CPU (SimpleITK) | Medium | High | General use (default) |
| Level Set | CPU (SimpleITK) | Slow | Very High | High precision needs |
| Connected Threshold | CPU (SimpleITK) | Very Fast | Low | Quick rough segmentation |
| Region Growing | CPU (SimpleITK) | Fast | Medium | Homogeneous regions |
| Threshold Brush | CPU (SimpleITK) | Very Fast | Variable | Simple threshold painting |

### User Interface

```
Algorithm: [Dropdown]
├── Watershed (Recommended)
├── Level Set (CPU)
├── Connected Threshold (Fast)
├── Region Growing
└── Threshold Brush (Simple)

Backend: [Auto / CPU / GPU*]
* GPU backends planned for Phase 2
```

### Algorithm Implementations

#### 1. Watershed (Default)
- **Implementation**: SimpleITK `MorphologicalWatershedFromMarkers`
- **Process**:
  1. Get initial mask from connected threshold
  2. Compute gradient magnitude
  3. Scale gradient by edge sensitivity parameter
  4. Create markers: eroded mask (foreground), dilated border (background)
  5. Run watershed from markers
- Good balance of speed and accuracy

#### 2. Level Set
- **Implementation**: SimpleITK `GeodesicActiveContourLevelSetImageFilter`
- **Process**:
  1. Get initial mask from connected threshold
  2. Create signed distance map from initial mask
  3. Compute speed image from inverted gradient
  4. Evolve level set with propagation, curvature, and advection terms
  5. Threshold at zero to get binary result
- Best boundary accuracy, adapts to weak edges

#### 3. Connected Threshold
- **Implementation**: SimpleITK `ConnectedThreshold`
- Fastest option, uses intensity range from GMM analysis
- May leak across weak boundaries
- Good baseline for simple regions

#### 4. Region Growing
- **Implementation**: SimpleITK `ConfidenceConnected`
- Iterative region growing based on statistical similarity
- Multiplier scales with edge sensitivity (3.5 at 0%, 1.0 at 100%)
- Good for homogeneous tissues

#### 5. Threshold Brush
- **Implementation**: Simple NumPy thresholding with optional SimpleITK filters
- **Auto-Threshold Methods**:
  - Otsu (`OtsuThresholdImageFilter`)
  - Huang (`HuangThresholdImageFilter`)
  - Triangle (`TriangleThresholdImageFilter`)
  - Maximum Entropy (`MaximumEntropyThresholdImageFilter`)
  - IsoData (`IsoDataThresholdImageFilter`)
  - Li (`LiThresholdImageFilter`)
- **Auto-Detection**: Compares seed intensity to computed threshold to determine whether to segment above or below
- **Manual Mode**: User-specified lower/upper thresholds with set-from-seed helper
- No connectivity analysis - paints all voxels in threshold range within brush

### Threshold Brush UI

When Threshold Brush is selected, additional controls appear:

```
[x] Auto threshold
Method: [Otsu ▼]

--- OR when Auto unchecked: ---

Lower: [-100 ════════════○════] 5000
Upper: [-100 ════════════════○] 5000
[Set from seed intensity]
Tolerance: [0 ══════○══════════] 100%
```

### Shared Pipeline

All algorithms share common stages:

```
1. ROI Extraction
   └── Extract region around cursor (1.2x brush radius margin)

2. Intensity Analysis
   └── GMM-based threshold estimation (IntensityAnalyzer)

3. Algorithm-Specific Segmentation
   └── Watershed / Level Set / Threshold Brush / etc.

4. Post-processing
   └── Apply circular/spherical brush mask constraint

5. Apply to Segment
   └── Modify labelmap via OR operation
```

### Backend Selection

The Backend dropdown shows Auto/CPU/GPU options. Currently:
- **Auto**: Selects CPU (GPU planned for Phase 2)
- **CPU**: All algorithms available
- **GPU**: UI present, implementation planned for Phase 2

## Consequences

### Positive

- **User choice**: Different algorithms suit different use cases
- **Simplicity option**: Threshold Brush for users who want direct control
- **Future extensibility**: Easy to add new algorithms or GPU backends
- **Graceful degradation**: All algorithms work on any system via CPU

### Negative

- **More code to maintain**: Multiple implementations in single file
- **Testing complexity**: Must test all algorithm combinations
- **User confusion potential**: Many options available

### Mitigation

- Default to "Watershed" for best general-purpose results
- Algorithm section collapsed by default
- Comprehensive tooltips explaining each option
- Threshold Brush provides escape hatch for difficult cases

## Implementation Details

### Watershed Implementation

```python
def _watershed(self, roi, localSeed, thresholds, params):
    # Get initial region with connected threshold
    initialMask = self._connectedThreshold(roi, localSeed, thresholds)

    # Compute gradient magnitude
    sitkRoi = sitk.GetImageFromArray(roi.astype(np.float32))
    gradient = sitk.GradientMagnitude(sitkRoi)

    # Scale by edge sensitivity
    gradArray = sitk.GetArrayFromImage(gradient)
    sensitivity = params.get("edge_sensitivity", 0.5)
    gradArray = (gradArray / gradMax * 255 * sensitivity).astype(np.float32)

    # Create markers from initial mask
    initialSitk = sitk.GetImageFromArray(initialMask)
    foreground = sitk.BinaryErode(initialSitk, [2, 2, 2])
    dilated = sitk.BinaryDilate(initialSitk, [3, 3, 3])
    background = sitk.Subtract(dilated, sitk.BinaryDilate(initialSitk, [1, 1, 1]))
    markers = sitk.Add(foreground, sitk.Multiply(background, 2))

    # Run watershed
    watershed = sitk.MorphologicalWatershedFromMarkers(
        gradient, markers, markWatershedLine=False, fullyConnected=True
    )
    return sitk.BinaryThreshold(watershed, 1, 1, 1, 0)
```

### Threshold Brush Auto-Detection

```python
def _computeAutoThreshold(self, roi, localSeed, method="otsu"):
    # Get threshold using selected method
    if method == "otsu":
        filterObj = sitk.OtsuThresholdImageFilter()
    # ... other methods

    filterObj.Execute(sitkRoi)
    threshold = filterObj.GetThreshold()

    # Auto-detect: check if seed is above or below threshold
    seed_intensity = roi[localSeed[2], localSeed[1], localSeed[0]]

    if seed_intensity >= threshold:
        # Seed in brighter region - segment above threshold
        return (threshold, data_max)
    else:
        # Seed in darker region - segment below threshold
        return (data_min, threshold)
```

## Phase 2 - Future Enhancements

### GPU Level Set (Planned)

```python
def gpu_level_set_segment(roi, seed, params):
    # Initialize level set from seed
    phi = initialize_level_set(roi, seed)

    # Sparse field GPU solver (OpenCL/CUDA)
    solver = GPUSparsefieldSolver(roi)
    for _ in range(params.iterations):
        phi = solver.evolve(phi, params)
        if solver.converged():
            break

    return threshold_level_set(phi)
```

### Auto-Selection Logic (Planned)

```python
def select_backend(algorithm, roi_size):
    if algorithm == "Level Set" and gpu_available():
        return "GPU"
    if roi_size > LARGE_ROI_THRESHOLD and gpu_available():
        return "GPU"  # Large ROIs benefit from GPU
    return "CPU"
```

## References

- ITK-SNAP source: [PaintbrushSettingsModel.cxx](https://github.com/pyushkevich/itksnap/blob/master/GUI/Model/PaintbrushSettingsModel.cxx)
- Smart Brush paper: [IEEE](https://ieeexplore.ieee.org/document/6703826)
- GPU Level Set: [Stanford](http://graphics.stanford.edu/papers/gpu_level_set/)
- ITK Watershed: [ITK Examples](https://examples.itk.org/src/segmentation/watersheds/segmentwithwatershedimagefilter/documentation)
- SimpleITK Threshold Filters: [SimpleITK Docs](https://simpleitk.readthedocs.io/en/master/filters.html)
