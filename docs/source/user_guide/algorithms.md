# Algorithms

SlicerAdaptiveBrush provides eight segmentation algorithms. Each has different characteristics
suited to different segmentation tasks.

## Algorithm Comparison

| Algorithm | Speed | Precision | Boundary | Best For |
|-----------|-------|-----------|----------|----------|
| Watershed | Medium | High | Excellent | General use (default) |
| Connected Threshold | Very Fast | Low | Poor | Quick rough segmentation |
| Region Growing | Fast | Medium | Fair | Homogeneous regions |
| Threshold Brush | Very Fast | Variable | N/A | Simple threshold painting |
| Geodesic Distance | Medium | High | Good | Structures with clear edges |
| Level Set | Slow | Very High | Excellent | High precision needs |
| Random Walker | Slow | High | Good | Complex boundaries |

## Watershed

**Recommended for most use cases.**

The watershed algorithm treats the image as a topographic surface where intensity values
represent elevation. It finds boundaries by simulating water flooding from marker points.

### Parameters

- **Gradient Scale** (0.5-3.0): Controls smoothing of gradient magnitude
  - Lower values: More detailed boundaries
  - Higher values: Smoother boundaries, less noise sensitivity

- **Smoothing** (0.1-1.0): Gaussian smoothing before gradient computation
  - Lower values: Preserve fine details
  - Higher values: Reduce noise

### When to Use

- General segmentation tasks
- Structures with clear boundaries
- When boundary adherence is important

### Example

Watershed performs well on brain tumor segmentation where the tumor has a clear
boundary with surrounding tissue:

```
Best parameters from optimization:
- Algorithm: watershed
- Edge sensitivity: 70
- Gradient scale: 1.06
- Smoothing: 0.78
- Dice score: 99.91%
```

## Connected Threshold

**Fastest algorithm, lowest precision.**

Segments all connected voxels within an intensity range from the seed point.
Simple but effective for homogeneous regions.

### Parameters

Uses the global **Edge Sensitivity** to determine threshold range around seed intensity.

### When to Use

- Quick rough segmentation
- Highly homogeneous regions (e.g., air, fluid)
- When speed is more important than precision

### Limitations

- No boundary refinement
- Sensitive to intensity variations
- May leak through weak boundaries

## Region Growing

**Fast with confidence-based growing.**

Uses SimpleITK's ConfidenceConnected filter which iteratively grows the region
based on mean and standard deviation of included voxels.

### Parameters

- **Multiplier**: Implicit from edge sensitivity
- **Iterations**: Number of growing iterations (default: 5)

### When to Use

- Regions with consistent intensity statistics
- When connected threshold leaks too much
- Moderate precision requirements

## Threshold Brush

**Simple intensity thresholding with auto-detection.**

Paints all voxels within a threshold range under the brush. Can automatically
determine whether to segment above or below threshold based on seed intensity.

### Parameters

- **Auto-threshold method**: Otsu, Huang, Triangle, Maximum Entropy, IsoData, Li
- **Manual mode**: Set lower/upper thresholds directly
- **Set from seed**: Calculate thresholds from seed intensity with tolerance

### When to Use

- Simple intensity-based segmentation
- When you know the exact intensity range
- Quick selection of high/low intensity structures

## Geodesic Distance

**Good for structures with clear edges.**

Combines intensity similarity and gradient information using fast marching.
The "speed" function rewards paths through similar intensities and penalizes
crossing edges.

### Parameters

- **Edge Weight** (0-1): Balance between intensity and gradient
  - 0: Pure intensity-based (like region growing)
  - 1: Pure edge-based (stops at all edges)
  - 0.5: Balanced (default)

### When to Use

- Structures with clear edges
- When watershed over-segments
- Tubular structures (vessels)

## Level Set

**Highest precision, slowest speed.**

Uses geodesic active contours that evolve a curve/surface to minimize an energy
function. Provides smooth, accurate boundaries.

### Parameters

- **Iterations** (30-150): Number of evolution iterations
  - More iterations: More accurate but slower

- **Propagation** (0.5-2.0): Expansion/contraction force
  - \> 1: Tends to expand
  - < 1: Tends to contract

### When to Use

- When precision is paramount
- Smooth, well-defined boundaries needed
- Willing to wait for results

### Limitations

- Significantly slower than other methods
- May require parameter tuning
- Can get stuck in local minima

## Random Walker

**Probabilistic segmentation for complex cases.**

Treats segmentation as a probability diffusion problem. Labels propagate from
seed points based on image gradients.

### Parameters

- **Beta** (10-200): Controls sensitivity to gradients
  - Higher values: More sensitive to edges
  - Lower values: Smoother propagation

### When to Use

- Complex, intertwined structures
- Multiple competing regions
- When deterministic methods fail

### Requirements

Requires scikit-image. Falls back to region growing if not available.

## Algorithm Selection Guide

### By Structure Type

| Structure | Recommended Algorithm |
|-----------|----------------------|
| Tumor/Lesion | Watershed |
| Bone (CT) | Threshold Brush or Connected Threshold |
| Brain tissue | Watershed or Level Set |
| Blood vessels | Geodesic Distance |
| Fluid/CSF | Connected Threshold |
| Complex anatomy | Random Walker |

### By Priority

| Priority | Recommended Algorithm |
|----------|----------------------|
| Speed | Connected Threshold or Threshold Brush |
| Precision | Level Set |
| Balanced | Watershed (default) |
| Boundary accuracy | Watershed or Geodesic Distance |

### From Optimization Results

Based on automated optimization against gold standards:

```
Parameter Importance:
- Algorithm choice: 73.1%
- Brush radius: 18.6%
- Threshold zone: 5.1%
- Edge sensitivity: 3.2%
```

**Key insight**: Algorithm selection is by far the most important parameter.
Watershed consistently outperforms other algorithms for boundary-sensitive tasks.
