# ADR-004: Caching Strategy for Drag Operations

## Status

Accepted (Infrastructure Complete, Optimization Pending)

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| PerformanceCache class | Complete | Structure and interface ready |
| Gradient cache fields | Scaffolded | Fields exist, not actively used |
| ROI cache fields | Scaffolded | Fields exist, not actively used |
| Threshold cache | Partial | Computed fresh each time |
| Preview mode | Not started | Full resolution always computed |
| Cache statistics | Complete | Logging infrastructure ready |
| Cache invalidation | Complete | Triggers on parameter changes |

### Current Behavior

The cache currently:
- Times each brush computation (for performance monitoring)
- Computes fresh thresholds for each point
- Invalidates on parameter changes (radius, sensitivity, algorithm)
- Clears state on mouse release

The following optimizations are designed but not active:
- Gradient magnitude caching across slice
- ROI reuse when seed moves within cached region
- Preview mode during drag operations

## Context

During brush drag operations, users expect smooth, lag-free painting. Computing the full adaptive segmentation for each mouse movement would be too slow for interactive use.

Key observations:
- Mouse events fire at 60+ Hz during drag
- Full algorithm takes 50-200ms
- Adjacent brush positions share computation
- Gradient magnitude is expensive but reusable

## Decision

Implement a tiered caching strategy with infrastructure ready for optimization:

### Tier 1: Gradient Cache (Long-lived)

- Cache gradient magnitude for the visible slice
- Invalidate when:
  - Slice index changes
  - Source volume changes
  - View orientation changes
- Memory: ~8MB per slice for typical 512x512 volume

```python
class GradientCache:
    def __init__(self):
        self.gradient = None
        self.slice_index = None
        self.volume_id = None

    def get_or_compute(self, volume, slice_index):
        if (self.slice_index != slice_index or
            self.volume_id != volume.GetID()):
            self.gradient = compute_gradient(volume, slice_index)
            self.slice_index = slice_index
            self.volume_id = volume.GetID()
        return self.gradient
```

### Tier 2: ROI Cache (Medium-lived)

- Cache extracted ROI and intermediate watershed results
- Invalidate when cursor moves outside cached region
- Cache region = 2x brush radius (for continuity during drag)

```python
class ROICache:
    def __init__(self):
        self.roi = None
        self.bounds = None  # (min_ijk, max_ijk)
        self.watershed_result = None

    def is_valid_for(self, seed_ijk, radius):
        if self.bounds is None:
            return False
        # Check if seed + radius fits within cached bounds
        return self._contains(seed_ijk, radius)
```

### Tier 3: Result Preview (Short-lived)

- Show downsampled preview during drag
- Full resolution computed on mouse release
- Preview resolution: 1/4 native resolution (16x faster)

```python
def compute_preview(self, roi, seed, radius):
    # Downsample for fast preview
    downsampled = sitk.Shrink(roi, [4, 4, 1])
    scaled_seed = (seed[0]//4, seed[1]//4, seed[2])
    scaled_radius = radius / 4

    # Fast computation on small image
    preview_mask = self.algorithm.compute(
        downsampled, scaled_seed, scaled_radius
    )

    # Upsample result
    return sitk.Expand(preview_mask, [4, 4, 1])
```

### Cache Lifecycle

```
Mouse Press:
├── Clear Tier 2 (ROI) cache
├── Check Tier 1 (Gradient) cache validity
└── Compute full resolution result

Mouse Move (drag):
├── Check Tier 2 cache validity
├── If valid: reuse cached watershed markers
├── If invalid: compute preview only
└── Show result immediately

Mouse Release:
├── Compute full resolution if preview was shown
├── Update Tier 2 cache for next stroke
└── Keep Tier 1 cache
```

## Current Implementation

```python
class PerformanceCache:
    """Cache for brush computations during drag operations.

    Current implementation provides the interface and invalidation logic.
    Actual caching optimization is pending.
    """

    def __init__(self):
        # Cache structures (ready for optimization)
        self._gradient_cache = None
        self._gradient_slice = None
        self._roi_cache = None
        self._roi_bounds = None
        self._thresholds = None

        # Statistics
        self._compute_times = []

    def computeOrGetCached(self, volumeArray, seedIjk, params,
                           intensityAnalyzer, computeFunc):
        """Compute result, using cache when valid.

        Currently computes fresh each time. Caching logic is scaffolded
        but not active.
        """
        start_time = time.time()

        # Get thresholds (computed fresh currently)
        thresholds = self._getOrComputeThresholds(
            volumeArray, seedIjk, params, intensityAnalyzer
        )

        # Compute result (no caching currently)
        result = computeFunc(volumeArray, seedIjk, params, thresholds)

        elapsed = time.time() - start_time
        self._compute_times.append(elapsed)

        return result

    def invalidate(self):
        """Clear cache on parameter changes."""
        self._thresholds = None
        self._roi_cache = None

    def onMouseRelease(self):
        """Clear per-stroke cache."""
        self._roi_cache = None
        self._roi_bounds = None
```

## Consequences

### Positive

- **Infrastructure ready**: Cache structure in place for optimization
- **Clean interface**: `computeOrGetCached` abstracts caching details
- **Invalidation working**: Parameters changes clear cache correctly
- **Statistics collection**: Performance data available for analysis

### Negative

- **Not optimized yet**: Full computation on every point
- **Drag may lag**: No preview mode for fast feedback
- **Memory not utilized**: Cache structures allocated but unused

### Optimization Path

When performance optimization is needed:

1. **Enable gradient caching**: Reuse gradient between points on same slice
2. **Enable ROI caching**: Reuse watershed results for nearby seeds
3. **Add preview mode**: Downsample during drag, full resolution on release
4. **Profile and tune**: Use collected statistics to identify bottlenecks

## Cache Statistics (Debug Mode)

```python
class CacheStats:
    def __init__(self):
        self.gradient_hits = 0
        self.gradient_misses = 0
        self.roi_hits = 0
        self.roi_misses = 0
        self.preview_count = 0

    def log_summary(self):
        gradient_rate = self.gradient_hits / (self.gradient_hits + self.gradient_misses)
        roi_rate = self.roi_hits / (self.roi_hits + self.roi_misses)
        logging.debug(f"Cache hit rates: gradient={gradient_rate:.1%}, roi={roi_rate:.1%}")
```

## Memory Budget

| Cache | Size Estimate | Lifetime |
|-------|--------------|----------|
| Gradient (512x512 slice) | 8 MB | Until slice change |
| ROI (100x100x10 region) | 4 MB | Until drag ends |
| Preview mask | 0.5 MB | Single frame |
| **Total (typical)** | **~12 MB** | |

## References

- [Game Programming Patterns: Object Pool](https://gameprogrammingpatterns.com/object-pool.html)
- [VTK Pipeline Caching](https://vtk.org/doc/nightly/html/classvtkAlgorithm.html)
