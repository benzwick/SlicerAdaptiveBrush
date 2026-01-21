# ADR-004: Caching Strategy for Drag Operations

## Status

Accepted

## Context

During brush drag operations, users expect smooth, lag-free painting. Computing the full adaptive segmentation for each mouse movement would be too slow for interactive use.

Key observations:
- Mouse events fire at 60+ Hz during drag
- Full algorithm takes 50-200ms
- Adjacent brush positions share computation
- Gradient magnitude is expensive but reusable

## Decision

Implement a tiered caching strategy:

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

## Consequences

### Positive

- **Smooth drag performance** (< 30ms per frame with caching)
- **Memory-efficient** (only caches what's needed)
- **Graceful degradation** (preview mode when cache misses)
- **Seamless transitions** between cached and computed results

### Negative

- **Added complexity** for cache management
- **Potential for stale cache bugs** if invalidation is wrong
- **Memory pressure** on low-end systems with large volumes
- **Preview may differ slightly** from final result

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

## Implementation

```python
class PerformanceCache:
    def __init__(self):
        self.gradient_cache = GradientCache()
        self.roi_cache = ROICache()
        self.stats = CacheStats()

    def process_point(self, volume, seed_ijk, radius, is_dragging):
        gradient = self.gradient_cache.get_or_compute(volume, seed_ijk[2])

        if is_dragging and not self.roi_cache.is_valid_for(seed_ijk, radius):
            # Use preview mode during drag when cache misses
            return self.compute_preview(volume, seed_ijk, radius, gradient)
        else:
            # Full computation
            return self.compute_full(volume, seed_ijk, radius, gradient)

    def on_mouse_release(self):
        # Compute full resolution if we were in preview mode
        # Clear ROI cache for next stroke
        self.roi_cache.clear()
```

## References

- [Game Programming Patterns: Object Pool](https://gameprogrammingpatterns.com/object-pool.html)
- [VTK Pipeline Caching](https://vtk.org/doc/nightly/html/classvtkAlgorithm.html)
