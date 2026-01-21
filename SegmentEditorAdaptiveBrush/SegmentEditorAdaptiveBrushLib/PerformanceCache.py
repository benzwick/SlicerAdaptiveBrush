"""Performance cache for drag operations.

This module provides caching to enable smooth drag operations by
reusing intermediate computations when the brush moves within
a cached region.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class PerformanceCache:
    """Cache for intermediate computations during drag operations.

    This cache stores gradient computations and ROI data to speed up
    consecutive brush operations during drag events.
    """

    def __init__(self, cache_margin: float = 2.0):
        """Initialize the cache.

        Args:
            cache_margin: Multiplier for ROI size to cache (e.g., 2.0 means
                cache 2x the brush radius).
        """
        self.cache_margin = cache_margin

        # Gradient cache (persists across drag)
        self.gradient_cache: Optional[np.ndarray] = None
        self.gradient_slice_index: Optional[int] = None
        self.gradient_volume_id: Optional[str] = None

        # ROI cache (cleared on mouse release)
        self.roi_cache: Optional[np.ndarray] = None
        self.roi_bounds: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
        self.threshold_cache: Optional[Dict] = None

        # Statistics
        self.stats = CacheStats()

    def computeOrGetCached(
        self,
        volumeArray: np.ndarray,
        seedIjk: Tuple[int, int, int],
        params: Dict,
        intensityAnalyzer: Any,
        segmentationFunc: Callable,
    ) -> np.ndarray:
        """Compute segmentation mask, using cache when possible.

        Args:
            volumeArray: Full volume array.
            seedIjk: Seed point (i, j, k).
            params: Segmentation parameters.
            intensityAnalyzer: IntensityAnalyzer instance.
            segmentationFunc: Function to run segmentation.

        Returns:
            Binary mask array.
        """
        start_time = time.perf_counter()

        # Check if we can reuse threshold estimation
        thresholds = self._getOrComputeThresholds(volumeArray, seedIjk, params, intensityAnalyzer)

        # Run segmentation
        mask = segmentationFunc(volumeArray, seedIjk, params, thresholds)

        elapsed = (time.perf_counter() - start_time) * 1000
        logging.debug(f"Adaptive brush computation: {elapsed:.1f}ms")

        return mask

    def _getOrComputeThresholds(
        self,
        volumeArray: np.ndarray,
        seedIjk: Tuple[int, int, int],
        params: Dict,
        intensityAnalyzer: Any,
    ) -> Dict:
        """Get cached thresholds or compute new ones.

        Args:
            volumeArray: Full volume array.
            seedIjk: Seed point.
            params: Parameters dict.
            intensityAnalyzer: Analyzer instance.

        Returns:
            Thresholds dictionary.
        """
        # For now, always recompute thresholds
        # Future: cache when seed is in similar intensity region
        radius_voxels = params.get("radius_voxels", (20, 20, 20))
        edge_sensitivity = params.get("edge_sensitivity", 0.5)

        thresholds = intensityAnalyzer.analyze(
            volumeArray,
            seedIjk,
            radius_voxels=tuple(radius_voxels),
            edge_sensitivity=edge_sensitivity,
        )

        self.threshold_cache = thresholds
        return thresholds

    def isValidFor(
        self, seedIjk: Tuple[int, int, int], radiusVoxels: Tuple[float, float, float]
    ) -> bool:
        """Check if cache is valid for given seed and radius.

        Args:
            seedIjk: Seed point (i, j, k).
            radiusVoxels: Radius in voxels.

        Returns:
            True if cache can be used.
        """
        if self.roi_bounds is None:
            return False

        start, end = self.roi_bounds

        # Check if seed + radius fits within cached bounds
        for i in range(3):
            if seedIjk[i] - radiusVoxels[i] < start[i]:
                return False
            if seedIjk[i] + radiusVoxels[i] > end[i]:
                return False

        return True

    def onMouseRelease(self):
        """Called when mouse is released - clear short-lived caches."""
        self.roi_cache = None
        self.roi_bounds = None
        # Keep gradient cache for next stroke

    def invalidate(self):
        """Invalidate all caches (e.g., when parameters change)."""
        self.gradient_cache = None
        self.gradient_slice_index = None
        self.gradient_volume_id = None
        self.roi_cache = None
        self.roi_bounds = None
        self.threshold_cache = None

    def clear(self):
        """Clear all caches completely."""
        self.invalidate()
        self.stats.reset()


class CacheStats:
    """Statistics for cache performance monitoring."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.gradient_hits = 0
        self.gradient_misses = 0
        self.roi_hits = 0
        self.roi_misses = 0
        self.threshold_hits = 0
        self.threshold_misses = 0
        self.total_compute_time_ms = 0.0
        self.total_cached_time_ms = 0.0

    def log_summary(self):
        """Log cache statistics summary."""
        total_gradient = self.gradient_hits + self.gradient_misses
        total_roi = self.roi_hits + self.roi_misses

        if total_gradient > 0:
            gradient_rate = self.gradient_hits / total_gradient
            logging.info(f"Gradient cache hit rate: {gradient_rate:.1%}")

        if total_roi > 0:
            roi_rate = self.roi_hits / total_roi
            logging.info(f"ROI cache hit rate: {roi_rate:.1%}")

        if self.total_compute_time_ms > 0:
            speedup = (
                self.total_compute_time_ms + self.total_cached_time_ms
            ) / self.total_compute_time_ms
            logging.info(f"Cache speedup factor: {speedup:.2f}x")
