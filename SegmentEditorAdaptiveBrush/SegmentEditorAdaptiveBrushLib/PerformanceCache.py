"""Performance cache for drag operations.

This module provides caching to enable smooth drag operations by
reusing intermediate computations when the brush moves within
a cached region.
"""

import logging
import time
from typing import Any, Callable, Optional

import numpy as np


class PerformanceCache:
    """Cache for intermediate computations during drag operations.

    This cache stores gradient computations and threshold data to speed up
    consecutive brush operations during drag events.

    Caching tiers:
    - Tier 1 (Gradient): Long-lived, persists until slice changes
    - Tier 2 (Thresholds): Medium-lived, reused when seed intensity similar
    - Tier 3 (ROI): Short-lived, cleared on mouse release
    """

    def __init__(self, cache_margin: float = 2.0, threshold_caching_enabled: bool = False):
        """Initialize the cache.

        Args:
            cache_margin: Multiplier for ROI size to cache (e.g., 2.0 means
                cache 2x the brush radius).
            threshold_caching_enabled: Whether to enable threshold caching.
                Disabled by default for maximum accuracy.
        """
        self.cache_margin = cache_margin
        self.threshold_caching_enabled = threshold_caching_enabled

        # Gradient cache (Tier 1 - persists across drag, until slice changes)
        self.gradient_cache: Optional[np.ndarray] = None
        self.gradient_slice_index: Optional[int] = None
        self.gradient_volume_id: Optional[str] = None

        # Threshold cache (Tier 2 - reused when seed intensity is similar)
        # Only active when threshold_caching_enabled is True
        self.threshold_cache: Optional[dict] = None
        self.threshold_seed_intensity: Optional[float] = None
        self.threshold_tolerance: float = 0.0  # Will be set based on std

        # ROI cache (Tier 3 - cleared on mouse release)
        self.roi_cache: Optional[np.ndarray] = None
        self.roi_bounds: Optional[tuple[tuple[int, int, int], tuple[int, int, int]]] = None

        # Statistics
        self.stats = CacheStats()

    def computeOrGetCached(
        self,
        volumeArray: np.ndarray,
        seedIjk: tuple[int, int, int],
        params: dict,
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

        # Get thresholds (may be cached)
        thresholds = self._getOrComputeThresholds(volumeArray, seedIjk, params, intensityAnalyzer)

        # Run segmentation
        mask = segmentationFunc(volumeArray, seedIjk, params, thresholds)

        elapsed = (time.perf_counter() - start_time) * 1000
        self.stats.total_compute_time_ms += elapsed
        logging.debug(
            f"Adaptive brush computation: {elapsed:.1f}ms (algorithm={params['algorithm']})"
        )

        return mask

    def _getOrComputeThresholds(
        self,
        volumeArray: np.ndarray,
        seedIjk: tuple[int, int, int],
        params: dict,
        intensityAnalyzer: Any,
    ) -> dict:
        """Get cached thresholds or compute new ones.

        Thresholds are cached and reused when the new seed intensity is
        within the tolerance range of the cached seed intensity. This
        provides smooth drag behavior when painting within similar regions.

        Args:
            volumeArray: Full volume array.
            seedIjk: Seed point (i, j, k).
            params: Parameters dict.
            intensityAnalyzer: Analyzer instance.

        Returns:
            Thresholds dictionary.
        """
        # Get current seed intensity (array is z,y,x but seedIjk is i,j,k = x,y,z)
        try:
            seed_intensity = float(volumeArray[seedIjk[2], seedIjk[1], seedIjk[0]])
        except IndexError:
            seed_intensity = None

        # Check if we can reuse cached thresholds
        if self._canReuseThresholds(seed_intensity):
            self.stats.threshold_hits += 1
            logging.debug(
                f"Threshold cache hit: seed={seed_intensity:.1f}, "
                f"cached={self.threshold_seed_intensity:.1f}"
            )
            return self.threshold_cache

        # Compute fresh thresholds
        self.stats.threshold_misses += 1
        radius_voxels = params.get("radius_voxels", (20, 20, 20))
        edge_sensitivity = params.get("edge_sensitivity", 0.5)

        thresholds = intensityAnalyzer.analyze(
            volumeArray,
            seedIjk,
            radius_voxels=tuple(radius_voxels),
            edge_sensitivity=edge_sensitivity,
        )

        # Cache the thresholds
        self.threshold_cache = thresholds
        self.threshold_seed_intensity = seed_intensity

        # Set tolerance based on the computed std (within 1 std = similar region)
        std = thresholds.get("std", 20.0)
        self.threshold_tolerance = max(std * 1.5, 10.0)  # At least 10 intensity units

        logging.debug(
            f"Threshold cache miss: seed={seed_intensity:.1f}, "
            f"thresholds=[{thresholds['lower']:.1f}, {thresholds['upper']:.1f}], "
            f"tolerance={self.threshold_tolerance:.1f}"
        )

        return thresholds

    def _canReuseThresholds(self, seed_intensity: Optional[float]) -> bool:
        """Check if cached thresholds can be reused.

        Args:
            seed_intensity: Current seed point intensity.

        Returns:
            True if caching is enabled, cache is valid, and seed intensity
            is within tolerance.
        """
        # Check if caching is enabled
        if not self.threshold_caching_enabled:
            return False

        if self.threshold_cache is None:
            return False

        if seed_intensity is None or self.threshold_seed_intensity is None:
            return False

        # Check if seed intensity is within tolerance of cached intensity
        intensity_diff = abs(seed_intensity - self.threshold_seed_intensity)
        return intensity_diff <= self.threshold_tolerance

    def getOrComputeGradient(
        self,
        volumeArray: np.ndarray,
        sliceIndex: int,
        volumeId: str,
        computeFunc: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Get cached gradient or compute new one.

        The gradient is cached per slice and reused when the brush moves
        within the same slice. This significantly speeds up watershed and
        level set algorithms.

        Args:
            volumeArray: Full volume array (z, y, x).
            sliceIndex: Current slice index (z).
            volumeId: Volume node ID for invalidation.
            computeFunc: Function to compute gradient from slice.

        Returns:
            Gradient magnitude array for the slice.
        """
        # Check if cache is valid
        if (
            self.gradient_cache is not None
            and self.gradient_slice_index == sliceIndex
            and self.gradient_volume_id == volumeId
        ):
            self.stats.gradient_hits += 1
            logging.debug(f"Gradient cache hit for slice {sliceIndex}")
            return self.gradient_cache

        # Compute fresh gradient
        self.stats.gradient_misses += 1
        start_time = time.perf_counter()

        # Extract slice and compute gradient
        slice_array = volumeArray[sliceIndex, :, :]
        gradient = computeFunc(slice_array)

        elapsed = (time.perf_counter() - start_time) * 1000
        logging.debug(f"Gradient cache miss for slice {sliceIndex}: {elapsed:.1f}ms")

        # Cache the result
        self.gradient_cache = gradient
        self.gradient_slice_index = sliceIndex
        self.gradient_volume_id = volumeId

        return gradient

    def isValidFor(
        self, seedIjk: tuple[int, int, int], radiusVoxels: tuple[float, float, float]
    ) -> bool:
        """Check if ROI cache is valid for given seed and radius.

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
        # Keep gradient cache for next stroke on same slice
        # Keep threshold cache - useful if next click is in similar region

        # Log statistics for this stroke
        self.stats.log_summary()

    def invalidate(self):
        """Invalidate all caches (e.g., when parameters change)."""
        self.gradient_cache = None
        self.gradient_slice_index = None
        self.gradient_volume_id = None
        self.roi_cache = None
        self.roi_bounds = None
        self.threshold_cache = None
        self.threshold_seed_intensity = None
        self.threshold_tolerance = 0.0

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
        total_threshold = self.threshold_hits + self.threshold_misses

        if total_gradient > 0:
            gradient_rate = self.gradient_hits / total_gradient
            logging.debug(f"Gradient cache hit rate: {gradient_rate:.1%}")

        if total_threshold > 0:
            threshold_rate = self.threshold_hits / total_threshold
            logging.debug(
                f"Threshold cache hit rate: {threshold_rate:.1%} "
                f"({self.threshold_hits}/{total_threshold})"
            )

        if self.total_compute_time_ms > 0:
            logging.debug(f"Total computation time: {self.total_compute_time_ms:.1f}ms")
