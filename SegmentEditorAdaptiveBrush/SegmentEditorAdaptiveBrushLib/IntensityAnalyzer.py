"""Intensity Analyzer for adaptive brush threshold estimation.

This module provides automatic intensity threshold estimation using
Gaussian Mixture Models (GMM) or simple statistics as a fallback.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

# Try to import sklearn for GMM
try:
    from sklearn.mixture import GaussianMixture

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.info("sklearn not available - using simple statistics for threshold estimation")


class IntensityAnalyzer:
    """Analyze image intensities to estimate segmentation thresholds.

    This class fits a Gaussian Mixture Model to the intensity distribution
    within a region of interest and identifies the component containing
    the seed point to compute optimal thresholds.
    """

    def __init__(self, use_gmm: bool = True, n_components_range: Tuple[int, int] = (2, 4)):
        """Initialize the analyzer.

        Args:
            use_gmm: Whether to use GMM (requires sklearn). Falls back to
                simple statistics if False or sklearn unavailable.
            n_components_range: Range of GMM components to try (min, max).
        """
        self.use_gmm = use_gmm and HAS_SKLEARN
        self.n_components_range = n_components_range

    def analyze(
        self,
        image: np.ndarray,
        seed_point: Tuple[int, int, int],
        radius_voxels: Optional[Tuple[float, float, float]] = None,
    ) -> Dict:
        """Analyze intensity distribution around seed point.

        Args:
            image: Image array (z, y, x ordering for 3D).
            seed_point: Seed point coordinates (x, y, z).
            radius_voxels: Optional ROI radius in voxels. If None, uses
                a default radius of 20 voxels.

        Returns:
            Dictionary with:
                - 'lower': Lower intensity threshold
                - 'upper': Upper intensity threshold
                - 'mean': Estimated mean of seed region
                - 'std': Estimated std of seed region
                - 'n_components': Number of GMM components (if GMM used)
        """
        # Extract ROI
        if radius_voxels is None:
            radius_voxels = (20.0, 20.0, 20.0)

        roi = self._extract_roi(image, seed_point, radius_voxels)

        # Get seed intensity
        # Note: image is (z, y, x), seed_point is (x, y, z)
        try:
            seed_intensity = float(image[seed_point[2], seed_point[1], seed_point[0]])
        except IndexError:
            logging.warning(f"Seed point {seed_point} out of bounds")
            # Return conservative thresholds based on full image
            return self._simple_statistics(image.flatten(), np.mean(image))

        if self.use_gmm:
            return self._gmm_analysis(roi, seed_intensity)
        else:
            return self._simple_statistics(roi, seed_intensity)

    def _extract_roi(
        self,
        image: np.ndarray,
        seed_point: Tuple[int, int, int],
        radius_voxels: Tuple[float, float, float],
    ) -> np.ndarray:
        """Extract region of interest around seed point.

        Args:
            image: Full image array (z, y, x).
            seed_point: Seed point (x, y, z).
            radius_voxels: Radius in voxels (x, y, z).

        Returns:
            Flattened ROI intensities as 1D array.
        """
        shape = image.shape  # (z, y, x)

        # Convert seed_point (x, y, z) to array indices
        cx, cy, cz = seed_point
        rx, ry, rz = [int(r) for r in radius_voxels]

        # Calculate bounds
        z_start = max(0, cz - rz)
        z_end = min(shape[0], cz + rz + 1)
        y_start = max(0, cy - ry)
        y_end = min(shape[1], cy + ry + 1)
        x_start = max(0, cx - rx)
        x_end = min(shape[2], cx + rx + 1)

        roi = image[z_start:z_end, y_start:y_end, x_start:x_end]

        return roi.flatten().astype(np.float64)

    def _gmm_analysis(self, roi: np.ndarray, seed_intensity: float) -> Dict:
        """Analyze using Gaussian Mixture Model.

        Args:
            roi: Flattened ROI intensities.
            seed_intensity: Intensity at seed point.

        Returns:
            Analysis results dictionary.
        """
        if len(roi) < 100:
            # Too few samples for GMM
            return self._simple_statistics(roi, seed_intensity)

        # Subsample if too large (for speed)
        if len(roi) > 10000:
            indices = np.random.choice(len(roi), 10000, replace=False)
            roi_sample = roi[indices]
        else:
            roi_sample = roi

        # Reshape for sklearn
        X = roi_sample.reshape(-1, 1)

        # Try different numbers of components and select best by BIC
        best_gmm = None
        best_bic = np.inf

        for n_components in range(self.n_components_range[0], self.n_components_range[1] + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n_components, random_state=42, max_iter=100, n_init=1
                )
                gmm.fit(X)
                bic = gmm.bic(X)

                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception as e:
                logging.debug(f"GMM with {n_components} components failed: {e}")
                continue

        if best_gmm is None:
            return self._simple_statistics(roi, seed_intensity)

        # Find component containing seed intensity
        seed_component = best_gmm.predict([[seed_intensity]])[0]

        # Get component parameters
        component_mean = float(best_gmm.means_[seed_component][0])
        component_std = float(np.sqrt(best_gmm.covariances_[seed_component][0][0]))

        # Compute thresholds (2.5 sigma range)
        sigma_multiplier = 2.5
        data_min, data_max = float(roi.min()), float(roi.max())

        lower = max(data_min, component_mean - sigma_multiplier * component_std)
        upper = min(data_max, component_mean + sigma_multiplier * component_std)

        return {
            "lower": lower,
            "upper": upper,
            "mean": component_mean,
            "std": component_std,
            "n_components": best_gmm.n_components,
        }

    def _simple_statistics(self, roi: np.ndarray, seed_intensity: float) -> Dict:
        """Fallback analysis using simple statistics.

        Uses intensities similar to the seed to estimate thresholds.

        Args:
            roi: Flattened ROI intensities.
            seed_intensity: Intensity at seed point.

        Returns:
            Analysis results dictionary.
        """
        if len(roi) == 0:
            return {
                "lower": seed_intensity - 50,
                "upper": seed_intensity + 50,
                "mean": seed_intensity,
                "std": 50.0,
                "n_components": 1,
            }

        # Compute global statistics
        global_mean = float(np.mean(roi))
        global_std = float(np.std(roi))

        if global_std < 1e-6:
            # Constant region
            return {
                "lower": seed_intensity - 1,
                "upper": seed_intensity + 1,
                "mean": seed_intensity,
                "std": 0.0,
                "n_components": 1,
            }

        # Find voxels with similar intensity to seed (within 2 std)
        tolerance = max(global_std * 2, 20)  # At least 20 HU tolerance
        similar_mask = np.abs(roi - seed_intensity) < tolerance
        similar_values = roi[similar_mask]

        if len(similar_values) < 10:
            # Fall back to global statistics
            similar_values = roi

        local_mean = float(np.mean(similar_values))
        local_std = float(np.std(similar_values))

        # Use percentiles for robustness
        lower = float(np.percentile(similar_values, 2))
        upper = float(np.percentile(similar_values, 98))

        return {
            "lower": lower,
            "upper": upper,
            "mean": local_mean,
            "std": local_std,
            "n_components": 1,
        }
