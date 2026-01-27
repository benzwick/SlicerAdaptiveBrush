"""Smooth contour rendering using marching squares.

Provides sub-pixel accurate contour extraction and visualization
using skimage.measure.find_contours.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ContourRenderer:
    """Render segmentation contours with multiple visualization modes.

    Supports two outline modes:
    - **Smooth**: Marching squares (skimage.measure.find_contours) for sub-pixel
      accurate contour extraction. Shows the interpolated boundary.
    - **Pixel**: Morphological dilation-erosion for jagged outlines that show
      the actual pixel boundaries of the segmentation.

    Both modes have their uses:
    - Smooth: Better for seeing true boundary shape, comparing curves
    - Pixel: Shows actual voxel boundaries, important for understanding
      what the algorithm actually segmented at pixel level
    """

    # Default colors (RGB 0-255)
    TRIAL_COLOR = (0, 255, 0)  # Green
    GOLD_COLOR = (255, 0, 255)  # Magenta
    TP_COLOR = (0, 200, 0)  # Green for agreement
    FP_COLOR = (255, 50, 50)  # Red for over-segmentation
    FN_COLOR = (50, 50, 255)  # Blue for under-segmentation

    def __init__(self):
        """Initialize renderer."""
        self._contour_cache: dict[str, list[np.ndarray]] = {}

    def get_morphological_outline(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Get pixel-level outline using morphological operations.

        Creates a jagged outline that shows the actual pixel boundaries
        of the segmentation (dilation - erosion).

        Args:
            mask: 2D binary mask.
            iterations: Thickness of outline in pixels.

        Returns:
            Binary mask of outline pixels.
        """
        from scipy import ndimage

        binary = mask.astype(bool)
        dilated = ndimage.binary_dilation(binary, iterations=iterations)
        eroded = ndimage.binary_erosion(binary, iterations=iterations)
        return dilated & ~eroded

    def find_contours(self, mask: np.ndarray) -> list[np.ndarray]:
        """Find smooth sub-pixel contours using marching squares.

        Args:
            mask: 2D binary mask (any dtype, will be converted to bool).

        Returns:
            List of contour arrays, each with shape (N, 2) in (row, col) format.
        """
        from skimage import measure

        # Ensure binary mask
        binary = mask.astype(bool).astype(np.float64)

        # Find contours at 0.5 level (sub-pixel accuracy via marching squares)
        contours = measure.find_contours(binary, level=0.5)

        return list(contours)

    def draw_contours(
        self,
        rgb: np.ndarray,
        contours: list[np.ndarray],
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw smooth contour polylines on RGB image.

        Args:
            rgb: RGB image (H, W, 3) as float64 or uint8.
            contours: List of contour arrays from find_contours.
            color: RGB color tuple (0-255).
            thickness: Line thickness in pixels.

        Returns:
            RGB image with contours drawn.
        """
        from skimage import draw

        result = rgb.astype(np.float64)

        for contour in contours:
            if len(contour) < 2:
                continue

            # Draw each segment of the polyline
            for i in range(len(contour) - 1):
                r0, c0 = contour[i]
                r1, c1 = contour[i + 1]

                # Get pixel coordinates for line
                rr, cc = draw.line(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1)))

                # Clip to image bounds
                valid = (rr >= 0) & (rr < result.shape[0]) & (cc >= 0) & (cc < result.shape[1])
                rr, cc = rr[valid], cc[valid]

                # Draw with specified thickness
                if thickness == 1:
                    for ch in range(3):
                        result[rr, cc, ch] = color[ch]
                else:
                    # For thickness > 1, draw to neighbors
                    for dr in range(-thickness // 2, thickness // 2 + 1):
                        for dc in range(-thickness // 2, thickness // 2 + 1):
                            rr_t = np.clip(rr + dr, 0, result.shape[0] - 1)
                            cc_t = np.clip(cc + dc, 0, result.shape[1] - 1)
                            for ch in range(3):
                                result[rr_t, cc_t, ch] = color[ch]

            # Close the contour if it's nearly closed
            if len(contour) > 2:
                dist = np.sqrt(
                    (contour[0][0] - contour[-1][0]) ** 2 + (contour[0][1] - contour[-1][1]) ** 2
                )
                if dist < 3:  # Close enough to close
                    r0, c0 = contour[-1]
                    r1, c1 = contour[0]
                    rr, cc = draw.line(
                        int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1))
                    )
                    valid = (rr >= 0) & (rr < result.shape[0]) & (cc >= 0) & (cc < result.shape[1])
                    rr, cc = rr[valid], cc[valid]
                    for ch in range(3):
                        result[rr, cc, ch] = color[ch]

        return result

    def create_comparison_image(
        self,
        image_slice: np.ndarray,
        trial_slice: np.ndarray | None = None,
        gold_slice: np.ndarray | None = None,
        trial_color: tuple[int, int, int] | None = None,
        gold_color: tuple[int, int, int] | None = None,
        line_thickness: int = 1,
        mode: str = "smooth",
    ) -> np.ndarray:
        """Create comparison image with both segmentations as outlines.

        Args:
            image_slice: 2D grayscale image (uint8)
            trial_slice: 2D binary trial segmentation mask
            gold_slice: 2D binary gold standard mask
            trial_color: RGB color for trial contour (default: green)
            gold_color: RGB color for gold contour (default: magenta)
            line_thickness: Contour line thickness in pixels
            mode: Outline mode - "smooth" (marching squares) or "pixel" (morphological)

        Returns:
            RGB image as uint8 array (H, W, 3)
        """
        if trial_color is None:
            trial_color = self.TRIAL_COLOR
        if gold_color is None:
            gold_color = self.GOLD_COLOR

        rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float64)

        if mode == "pixel":
            # Morphological outline - shows actual pixel boundaries
            if gold_slice is not None and np.any(gold_slice > 0):
                outline = self.get_morphological_outline(gold_slice > 0, iterations=line_thickness)
                for c in range(3):
                    rgb[:, :, c] = np.where(outline, gold_color[c], rgb[:, :, c])

            if trial_slice is not None and np.any(trial_slice > 0):
                outline = self.get_morphological_outline(trial_slice > 0, iterations=line_thickness)
                for c in range(3):
                    rgb[:, :, c] = np.where(outline, trial_color[c], rgb[:, :, c])
        else:
            # Smooth contours via marching squares - shows interpolated boundary
            if gold_slice is not None and np.any(gold_slice > 0):
                contours = self.find_contours(gold_slice > 0)
                rgb = self.draw_contours(rgb, contours, gold_color, line_thickness)

            if trial_slice is not None and np.any(trial_slice > 0):
                contours = self.find_contours(trial_slice > 0)
                rgb = self.draw_contours(rgb, contours, trial_color, line_thickness)

        return rgb.astype(np.uint8)

    def create_error_image(
        self,
        image_slice: np.ndarray,
        trial_slice: np.ndarray | None = None,
        gold_slice: np.ndarray | None = None,
        tp_color: tuple[int, int, int] | None = None,
        fp_color: tuple[int, int, int] | None = None,
        fn_color: tuple[int, int, int] | None = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create error analysis image with TP/FP/FN coloring.

        Color coding (assuming gold standard is truth):
        - Green: True Positive (both agree)
        - Red: False Positive (trial only - over-segmentation)
        - Blue: False Negative (gold only - under-segmentation)

        Args:
            image_slice: 2D grayscale image (uint8)
            trial_slice: 2D binary trial segmentation mask
            gold_slice: 2D binary gold standard mask
            tp_color: RGB for true positives (agreement)
            fp_color: RGB for false positives (over-segmentation)
            fn_color: RGB for false negatives (under-segmentation)
            alpha: Opacity of overlay

        Returns:
            RGB image as uint8 array (H, W, 3)
        """
        if tp_color is None:
            tp_color = self.TP_COLOR
        if fp_color is None:
            fp_color = self.FP_COLOR
        if fn_color is None:
            fn_color = self.FN_COLOR

        rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float64)

        trial_mask = (
            trial_slice > 0 if trial_slice is not None else np.zeros_like(image_slice, dtype=bool)
        )
        gold_mask = (
            gold_slice > 0 if gold_slice is not None else np.zeros_like(image_slice, dtype=bool)
        )

        # Calculate TP, FP, FN regions
        tp = trial_mask & gold_mask  # Both agree
        fp = trial_mask & ~gold_mask  # Trial only (over-segmentation)
        fn = ~trial_mask & gold_mask  # Gold only (under-segmentation)

        # Apply colors with alpha blending
        for c in range(3):
            rgb[:, :, c] = np.where(
                tp, (1 - alpha) * rgb[:, :, c] + alpha * tp_color[c], rgb[:, :, c]
            )
            rgb[:, :, c] = np.where(
                fp, (1 - alpha) * rgb[:, :, c] + alpha * fp_color[c], rgb[:, :, c]
            )
            rgb[:, :, c] = np.where(
                fn, (1 - alpha) * rgb[:, :, c] + alpha * fn_color[c], rgb[:, :, c]
            )

        return rgb.astype(np.uint8)

    def create_combined_image(
        self,
        image_slice: np.ndarray,
        trial_slice: np.ndarray | None = None,
        gold_slice: np.ndarray | None = None,
        show_contours: bool = True,
        show_error_regions: bool = True,
        error_alpha: float = 0.3,
    ) -> np.ndarray:
        """Create combined visualization with both contours and error regions.

        Args:
            image_slice: 2D grayscale image (uint8)
            trial_slice: 2D binary trial segmentation mask
            gold_slice: 2D binary gold standard mask
            show_contours: Whether to draw smooth contours
            show_error_regions: Whether to show TP/FP/FN regions
            error_alpha: Opacity of error regions

        Returns:
            RGB image as uint8 array (H, W, 3)
        """
        rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float64)

        # Draw error regions first (underneath contours)
        if show_error_regions:
            trial_mask = (
                trial_slice > 0
                if trial_slice is not None
                else np.zeros_like(image_slice, dtype=bool)
            )
            gold_mask = (
                gold_slice > 0 if gold_slice is not None else np.zeros_like(image_slice, dtype=bool)
            )

            tp = trial_mask & gold_mask
            fp = trial_mask & ~gold_mask
            fn = ~trial_mask & gold_mask

            for c in range(3):
                rgb[:, :, c] = np.where(
                    tp,
                    (1 - error_alpha) * rgb[:, :, c] + error_alpha * self.TP_COLOR[c],
                    rgb[:, :, c],
                )
                rgb[:, :, c] = np.where(
                    fp,
                    (1 - error_alpha) * rgb[:, :, c] + error_alpha * self.FP_COLOR[c],
                    rgb[:, :, c],
                )
                rgb[:, :, c] = np.where(
                    fn,
                    (1 - error_alpha) * rgb[:, :, c] + error_alpha * self.FN_COLOR[c],
                    rgb[:, :, c],
                )

        # Draw contours on top
        if show_contours:
            if gold_slice is not None and np.any(gold_slice > 0):
                contours = self.find_contours(gold_slice > 0)
                rgb = self.draw_contours(rgb, contours, self.GOLD_COLOR, thickness=1)

            if trial_slice is not None and np.any(trial_slice > 0):
                contours = self.find_contours(trial_slice > 0)
                rgb = self.draw_contours(rgb, contours, self.TRIAL_COLOR, thickness=1)

        return rgb.astype(np.uint8)

    def clear_cache(self) -> None:
        """Clear the contour cache."""
        self._contour_cache.clear()
