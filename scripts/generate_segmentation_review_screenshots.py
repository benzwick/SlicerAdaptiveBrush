#!/usr/bin/env python3
"""Generate slice-by-slice images of segmentation for VLM review.

Exports actual image slices at native resolution with segmentation overlay
and optional gold standard outline for comparison.

Usage:
    Slicer --python-script scripts/run_review_screenshots.py <segmentation_path> [output_dir]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255 uint8 range using percentile windowing."""
    arr = array.astype(np.float64)
    p5, p95 = np.percentile(arr, [5, 95])
    window_center = (p5 + p95) / 2
    window_width = (p95 - p5) * 1.2

    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2

    arr = np.clip(arr, min_val, max_val)
    arr = (arr - min_val) / (max_val - min_val) * 255

    return arr.astype(np.uint8)


def get_morphological_outline(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Get pixel-level outline using morphological operations.

    Creates a jagged outline that shows the actual pixel boundaries
    of the segmentation (dilation - erosion). This is useful for seeing
    exactly what voxels were segmented.

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


def find_smooth_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Find smooth sub-pixel contours using marching squares.

    Creates smooth, interpolated contours that show the true boundary
    shape rather than pixel staircasing. Better for seeing the overall
    boundary shape.

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


def draw_contour_polylines(
    rgb: np.ndarray,
    contours: list[np.ndarray],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    """Draw smooth contour polylines on RGB image.

    Args:
        rgb: RGB image (H, W, 3) as float64.
        contours: List of contour arrays from find_smooth_contours.
        color: RGB color tuple (0-255).
        thickness: Line thickness in pixels.

    Returns:
        RGB image with contours drawn.
    """
    from skimage import draw

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
            valid = (rr >= 0) & (rr < rgb.shape[0]) & (cc >= 0) & (cc < rgb.shape[1])
            rr, cc = rr[valid], cc[valid]

            # Draw with specified thickness
            if thickness == 1:
                for ch in range(3):
                    rgb[rr, cc, ch] = color[ch]
            else:
                # For thickness > 1, also draw to neighbors
                for dr in range(-thickness // 2, thickness // 2 + 1):
                    for dc in range(-thickness // 2, thickness // 2 + 1):
                        rr_t = np.clip(rr + dr, 0, rgb.shape[0] - 1)
                        cc_t = np.clip(cc + dc, 0, rgb.shape[1] - 1)
                        for ch in range(3):
                            rgb[rr_t, cc_t, ch] = color[ch]

        # Close the contour if it's nearly closed
        if len(contour) > 2:
            dist = np.sqrt(
                (contour[0][0] - contour[-1][0]) ** 2 + (contour[0][1] - contour[-1][1]) ** 2
            )
            if dist < 3:  # Close enough to close
                r0, c0 = contour[-1]
                r1, c1 = contour[0]
                rr, cc = draw.line(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1)))
                valid = (rr >= 0) & (rr < rgb.shape[0]) & (cc >= 0) & (cc < rgb.shape[1])
                rr, cc = rr[valid], cc[valid]
                for ch in range(3):
                    rgb[rr, cc, ch] = color[ch]

    return rgb


def create_comparison_image_smooth(
    image_slice: np.ndarray,
    trial_slice: np.ndarray | None = None,
    gold_slice: np.ndarray | None = None,
    trial_color: tuple[int, int, int] = (0, 255, 0),  # Green for trial
    gold_color: tuple[int, int, int] = (255, 0, 255),  # Magenta for gold
    line_thickness: int = 1,
) -> np.ndarray:
    """Create comparison image with smooth contours (marching squares).

    Uses marching squares (skimage.measure.find_contours) for sub-pixel accurate,
    smooth contour lines. Better for seeing the overall boundary shape.

    No judgment about which is correct - just shows both for comparison.

    Args:
        image_slice: 2D grayscale image (uint8)
        trial_slice: 2D binary trial segmentation mask
        gold_slice: 2D binary gold standard mask
        trial_color: RGB color for trial contour
        gold_color: RGB color for gold contour
        line_thickness: Contour line thickness in pixels

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float64)

    # Draw gold contour first (so trial is on top where they overlap)
    if gold_slice is not None and np.any(gold_slice > 0):
        contours = find_smooth_contours(gold_slice > 0)
        rgb = draw_contour_polylines(rgb, contours, gold_color, line_thickness)

    # Draw trial contour on top
    if trial_slice is not None and np.any(trial_slice > 0):
        contours = find_smooth_contours(trial_slice > 0)
        rgb = draw_contour_polylines(rgb, contours, trial_color, line_thickness)

    return rgb.astype(np.uint8)


def create_comparison_image_pixel(
    image_slice: np.ndarray,
    trial_slice: np.ndarray | None = None,
    gold_slice: np.ndarray | None = None,
    trial_color: tuple[int, int, int] = (0, 255, 0),  # Green for trial
    gold_color: tuple[int, int, int] = (255, 0, 255),  # Magenta for gold
) -> np.ndarray:
    """Create comparison image with pixel-level outlines (morphological).

    Uses morphological dilation-erosion for jagged outlines that show the
    actual pixel boundaries of the segmentation. Better for seeing exactly
    what voxels were segmented.

    No judgment about which is correct - just shows both for comparison.

    Args:
        image_slice: 2D grayscale image (uint8)
        trial_slice: 2D binary trial segmentation mask
        gold_slice: 2D binary gold standard mask
        trial_color: RGB color for trial outline
        gold_color: RGB color for gold outline

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float64)

    # Draw gold outline first (so trial is on top where they overlap)
    if gold_slice is not None and np.any(gold_slice > 0):
        outline = get_morphological_outline(gold_slice > 0)
        for c in range(3):
            rgb[:, :, c] = np.where(outline, gold_color[c], rgb[:, :, c])

    # Draw trial outline on top
    if trial_slice is not None and np.any(trial_slice > 0):
        outline = get_morphological_outline(trial_slice > 0)
        for c in range(3):
            rgb[:, :, c] = np.where(outline, trial_color[c], rgb[:, :, c])

    return rgb.astype(np.uint8)


def create_error_image(
    image_slice: np.ndarray,
    trial_slice: np.ndarray | None = None,
    gold_slice: np.ndarray | None = None,
    tp_color: tuple[int, int, int] = (0, 200, 0),  # Green = agreement
    fp_color: tuple[int, int, int] = (255, 50, 50),  # Red = over-segmentation
    fn_color: tuple[int, int, int] = (50, 50, 255),  # Blue = under-segmentation
    alpha: float = 0.5,
) -> np.ndarray:
    """Create error analysis image assuming gold standard is truth.

    Color coding:
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
    rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float64)

    trial_mask = (
        trial_slice > 0 if trial_slice is not None else np.zeros_like(image_slice, dtype=bool)
    )
    gold_mask = gold_slice > 0 if gold_slice is not None else np.zeros_like(image_slice, dtype=bool)

    # Calculate TP, FP, FN regions
    tp = trial_mask & gold_mask  # Both agree
    fp = trial_mask & ~gold_mask  # Trial only (over-segmentation)
    fn = ~trial_mask & gold_mask  # Gold only (under-segmentation)

    # Apply colors
    for c in range(3):
        rgb[:, :, c] = np.where(tp, (1 - alpha) * rgb[:, :, c] + alpha * tp_color[c], rgb[:, :, c])
        rgb[:, :, c] = np.where(fp, (1 - alpha) * rgb[:, :, c] + alpha * fp_color[c], rgb[:, :, c])
        rgb[:, :, c] = np.where(fn, (1 - alpha) * rgb[:, :, c] + alpha * fn_color[c], rgb[:, :, c])

    return rgb.astype(np.uint8)


def export_segmentation_to_labelmap(segmentation_node, reference_volume, segment_id=None):
    """Export segmentation to labelmap array aligned with reference volume.

    Uses Slicer's proper resampling to align segmentation with volume geometry.

    Returns:
        Numpy array in [k, j, i] order matching reference volume shape
    """
    import slicer
    import vtk

    segmentation = segmentation_node.GetSegmentation()
    if segment_id is None:
        if segmentation.GetNumberOfSegments() == 0:
            return None
        segment_id = segmentation.GetNthSegmentID(0)

    # Create a temporary labelmap volume node
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

    # Export segmentation to labelmap using reference volume geometry
    segment_ids = vtk.vtkStringArray()
    segment_ids.InsertNextValue(segment_id)

    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segmentation_node, segment_ids, labelmap_node, reference_volume
    )

    # Get the array
    array = slicer.util.arrayFromVolume(labelmap_node)

    # Clean up
    slicer.mrmlScene.RemoveNode(labelmap_node)

    return array


def generate_review_images(
    segmentation_path: str | Path,
    output_dir: str | Path,
    gold_standard_path: str | Path | None = None,
    volume_path: str | Path | None = None,
    axis: int = 2,
    sample_every_n: int = 1,
    margin_slices: int = 2,
):
    """Generate slice images at native resolution for VLM review.

    Creates two visualization modes:
    1. compare/ - Neutral comparison with both as outlines (no judgment)
    2. error/ - Error analysis assuming gold is truth (TP=green, FP=red, FN=blue)

    Args:
        segmentation_path: Path to segmentation file (trial result)
        output_dir: Directory for output images
        gold_standard_path: Path to gold standard segmentation
        volume_path: Path to volume (optional, uses sample data if needed)
        axis: Slice axis (0=sagittal, 1=coronal, 2=axial)
        sample_every_n: Sample every Nth slice (1 = all slices)
        margin_slices: Extra slices to include before/after segmentation

    Returns:
        Path to manifest file
    """
    import slicer
    from PIL import Image

    segmentation_path = Path(segmentation_path).resolve()  # Absolute path
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation
    logger.info(f"Loading trial segmentation: {segmentation_path}")
    segmentation_node = slicer.util.loadSegmentation(str(segmentation_path))

    if segmentation_node is None:
        raise ValueError(f"Failed to load segmentation: {segmentation_path}")

    # Load gold standard if provided
    gold_node = None
    if gold_standard_path:
        gold_standard_path = Path(gold_standard_path).resolve()  # Absolute path
        logger.info(f"Loading gold standard: {gold_standard_path}")
        gold_node = slicer.util.loadSegmentation(str(gold_standard_path))

    # Get or load volume
    volume_node = None
    if volume_path:
        logger.info(f"Loading volume: {volume_path}")
        volume_node = slicer.util.loadVolume(str(volume_path))
    else:
        # Try to find associated volume from sample data
        if "MRBrainTumor1" in str(segmentation_path):
            import SampleData

            logger.info("Loading MRBrainTumor1 sample data")
            volume_node = SampleData.downloadSample("MRBrainTumor1")

    if volume_node is None:
        raise ValueError("No volume available - cannot generate images")

    # Get volume array
    volume_array = slicer.util.arrayFromVolume(volume_node)  # [k, j, i]
    logger.info(f"Volume shape: {volume_array.shape}")

    # Export segmentation aligned to volume geometry
    logger.info("Exporting trial segmentation to volume geometry...")
    seg_array = export_segmentation_to_labelmap(segmentation_node, volume_node)

    if seg_array is None:
        raise ValueError("Failed to export segmentation")

    logger.info(f"Segmentation shape: {seg_array.shape}")

    # Export gold standard if provided
    gold_array = None
    if gold_node:
        logger.info("Exporting gold standard to volume geometry...")
        gold_array = export_segmentation_to_labelmap(gold_node, volume_node)
        if gold_array is not None:
            logger.info(f"Gold standard shape: {gold_array.shape}")

    # Normalize volume for display
    volume_uint8 = normalize_to_uint8(volume_array)

    # Map axis to numpy axis
    numpy_axis = 2 - axis
    axis_names = {0: "sagittal", 1: "coronal", 2: "axial"}

    # Find slices with segmentation data
    n_slices = volume_array.shape[numpy_axis]
    slices_with_seg = []
    slices_with_gold = []

    for i in range(n_slices):
        if numpy_axis == 0:
            has_seg = np.any(seg_array[i, :, :] > 0)
            has_gold = gold_array is not None and np.any(gold_array[i, :, :] > 0)
        elif numpy_axis == 1:
            has_seg = np.any(seg_array[:, i, :] > 0)
            has_gold = gold_array is not None and np.any(gold_array[:, i, :] > 0)
        else:
            has_seg = np.any(seg_array[:, :, i] > 0)
            has_gold = gold_array is not None and np.any(gold_array[:, :, i] > 0)

        if has_seg:
            slices_with_seg.append(i)
        if has_gold:
            slices_with_gold.append(i)

    # Combine both for range calculation
    all_relevant_slices = set(slices_with_seg) | set(slices_with_gold)
    if not all_relevant_slices:
        raise ValueError("No slices contain segmentation data")

    # Expand range with margin
    min_slice = max(0, min(all_relevant_slices) - margin_slices)
    max_slice = min(n_slices - 1, max(all_relevant_slices) + margin_slices)

    slice_range = list(range(min_slice, max_slice + 1))
    if sample_every_n > 1:
        slice_range = slice_range[::sample_every_n]

    logger.info(
        f"Generating {len(slice_range)} {axis_names[axis]} slices ({min_slice}-{max_slice})"
    )

    # Create subdirectories for all visualization modes
    compare_smooth_dir = output_dir / "compare_smooth"
    compare_pixel_dir = output_dir / "compare_pixel"
    error_dir = output_dir / "error"
    compare_smooth_dir.mkdir(exist_ok=True)
    compare_pixel_dir.mkdir(exist_ok=True)
    error_dir.mkdir(exist_ok=True)

    # Prepare manifest
    images_list: list[dict] = []
    manifest: dict = {
        "generated": datetime.now().isoformat(),
        "segmentation_path": str(segmentation_path),
        "gold_standard_path": str(gold_standard_path) if gold_standard_path else None,
        "volume_path": str(volume_path) if volume_path else "MRBrainTumor1 (sample)",
        "axis": axis_names[axis],
        "volume_shape": list(volume_array.shape),
        "slices_with_trial": slices_with_seg,
        "slices_with_gold": slices_with_gold,
        "slice_range": [min_slice, max_slice],
        "total_slices": len(slice_range),
        "modes": {
            "compare_smooth": {
                "description": "Smooth contours via marching squares - shows interpolated boundary shape",
                "trial_color": "green contour (1px)",
                "gold_color": "magenta contour (1px)",
                "best_for": "Comparing overall boundary curves",
            },
            "compare_pixel": {
                "description": "Pixel-level outlines via morphology - shows actual voxel boundaries",
                "trial_color": "green outline (1px)",
                "gold_color": "magenta outline (1px)",
                "best_for": "Seeing exactly what voxels were segmented",
            },
            "error": {
                "description": "Error analysis - assumes gold standard is truth",
                "green": "Agreement (true positive)",
                "red": "Over-segmentation (false positive - trial only)",
                "blue": "Under-segmentation (false negative - gold only)",
            },
        },
        "images": images_list,
    }

    # Generate images in all modes
    for i, slice_idx in enumerate(slice_range):
        # Extract slices
        if numpy_axis == 0:
            img_slice = volume_uint8[slice_idx, :, :]
            trial_slice = seg_array[slice_idx, :, :]
            gold_slice = gold_array[slice_idx, :, :] if gold_array is not None else None
        elif numpy_axis == 1:
            img_slice = volume_uint8[:, slice_idx, :]
            trial_slice = seg_array[:, slice_idx, :]
            gold_slice = gold_array[:, slice_idx, :] if gold_array is not None else None
        else:
            img_slice = volume_uint8[:, :, slice_idx]
            trial_slice = seg_array[:, :, slice_idx]
            gold_slice = gold_array[:, :, slice_idx] if gold_array is not None else None

        filename = f"{slice_idx:04d}.png"

        # Mode 1: Smooth contours (marching squares)
        smooth_img = create_comparison_image_smooth(img_slice, trial_slice, gold_slice)
        Image.fromarray(smooth_img).save(compare_smooth_dir / filename)

        # Mode 2: Pixel outlines (morphological)
        pixel_img = create_comparison_image_pixel(img_slice, trial_slice, gold_slice)
        Image.fromarray(pixel_img).save(compare_pixel_dir / filename)

        # Mode 3: Error analysis (TP/FP/FN coloring)
        error_img = create_error_image(img_slice, trial_slice, gold_slice)
        Image.fromarray(error_img).save(error_dir / filename)

        has_trial = slice_idx in slices_with_seg
        has_gold = slice_idx in slices_with_gold

        images_list.append(
            {
                "filename": filename,
                "slice_index": int(slice_idx),
                "has_trial": has_trial,
                "has_gold": has_gold,
                "dimensions": list(smooth_img.shape[:2]),
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(f"  Saved {i + 1}/{len(slice_range)} images")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated {len(slice_range)} images")
    logger.info(f"Manifest: {manifest_path}")

    # Clean up nodes
    slicer.mrmlScene.RemoveNode(segmentation_node)
    if gold_node:
        slicer.mrmlScene.RemoveNode(gold_node)

    return manifest_path


# This module is meant to be imported by run_review_screenshots.py
# For CLI usage: Slicer --python-script scripts/run_review_screenshots.py 79
