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


def get_outline(mask: np.ndarray) -> np.ndarray:
    """Get outline of binary mask."""
    from scipy import ndimage

    dilated = ndimage.binary_dilation(mask, iterations=1)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    return dilated & ~eroded


def create_overlay_image(
    image_slice: np.ndarray,
    seg_slice: np.ndarray | None = None,
    gold_slice: np.ndarray | None = None,
    seg_color: tuple[int, int, int] = (0, 255, 0),  # Green for trial
    gold_color: tuple[int, int, int] = (255, 255, 0),  # Yellow for gold outline
    seg_alpha: float = 0.4,
) -> np.ndarray:
    """Create RGB image with segmentation overlay and gold standard outline.

    Args:
        image_slice: 2D grayscale image (uint8)
        seg_slice: 2D binary segmentation mask (trial result)
        gold_slice: 2D binary gold standard mask (shown as outline)
        seg_color: RGB color for trial segmentation
        gold_color: RGB color for gold standard outline
        seg_alpha: Opacity of trial segmentation fill

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    # Convert grayscale to RGB
    rgb = np.stack([image_slice, image_slice, image_slice], axis=-1).astype(np.float64)

    # Apply trial segmentation as semi-transparent fill
    if seg_slice is not None and np.any(seg_slice > 0):
        mask = seg_slice > 0
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask, (1 - seg_alpha) * rgb[:, :, c] + seg_alpha * seg_color[c], rgb[:, :, c]
            )

    # Apply gold standard as outline only
    if gold_slice is not None and np.any(gold_slice > 0):
        outline = get_outline(gold_slice > 0)
        for c in range(3):
            rgb[:, :, c] = np.where(outline, gold_color[c], rgb[:, :, c])

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
    seg_color: tuple[int, int, int] = (0, 255, 0),  # Green for trial
    gold_color: tuple[int, int, int] = (255, 255, 0),  # Yellow for gold
    seg_alpha: float = 0.4,
):
    """Generate slice images at native resolution for VLM review.

    Args:
        segmentation_path: Path to segmentation file (trial result)
        output_dir: Directory for output images
        gold_standard_path: Path to gold standard segmentation (shown as outline)
        volume_path: Path to volume (optional, uses sample data if needed)
        axis: Slice axis (0=sagittal, 1=coronal, 2=axial)
        sample_every_n: Sample every Nth slice (1 = all slices)
        margin_slices: Extra slices to include before/after segmentation
        seg_color: RGB color for trial segmentation (default green)
        gold_color: RGB color for gold standard outline (default yellow)
        seg_alpha: Opacity of trial segmentation (0-1)

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
        "legend": {
            "trial_segmentation": f"RGB{seg_color} semi-transparent fill",
            "gold_standard": f"RGB{gold_color} outline" if gold_standard_path else None,
        },
        "images": images_list,
    }

    # Generate images
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

        # Create overlay
        rgb = create_overlay_image(
            img_slice, trial_slice, gold_slice, seg_color, gold_color, seg_alpha
        )

        # Save image
        filename = f"{slice_idx:04d}.png"
        filepath = output_dir / filename

        img = Image.fromarray(rgb)
        img.save(filepath)

        has_trial = slice_idx in slices_with_seg
        has_gold = slice_idx in slices_with_gold

        images_list.append(
            {
                "filename": filename,
                "slice_index": int(slice_idx),
                "has_trial": has_trial,
                "has_gold": has_gold,
                "dimensions": list(rgb.shape[:2]),
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
