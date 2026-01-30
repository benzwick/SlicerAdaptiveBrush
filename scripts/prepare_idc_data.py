#!/usr/bin/env python
"""Prepare IDC data for parameter optimization.

This script processes downloaded IDC DICOM data and creates:
1. Gold standard directories with segmentation files
2. Recipes with click locations
3. Optimization configs

Run with:
    uv run python scripts/prepare_idc_data.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()


def load_volume(volume_dir: Path) -> sitk.Image:
    """Load DICOM volume from directory."""
    reader = sitk.ImageSeriesReader()
    series_files = reader.GetGDCMSeriesFileNames(str(volume_dir))
    if not series_files:
        raise FileNotFoundError(f"No DICOM series found in {volume_dir}")
    reader.SetFileNames(series_files)
    return reader.Execute()


def load_segmentation(seg_dir: Path, reference_volume: sitk.Image) -> tuple[np.ndarray, list[str]]:
    """Load DICOM SEG and resample to reference volume geometry.

    Uses SimpleITK's resampling to properly align SEG with reference volume,
    handling all coordinate system transformations automatically.

    Args:
        seg_dir: Directory containing DICOM SEG file.
        reference_volume: Reference volume for geometry.

    Returns:
        Tuple of (labelmap_array, segment_names).
    """
    seg_files = list(seg_dir.glob("*.dcm"))
    if not seg_files:
        raise FileNotFoundError(f"No DICOM SEG files in {seg_dir}")

    # Read DICOM SEG using pydicom for segment names
    seg_dcm = pydicom.dcmread(seg_files[0])
    segment_names = [s.SegmentLabel for s in seg_dcm.SegmentSequence]
    logger.info(f"Found {len(segment_names)} segments: {segment_names}")

    # Get pixel array and build 3D volume from DICOM SEG
    seg_array = seg_dcm.pixel_array  # (frames, rows, cols)
    total_frames = seg_array.shape[0]
    logger.info(f"Seg array shape: {seg_array.shape}, total frames: {total_frames}")

    # Get SEG geometry from SharedFunctionalGroupsSequence
    shared_fg = seg_dcm.SharedFunctionalGroupsSequence[0]
    pixel_measures = shared_fg.PixelMeasuresSequence[0]
    seg_spacing_xy = [float(pixel_measures.PixelSpacing[0]), float(pixel_measures.PixelSpacing[1])]
    seg_slice_thickness = float(pixel_measures.SliceThickness)

    # Get orientation from PlaneOrientationSequence
    plane_orient = shared_fg.PlaneOrientationSequence[0]
    orientation = [float(x) for x in plane_orient.ImageOrientationPatient]

    # Extract all frame positions to determine z-range and ordering
    frame_positions = []
    for frame_fg in seg_dcm.PerFrameFunctionalGroupsSequence:
        pos = frame_fg.PlanePositionSequence[0].ImagePositionPatient
        frame_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

    frame_positions = np.array(frame_positions)

    # Compute slice direction from orientation vectors
    row_dir = np.array(orientation[:3])
    col_dir = np.array(orientation[3:])
    slice_dir = np.cross(row_dir, col_dir)

    # Project positions onto slice direction to get z-coordinates
    z_coords = frame_positions @ slice_dir

    # Sort frames by z-coordinate
    sorted_indices = np.argsort(z_coords)
    sorted_positions = frame_positions[sorted_indices]

    # Determine spacing between slices
    if len(sorted_positions) > 1:
        slice_spacing = np.abs(z_coords[sorted_indices[1]] - z_coords[sorted_indices[0]])
    else:
        slice_spacing = seg_slice_thickness

    # Build 3D segmentation volume
    # Combine all segments (OR operation)
    n_frames = len(sorted_indices)
    rows, cols = seg_array.shape[1], seg_array.shape[2]

    seg_3d = np.zeros((n_frames, rows, cols), dtype=np.uint8)
    for new_idx, old_idx in enumerate(sorted_indices):
        seg_3d[new_idx] = (seg_array[old_idx] > 0).astype(np.uint8)

    # Create SimpleITK image from segmentation
    seg_image = sitk.GetImageFromArray(seg_3d)

    # Set geometry
    seg_origin = sorted_positions[0].tolist()
    seg_spacing = [seg_spacing_xy[1], seg_spacing_xy[0], slice_spacing]  # col, row, slice

    # Direction: row_dir is along columns (x), col_dir is along rows (y), slice_dir is z
    direction = [
        col_dir[0],
        row_dir[0],
        slice_dir[0],
        col_dir[1],
        row_dir[1],
        slice_dir[1],
        col_dir[2],
        row_dir[2],
        slice_dir[2],
    ]

    seg_image.SetOrigin(seg_origin)
    seg_image.SetSpacing(seg_spacing)
    seg_image.SetDirection(direction)

    logger.info(f"SEG geometry: origin={seg_origin}, spacing={seg_spacing}")
    logger.info(
        f"REF geometry: origin={reference_volume.GetOrigin()}, spacing={reference_volume.GetSpacing()}"
    )

    # Resample segmentation to reference volume geometry
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_volume)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    resampled_seg = resampler.Execute(seg_image)

    # Convert to numpy array
    labelmap = sitk.GetArrayFromImage(resampled_seg).astype(np.uint8)

    # Verify alignment
    foreground_slices = np.sum(np.any(labelmap > 0, axis=(1, 2)))
    logger.info(
        f"Resampled labelmap shape: {labelmap.shape}, {foreground_slices} slices with foreground"
    )

    if foreground_slices == 0:
        logger.warning("No foreground voxels after resampling - geometry mismatch?")

    return labelmap, segment_names


def generate_clicks_from_dicom_seg(seg_dir: Path, n_clicks: int = 5, seed: int = 42) -> list[dict]:
    """Generate click locations from DICOM SEG interior points.

    Args:
        seg_dir: Directory containing DICOM SEG file.
        n_clicks: Number of clicks to generate.
        seed: Random seed.

    Returns:
        List of click dictionaries with 'ras' coordinates.
    """
    from scipy import ndimage

    rng = np.random.default_rng(seed)

    seg_files = list(seg_dir.glob("*.dcm"))
    if not seg_files:
        raise FileNotFoundError(f"No DICOM SEG files in {seg_dir}")

    seg_dcm = pydicom.dcmread(seg_files[0])
    seg_array = seg_dcm.pixel_array

    # Get orientation from SharedFunctionalGroupsSequence
    shared_fg = seg_dcm.SharedFunctionalGroupsSequence[0]
    plane_orient = shared_fg.PlaneOrientationSequence[0]
    orientation = [float(x) for x in plane_orient.ImageOrientationPatient]
    row_dir = np.array(orientation[:3])
    col_dir = np.array(orientation[3:])

    # Get pixel spacing
    pixel_measures = shared_fg.PixelMeasuresSequence[0]
    pixel_spacing = [float(pixel_measures.PixelSpacing[0]), float(pixel_measures.PixelSpacing[1])]

    # Collect interior foreground positions
    foreground_positions = []

    for frame_idx, frame_fg in enumerate(seg_dcm.PerFrameFunctionalGroupsSequence):
        frame_data = seg_array[frame_idx]
        if not np.any(frame_data > 0):
            continue

        # Erode to find interior points
        eroded = ndimage.binary_erosion(frame_data > 0, iterations=2)
        if not np.any(eroded):
            eroded = frame_data > 0

        plane_pos = frame_fg.PlanePositionSequence[0]
        frame_origin = np.array([float(x) for x in plane_pos.ImagePositionPatient])

        for row, col in np.argwhere(eroded):
            world_pos = (
                frame_origin + col * pixel_spacing[1] * row_dir + row * pixel_spacing[0] * col_dir
            )
            foreground_positions.append(world_pos)

    if len(foreground_positions) == 0:
        logger.warning("No foreground voxels in DICOM SEG")
        return []

    positions_array = np.array(foreground_positions)
    logger.info(f"Found {len(positions_array)} interior positions")

    # Weight by distance from centroid (prefer central points)
    centroid = positions_array.mean(axis=0)
    distances = np.linalg.norm(positions_array - centroid, axis=1)
    weights = 1.0 / (distances + 1.0)
    weights = weights / weights.sum()

    # Sample positions
    n_available = min(n_clicks, len(positions_array))
    selected = rng.choice(len(positions_array), size=n_available, replace=False, p=weights)

    clicks = []
    for idx in selected:
        lps = positions_array[idx]
        ras = [-lps[0], -lps[1], lps[2]]  # LPS to RAS
        clicks.append({"ras": ras})

    logger.info(f"Generated {len(clicks)} click locations")
    return clicks


# Dataset-specific configurations
DATASET_CONFIGS = {
    "ct_lung": {"n_clicks": 8, "brush_radius_mm": 20.0},
    "ct_bone": {"n_clicks": 6, "brush_radius_mm": 12.0},
    "ct_soft_tissue": {"n_clicks": 8, "brush_radius_mm": 10.0},
    "ct_vessel_contrast": {"n_clicks": 6, "brush_radius_mm": 18.0},
    "mri_t2_lesion": {"n_clicks": 5, "brush_radius_mm": 12.0},
    "mri_t1gd_tumor": {"n_clicks": 6, "brush_radius_mm": 18.0},
    "pet_tumor": {"n_clicks": 6, "brush_radius_mm": 25.0},
}


def create_gold_standard(
    preset_name: str,
    labelmap: np.ndarray,
    volume: sitk.Image,
    clicks: list[dict],
    idc_info: dict,
    segment_names: list[str],
) -> Path:
    """Create gold standard directory with metadata only (DICOM SEG loaded directly)."""
    gold_dir = (
        PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester" / "GoldStandards" / f"idc_{preset_name}"
    )
    gold_dir.mkdir(parents=True, exist_ok=True)

    # No NRRD conversion - DICOM SEG will be loaded directly by Slicer

    # Compute statistics
    voxel_count = int(np.sum(labelmap > 0))

    # Create metadata (source=IDC signals to load from DICOM)
    metadata = {
        "name": f"idc_{preset_name}",
        "description": f"IDC gold standard for {preset_name} preset",
        "source": "NCI Imaging Data Commons",
        "idc_collection": idc_info.get("collection", "unknown"),
        "idc_patient": idc_info.get("patient", "unknown"),
        "preset": preset_name,
        "created_at": datetime.now().isoformat(),
        "clicks": clicks,
        "parameters": {
            "brush_radius_mm": 15.0,
        },
        "statistics": {
            "voxel_count": voxel_count,
            "segment_names": segment_names,
        },
    }

    metadata_file = gold_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created gold standard metadata: {gold_dir}")
    return gold_dir


def create_recipe(preset_name: str, clicks: list[dict]) -> Path:
    """Create recipe file for preset."""
    recipes_dir = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester" / "recipes"
    recipe_file = recipes_dir / f"idc_{preset_name}.py"

    click_lines = []
    for click in clicks:
        ras = click["ras"]
        click_lines.append(f"    effect.paintAt({ras[0]:.2f}, {ras[1]:.2f}, {ras[2]:.2f})")

    click_code = "\n".join(click_lines)

    content = f'''"""
Recipe: idc_{preset_name}

Auto-generated recipe for {preset_name} parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/{preset_name}"
segment_name = "{preset_name.replace("_", " ").title()}"
gold_standard = "idc_{preset_name}"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("{preset_name}")
    effect.brushRadiusMm = 15.0

{click_code}
'''

    with open(recipe_file, "w") as f:
        f.write(content)

    logger.info(f"Created recipe: {recipe_file}")
    return recipe_file


def create_config(preset_name: str) -> Path:
    """Create optimization config for preset."""
    configs_dir = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester" / "configs"
    config_file = configs_dir / f"idc_{preset_name}.yaml"

    content = f"""# IDC Optimization: {preset_name}
version: "1.0"
name: "IDC {preset_name.replace("_", " ").title()}"
description: "Parameter optimization for {preset_name} using IDC data"

settings:
  n_trials: 30
  pruning: true
  pruner: "hyperband"
  sampler: "tpe"
  primary_metric: "dice"
  save_segmentations: true
  save_screenshots: true

recipes:
  - path: "recipes/idc_{preset_name}.py"
    gold_standard: "GoldStandards/idc_{preset_name}/gold.seg.nrrd"
    dicom_source: "idc_data/{preset_name}"

parameter_space:
  global:
    edge_sensitivity:
      type: int
      range: [20, 80]
      step: 10
    threshold_zone:
      type: int
      range: [30, 70]
      step: 10
    brush_radius_mm:
      type: float
      range: [8.0, 30.0]
      step: 2.0

  algorithm_substitution:
    enabled: true
    candidates:
      - geodesic_distance
      - watershed
      - random_walker
      - connected_threshold
      - region_growing
      - threshold_brush

  algorithms:
    geodesic_distance:
      geodesic_edge_weight:
        type: float
        range: [4.0, 15.0]
    watershed:
      watershed_gradient_scale:
        type: float
        range: [0.5, 2.5]
    random_walker:
      random_walker_beta:
        type: float
        range: [50.0, 200.0]
    region_growing:
      region_growing_multiplier:
        type: float
        range: [1.5, 4.0]
    connected_threshold:
      pass: true
    threshold_brush:
      pass: true

output:
  reports: [json, markdown]
  algorithm_profiles: true
"""

    with open(config_file, "w") as f:
        f.write(content)

    logger.info(f"Created config: {config_file}")
    return config_file


def process_dataset(preset_name: str, info: dict) -> dict:
    """Process a single IDC dataset."""
    preset_dir = PROJECT_ROOT / "idc_data" / preset_name

    # Find directories
    volume_dir = None
    seg_dir = None
    for subdir in preset_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.startswith(("CT_", "MR_", "PT_")):
            volume_dir = subdir
        elif subdir.name.startswith("SEG_"):
            seg_dir = subdir

    if not volume_dir or not seg_dir:
        raise FileNotFoundError(f"Missing volume or seg directory in {preset_dir}")

    # Load data
    logger.info(f"Loading volume from {volume_dir.name}")
    volume = load_volume(volume_dir)
    logger.info(f"Volume size: {volume.GetSize()}, spacing: {volume.GetSpacing()}")

    logger.info(f"Loading segmentation from {seg_dir.name}")
    labelmap, segment_names = load_segmentation(seg_dir, volume)
    logger.info(f"Labelmap shape: {labelmap.shape}, voxels: {np.sum(labelmap > 0)}")

    # Generate clicks directly from DICOM SEG (matches Slicer's DICOM loading)
    clicks = generate_clicks_from_dicom_seg(seg_dir, n_clicks=5)

    # Create gold standard
    gold_dir = create_gold_standard(preset_name, labelmap, volume, clicks, info, segment_names)

    # Create recipe and config
    recipe_path = create_recipe(preset_name, clicks)
    config_path = create_config(preset_name)

    return {
        "status": "success",
        "gold_dir": str(gold_dir),
        "recipe": str(recipe_path),
        "config": str(config_path),
        "n_clicks": len(clicks),
        "voxels": int(np.sum(labelmap > 0)),
    }


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Preparing IDC data for parameter optimization")
    logger.info("=" * 60)

    # Load manifest
    manifest_file = PROJECT_ROOT / "idc_optimization_data.json"
    with open(manifest_file) as f:
        idc_data = json.load(f)

    datasets = idc_data.get("datasets", {})
    logger.info(f"Found {len(datasets)} datasets")

    results = {}
    for preset_name, info in datasets.items():
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Processing: {preset_name}")
        logger.info("=" * 60)

        try:
            results[preset_name] = process_dataset(preset_name, info)
            logger.info(f"✓ {preset_name} complete")
        except Exception as e:
            logger.exception(f"✗ {preset_name} failed: {e}")
            results[preset_name] = {"status": "error", "error": str(e)}

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    success = [k for k, v in results.items() if v.get("status") == "success"]
    failed = [k for k, v in results.items() if v.get("status") != "success"]

    logger.info(f"Successful: {len(success)}")
    for name in success:
        logger.info(f"  ✓ {name}")

    if failed:
        logger.info(f"Failed: {len(failed)}")
        for name in failed:
            logger.info(f"  ✗ {name}: {results[name].get('error', 'unknown')}")

    # Save results
    results_file = PROJECT_ROOT / "idc_preparation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Print run commands
    if success:
        logger.info("\nTo run optimization in Slicer:")
        for name in success[:3]:  # Show first 3
            logger.info(
                f"  $SLICER_PATH --python-script scripts/run_optimization.py "
                f"SegmentEditorAdaptiveBrushTester/configs/idc_{name}.yaml"
            )


if __name__ == "__main__":
    main()
