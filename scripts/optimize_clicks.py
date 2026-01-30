#!/usr/bin/env python
"""First-pass optimization: find best click count, locations, and brush size per algorithm.

Run with:
    Slicer --python-script scripts/optimize_clicks.py <dataset_name> [--trials N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("optimize_clicks")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()

ALGORITHMS = [
    "geodesic_distance",
    "watershed",
    "region_growing",
    "connected_threshold",
    "threshold_brush",
]


def load_dicom_data(dataset_name: str):
    """Load DICOM volume and segmentation."""
    import slicer
    from DICOMLib import DICOMUtils

    dicom_dir = PROJECT_ROOT / "idc_data" / dataset_name

    volume_subdir = None
    seg_subdir = None
    for subdir in dicom_dir.iterdir():
        if subdir.is_dir():
            if subdir.name.startswith(("CT_", "MR_", "PT_")):
                volume_subdir = subdir
            elif subdir.name.startswith("SEG_"):
                seg_subdir = subdir

    if not volume_subdir or not seg_subdir:
        raise FileNotFoundError(f"Missing data in {dicom_dir}")

    slicer.mrmlScene.Clear(0)

    # Load volume
    logger.info(f"Loading volume from {volume_subdir.name}")
    volume_node = None
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(str(volume_subdir), db)
        for patient in db.patients():
            for study in db.studiesForPatient(patient):
                for series in db.seriesForStudy(study):
                    files = db.filesForSeries(series)
                    plugin = slicer.modules.dicomPlugins["DICOMScalarVolumePlugin"]()
                    loadables = plugin.examine([files])
                    if loadables:
                        volume_node = plugin.load(loadables[0])
                        if volume_node:
                            break

    if not volume_node:
        raise RuntimeError("Failed to load volume")
    logger.info(f"Loaded volume: {volume_node.GetName()}")

    # Load segmentation
    logger.info(f"Loading segmentation from {seg_subdir.name}")
    num_seg_before = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode")

    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(str(seg_subdir), db)
        for patient in db.patients():
            for study in db.studiesForPatient(patient):
                for series in db.seriesForStudy(study):
                    files = db.filesForSeries(series)
                    if "DICOMSegmentationPlugin" in slicer.modules.dicomPlugins:
                        plugin = slicer.modules.dicomPlugins["DICOMSegmentationPlugin"]()
                        loadables = plugin.examine([files])
                        if loadables:
                            plugin.load(loadables[0])
                            break

    num_seg_after = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode")
    if num_seg_after <= num_seg_before:
        raise RuntimeError("Failed to load segmentation")

    gold_seg_node = slicer.mrmlScene.GetNthNodeByClass(num_seg_after - 1, "vtkMRMLSegmentationNode")
    logger.info(f"Loaded gold standard: {gold_seg_node.GetName()}")

    return volume_node, gold_seg_node


def setup_effect(segmentation_node, volume_node):
    """Set up segment editor and activate Adaptive Brush."""
    import slicer

    slicer.util.selectModule("SegmentEditor")
    slicer.app.processEvents()

    segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
    segment_editor_widget = segment_editor_module.editor

    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setSourceVolumeNode(volume_node)
    slicer.app.processEvents()

    segment_editor_widget.setActiveEffectByName("Adaptive Brush")
    effect = segment_editor_widget.activeEffect()

    if effect is None:
        raise RuntimeError("Failed to activate Adaptive Brush effect")

    return effect.self(), segment_editor_widget


def generate_clicks(gold_seg_node, volume_node, n_clicks: int, seed: int) -> list[list[float]]:
    """Generate click RAS coordinates from segmentation interior."""
    import slicer
    import vtk
    from scipy import ndimage

    rng = np.random.default_rng(seed)

    # Get labelmap array from first segment
    seg = gold_seg_node.GetSegmentation()
    segment_id = seg.GetNthSegmentID(0)
    labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(gold_seg_node, segment_id, volume_node)

    if labelmap is None or not np.any(labelmap > 0):
        logger.warning("Empty labelmap")
        return []

    # Erode to find interior
    eroded = ndimage.binary_erosion(labelmap > 0, iterations=2)
    if not np.any(eroded):
        eroded = labelmap > 0

    foreground = np.argwhere(eroded)
    if len(foreground) == 0:
        return []

    # Weight by distance from centroid
    centroid = foreground.mean(axis=0)
    distances = np.linalg.norm(foreground - centroid, axis=1)
    weights = 1.0 / (distances + 1.0)
    weights = weights / weights.sum()

    n_available = min(n_clicks, len(foreground))
    selected = rng.choice(len(foreground), size=n_available, replace=False, p=weights)

    # Convert to RAS
    ijkToRas = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijkToRas)

    clicks = []
    for idx in selected:
        k, j, i = foreground[idx]
        ijk = [float(i), float(j), float(k), 1.0]
        ras = [0.0, 0.0, 0.0, 1.0]
        ijkToRas.MultiplyPoint(ijk, ras)
        clicks.append([ras[0], ras[1], ras[2]])

    return clicks


def clear_segment(segmentation_node, segment_id: str) -> str:
    """Clear segment by removing and re-adding."""
    segmentation = segmentation_node.GetSegmentation()
    segment = segmentation.GetSegment(segment_id)
    segment_name = segment.GetName()
    color = segment.GetColor()
    segmentation.RemoveSegment(segment_id)
    new_segment_id = segmentation.AddEmptySegment(segment_name, segment_name, color)
    return str(new_segment_id)


def compute_dice_single(test_arr, gold_arr) -> float:
    """Compute Dice coefficient between two binary arrays."""
    test_binary = test_arr > 0
    gold_binary = gold_arr > 0

    intersection = int(np.sum(test_binary & gold_binary))
    union = int(np.sum(test_binary)) + int(np.sum(gold_binary))

    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


def compute_dice(
    test_seg_node, test_segment_id, gold_seg_node, gold_segment_id, volume_node
) -> float:
    """Compute Dice coefficient.

    For multi-segment gold standards, computes Dice against each segment
    independently and returns the maximum (best matching segment).
    """
    import slicer

    try:
        test_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
            test_seg_node, test_segment_id, volume_node
        )

        if test_arr is None or not np.any(test_arr > 0):
            return 0.0

        # Get all segment IDs from gold standard
        gold_segmentation = gold_seg_node.GetSegmentation()
        num_segments = gold_segmentation.GetNumberOfSegments()

        if num_segments == 1:
            # Single segment - use directly
            gold_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                gold_seg_node, gold_segment_id, volume_node
            )
            if gold_arr is None:
                return 0.0
            return compute_dice_single(test_arr, gold_arr)

        # Multi-segment - compute Dice for each and return best match
        best_dice = 0.0

        for i in range(num_segments):
            seg_id = gold_segmentation.GetNthSegmentID(i)
            gold_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                gold_seg_node, seg_id, volume_node
            )

            if gold_arr is None or not np.any(gold_arr > 0):
                continue

            dice = compute_dice_single(test_arr, gold_arr)
            if dice > best_dice:
                best_dice = dice

        return best_dice

    except Exception as e:
        logger.error(f"Dice error: {e}")
        return 0.0


def run_trial(
    effect,
    segment_editor_widget,
    segmentation_node,
    segment_id: str,
    gold_seg_node,
    volume_node,
    clicks: list[list[float]],
    algorithm: str,
    brush_radius: float,
) -> tuple[float, str]:
    """Run a single trial."""
    import slicer

    # Clear segment
    segment_id = clear_segment(segmentation_node, segment_id)

    # Select the segment
    segment_editor_widget.setCurrentSegmentID(segment_id)
    slicer.app.processEvents()

    # Set algorithm
    effect.setAlgorithm(algorithm)

    # Set brush radius
    effect.brushRadiusMm = brush_radius
    slicer.app.processEvents()

    # Paint at each click
    for ras in clicks:
        effect.paintAt(ras[0], ras[1], ras[2])
        slicer.app.processEvents()

    # Compute Dice
    gold_segment_id = gold_seg_node.GetSegmentation().GetNthSegmentID(0)
    dice = compute_dice(segmentation_node, segment_id, gold_seg_node, gold_segment_id, volume_node)

    return dice, segment_id


def optimize_dataset(dataset_name: str, n_trials_per_algo: int = 12):
    """Optimize click parameters for a dataset."""
    import slicer

    logger.info(f"{'=' * 60}")
    logger.info(f"Optimizing: {dataset_name}")
    logger.info(f"{'=' * 60}")

    # Load data
    volume_node, gold_seg_node = load_dicom_data(dataset_name)

    # Create test segmentation
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
    segment_id = segmentation_node.GetSegmentation().AddEmptySegment("Test")

    # Set up effect
    effect, segment_editor_widget = setup_effect(segmentation_node, volume_node)
    logger.info("Adaptive Brush activated")

    # Parameter ranges
    n_clicks_options = [3, 4, 5, 6, 8, 10]
    brush_radius_options = [8.0, 12.0, 15.0, 18.0, 22.0, 28.0]
    seed_options = list(range(5))

    rng = np.random.default_rng(42)
    best_per_algorithm = {}

    for algorithm in ALGORITHMS:
        logger.info(f"\n--- Testing: {algorithm} ---")
        best_dice = 0.0
        best_params = None

        for trial in range(n_trials_per_algo):
            n_clicks = int(rng.choice(n_clicks_options))
            brush_radius = float(rng.choice(brush_radius_options))
            seed = int(rng.choice(seed_options))

            clicks = generate_clicks(gold_seg_node, volume_node, n_clicks, seed)
            if not clicks:
                logger.warning("No clicks generated")
                continue

            try:
                dice, segment_id = run_trial(
                    effect,
                    segment_editor_widget,
                    segmentation_node,
                    segment_id,
                    gold_seg_node,
                    volume_node,
                    clicks,
                    algorithm,
                    brush_radius,
                )

                logger.info(
                    f"  {algorithm} trial {trial + 1}: n={n_clicks}, r={brush_radius}mm, "
                    f"seed={seed} -> Dice={dice:.4f}"
                )

                if dice > best_dice:
                    best_dice = dice
                    best_params = {
                        "algorithm": algorithm,
                        "dice": float(dice),
                        "n_clicks": n_clicks,
                        "brush_radius_mm": brush_radius,
                        "seed": seed,
                        "clicks": [[float(c) for c in click] for click in clicks],
                    }

            except Exception as e:
                logger.warning(f"  Trial failed: {e}")

        if best_params:
            best_per_algorithm[algorithm] = best_params
            logger.info(f"  BEST {algorithm}: Dice={best_dice:.4f}")

    # Save results
    output_dir = PROJECT_ROOT / "click_optimization_results" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "best_per_algorithm.json", "w") as f:
        json.dump(best_per_algorithm, f, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY: {dataset_name}")
    logger.info(f"{'=' * 60}")
    for alg, params in sorted(
        best_per_algorithm.items(),
        key=lambda x: -float(x[1]["dice"]),  # type: ignore[arg-type]
    ):
        logger.info(
            f"  {alg}: Dice={params['dice']:.4f}, "
            f"n_clicks={params['n_clicks']}, brush={params['brush_radius_mm']}mm"
        )

    logger.info(f"\nSaved to: {output_dir}")
    return best_per_algorithm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--trials", type=int, default=12, help="Trials per algorithm")
    args = parser.parse_args()

    try:
        optimize_dataset(args.dataset, args.trials)
    except Exception as e:
        logger.exception(f"Failed: {e}")
        sys.exit(1)

    import slicer

    slicer.app.quit()


if __name__ == "__main__":
    main()
