#!/usr/bin/env python
"""Explore Slicer sample data to find good seed points for gold standards.

Run via:
    Slicer --python-script scripts/explore_sample_data.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_volume_info(volume_node):
    """Get basic info about a volume."""
    import slicer

    arr = slicer.util.arrayFromVolume(volume_node)
    spacing = volume_node.GetSpacing()
    origin = volume_node.GetOrigin()
    dims = volume_node.GetImageData().GetDimensions()

    return {
        "name": volume_node.GetName(),
        "dimensions": dims,
        "spacing": spacing,
        "origin": origin,
        "intensity_range": [float(arr.min()), float(arr.max())],
        "intensity_mean": float(arr.mean()),
        "intensity_std": float(arr.std()),
    }


def ijk_to_ras(volume_node, ijk):
    """Convert IJK coordinates to RAS."""
    import vtk

    ijk_to_ras_matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras_matrix)

    ijk_homogeneous = [ijk[0], ijk[1], ijk[2], 1.0]
    ras_homogeneous = [0.0, 0.0, 0.0, 0.0]

    for i in range(4):
        for j in range(4):
            ras_homogeneous[i] += ijk_to_ras_matrix.GetElement(i, j) * ijk_homogeneous[j]

    return [ras_homogeneous[0], ras_homogeneous[1], ras_homogeneous[2]]


def find_structure_seeds(volume_node, structure_type):
    """Find good seed points for a structure type."""
    import numpy as np
    import slicer

    arr = slicer.util.arrayFromVolume(volume_node)
    dims = arr.shape  # (K, J, I) in numpy

    seeds = []

    if structure_type == "ventricles":
        # Ventricles are dark (CSF) in T1 MRI, roughly in center of brain
        # Look for dark regions in central slices
        center_k = dims[0] // 2
        center_j = dims[1] // 2
        center_i = dims[2] // 2

        # Search around center for low intensity
        threshold = np.percentile(arr, 20)  # Dark regions
        for dk in range(-10, 11, 5):
            for dj in range(-10, 11, 5):
                k, j, i = center_k + dk, center_j + dj, center_i
                if 0 <= k < dims[0] and 0 <= j < dims[1]:
                    if arr[k, j, i] < threshold:
                        ras = ijk_to_ras(volume_node, [i, j, k])
                        seeds.append(
                            {"ijk": [i, j, k], "ras": ras, "intensity": float(arr[k, j, i])}
                        )

    elif structure_type == "white_matter":
        # White matter is bright in T1 MRI
        center_k = dims[0] // 2
        threshold = np.percentile(arr, 70)  # Bright regions

        # Sample in frontal region
        for dj in range(-20, 21, 10):
            for di in range(-20, 21, 10):
                j = dims[1] // 2 + dj
                i = dims[2] // 2 + di
                k = center_k
                if 0 <= j < dims[1] and 0 <= i < dims[2]:
                    if arr[k, j, i] > threshold:
                        ras = ijk_to_ras(volume_node, [i, j, k])
                        seeds.append(
                            {"ijk": [i, j, k], "ras": ras, "intensity": float(arr[k, j, i])}
                        )

    elif structure_type == "lung":
        # Lung is very dark on CT (air ~ -1000 HU)
        threshold = -500  # Air threshold
        center_k = dims[0] // 2

        # Search for air regions (avoid trachea in center)
        for side in [-1, 1]:  # Left and right lung
            for dk in range(-20, 21, 10):
                k = center_k + dk
                j = dims[1] // 2
                i = dims[2] // 2 + side * (dims[2] // 4)  # Offset to side
                if 0 <= k < dims[0] and 0 <= i < dims[2]:
                    if arr[k, j, i] < threshold:
                        ras = ijk_to_ras(volume_node, [i, j, k])
                        seeds.append(
                            {"ijk": [i, j, k], "ras": ras, "intensity": float(arr[k, j, i])}
                        )

    elif structure_type == "bone":
        # Bone is bright on CT (> 300 HU typically)
        threshold = 200
        center_k = dims[0] // 2

        # Search for bone (vertebra in center-posterior)
        for dk in range(-10, 11, 5):
            k = center_k + dk
            j = int(dims[1] * 0.3)  # Posterior
            i = dims[2] // 2  # Center
            if 0 <= k < dims[0]:
                if arr[k, j, i] > threshold:
                    ras = ijk_to_ras(volume_node, [i, j, k])
                    seeds.append({"ijk": [i, j, k], "ras": ras, "intensity": float(arr[k, j, i])})

    elif structure_type == "tumor":
        # Tumor - similar intensity to gray matter, look for enhancement
        center_k = dims[0] // 2
        mid_intensity = np.percentile(arr, 50)

        # Search around center
        for dk in range(-15, 16, 5):
            for dj in range(-15, 16, 5):
                for di in range(-15, 16, 5):
                    k = center_k + dk
                    j = dims[1] // 2 + dj
                    i = dims[2] // 2 + di
                    if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2]:
                        val = arr[k, j, i]
                        if mid_intensity * 0.8 < val < mid_intensity * 1.2:
                            ras = ijk_to_ras(volume_node, [i, j, k])
                            seeds.append({"ijk": [i, j, k], "ras": ras, "intensity": float(val)})

    elif structure_type == "left_ventricle":
        # Left ventricle in cardiac CT - contrast enhanced, center-left
        center_k = dims[0] // 2
        threshold = np.percentile(arr, 80)  # Bright (contrast)

        for dk in range(-10, 11, 5):
            k = center_k + dk
            j = dims[1] // 2
            i = int(dims[2] * 0.6)  # Slightly left of center
            if 0 <= k < dims[0]:
                if arr[k, j, i] > threshold:
                    ras = ijk_to_ras(volume_node, [i, j, k])
                    seeds.append({"ijk": [i, j, k], "ras": ras, "intensity": float(arr[k, j, i])})

    # Return top 5 seeds by intensity criteria
    return seeds[:5] if seeds else []


def explore_dataset(sample_name, structures):
    """Explore a sample dataset and find seed points."""
    import SampleData
    import slicer

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Exploring: {sample_name}")
    logger.info("=" * 60)

    # Load sample data
    slicer.mrmlScene.Clear(0)
    volume_node = SampleData.downloadSample(sample_name)

    if volume_node is None:
        logger.error(f"Failed to load {sample_name}")
        return None

    # Get volume info
    info = get_volume_info(volume_node)
    logger.info(f"Dimensions: {info['dimensions']}")
    logger.info(f"Spacing: {info['spacing']}")
    logger.info(f"Intensity range: {info['intensity_range']}")
    logger.info(f"Intensity mean/std: {info['intensity_mean']:.1f} / {info['intensity_std']:.1f}")

    # Find seeds for each structure
    results = {"sample_data": sample_name, "volume_info": info, "structures": {}}

    for structure in structures:
        logger.info(f"\nFinding seeds for: {structure}")
        seeds = find_structure_seeds(volume_node, structure)
        results["structures"][structure] = seeds
        logger.info(f"  Found {len(seeds)} potential seeds")
        for i, seed in enumerate(seeds[:3]):
            logger.info(
                f"    {i + 1}. RAS: [{seed['ras'][0]:.1f}, {seed['ras'][1]:.1f}, {seed['ras'][2]:.1f}], intensity: {seed['intensity']:.1f}"
            )

    return results


def main():
    """Main entry point."""
    import slicer

    datasets = [
        ("MRHead", ["ventricles", "white_matter"]),
        ("CTChest", ["lung", "bone"]),
        ("MRBrainTumor2", ["tumor"]),
        ("CTACardio", ["left_ventricle"]),
    ]

    all_results = {}

    for sample_name, structures in datasets:
        try:
            results = explore_dataset(sample_name, structures)
            if results:
                all_results[sample_name] = results
        except Exception as e:
            logger.error(f"Error exploring {sample_name}: {e}")

    # Save results
    output_path = Path(__file__).parent.parent / "gold_standard_seeds.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    # Exit Slicer
    slicer.app.quit()


if __name__ == "__main__":
    main()
