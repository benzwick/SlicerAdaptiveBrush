#!/usr/bin/env python
"""Find tumor coordinates in MRBrainTumor1 by intensity analysis.

Run in Slicer:
    Slicer --python-script scripts/find_tumor_coordinates.py
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    """Find bright regions that might be the tumor."""
    import SampleData
    import slicer

    # Load MRBrainTumor1
    print("Loading MRBrainTumor1...")
    volume_node = SampleData.downloadSample("MRBrainTumor1")

    # Get the image array
    array = slicer.util.arrayFromVolume(volume_node)
    print(f"Array shape: {array.shape}")
    print(f"Array dtype: {array.dtype}")
    print(f"Intensity range: {array.min()} to {array.max()}")

    # Find the brightest region (tumor is hyperintense)
    threshold = np.percentile(array, 99)  # Top 1% brightest
    print(f"99th percentile threshold: {threshold}")

    # Find coordinates of bright voxels
    bright_voxels = np.where(array > threshold)

    if len(bright_voxels[0]) > 0:
        # Convert IJK to RAS
        # Get the transformation matrix
        ras_to_ijk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(ras_to_ijk)

        ijk_to_ras = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ras_to_ijk, ijk_to_ras)

        # Find centroid of bright region
        k_coords = bright_voxels[0]  # Slices (S)
        j_coords = bright_voxels[1]  # Rows (A)
        i_coords = bright_voxels[2]  # Columns (R)

        # Find centroid in IJK
        i_center = np.mean(i_coords)
        j_center = np.mean(j_coords)
        k_center = np.mean(k_coords)

        print(f"\nBright region centroid (IJK): ({i_center:.1f}, {j_center:.1f}, {k_center:.1f})")

        # Convert to RAS
        ijk_point = [i_center, j_center, k_center, 1]
        ras_point = [0, 0, 0, 1]
        ijk_to_ras.MultiplyPoint(ijk_point, ras_point)

        print(
            f"Bright region centroid (RAS): ({ras_point[0]:.1f}, {ras_point[1]:.1f}, {ras_point[2]:.1f})"
        )

        # Find the bounds
        print("\nBright region bounds (IJK):")
        print(f"  I: {i_coords.min()} to {i_coords.max()}")
        print(f"  J: {j_coords.min()} to {j_coords.max()}")
        print(f"  K: {k_coords.min()} to {k_coords.max()}")

        # Sample some specific bright points
        print("\nSample bright voxel locations (RAS):")
        num_samples = min(10, len(i_coords))
        indices = np.random.choice(len(i_coords), num_samples, replace=False)

        for idx in indices:
            i, j, k = i_coords[idx], j_coords[idx], k_coords[idx]
            intensity = array[k, j, i]

            ijk_point = [i, j, k, 1]
            ras_point = [0, 0, 0, 1]
            ijk_to_ras.MultiplyPoint(ijk_point, ras_point)

            print(
                f"  RAS ({ras_point[0]:.1f}, {ras_point[1]:.1f}, {ras_point[2]:.1f}) intensity={intensity}"
            )

        # Look for the tumor more specifically - it should be a bright ring
        # Find slices with high-intensity ring patterns
        print("\n\nSlice-by-slice intensity analysis:")
        for k in range(array.shape[0]):
            slice_max = array[k].max()
            if slice_max > threshold * 0.8:
                # Find centroid of bright region in this slice
                bright_in_slice = np.where(array[k] > threshold * 0.9)
                if len(bright_in_slice[0]) > 50:  # At least 50 bright pixels
                    j_mean = np.mean(bright_in_slice[0])
                    i_mean = np.mean(bright_in_slice[1])

                    ijk_point = [i_mean, j_mean, k, 1]
                    ras_point = [0, 0, 0, 1]
                    ijk_to_ras.MultiplyPoint(ijk_point, ras_point)

                    print(
                        f"  Slice {k}: RAS ({ras_point[0]:.1f}, {ras_point[1]:.1f}, {ras_point[2]:.1f}) "
                        f"- {len(bright_in_slice[0])} bright pixels, max={slice_max}"
                    )

    print("\n\nDone! Use the RAS coordinates above for tumor painting.")

    # Exit Slicer
    slicer.app.exit(0)


if __name__ == "__main__":
    import vtk

    main()
