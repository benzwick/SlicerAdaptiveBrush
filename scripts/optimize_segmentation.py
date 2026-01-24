#!/usr/bin/env python
"""Interactive segmentation optimization script.

Run in Slicer:
    Slicer --python-script scripts/optimize_segmentation.py [sample_data]

This script:
1. Loads sample data
2. Waits for user to place 5 fiducial points
3. Runs all algorithms with those points
4. Generates optimization report
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# All available algorithms
ALGORITHMS = [
    "connected_threshold",
    "region_growing",
    "threshold_brush",
    "watershed",
    "geodesic_distance",
    "random_walker",
    "level_set_cpu",
    "level_set_gpu",
]

# Default parameters
DEFAULT_PARAMS = {
    "brush_radius_mm": 20.0,
    "edge_sensitivity": 35,
}


def load_sample_data(sample_name: str = "MRBrainTumor1"):
    """Load sample data into Slicer."""
    import SampleData

    logger.info(f"Loading sample data: {sample_name}")
    volume = SampleData.downloadSample(sample_name)
    if volume is None:
        raise RuntimeError(f"Failed to load {sample_name}")
    return volume


def wait_for_fiducials(min_points: int = 5):
    """Wait for user to place fiducial points and return their RAS coordinates."""
    import slicer

    print("\n" + "=" * 60)
    print("PLACE REFERENCE POINTS ON TARGET STRUCTURE")
    print("=" * 60)
    print("1. Go to Markups module")
    print("2. Create a new Point List")
    print(f"3. Place {min_points} points on your target structure:")
    print("   - Point 1: Center")
    print("   - Point 2: Superior edge")
    print("   - Point 3: Anterior edge")
    print("   - Point 4: Posterior edge")
    print("   - Point 5: Lateral edge")
    print("\nPress ENTER in the Python console when done...")
    print("=" * 60 + "\n")

    # This will block in interactive mode
    # In non-interactive mode, we'll check for existing fiducials

    # Find markup nodes
    markup_nodes = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")

    if not markup_nodes:
        logger.warning("No fiducial nodes found. Creating instructions...")
        return None

    # Get the most recently modified markup node
    latest_node = markup_nodes[-1]

    n_points = latest_node.GetNumberOfControlPoints()
    if n_points < min_points:
        logger.warning(f"Only {n_points} points found, need {min_points}")
        return None

    # Extract RAS coordinates
    points = []
    for i in range(n_points):
        pos = [0.0, 0.0, 0.0]
        latest_node.GetNthControlPointPosition(i, pos)
        points.append(tuple(pos))
        logger.info(f"Point {i+1}: RAS {pos}")

    return points[:min_points]


def run_optimization(volume_node, points: list[tuple], params: dict):
    """Run all algorithms with the given points and return results."""
    import time

    import numpy as np
    import slicer
    import vtk

    results = []

    # Create segmentation node
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

    # Set up segment editor
    segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
    segment_editor_widget.setMRMLScene(slicer.mrmlScene)
    segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setSourceVolumeNode(volume_node)

    # Activate Adaptive Brush
    segment_editor_widget.setActiveEffectByName("Adaptive Brush")
    effect = segment_editor_widget.activeEffect()

    if effect is None:
        raise RuntimeError("Failed to activate Adaptive Brush effect")

    scripted_effect = effect.self()

    # Set parameters
    scripted_effect.radiusMm = params["brush_radius_mm"]
    scripted_effect.radiusSlider.value = params["brush_radius_mm"]
    scripted_effect.edgeSensitivity = params["edge_sensitivity"]
    scripted_effect.sensitivitySlider.value = params["edge_sensitivity"]

    # Get slice widget for coordinate conversion
    layoutManager = slicer.app.layoutManager()
    redWidget = layoutManager.sliceWidget("Red")
    redLogic = redWidget.sliceLogic()

    def ras_to_xy(ras, slice_widget):
        """Convert RAS to screen XY."""
        slice_logic = slice_widget.sliceLogic()
        slice_node = slice_logic.GetSliceNode()
        xy_to_ras = slice_node.GetXYToRAS()
        ras_to_xy_mat = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy_mat)
        ras_point = [ras[0], ras[1], ras[2], 1]
        xy_point = [0, 0, 0, 1]
        ras_to_xy_mat.MultiplyPoint(ras_point, xy_point)
        return (int(xy_point[0]), int(xy_point[1]))

    def count_voxels(seg_node, seg_id, vol_node):
        """Count voxels in a segment."""
        try:
            labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, vol_node)
            return int(np.sum(labelmap > 0))
        except Exception:
            return 0

    # Test each algorithm
    for algo in ALGORITHMS:
        logger.info(f"Testing algorithm: {algo}")

        # Clear segments
        segmentation_node.GetSegmentation().RemoveAllSegments()
        segment_id = segmentation_node.GetSegmentation().AddEmptySegment("Test")
        segment_editor_widget.setCurrentSegmentID(segment_id)

        # Set algorithm
        scripted_effect.algorithm = algo
        scripted_effect._updateAlgorithmParamsVisibility()
        slicer.app.processEvents()

        # Paint all points
        total_time = 0.0
        for ras in points:
            redLogic.SetSliceOffset(ras[2])
            slicer.app.processEvents()

            xy = ras_to_xy(ras, redWidget)

            start = time.time()
            scripted_effect.scriptedEffect.saveStateForUndo()
            scripted_effect.isDrawing = True
            scripted_effect._currentStrokeEraseMode = False
            scripted_effect.processPoint(xy, redWidget)
            scripted_effect.isDrawing = False
            total_time += time.time() - start

            slicer.app.processEvents()

        voxel_count = count_voxels(segmentation_node, segment_id, volume_node)

        results.append(
            {
                "algorithm": algo,
                "voxels": voxel_count,
                "time_ms": total_time * 1000,
                "voxels_per_ms": voxel_count / (total_time * 1000) if total_time > 0 else 0,
            }
        )

        logger.info(f"  {algo}: {voxel_count} voxels in {total_time*1000:.0f}ms")

    # Cleanup
    segment_editor_widget.setActiveEffect(None)

    return results


def generate_report(sample_name: str, points: list, params: dict, results: list) -> str:
    """Generate optimization report."""
    # Sort by voxels descending
    sorted_results = sorted(results, key=lambda x: x["voxels"], reverse=True)

    report = []
    report.append(f"\nOPTIMIZATION RESULTS: {sample_name}")
    report.append("=" * 50)
    report.append(f"Reference Points: {len(points)}")
    report.append(f"Brush Radius: {params['brush_radius_mm']}mm")
    report.append(f"Edge Sensitivity: {params['edge_sensitivity']}")
    report.append("")

    report.append("REFERENCE POINTS (RAS):")
    report.append("-" * 30)
    for i, p in enumerate(points):
        report.append(f"  {i+1}. ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
    report.append("")

    report.append("ALGORITHM COMPARISON:")
    report.append("-" * 30)
    for i, r in enumerate(sorted_results):
        rank_label = ""
        if i == 0:
            rank_label = " <- BEST"
        report.append(
            f"{i+1}. {r['algorithm']:20s}: {r['voxels']:,} voxels "
            f"({r['time_ms']:.0f}ms){rank_label}"
        )
    report.append("")

    # Recommendations
    best = sorted_results[0]
    report.append("RECOMMENDATIONS:")
    report.append("-" * 30)
    report.append(f"Best algorithm: {best['algorithm']}")
    report.append(f"Expected voxels: ~{best['voxels']:,}")
    report.append("")

    if best["algorithm"] == "connected_threshold":
        report.append("Note: connected_threshold may over-segment.")
        report.append("Consider threshold_brush for more controlled results.")

    return "\n".join(report)


def main():
    """Main optimization workflow."""

    # Get sample data name from args
    sample_name = sys.argv[1] if len(sys.argv) > 1 else "MRBrainTumor1"

    print("\n" + "=" * 60)
    print("ADAPTIVE BRUSH SEGMENTATION OPTIMIZER")
    print("=" * 60)

    # Load data
    volume = load_sample_data(sample_name)

    # Check for existing fiducials
    points = wait_for_fiducials(5)

    if points is None:
        print("\nNo fiducials found. Please:")
        print("1. Place 5 points on your target structure")
        print("2. Run this script again")
        print("\nSlicer will stay open for you to place points.")
        return

    # Run optimization
    print("\nRunning optimization...")
    results = run_optimization(volume, points, DEFAULT_PARAMS)

    # Generate report
    report = generate_report(sample_name, points, DEFAULT_PARAMS, results)
    print(report)

    # Save report
    output_dir = Path(__file__).parent.parent / "test_runs"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_path = output_dir / f"{timestamp}_optimization_report.txt"

    with open(report_path, "w") as f:
        f.write(report)
        f.write("\n\nRaw Results:\n")
        f.write(json.dumps(results, indent=2))

    print(f"\nReport saved to: {report_path}")
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
