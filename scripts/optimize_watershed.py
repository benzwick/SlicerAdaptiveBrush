#!/usr/bin/env python
"""Optimize watershed parameters to find the best tumor segmentation.

Tests different edge_sensitivity values and captures screenshots for comparison.

Usage (run in Slicer):
    exec(open('scripts/optimize_watershed.py').read())
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import slicer
import vtk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "test_runs" / f"watershed_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

TUMOR_CLICK_POINTS = [
    (-5.31, 34.77, 20.83),
    (-5.31, 25.12, 35.97),
    (-5.31, 20.70, 22.17),
    (-6.16, 38.28, 30.61),
    (-1.35, 28.65, 18.90),
]

# Parameters to test
PARAM_SETS = [
    {"edge_sensitivity": 20, "brush_radius_mm": 25.0},
    {"edge_sensitivity": 30, "brush_radius_mm": 25.0},
    {"edge_sensitivity": 40, "brush_radius_mm": 25.0},  # baseline
    {"edge_sensitivity": 50, "brush_radius_mm": 25.0},
    {"edge_sensitivity": 60, "brush_radius_mm": 25.0},
]

EXIT_WHEN_DONE = True


def get_red_widget():
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
    slicer.app.processEvents()
    return layoutManager.sliceWidget("Red")


def center_view(red_widget, center_ras, fov_mm=100):
    sliceNode = red_widget.sliceLogic().GetSliceNode()
    sliceNode.SetOrientationToAxial()
    red_widget.sliceLogic().SetSliceOffset(center_ras[2])
    sliceNode.SetFieldOfView(fov_mm, fov_mm, 1.0)
    sliceToRAS = sliceNode.GetSliceToRAS()
    sliceToRAS.SetElement(0, 3, center_ras[0])
    sliceToRAS.SetElement(1, 3, center_ras[1])
    sliceToRAS.SetElement(2, 3, center_ras[2])
    sliceNode.UpdateMatrices()
    slicer.app.processEvents()
    time.sleep(0.1)


def capture_screenshot(red_widget, output_path):
    red_widget.sliceView().forceRender()
    slicer.app.processEvents()
    time.sleep(0.15)
    windowToImage = vtk.vtkWindowToImageFilter()
    windowToImage.SetInput(red_widget.sliceView().renderWindow())
    windowToImage.ReadFrontBufferOff()
    windowToImage.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(windowToImage.GetOutputPort())
    writer.Write()
    logger.info(f"Screenshot: {output_path.name}")


def set_display_outline(seg_node):
    dn = seg_node.GetDisplayNode()
    dn.SetVisibility(True)
    dn.SetVisibility2DFill(False)
    dn.SetVisibility2DOutline(True)
    dn.SetSliceIntersectionThickness(3)
    seg = seg_node.GetSegmentation()
    if seg.GetNumberOfSegments() > 0:
        dn.SetSegmentOverrideColor(seg.GetNthSegmentID(0), 1.0, 1.0, 0.0)
    slicer.app.processEvents()


def ras_to_xy(ras, red_widget):
    sliceNode = red_widget.sliceLogic().GetSliceNode()
    xyToRas = sliceNode.GetXYToRAS()
    rasToXy = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(xyToRas, rasToXy)
    ras4 = [ras[0], ras[1], ras[2], 1]
    xy4 = [0, 0, 0, 1]
    rasToXy.MultiplyPoint(ras4, xy4)
    return (int(xy4[0]), int(xy4[1]))


def run_watershed(red_widget, volume_node, seg_node, segment_id, params):
    editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
    editor.setSegmentationNode(seg_node)
    editor.setSourceVolumeNode(volume_node)
    editor.setCurrentSegmentID(segment_id)
    editor.setActiveEffectByName("Adaptive Brush")
    slicer.app.processEvents()

    effect = editor.activeEffect()
    if not effect:
        return 0, 0

    se = effect.self()
    idx = se.algorithmCombo.findData("watershed")
    if idx >= 0:
        se.algorithmCombo.setCurrentIndex(idx)
    se.radiusSlider.value = params["brush_radius_mm"]
    se.sensitivitySlider.value = params["edge_sensitivity"]
    slicer.app.processEvents()

    total_time = 0
    for ras in TUMOR_CLICK_POINTS:
        red_widget.sliceLogic().SetSliceOffset(ras[2])
        slicer.app.processEvents()
        xy = ras_to_xy(ras, red_widget)

        start = time.time()
        se.scriptedEffect.saveStateForUndo()
        se.isDrawing = True
        se._currentStrokeEraseMode = False
        se.processPoint(xy, red_widget)
        se.isDrawing = False
        total_time += time.time() - start
        slicer.app.processEvents()

    arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, segment_id, volume_node)
    return int(np.sum(arr > 0)), total_time * 1000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {OUTPUT_DIR}")

    slicer.mrmlScene.Clear(0)

    import SampleData

    volume_node = SampleData.downloadSample("MRBrainTumor1")

    slicer.util.selectModule("SegmentEditor")
    slicer.app.processEvents()
    time.sleep(0.3)

    red_widget = get_red_widget()

    tumor_center = (
        sum(p[0] for p in TUMOR_CLICK_POINTS) / len(TUMOR_CLICK_POINTS),
        sum(p[1] for p in TUMOR_CLICK_POINTS) / len(TUMOR_CLICK_POINTS),
        sum(p[2] for p in TUMOR_CLICK_POINTS) / len(TUMOR_CLICK_POINTS),
    )

    results = []
    all_seg_nodes = []

    for params in PARAM_SETS:
        name = f"sens{params['edge_sensitivity']}"
        logger.info(f"\n=== {name} ===")

        for prev_seg in all_seg_nodes:
            prev_seg.GetDisplayNode().SetVisibility(False)
        slicer.app.processEvents()

        seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        seg_node.SetName(f"Watershed_{name}")
        segment_id = seg_node.GetSegmentation().AddEmptySegment("Tumor")
        all_seg_nodes.append(seg_node)
        slicer.app.processEvents()

        voxels, time_ms = run_watershed(red_widget, volume_node, seg_node, segment_id, params)
        logger.info(f"Result: {voxels:,} voxels in {time_ms:.0f}ms")

        param_dir = OUTPUT_DIR / name
        param_dir.mkdir(exist_ok=True)

        center_view(red_widget, tumor_center, fov_mm=100)
        set_display_outline(seg_node)
        capture_screenshot(red_widget, param_dir / "outline.png")

        results.append(
            {
                "params": params,
                "name": name,
                "voxel_count": voxels,
                "time_ms": time_ms,
                "screenshot": str(param_dir / "outline.png"),
            }
        )

    summary = {
        "created": datetime.now().isoformat(),
        "algorithm": "watershed",
        "sample_data": "MRBrainTumor1",
        "tumor_center": tumor_center,
        "click_points": TUMOR_CLICK_POINTS,
        "results": results,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== OPTIMIZATION COMPLETE ===")
    logger.info(f"Output: {OUTPUT_DIR}")
    for r in sorted(results, key=lambda x: x["voxel_count"]):
        logger.info(f"  {r['name']}: {r['voxel_count']:,} voxels")

    print(f"\nDONE! Results at: {OUTPUT_DIR}")

    if EXIT_WHEN_DONE:
        slicer.util.exit()


if __name__ == "__main__":
    main()
