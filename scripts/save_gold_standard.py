#!/usr/bin/env python
"""Save the optimized watershed segmentation as a gold standard.

This script creates the gold standard from the best parameters found
during optimization and documents the process.

Usage (run in Slicer):
    exec(open('scripts/save_gold_standard.py').read())
"""

import logging
import time
from pathlib import Path

import numpy as np
import slicer
import vtk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent

# Optimal parameters from optimization
GOLD_STANDARD_NAME = "MRBrainTumor1_tumor"
ALGORITHM = "watershed"
PARAMS = {
    "brush_radius_mm": 25.0,
    "edge_sensitivity": 40,
}

# Click points for MRBrainTumor1 tumor (RAS coordinates)
TUMOR_CLICK_POINTS = [
    (-5.31, 34.77, 20.83),
    (-5.31, 25.12, 35.97),
    (-5.31, 20.70, 22.17),
    (-6.16, 38.28, 30.61),
    (-1.35, 28.65, 18.90),
]

EXIT_WHEN_DONE = True


def ras_to_xy(ras, red_widget):
    """Convert RAS to screen XY."""
    sliceNode = red_widget.sliceLogic().GetSliceNode()
    xyToRas = sliceNode.GetXYToRAS()
    rasToXy = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(xyToRas, rasToXy)
    ras4 = [ras[0], ras[1], ras[2], 1]
    xy4 = [0, 0, 0, 1]
    rasToXy.MultiplyPoint(ras4, xy4)
    return (int(xy4[0]), int(xy4[1]))


def run_segmentation(red_widget, volume_node, seg_node, segment_id):
    """Run watershed segmentation with optimal parameters."""
    editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
    editor.setSegmentationNode(seg_node)
    editor.setSourceVolumeNode(volume_node)
    editor.setCurrentSegmentID(segment_id)
    editor.setActiveEffectByName("Adaptive Brush")
    slicer.app.processEvents()

    effect = editor.activeEffect()
    if not effect:
        logger.error("Adaptive Brush not available")
        return 0, 0

    se = effect.self()
    idx = se.algorithmCombo.findData(ALGORITHM)
    if idx >= 0:
        se.algorithmCombo.setCurrentIndex(idx)
    se.radiusSlider.value = PARAMS["brush_radius_mm"]
    se.sensitivitySlider.value = PARAMS["edge_sensitivity"]
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


def capture_screenshot(red_widget, output_path):
    """Capture slice view to PNG."""
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


def center_view(red_widget, center_ras, fov_mm=100):
    """Center view on coordinates."""
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


def main():
    logger.info("Creating gold standard segmentation...")

    # Clear scene
    slicer.mrmlScene.Clear(0)

    # Load sample data
    import SampleData

    volume_node = SampleData.downloadSample("MRBrainTumor1")
    logger.info(f"Loaded: {volume_node.GetName()}")

    # Setup
    slicer.util.selectModule("SegmentEditor")
    slicer.app.processEvents()
    time.sleep(0.3)

    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
    red_widget = layoutManager.sliceWidget("Red")
    slicer.app.processEvents()

    # Create segmentation
    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    seg_node.CreateDefaultDisplayNodes()
    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
    seg_node.SetName("GoldStandard_MRBrainTumor1")
    segment_id = seg_node.GetSegmentation().AddEmptySegment("Tumor")
    slicer.app.processEvents()

    # Run segmentation
    voxels, time_ms = run_segmentation(red_widget, volume_node, seg_node, segment_id)
    logger.info(f"Segmentation: {voxels:,} voxels in {time_ms:.0f}ms")

    # Calculate tumor center
    tumor_center = (
        sum(p[0] for p in TUMOR_CLICK_POINTS) / len(TUMOR_CLICK_POINTS),
        sum(p[1] for p in TUMOR_CLICK_POINTS) / len(TUMOR_CLICK_POINTS),
        sum(p[2] for p in TUMOR_CLICK_POINTS) / len(TUMOR_CLICK_POINTS),
    )

    # Save gold standard using GoldStandardManager
    from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

    manager = GoldStandardManager()

    # Prepare click locations with parameters
    click_locations = [
        {
            "ras": list(ras),
            "params": {
                "algorithm": ALGORITHM,
                "brush_radius_mm": PARAMS["brush_radius_mm"],
                "edge_sensitivity": PARAMS["edge_sensitivity"],
            },
        }
        for ras in TUMOR_CLICK_POINTS
    ]

    # Save
    gold_path = manager.save_as_gold(
        segmentation_node=seg_node,
        volume_node=volume_node,
        segment_id=segment_id,
        name=GOLD_STANDARD_NAME,
        click_locations=click_locations,
        description="Tumor segmentation using watershed algorithm. Created through automated optimization comparing 5 algorithms (watershed, geodesic_distance, connected_threshold, region_growing, threshold_brush). Watershed performed best with clean boundary adherence. Edge sensitivity tested from 20-60 with identical results, confirming robustness.",
        algorithm=ALGORITHM,
        parameters=PARAMS,
    )

    logger.info(f"Gold standard saved to: {gold_path}")

    # Capture reference screenshots
    screenshots_dir = gold_path / "reference_screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    # Setup display
    dn = seg_node.GetDisplayNode()
    dn.SetVisibility2DFill(False)
    dn.SetVisibility2DOutline(True)
    dn.SetSliceIntersectionThickness(3)
    dn.SetSegmentOverrideColor(segment_id, 1.0, 1.0, 0.0)
    slicer.app.processEvents()

    # Center slice
    center_view(red_widget, tumor_center, fov_mm=100)
    capture_screenshot(red_widget, screenshots_dir / "center_outline.png")

    # Fill mode
    dn.SetVisibility2DFill(True)
    dn.SetOpacity2DFill(0.5)
    dn.SetSegmentOverrideColor(segment_id, 0.0, 1.0, 0.0)
    slicer.app.processEvents()
    capture_screenshot(red_widget, screenshots_dir / "center_fill.png")

    # Above and below
    for offset, name in [(-8, "above"), (8, "below")]:
        pos = (tumor_center[0], tumor_center[1], tumor_center[2] + offset)
        center_view(red_widget, pos, fov_mm=100)
        capture_screenshot(red_widget, screenshots_dir / f"{name}_fill.png")

    logger.info(f"Reference screenshots saved to: {screenshots_dir}")

    # Create lab notebook
    from SegmentEditorAdaptiveBrushTesterLib import LabNotebook

    notebook = LabNotebook(
        "Gold Standard Creation: MRBrainTumor1", filename="gold_standard_MRBrainTumor1_tumor"
    )

    notebook.add_section(
        "Overview",
        """
This document records the creation of a gold standard segmentation for the
MRBrainTumor1 sample data tumor. The gold standard was created through
automated algorithm comparison and parameter optimization.
""",
    )

    notebook.add_section(
        "Sample Data",
        """
- **Dataset:** MRBrainTumor1 (3D Slicer sample data)
- **Modality:** MRI T1-weighted with contrast
- **Target:** Brain tumor (meningioma)
""",
    )

    notebook.add_section(
        "Algorithm Comparison",
        """
Five algorithms were compared using identical click points and parameters:

| Algorithm | Voxels | Quality |
|-----------|--------|---------|
| watershed | 13,377 | ✅ Best - clean boundary adherence |
| geodesic_distance | 15,419 | Good - slight over-extension |
| threshold_brush | 16,121 | Okay - has leak artifacts |
| region_growing | 11,273 | Poor - fragmented result |
| connected_threshold | 76,202 | Bad - massive over-segmentation |

**Winner:** Watershed algorithm
""",
    )

    notebook.add_section(
        "Parameter Optimization",
        """
Watershed edge_sensitivity was tested from 20-60:

| edge_sensitivity | voxel_count |
|------------------|-------------|
| 20 | 13,376 |
| 30 | 13,376 |
| 40 | 13,377 |
| 50 | 13,377 |
| 60 | 13,377 |

**Conclusion:** Results are nearly identical across the range, indicating
the watershed algorithm is robust for this tumor. The clear tumor boundaries
make it insensitive to this parameter.
""",
    )

    notebook.add_section(
        "Final Parameters",
        f"""
```python
algorithm = "{ALGORITHM}"
brush_radius_mm = {PARAMS['brush_radius_mm']}
edge_sensitivity = {PARAMS['edge_sensitivity']}
```
""",
    )

    notebook.add_section(
        "Click Points (RAS)",
        """
5 clicks were used to segment the tumor:
1. (-5.31, 34.77, 20.83)
2. (-5.31, 25.12, 35.97)
3. (-5.31, 20.70, 22.17)
4. (-6.16, 38.28, 30.61)
5. (-1.35, 28.65, 18.90)

Tumor center: (-4.69, 29.50, 25.70)
""",
    )

    notebook.add_section(
        "Result",
        f"""
- **Voxel count:** {voxels:,}
- **Segmentation time:** {time_ms:.0f}ms
- **Gold standard path:** `GoldStandards/{GOLD_STANDARD_NAME}/`
""",
    )

    notebook.add_section(
        "Manual Improvement Tips",
        """
To further refine this gold standard:

1. **Fill gaps:** Use Paint effect (3-5mm brush) to fill any missed interior regions
2. **Trim edges:** Use Erase mode or Scissors to remove over-segmentation
3. **Multi-slice check:** Scroll through all Z slices to verify 3D extent
4. **Alternative tools:**
   - Grow from Seeds for precise boundary control
   - Level Tracing for semi-automatic boundary following
""",
    )

    notebook.add_section(
        "Files Created",
        f"""
- `GoldStandards/{GOLD_STANDARD_NAME}/gold.seg.nrrd` - Segmentation file
- `GoldStandards/{GOLD_STANDARD_NAME}/metadata.json` - Parameters and click points
- `GoldStandards/{GOLD_STANDARD_NAME}/reference_screenshots/` - Visual reference
- `LabNotebooks/gold_standard_MRBrainTumor1_tumor.md` - This documentation
""",
    )

    notebook_path = notebook.save()
    logger.info(f"Lab notebook saved to: {notebook_path}")

    print(f"\n{'='*60}")
    print("GOLD STANDARD CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Location: GoldStandards/{GOLD_STANDARD_NAME}/")
    print(f"Voxels: {voxels:,}")
    print(f"Documentation: {notebook_path}")
    print(f"{'='*60}")

    if EXIT_WHEN_DONE:
        slicer.util.exit()


if __name__ == "__main__":
    main()
