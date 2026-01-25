#!/usr/bin/env python
"""Comprehensive parameter optimization for all algorithms.

Tests all algorithms with their specific parameters, varying:
- Brush radius and threshold zone (inner circle)
- Algorithm-specific parameters
- Number and position of clicks
- Sampling methods

Compares against gold standard using Dice and Hausdorff metrics.

Usage (run in Slicer):
    exec(open('scripts/comprehensive_optimization.py').read())
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import slicer
import vtk

SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = (
    SCRIPT_DIR / "test_runs" / f"comprehensive_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# Create output dir early so we can log there
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log to both console and file in output folder
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "console.log"),
    ],
)
logger = logging.getLogger(__name__)

# All click points we might use (different subsets for different tests)
ALL_CLICK_POINTS = [
    (-5.31, 34.77, 20.83),  # 0: center-ish
    (-5.31, 25.12, 35.97),  # 1: superior
    (-5.31, 20.70, 22.17),  # 2: inferior-anterior
    (-6.16, 38.28, 30.61),  # 3: posterior
    (-1.35, 28.65, 18.90),  # 4: lateral
    (-8.0, 30.0, 25.0),  # 5: medial edge
    (-3.0, 32.0, 28.0),  # 6: center
    (-5.0, 28.0, 23.0),  # 7: lower center
]

# Click configurations to test
CLICK_CONFIGS = {
    "1_center": [6],
    "3_spread": [0, 1, 4],
    "5_standard": [0, 1, 2, 3, 4],
    "5_dense_center": [0, 5, 6, 7, 4],
    "7_thorough": [0, 1, 2, 3, 4, 6, 7],
}

# Algorithm configurations with parameter ranges to test
ALGORITHM_CONFIGS = {
    "watershed": {
        "base_params": {},
        "variations": [
            {"edge_sensitivity": 30, "watershedGradientScale": 1.0, "watershedSmoothing": 0.3},
            {"edge_sensitivity": 40, "watershedGradientScale": 1.5, "watershedSmoothing": 0.5},
            {"edge_sensitivity": 50, "watershedGradientScale": 2.0, "watershedSmoothing": 0.5},
            {"edge_sensitivity": 60, "watershedGradientScale": 2.5, "watershedSmoothing": 0.8},
        ],
    },
    "geodesic_distance": {
        "base_params": {},
        "variations": [
            {
                "edge_sensitivity": 40,
                "geodesicEdgeWeight": 5.0,
                "geodesicDistanceScale": 0.8,
                "geodesicSmoothing": 0.3,
            },
            {
                "edge_sensitivity": 50,
                "geodesicEdgeWeight": 8.0,
                "geodesicDistanceScale": 1.0,
                "geodesicSmoothing": 0.5,
            },
            {
                "edge_sensitivity": 50,
                "geodesicEdgeWeight": 12.0,
                "geodesicDistanceScale": 1.2,
                "geodesicSmoothing": 0.5,
            },
            {
                "edge_sensitivity": 60,
                "geodesicEdgeWeight": 15.0,
                "geodesicDistanceScale": 1.5,
                "geodesicSmoothing": 0.8,
            },
        ],
    },
    "connected_threshold": {
        "base_params": {},
        "variations": [
            {"edge_sensitivity": 30, "thresholdZone": 40},
            {"edge_sensitivity": 40, "thresholdZone": 50},
            {"edge_sensitivity": 50, "thresholdZone": 60},
            {"edge_sensitivity": 60, "thresholdZone": 70},
        ],
    },
    "region_growing": {
        "base_params": {},
        "variations": [
            {"edge_sensitivity": 40, "regionGrowingMultiplier": 2.0, "regionGrowingIterations": 3},
            {"edge_sensitivity": 50, "regionGrowingMultiplier": 2.5, "regionGrowingIterations": 4},
            {"edge_sensitivity": 50, "regionGrowingMultiplier": 3.0, "regionGrowingIterations": 5},
            {"edge_sensitivity": 60, "regionGrowingMultiplier": 3.5, "regionGrowingIterations": 6},
        ],
    },
    "threshold_brush": {
        "base_params": {"autoThreshold": True},
        "variations": [
            {"edge_sensitivity": 40, "thresholdMethod": "otsu"},
            {"edge_sensitivity": 50, "thresholdMethod": "huang"},
            {"edge_sensitivity": 50, "thresholdMethod": "triangle"},
            {"edge_sensitivity": 50, "thresholdMethod": "li"},
        ],
    },
    "level_set_cpu": {
        "base_params": {},
        "variations": [
            {
                "edge_sensitivity": 40,
                "levelSetPropagation": 0.8,
                "levelSetCurvature": 0.8,
                "levelSetIterations": 30,
            },
            {
                "edge_sensitivity": 50,
                "levelSetPropagation": 1.0,
                "levelSetCurvature": 1.0,
                "levelSetIterations": 50,
            },
            {
                "edge_sensitivity": 50,
                "levelSetPropagation": 1.2,
                "levelSetCurvature": 1.2,
                "levelSetIterations": 80,
            },
            {
                "edge_sensitivity": 60,
                "levelSetPropagation": 1.5,
                "levelSetCurvature": 1.5,
                "levelSetIterations": 100,
            },
        ],
    },
    "random_walker": {
        "base_params": {},
        "variations": [
            {"edge_sensitivity": 40, "randomWalkerBeta": 80},
            {"edge_sensitivity": 50, "randomWalkerBeta": 130},
            {"edge_sensitivity": 50, "randomWalkerBeta": 200},
            {"edge_sensitivity": 60, "randomWalkerBeta": 300},
        ],
    },
}

# Brush radius configurations
RADIUS_CONFIGS = [
    {"brush_radius_mm": 15.0, "thresholdZone": 50},
    {"brush_radius_mm": 20.0, "thresholdZone": 50},
    {"brush_radius_mm": 25.0, "thresholdZone": 50},
    {"brush_radius_mm": 30.0, "thresholdZone": 50},
    {"brush_radius_mm": 25.0, "thresholdZone": 30},  # smaller inner circle
    {"brush_radius_mm": 25.0, "thresholdZone": 70},  # larger inner circle
]

EXIT_WHEN_DONE = True


class MetricsCalculator:
    """Calculate Dice and Hausdorff metrics."""

    def __init__(self, gold_seg_node, gold_segment_id, volume_node):
        self.volume_node = volume_node
        self.spacing = volume_node.GetSpacing()

        # Get gold standard array
        self.gold_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
            gold_seg_node, gold_segment_id, volume_node
        ).astype(np.uint8)
        self.gold_arr = (self.gold_arr > 0).astype(np.uint8)
        self.gold_voxels = int(np.sum(self.gold_arr))

    def compute(self, test_seg_node, test_segment_id):
        """Compute metrics between test and gold standard."""
        test_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
            test_seg_node, test_segment_id, self.volume_node
        ).astype(np.uint8)
        test_arr = (test_arr > 0).astype(np.uint8)
        test_voxels = int(np.sum(test_arr))

        if test_voxels == 0:
            return {
                "dice": 0.0,
                "hausdorff_95": float("inf"),
                "test_voxels": 0,
                "gold_voxels": self.gold_voxels,
            }

        # Dice
        intersection = int(np.sum(test_arr & self.gold_arr))
        dice = (
            2.0 * intersection / (test_voxels + self.gold_voxels)
            if (test_voxels + self.gold_voxels) > 0
            else 0.0
        )

        # Hausdorff 95
        try:
            test_sitk = sitk.GetImageFromArray(test_arr)
            gold_sitk = sitk.GetImageFromArray(self.gold_arr)
            test_sitk.SetSpacing((self.spacing[2], self.spacing[1], self.spacing[0]))
            gold_sitk.SetSpacing((self.spacing[2], self.spacing[1], self.spacing[0]))

            hausdorff = sitk.HausdorffDistanceImageFilter()
            hausdorff.Execute(gold_sitk, test_sitk)
            hd_max = hausdorff.GetHausdorffDistance()

            # Compute HD95
            from scipy.ndimage import binary_erosion, distance_transform_edt

            test_dt = distance_transform_edt(
                test_arr == 0, sampling=(self.spacing[2], self.spacing[1], self.spacing[0])
            )
            gold_dt = distance_transform_edt(
                self.gold_arr == 0, sampling=(self.spacing[2], self.spacing[1], self.spacing[0])
            )

            test_surface = test_arr & ~binary_erosion(test_arr)
            gold_surface = self.gold_arr & ~binary_erosion(self.gold_arr)

            test_to_gold = gold_dt[test_surface > 0]
            gold_to_test = test_dt[gold_surface > 0]

            if len(test_to_gold) > 0 and len(gold_to_test) > 0:
                all_distances = np.concatenate([test_to_gold, gold_to_test])
                hd95 = float(np.percentile(all_distances, 95))
            else:
                hd95 = hd_max
        except Exception as e:
            logger.warning(f"Hausdorff calculation failed: {e}")
            hd95 = float("inf")

        return {
            "dice": dice,
            "hausdorff_95": hd95,
            "test_voxels": test_voxels,
            "gold_voxels": self.gold_voxels,
        }


def get_red_widget():
    """Get Red slice widget."""
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
    slicer.app.processEvents()
    return layoutManager.sliceWidget("Red")


def center_view(red_widget, center_ras, fov_mm=100):
    """Center view on coordinates with correct aspect ratio."""
    sliceNode = red_widget.sliceLogic().GetSliceNode()
    sliceNode.SetOrientationToAxial()
    red_widget.sliceLogic().SetSliceOffset(center_ras[2])

    # Get widget dimensions to maintain correct aspect ratio
    # Note: In PythonQt, .size is a property, not a method
    widget_size = red_widget.size
    width = widget_size.width()
    height = widget_size.height()
    aspect_ratio = width / height if height > 0 else 1.0

    # Set FOV with correct aspect ratio (FOV is width, height, depth)
    fov_width = fov_mm * aspect_ratio if aspect_ratio > 1 else fov_mm
    fov_height = fov_mm / aspect_ratio if aspect_ratio < 1 else fov_mm
    sliceNode.SetFieldOfView(fov_width, fov_height, 1.0)

    sliceToRAS = sliceNode.GetSliceToRAS()
    sliceToRAS.SetElement(0, 3, center_ras[0])
    sliceToRAS.SetElement(1, 3, center_ras[1])
    sliceToRAS.SetElement(2, 3, center_ras[2])
    sliceNode.UpdateMatrices()
    slicer.app.processEvents()


def capture_screenshot(red_widget, output_path):
    """Capture screenshot."""
    red_widget.sliceView().forceRender()
    slicer.app.processEvents()
    time.sleep(0.1)
    windowToImage = vtk.vtkWindowToImageFilter()
    windowToImage.SetInput(red_widget.sliceView().renderWindow())
    windowToImage.ReadFrontBufferOff()
    windowToImage.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(windowToImage.GetOutputPort())
    writer.Write()


def set_display_outline(seg_node, segment_id):
    """Set segmentation to outline mode."""
    dn = seg_node.GetDisplayNode()
    dn.SetVisibility(True)
    dn.SetVisibility2DFill(False)
    dn.SetVisibility2DOutline(True)
    dn.SetSliceIntersectionThickness(3)
    dn.SetSegmentOverrideColor(segment_id, 1.0, 1.0, 0.0)
    slicer.app.processEvents()


def set_display_fill(seg_node, segment_id):
    """Set segmentation to fill mode."""
    dn = seg_node.GetDisplayNode()
    dn.SetVisibility(True)
    dn.SetVisibility2DFill(True)
    dn.SetVisibility2DOutline(True)
    dn.SetSliceIntersectionThickness(2)
    dn.SetOpacity2DFill(0.5)
    dn.SetSegmentOverrideColor(segment_id, 0.0, 1.0, 0.0)
    slicer.app.processEvents()


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


def apply_params_to_effect(se, algorithm, params):
    """Apply parameters to the scripted effect."""
    # Set algorithm
    idx = se.algorithmCombo.findData(algorithm)
    if idx >= 0:
        se.algorithmCombo.setCurrentIndex(idx)
    slicer.app.processEvents()

    # Common parameters
    if "brush_radius_mm" in params:
        se.radiusSlider.value = params["brush_radius_mm"]
    if "edge_sensitivity" in params:
        se.sensitivitySlider.value = params["edge_sensitivity"]
    if "thresholdZone" in params:
        se.zoneSlider.value = params["thresholdZone"]

    # Algorithm-specific parameters
    if algorithm == "watershed":
        if "watershedGradientScale" in params:
            se.watershedGradientScaleSlider.value = params["watershedGradientScale"]
        if "watershedSmoothing" in params:
            se.watershedSmoothingSlider.value = params["watershedSmoothing"]

    elif algorithm == "geodesic_distance":
        if "geodesicEdgeWeight" in params:
            se.geodesicEdgeWeightSlider.value = params["geodesicEdgeWeight"]
        if "geodesicDistanceScale" in params:
            se.geodesicDistanceScaleSlider.value = params["geodesicDistanceScale"]
        if "geodesicSmoothing" in params:
            se.geodesicSmoothingSlider.value = params["geodesicSmoothing"]

    elif algorithm == "level_set_cpu" or algorithm == "level_set_gpu":
        if "levelSetPropagation" in params:
            se.levelSetPropagationSlider.value = params["levelSetPropagation"]
        if "levelSetCurvature" in params:
            se.levelSetCurvatureSlider.value = params["levelSetCurvature"]
        if "levelSetIterations" in params:
            se.levelSetIterationsSlider.value = params["levelSetIterations"]

    elif algorithm == "region_growing":
        if "regionGrowingMultiplier" in params:
            se.regionGrowingMultiplierSlider.value = params["regionGrowingMultiplier"]
        if "regionGrowingIterations" in params:
            se.regionGrowingIterationsSlider.value = params["regionGrowingIterations"]

    elif algorithm == "random_walker":
        if "randomWalkerBeta" in params:
            se.randomWalkerBetaSlider.value = params["randomWalkerBeta"]

    elif algorithm == "threshold_brush":
        if "thresholdMethod" in params:
            method_idx = se.thresholdMethodCombo.findData(params["thresholdMethod"])
            if method_idx >= 0:
                se.thresholdMethodCombo.setCurrentIndex(method_idx)

    slicer.app.processEvents()


def run_segmentation(
    red_widget, volume_node, seg_node, segment_id, algorithm, params, click_indices, trial_dir=None
):
    """Run segmentation with given parameters and clicks.

    If trial_dir is provided, captures per-stroke screenshots with brush circles visible.
    """
    editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
    editor.setSegmentationNode(seg_node)
    editor.setSourceVolumeNode(volume_node)
    editor.setCurrentSegmentID(segment_id)
    editor.setActiveEffectByName("Adaptive Brush")
    slicer.app.processEvents()

    effect = editor.activeEffect()
    if not effect:
        logger.error("Adaptive Brush not available")
        return 0, 0, []

    se = effect.self()
    apply_params_to_effect(se, algorithm, params)

    click_points = [ALL_CLICK_POINTS[i] for i in click_indices]
    errors = []
    total_time = 0
    stroke_screenshots = []

    for i, ras in enumerate(click_points):
        red_widget.sliceLogic().SetSliceOffset(ras[2])
        slicer.app.processEvents()
        xy = ras_to_xy(ras, red_widget)

        # Capture screenshot BEFORE stroke with brush circle at this location
        if trial_dir:
            strokes_dir = trial_dir / "strokes"
            strokes_dir.mkdir(exist_ok=True)

            # Move brush circle to this position (simulate mouse move to show circle)
            try:
                # Update brush circle position by calling the circle update method
                se._updateBrushPreview(xy, red_widget)
                slicer.app.processEvents()
                time.sleep(0.05)

                # Capture "before" screenshot with brush circle showing where we'll paint
                # Use "1_before" and "2_after" so they sort correctly
                before_path = strokes_dir / f"stroke_{i+1:02d}_1_before.png"
                capture_screenshot(red_widget, before_path)
                stroke_screenshots.append(str(before_path))
            except Exception as e:
                logger.warning(f"Could not capture before screenshot: {e}")

        try:
            start = time.time()
            se.scriptedEffect.saveStateForUndo()
            se.isDrawing = True
            se._currentStrokeEraseMode = False
            se.processPoint(xy, red_widget)
            se.isDrawing = False
            total_time += time.time() - start
        except Exception as e:
            error_msg = f"Click {i+1} failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        slicer.app.processEvents()

        # Capture screenshot AFTER stroke showing the result
        if trial_dir:
            try:
                # Keep brush circle visible at same position
                se._updateBrushPreview(xy, red_widget)
                slicer.app.processEvents()
                time.sleep(0.05)

                after_path = strokes_dir / f"stroke_{i+1:02d}_2_after.png"
                capture_screenshot(red_widget, after_path)
                stroke_screenshots.append(str(after_path))
            except Exception as e:
                logger.warning(f"Could not capture after screenshot: {e}")

    arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, segment_id, volume_node)
    voxels = int(np.sum(arr > 0))
    return voxels, total_time * 1000, errors, stroke_screenshots


def run_trial(
    trial_id, algorithm, params, click_config, red_widget, volume_node, metrics_calc, output_dir
):
    """Run a single trial and return results."""
    click_name, click_indices = click_config

    # Create trial output directory first (needed for per-stroke screenshots)
    trial_dir = output_dir / f"trial_{trial_id:04d}"
    trial_dir.mkdir(exist_ok=True)

    # Create segmentation
    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    seg_node.CreateDefaultDisplayNodes()
    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
    seg_node.SetName(f"Trial_{trial_id}")
    segment_id = seg_node.GetSegmentation().AddEmptySegment("Test")
    slicer.app.processEvents()

    # Set display to outline mode so we can see the brush circle and segmentation
    set_display_outline(seg_node, segment_id)

    # Run segmentation with per-stroke screenshots
    voxels, time_ms, errors, stroke_screenshots = run_segmentation(
        red_widget, volume_node, seg_node, segment_id, algorithm, params, click_indices, trial_dir
    )

    # Calculate metrics
    metrics = metrics_calc.compute(seg_node, segment_id)

    # Calculate tumor center for screenshots
    click_points = [ALL_CLICK_POINTS[i] for i in click_indices]
    tumor_center = (
        sum(p[0] for p in click_points) / len(click_points),
        sum(p[1] for p in click_points) / len(click_points),
        sum(p[2] for p in click_points) / len(click_points),
    )

    # Capture final screenshots (per-stroke screenshots already captured)
    center_view(red_widget, tumor_center, fov_mm=100)

    # Outline at center
    set_display_outline(seg_node, segment_id)
    capture_screenshot(red_widget, trial_dir / "outline_center.png")

    # Fill at center
    set_display_fill(seg_node, segment_id)
    capture_screenshot(red_widget, trial_dir / "fill_center.png")

    # Multiple Z slices for verification
    for offset in [-12, -6, 6, 12]:
        pos = (tumor_center[0], tumor_center[1], tumor_center[2] + offset)
        center_view(red_widget, pos, fov_mm=100)
        name = f"z{offset:+d}"
        capture_screenshot(red_widget, trial_dir / f"fill_{name}.png")

    # Hide segmentation
    seg_node.GetDisplayNode().SetVisibility(False)

    result = {
        "trial_id": trial_id,
        "algorithm": algorithm,
        "params": params,
        "click_config": click_name,
        "num_clicks": len(click_indices),
        "voxels": voxels,
        "time_ms": time_ms,
        "dice": metrics["dice"],
        "hausdorff_95": metrics["hausdorff_95"],
        "errors": errors,
        "screenshots": str(trial_dir),
        "stroke_screenshots": stroke_screenshots,
    }

    # Clean up
    slicer.mrmlScene.RemoveNode(seg_node)

    return result


def main():
    logger.info(f"Output: {OUTPUT_DIR}")

    # Clear scene
    slicer.mrmlScene.Clear(0)

    # Load sample data
    import SampleData

    volume_node = SampleData.downloadSample("MRBrainTumor1")
    logger.info(f"Loaded: {volume_node.GetName()}")

    # Load gold standard
    from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

    manager = GoldStandardManager()
    gold_seg, gold_meta = manager.load_gold("MRBrainTumor1_tumor")
    gold_segment_id = gold_meta["segment_id"]
    gold_seg.GetDisplayNode().SetVisibility(False)
    logger.info(f"Loaded gold standard: {gold_meta['voxel_count']} voxels")

    # Setup
    slicer.util.selectModule("SegmentEditor")
    slicer.app.processEvents()
    time.sleep(0.3)

    red_widget = get_red_widget()
    metrics_calc = MetricsCalculator(gold_seg, gold_segment_id, volume_node)

    results = []
    bugs_found = []
    trial_id = 0

    # Phase 1: Test all algorithms with default radius, varying algorithm params
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Algorithm parameter optimization")
    logger.info("=" * 60)

    for algorithm, config in ALGORITHM_CONFIGS.items():
        for variation in config["variations"]:
            params = {
                "brush_radius_mm": 25.0,
                "thresholdZone": 50,
                **config["base_params"],
                **variation,
            }

            for click_name, click_indices in CLICK_CONFIGS.items():
                trial_id += 1
                logger.info(
                    f"\nTrial {trial_id}: {algorithm} | {click_name} | sens={params.get('edge_sensitivity', 50)}"
                )

                try:
                    result = run_trial(
                        trial_id,
                        algorithm,
                        params,
                        (click_name, click_indices),
                        red_widget,
                        volume_node,
                        metrics_calc,
                        OUTPUT_DIR,
                    )
                    results.append(result)

                    logger.info(
                        f"  Dice={result['dice']:.3f}, HD95={result['hausdorff_95']:.1f}mm, voxels={result['voxels']}"
                    )

                    if result["errors"]:
                        for err in result["errors"]:
                            bugs_found.append(
                                {"trial": trial_id, "algorithm": algorithm, "error": err}
                            )

                except Exception as e:
                    error_msg = f"Trial {trial_id} failed completely: {str(e)}"
                    logger.error(error_msg)
                    bugs_found.append(
                        {"trial": trial_id, "algorithm": algorithm, "error": error_msg}
                    )

    # Phase 2: Test radius variations with best algorithm (watershed)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Radius and threshold zone optimization")
    logger.info("=" * 60)

    for radius_config in RADIUS_CONFIGS:
        params = {
            "edge_sensitivity": 40,
            "watershedGradientScale": 1.5,
            "watershedSmoothing": 0.5,
            **radius_config,
        }

        trial_id += 1
        click_name = "5_standard"
        click_indices = CLICK_CONFIGS[click_name]

        logger.info(
            f"\nTrial {trial_id}: watershed | radius={params['brush_radius_mm']}mm, zone={params['thresholdZone']}%"
        )

        try:
            result = run_trial(
                trial_id,
                "watershed",
                params,
                (click_name, click_indices),
                red_widget,
                volume_node,
                metrics_calc,
                OUTPUT_DIR,
            )
            results.append(result)
            logger.info(
                f"  Dice={result['dice']:.3f}, HD95={result['hausdorff_95']:.1f}mm, voxels={result['voxels']}"
            )

        except Exception as e:
            error_msg = f"Trial {trial_id} failed: {str(e)}"
            logger.error(error_msg)
            bugs_found.append({"trial": trial_id, "algorithm": "watershed", "error": error_msg})

    # Save results
    summary = {
        "created": datetime.now().isoformat(),
        "sample_data": "MRBrainTumor1",
        "gold_standard": "MRBrainTumor1_tumor",
        "gold_voxels": gold_meta["voxel_count"],
        "total_trials": len(results),
        "bugs_found": bugs_found,
        "results": results,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)

    # Best results per algorithm
    by_algorithm = {}
    for r in results:
        algo = r["algorithm"]
        if algo not in by_algorithm:
            by_algorithm[algo] = []
        by_algorithm[algo].append(r)

    best_per_algo = {}
    for algo, trials in by_algorithm.items():
        valid_trials = [t for t in trials if t["dice"] > 0]
        if valid_trials:
            best = max(valid_trials, key=lambda x: x["dice"])
            best_per_algo[algo] = best
            logger.info(f"\n{algo}:")
            logger.info(f"  Best Dice: {best['dice']:.3f}")
            logger.info(f"  HD95: {best['hausdorff_95']:.1f}mm")
            logger.info(f"  Clicks: {best['click_config']}")
            logger.info(f"  Params: {best['params']}")

    # Overall best
    all_valid = [r for r in results if r["dice"] > 0]
    if all_valid:
        overall_best = max(all_valid, key=lambda x: x["dice"])
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL BEST:")
        logger.info(f"  Algorithm: {overall_best['algorithm']}")
        logger.info(f"  Dice: {overall_best['dice']:.3f}")
        logger.info(f"  HD95: {overall_best['hausdorff_95']:.1f}mm")
        logger.info(f"  Clicks: {overall_best['click_config']}")
        logger.info(f"  Trial: {overall_best['trial_id']}")

    # Save analysis
    analysis = {
        "best_per_algorithm": {
            algo: {"trial_id": r["trial_id"], "dice": r["dice"], "params": r["params"]}
            for algo, r in best_per_algo.items()
        },
        "overall_best": {
            "trial_id": overall_best["trial_id"],
            "algorithm": overall_best["algorithm"],
            "dice": overall_best["dice"],
            "hausdorff_95": overall_best["hausdorff_95"],
            "params": overall_best["params"],
        }
        if all_valid
        else None,
        "bugs_summary": f"{len(bugs_found)} bugs/errors found",
    }

    with open(OUTPUT_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Create lab notebook
    from SegmentEditorAdaptiveBrushTesterLib import LabNotebook

    notebook = LabNotebook(
        "Comprehensive Algorithm Optimization",
        filename=f"comprehensive_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    notebook.add_section(
        "Overview",
        f"""
Comprehensive parameter optimization across all algorithms.

- **Total trials:** {len(results)}
- **Algorithms tested:** {len(ALGORITHM_CONFIGS)}
- **Click configurations:** {len(CLICK_CONFIGS)}
- **Radius configurations:** {len(RADIUS_CONFIGS)}
- **Bugs found:** {len(bugs_found)}
""",
    )

    notebook.add_section("Best Results per Algorithm", "")
    for algo, best in best_per_algo.items():
        notebook.add_subsection(
            algo,
            f"""
- **Dice:** {best['dice']:.3f}
- **Hausdorff 95%:** {best['hausdorff_95']:.1f}mm
- **Click config:** {best['click_config']}
- **Parameters:** `{best['params']}`
""",
        )

    if all_valid:
        notebook.add_section(
            "Overall Best",
            f"""
- **Algorithm:** {overall_best['algorithm']}
- **Dice:** {overall_best['dice']:.3f}
- **Hausdorff 95%:** {overall_best['hausdorff_95']:.1f}mm
- **Trial ID:** {overall_best['trial_id']}
- **Click config:** {overall_best['click_config']}
""",
        )

    if bugs_found:
        notebook.add_section("Bugs and Errors Found", "")
        for bug in bugs_found:
            notebook.add_bullet(f"Trial {bug['trial']}: {bug['algorithm']} - {bug['error']}")

    notebook.add_section(
        "Files",
        f"""
- Results: `{OUTPUT_DIR}/summary.json`
- Analysis: `{OUTPUT_DIR}/analysis.json`
- Screenshots: `{OUTPUT_DIR}/trial_*/`
""",
    )

    notebook_path = notebook.save()
    logger.info(f"\nLab notebook: {notebook_path}")

    print(f"\n{'='*60}")
    print("COMPREHENSIVE OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Trials: {len(results)}")
    print(f"Bugs: {len(bugs_found)}")
    if all_valid:
        print(f"Best: {overall_best['algorithm']} (Dice={overall_best['dice']:.3f})")
    print(f"{'='*60}")

    if EXIT_WHEN_DONE:
        slicer.util.exit()


if __name__ == "__main__":
    main()
