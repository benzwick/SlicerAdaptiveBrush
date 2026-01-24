"""Tumor segmentation optimization test.

Systematically tests all algorithms with multiple click points to find
optimal parameters for tumor segmentation in MRBrainTumor1 sample data.

This test is designed for iterative optimization - run it, analyze results,
adjust parameters, repeat.
"""

from __future__ import annotations

import logging

import numpy as np
import slicer
import vtk
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


# Candidate click points in RAS coordinates for MRBrainTumor1 tumor
# Manually placed by user on 2026-01-24 via Slicer markups (converted from LPS to RAS)
# LPS->RAS: R=-L, A=-P, S=S
# The tumor is a ring-enhancing lesion in the left hemisphere
# Format: (R, A, S) - Right, Anterior, Superior
TUMOR_CLICK_POINTS = [
    # F-1: LPS(5.31, -34.77, 20.83) -> RAS
    (-5.31, 34.77, 20.83),
    # F-2: LPS(5.31, -25.12, 35.97) -> RAS
    (-5.31, 25.12, 35.97),
    # F-3: LPS(5.31, -20.70, 22.17) -> RAS
    (-5.31, 20.70, 22.17),
    # F-4: LPS(6.16, -38.28, 30.61) -> RAS
    (-6.16, 38.28, 30.61),
    # F-6: LPS(1.35, -28.65, 18.90) -> RAS
    (-1.35, 28.65, 18.90),
]

# All available algorithms
# Note: level_set requires "level_set_cpu" or "level_set_gpu" suffix
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

# Default parameters for optimization (can be tuned)
# ITERATION 4: Larger brush, mid sensitivity
OPTIMIZATION_PARAMS = {
    "brush_radius_mm": 25.0,  # Larger brush to capture more tumor
    "edge_sensitivity": 40,  # Mid-range sensitivity
    "inner_radius_ratio": 0.3,  # Sampling zone ratio
}


@register_test(category="optimization")
class TestOptimizationTumor(TestCase):
    """Optimize tumor segmentation parameters."""

    name = "optimization_tumor"
    description = "Find optimal parameters for tumor segmentation in MRBrainTumor1"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.effect = None
        self.redWidget = None
        self.results: list[dict] = []

    def setup(self, ctx: TestContext) -> None:
        """Load MRBrainTumor1 and prepare for testing."""
        logger.info("Setting up tumor optimization test")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Load MRBrainTumor1 sample data
        import SampleData

        ctx.log("Loading MRBrainTumor1 sample data")
        self.volume_node = SampleData.downloadSample("MRBrainTumor1")

        if self.volume_node is None:
            raise RuntimeError("Failed to load MRBrainTumor1 sample data")

        ctx.log(f"Loaded volume: {self.volume_node.GetName()}")

        # Get volume info for documentation
        spacing = self.volume_node.GetSpacing()
        dimensions = self.volume_node.GetImageData().GetDimensions()
        ctx.log(f"Volume spacing: {spacing}")
        ctx.log(f"Volume dimensions: {dimensions}")

        # Create segmentation node
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        # Switch to Segment Editor module to see GUI
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        # Use the actual module's segment editor widget (not a hidden one)
        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        slicer.app.processEvents()

        # Activate Adaptive Brush by clicking the effect button
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        # Get slice widget and set up for brush visibility
        layoutManager = slicer.app.layoutManager()
        self.redWidget = layoutManager.sliceWidget("Red")
        redLogic = self.redWidget.sliceLogic()

        # Set brush radius for visibility in screenshots
        scripted_effect = self.effect.self()
        scripted_effect.radiusSlider.value = OPTIMIZATION_PARAMS["brush_radius_mm"]
        slicer.app.processEvents()

        # Navigate to first tumor point and show brush circle
        first_ras = TUMOR_CLICK_POINTS[0]
        redLogic.SetSliceOffset(first_ras[2])
        slicer.app.processEvents()

        first_xy = self._rasToXy(first_ras, self.redWidget)
        if first_xy:
            scripted_effect._updateBrushPreview(first_xy, self.redWidget, eraseMode=False)
            self.redWidget.sliceView().forceRender()
            slicer.app.processEvents()

        ctx.screenshot("[setup] MRBrainTumor1 loaded, Adaptive Brush active")

    def run(self, ctx: TestContext) -> None:
        """Test each algorithm with multiple click points."""
        logger.info("Running tumor optimization tests")

        scripted_effect = self.effect.self()

        # Set brush radius using the slider widget (like a user would)
        brush_radius = OPTIMIZATION_PARAMS["brush_radius_mm"]
        scripted_effect.radiusSlider.value = brush_radius
        ctx.log(f"Brush radius set to {brush_radius}mm")

        # Set edge sensitivity using the slider widget
        edge_sensitivity = OPTIMIZATION_PARAMS["edge_sensitivity"]
        scripted_effect.sensitivitySlider.value = edge_sensitivity
        ctx.log(f"Edge sensitivity set to {edge_sensitivity}")

        # Use the slice widget from setup
        redLogic = self.redWidget.sliceLogic()

        # Test each algorithm - clean slate for each
        for algo in ALGORITHMS:
            ctx.log(f"\n{'='*50}")
            ctx.log(f"Testing algorithm: {algo}")
            ctx.log(f"{'='*50}")

            # Remove all existing segments for clean comparison
            segmentation = self.segmentation_node.GetSegmentation()
            segmentation.RemoveAllSegments()

            # Create ONE segment for this algorithm
            segment_name = "Tumor"
            segment_id = segmentation.AddEmptySegment(segment_name)
            self.segment_editor_widget.setCurrentSegmentID(segment_id)
            slicer.app.processEvents()

            # Re-activate Adaptive Brush (segment removal may have deactivated it)
            self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
            self.effect = self.segment_editor_widget.activeEffect()
            scripted_effect = self.effect.self()
            slicer.app.processEvents()

            # Select algorithm using the combo box (like a user would)
            combo = scripted_effect.algorithmCombo
            idx = combo.findData(algo)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            slicer.app.processEvents()

            # Navigate to first click point and show brush circle for algorithm screenshot
            first_ras = TUMOR_CLICK_POINTS[0]
            redLogic.SetSliceOffset(first_ras[2])
            first_xy = self._rasToXy(first_ras, self.redWidget)
            if first_xy:
                scripted_effect._updateBrushPreview(first_xy, self.redWidget, eraseMode=False)
                self.redWidget.sliceView().forceRender()
            slicer.app.processEvents()

            ctx.screenshot(f"[{algo}] Algorithm selected, ready to paint")

            # Paint all 5 points into the SAME segment (cumulative)
            total_time = 0.0
            import time

            for i, ras in enumerate(TUMOR_CLICK_POINTS):
                # Navigate to click location
                redLogic.SetSliceOffset(ras[2])
                slicer.app.processEvents()

                # Convert RAS to XY
                xy = self._rasToXy(ras, self.redWidget)
                if xy is None:
                    ctx.log(f"  Point {i+1}: Could not convert RAS {ras}")
                    continue

                # Show brush at this location before painting
                scripted_effect._updateBrushPreview(xy, self.redWidget, eraseMode=False)
                self.redWidget.sliceView().forceRender()
                slicer.app.processEvents()

                ctx.log(f"  Point {i+1}: Painting at RAS {ras}")

                # Paint (cumulative into same segment)
                start = time.time()
                scripted_effect.scriptedEffect.saveStateForUndo()
                scripted_effect.isDrawing = True
                scripted_effect._currentStrokeEraseMode = False
                scripted_effect.processPoint(xy, self.redWidget)
                scripted_effect.isDrawing = False
                elapsed = time.time() - start
                total_time += elapsed

                slicer.app.processEvents()

                # Show brush and take screenshot after this stroke
                scripted_effect._updateBrushPreview(xy, self.redWidget, eraseMode=False)
                self.redWidget.sliceView().forceRender()
                slicer.app.processEvents()

                # Count voxels so far and capture result
                voxels_so_far = self._count_segment_voxels(segment_id)
                ctx.screenshot(f"[{algo}_click{i+1}] Click {i+1}: {voxels_so_far:,} voxels total")

            # Count total voxels after all 5 points
            voxel_count = self._count_segment_voxels(segment_id)
            ctx.log(f"  TOTAL for {algo}: {voxel_count} voxels in {total_time*1000:.0f}ms")

            # Record result
            self.results.append(
                {
                    "algorithm": algo,
                    "total_voxels": voxel_count,
                    "total_time_ms": total_time * 1000,
                    "brush_radius_mm": brush_radius,
                    "edge_sensitivity": edge_sensitivity,
                    "num_points": len(TUMOR_CLICK_POINTS),
                }
            )

            ctx.metric(f"{algo}_total_voxels", voxel_count)
            ctx.metric(f"{algo}_total_time_ms", total_time * 1000)

        # Show brush for summary screenshot
        first_ras = TUMOR_CLICK_POINTS[0]
        redLogic.SetSliceOffset(first_ras[2])
        first_xy = self._rasToXy(first_ras, self.redWidget)
        if first_xy:
            scripted_effect._updateBrushPreview(first_xy, self.redWidget, eraseMode=False)
            self.redWidget.sliceView().forceRender()
            slicer.app.processEvents()

        ctx.screenshot("[summary] All algorithms tested")

    def _rasToXy(self, ras, sliceWidget):
        """Convert RAS coordinates to screen XY for a slice widget."""
        sliceLogic = sliceWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()

        # Get XY to RAS matrix and invert it
        xyToRas = sliceNode.GetXYToRAS()
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xyToRas, rasToXy)

        rasPoint = [ras[0], ras[1], ras[2], 1]
        xyPoint = [0, 0, 0, 1]
        rasToXy.MultiplyPoint(rasPoint, xyPoint)

        return (int(xyPoint[0]), int(xyPoint[1]))

    def _count_segment_voxels(self, segment_id: str) -> int:
        """Count the number of voxels in a segment."""
        try:
            labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(
                self.segmentation_node, segment_id, self.volume_node
            )
            return int(np.sum(labelmap > 0))
        except Exception as e:
            logger.warning(f"Could not count voxels: {e}")
            return 0

    def verify(self, ctx: TestContext) -> None:
        """Analyze and report optimization results."""
        logger.info("Analyzing optimization results")

        if not self.results:
            ctx.log("No results to analyze")
            return

        # Report summary
        ctx.log("\n" + "=" * 60)
        ctx.log("OPTIMIZATION RESULTS SUMMARY")
        ctx.log("=" * 60)
        ctx.log(
            f"Parameters: brush={OPTIMIZATION_PARAMS['brush_radius_mm']}mm, "
            f"sensitivity={OPTIMIZATION_PARAMS['edge_sensitivity']}"
        )
        ctx.log("")

        best_algo = None
        best_voxels = 0

        # Sort by voxel count descending
        sorted_results = sorted(self.results, key=lambda x: x["total_voxels"], reverse=True)

        for r in sorted_results:
            algo = r["algorithm"]
            voxels = r["total_voxels"]
            time_ms = r["total_time_ms"]

            ctx.log(f"{algo}:")
            ctx.log(f"  Voxels: {voxels:,}")
            ctx.log(f"  Time: {time_ms:.0f}ms")
            ctx.log(f"  Voxels/ms: {voxels/time_ms:.1f}" if time_ms > 0 else "  Voxels/ms: N/A")

            if voxels > best_voxels:
                best_voxels = voxels
                best_algo = algo

        ctx.log("")
        ctx.log(f"BEST: {best_algo} with {best_voxels:,} voxels")

        # Assert we got some segmentation
        total_all = sum(r["total_voxels"] for r in self.results)
        ctx.assert_greater(total_all, 0, "Should have segmented some voxels")

        # Show brush for verify screenshot
        if self.redWidget and self.effect:
            first_ras = TUMOR_CLICK_POINTS[0]
            self.redWidget.sliceLogic().SetSliceOffset(first_ras[2])
            first_xy = self._rasToXy(first_ras, self.redWidget)
            if first_xy:
                scripted_effect = self.effect.self()
                scripted_effect._updateBrushPreview(first_xy, self.redWidget, eraseMode=False)
                self.redWidget.sliceView().forceRender()
                slicer.app.processEvents()

        ctx.screenshot(f"[verify] Best: {best_algo} with {best_voxels:,} voxels")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        logger.info("Tearing down optimization test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        # Write detailed results to log
        ctx.log("\nDetailed results saved for analysis")
        for r in self.results:
            ctx.log(f"  {r}")

        ctx.log("Teardown complete")
