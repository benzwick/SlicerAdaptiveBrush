"""Gold standard regression test.

Loads gold standard segmentations and compares test segmentations against them
to detect regressions. Uses Dice coefficient and Hausdorff distance metrics.
"""

from __future__ import annotations

import logging
import time

import slicer
import vtk
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test
from SegmentEditorAdaptiveBrushTesterLib.GoldStandardManager import GoldStandardManager
from SegmentEditorAdaptiveBrushTesterLib.SegmentationMetrics import (
    SegmentationMetrics,
    StrokeMetricsTracker,
)

logger = logging.getLogger(__name__)


# Minimum acceptable Dice coefficient (flag regression if below)
DICE_REGRESSION_THRESHOLD = 0.80

# Maximum acceptable Hausdorff 95% distance in mm
HAUSDORFF_REGRESSION_THRESHOLD = 10.0


@register_test(category="regression")
class TestRegressionGold(TestCase):
    """Test algorithms against gold standard segmentations."""

    name = "regression_gold"
    description = "Compare algorithm results against gold standard segmentations"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.effect = None
        self.redWidget = None
        self.gold_manager = GoldStandardManager()
        self.results: list[dict] = []

    def setup(self, ctx: TestContext) -> None:
        """Load gold standards and prepare for testing."""
        logger.info("Setting up gold standard regression test")

        # List available gold standards
        standards = self.gold_manager.list_gold_standards()
        ctx.log(f"Found {len(standards)} gold standard(s)")

        if not standards:
            ctx.log("No gold standards found - skipping regression test")
            ctx.log("Create gold standards using create-gold-standard skill")
            return

        for std in standards:
            ctx.log(f"  - {std['name']}: {std.get('voxel_count', 'N/A')} voxels")

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # We'll test each gold standard in the run phase
        ctx.screenshot("[setup] Gold standards loaded")

    def run(self, ctx: TestContext) -> None:
        """Test each gold standard."""
        logger.info("Running gold standard regression tests")

        standards = self.gold_manager.list_gold_standards()
        if not standards:
            ctx.log("No gold standards to test")
            return

        for std_info in standards:
            self._test_gold_standard(ctx, std_info)

    def _test_gold_standard(self, ctx: TestContext, std_info: dict) -> None:
        """Test a single gold standard."""
        name = std_info["name"]
        ctx.log(f"\n{'=' * 50}")
        ctx.log(f"Testing gold standard: {name}")
        ctx.log(f"{'=' * 50}")

        # Clear scene for each gold standard test
        slicer.mrmlScene.Clear(0)

        try:
            # Load gold standard
            gold_seg_node, metadata = self.gold_manager.load_gold(name)
            gold_segment_id = metadata.get("segment_id", "Segment_1")
            volume_name = metadata.get("volume", {}).get("name", "")
            algorithm = metadata.get("algorithm", "watershed")
            parameters = metadata.get("parameters", {})
            clicks = metadata.get("clicks", [])

            ctx.log(f"Volume: {volume_name}")
            ctx.log(f"Algorithm: {algorithm}")
            ctx.log(f"Gold voxel count: {metadata.get('voxel_count', 'N/A')}")
            ctx.log(f"Click points: {len(clicks)}")

            # Load sample data
            import SampleData

            # Map volume name to sample data name
            sample_map = {
                "MRBrainTumor1": "MRBrainTumor1",
                "MRHead": "MRHead",
            }
            sample_name = None
            for key, value in sample_map.items():
                if key in volume_name:
                    sample_name = value
                    break

            if sample_name is None:
                ctx.log(f"Unknown volume: {volume_name} - skipping")
                return

            self.volume_node = SampleData.downloadSample(sample_name)
            if self.volume_node is None:
                ctx.log(f"Failed to load sample data: {sample_name}")
                return

            # Create test segmentation
            self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            self.segmentation_node.CreateDefaultDisplayNodes()
            self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                self.volume_node
            )

            # Create segment
            segment_name = "Test"
            test_segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(segment_name)

            # Activate Segment Editor
            slicer.util.selectModule("SegmentEditor")
            slicer.app.processEvents()

            segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
            self.segment_editor_widget = segment_editor_module.editor
            self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
            self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
            self.segment_editor_widget.setCurrentSegmentID(test_segment_id)
            slicer.app.processEvents()

            # Activate Adaptive Brush
            self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
            self.effect = self.segment_editor_widget.activeEffect()

            if self.effect is None:
                ctx.log("Failed to activate Adaptive Brush effect")
                return

            scripted_effect = self.effect.self()

            # Set algorithm
            combo = scripted_effect.algorithmCombo
            idx = combo.findData(algorithm)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            slicer.app.processEvents()

            # Set parameters
            brush_radius = parameters.get("brush_radius_mm", 25.0)
            edge_sensitivity = parameters.get("edge_sensitivity", 40)

            scripted_effect.radiusSlider.value = brush_radius
            scripted_effect.sensitivitySlider.value = edge_sensitivity
            slicer.app.processEvents()

            # Get slice widget
            layoutManager = slicer.app.layoutManager()
            self.redWidget = layoutManager.sliceWidget("Red")

            # Initialize stroke metrics tracker
            tracker = StrokeMetricsTracker(gold_seg_node, gold_segment_id, self.volume_node)

            ctx.screenshot(f"[{name}] Ready to paint")

            # Paint at each click location
            total_time = 0.0
            for i, click in enumerate(clicks):
                ras = click.get("ras")
                if ras is None:
                    continue

                # Navigate to click location
                self.redWidget.sliceLogic().SetSliceOffset(ras[2])
                slicer.app.processEvents()

                # Convert RAS to XY
                xy = self._rasToXy(ras, self.redWidget)
                if xy is None:
                    ctx.log(f"  Click {i + 1}: Could not convert RAS {ras}")
                    continue

                # Paint
                start = time.time()
                scripted_effect.scriptedEffect.saveStateForUndo()
                scripted_effect.isDrawing = True
                scripted_effect._currentStrokeEraseMode = False
                scripted_effect.processPoint(xy, self.redWidget)
                scripted_effect.isDrawing = False
                elapsed = time.time() - start
                total_time += elapsed

                slicer.app.processEvents()

                # Record stroke metrics
                stroke_record = tracker.record_stroke(
                    self.segmentation_node,
                    test_segment_id,
                    {"click": i + 1, "ras": ras},
                )

                ctx.log(
                    f"  Click {i + 1}: Dice={stroke_record.dice:.3f}, "
                    f"HD95={stroke_record.hausdorff_95:.1f}mm"
                )

            # Get final metrics
            final_metrics = SegmentationMetrics.compute(
                self.segmentation_node,
                test_segment_id,
                gold_seg_node,
                gold_segment_id,
                self.volume_node,
            )

            ctx.screenshot(
                f"[{name}] Final: Dice={final_metrics.dice:.3f}, "
                f"HD95={final_metrics.hausdorff_95:.1f}mm"
            )

            # Record summary
            summary = tracker.get_summary()

            result = {
                "gold_standard": name,
                "algorithm": algorithm,
                "dice": final_metrics.dice,
                "hausdorff_95": final_metrics.hausdorff_95,
                "hausdorff_max": final_metrics.hausdorff_max,
                "test_voxels": final_metrics.test_voxels,
                "gold_voxels": final_metrics.reference_voxels,
                "total_strokes": len(clicks),
                "total_time_ms": total_time * 1000,
                "strokes_to_90pct": summary.get("strokes_to_90pct"),
                "regression": (
                    final_metrics.dice < DICE_REGRESSION_THRESHOLD
                    or final_metrics.hausdorff_95 > HAUSDORFF_REGRESSION_THRESHOLD
                ),
            }
            self.results.append(result)

            ctx.metric(f"{name}_dice", final_metrics.dice)
            ctx.metric(f"{name}_hausdorff_95", final_metrics.hausdorff_95)

            ctx.log(f"\nFinal metrics for {name}:")
            ctx.log(f"  Dice: {final_metrics.dice:.3f}")
            ctx.log(f"  Hausdorff 95%: {final_metrics.hausdorff_95:.1f}mm")
            ctx.log(f"  Test voxels: {final_metrics.test_voxels:,}")
            ctx.log(f"  Gold voxels: {final_metrics.reference_voxels:,}")

            if result["regression"]:
                ctx.log("  ** REGRESSION DETECTED **")

        except Exception as e:
            logger.exception(f"Error testing gold standard {name}: {e}")
            ctx.log(f"ERROR: {e}")

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

    def verify(self, ctx: TestContext) -> None:
        """Check for regressions."""
        logger.info("Verifying regression test results")

        if not self.results:
            ctx.log("No results to verify")
            return

        ctx.log("\n" + "=" * 60)
        ctx.log("REGRESSION TEST SUMMARY")
        ctx.log("=" * 60)
        ctx.log(f"Dice threshold: >= {DICE_REGRESSION_THRESHOLD}")
        ctx.log(f"Hausdorff 95% threshold: <= {HAUSDORFF_REGRESSION_THRESHOLD}mm")
        ctx.log("")

        regressions = []

        for r in self.results:
            name = r["gold_standard"]
            ctx.log(f"{name}:")
            ctx.log(f"  Dice: {r['dice']:.3f}")
            ctx.log(f"  Hausdorff 95%: {r['hausdorff_95']:.1f}mm")

            if r["regression"]:
                ctx.log("  ** REGRESSION **")
                regressions.append(name)
            else:
                ctx.log("  PASS")

        ctx.log("")
        if regressions:
            ctx.log(f"REGRESSIONS DETECTED: {len(regressions)}")
            for name in regressions:
                ctx.log(f"  - {name}")
            ctx.assert_true(False, f"Regressions detected in {len(regressions)} gold standard(s)")
        else:
            ctx.log("ALL GOLD STANDARDS PASSED")
            ctx.assert_true(True, "No regressions detected")

        ctx.screenshot("[verify] Regression test complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        logger.info("Tearing down regression test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")
