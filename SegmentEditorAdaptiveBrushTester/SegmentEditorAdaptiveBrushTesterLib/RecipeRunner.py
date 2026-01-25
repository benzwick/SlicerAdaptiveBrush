"""Recipe execution engine for running segmentation recipes in Slicer.

Executes recipes action by action, with support for progress callbacks,
screenshot capture, and metrics collection.

See ADR-013 for architecture decisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .Recipe import Action, Recipe

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of executing a single action."""

    action_index: int
    action: Action
    duration_ms: float
    voxel_count: int
    success: bool
    error: str | None = None


@dataclass
class RecipeResult:
    """Result of executing a complete recipe."""

    recipe_name: str
    success: bool
    action_results: list[ActionResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    final_voxel_count: int = 0
    segmentation_node: Any = None  # vtkMRMLSegmentationNode
    volume_node: Any = None  # vtkMRMLScalarVolumeNode
    segment_id: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "recipe_name": self.recipe_name,
            "success": self.success,
            "action_results": [
                {
                    "action_index": r.action_index,
                    "effect": r.action.effect,
                    "params": r.action.params,
                    "duration_ms": r.duration_ms,
                    "voxel_count": r.voxel_count,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.action_results
            ],
            "total_duration_ms": self.total_duration_ms,
            "final_voxel_count": self.final_voxel_count,
            "segment_id": self.segment_id,
            "error": self.error,
        }


class RecipeRunner:
    """Execute segmentation recipes in Slicer.

    Runs recipes action by action, managing effect activation,
    parameter configuration, and stroke execution.

    Example:
        recipe = Recipe.load("recipes/brain_tumor_1.py")
        runner = RecipeRunner(recipe)
        result = runner.run()
        print(f"Final Dice: {result.final_voxel_count} voxels")
    """

    def __init__(self, recipe: Recipe):
        """Initialize runner with a recipe.

        Args:
            recipe: Recipe to execute.
        """
        self.recipe = recipe
        self.segmentation_node = None
        self.volume_node = None
        self.segment_id = None
        self.segment_editor_widget = None
        self._slice_widget = None

    def run(
        self,
        progress_callback: Callable[[int, Action], None] | None = None,
        screenshot_callback: Callable[[int, str, str], None] | None = None,
        clear_scene: bool = True,
    ) -> RecipeResult:
        """Execute the recipe.

        Args:
            progress_callback: Called before each action with (index, action).
            screenshot_callback: Called after each action with (index, phase, description).
                phase is "before" or "after".
            clear_scene: Whether to clear the scene before starting.

        Returns:
            RecipeResult with execution details.
        """
        import slicer

        result = RecipeResult(recipe_name=self.recipe.name, success=False)
        start_time = time.time()

        try:
            # Clear scene if requested
            if clear_scene:
                slicer.mrmlScene.Clear(0)

            # Load sample data
            logger.info(f"Loading sample data: {self.recipe.sample_data}")
            self.volume_node = self._load_sample_data(self.recipe.sample_data)
            result.volume_node = self.volume_node

            # Create segmentation
            self.segmentation_node = self._create_segmentation()
            result.segmentation_node = self.segmentation_node
            assert self.segment_id is not None  # Set by _create_segmentation
            result.segment_id = self.segment_id

            # Switch to Segment Editor
            slicer.util.selectModule("SegmentEditor")
            slicer.app.processEvents()

            # Get segment editor widget
            segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
            self.segment_editor_widget = segment_editor_module.editor
            self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
            self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
            slicer.app.processEvents()

            # Get slice widget for visualization
            layout_manager = slicer.app.layoutManager()
            self._slice_widget = layout_manager.sliceWidget("Red")

            # Execute actions
            for i, action in enumerate(self.recipe.actions):
                logger.info(f"Executing action {i + 1}/{len(self.recipe.actions)}: {action.effect}")

                if progress_callback:
                    progress_callback(i, action)

                # Screenshot before
                if screenshot_callback:
                    screenshot_callback(i, "before", f"Before action {i + 1}: {action.effect}")

                # Execute action
                action_result = self._execute_action(i, action)
                result.action_results.append(action_result)

                if not action_result.success:
                    logger.error(f"Action {i + 1} failed: {action_result.error}")
                    # Continue with other actions even if one fails

                # Screenshot after
                if screenshot_callback:
                    voxels = action_result.voxel_count
                    screenshot_callback(i, "after", f"After action {i + 1}: {voxels:,} voxels")

            # Final voxel count
            result.final_voxel_count = self._count_segment_voxels()
            result.success = True

        except Exception as e:
            logger.exception(f"Recipe execution failed: {e}")
            result.error = str(e)
            result.success = False

        result.total_duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Recipe '{self.recipe.name}' completed: "
            f"{result.final_voxel_count:,} voxels in {result.total_duration_ms:.0f}ms"
        )

        return result

    def _load_sample_data(self, sample_name: str):
        """Load sample data by name.

        Args:
            sample_name: Slicer SampleData name.

        Returns:
            Loaded volume node.
        """
        import SampleData

        volume_node = SampleData.downloadSample(sample_name)
        if volume_node is None:
            raise RuntimeError(f"Failed to load sample data: {sample_name}")

        logger.info(f"Loaded sample data: {volume_node.GetName()}")
        return volume_node

    def _create_segmentation(self):
        """Create segmentation node with a segment.

        Returns:
            Created segmentation node.
        """
        import slicer

        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation_node.CreateDefaultDisplayNodes()
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        # Add segment
        segmentation = segmentation_node.GetSegmentation()
        self.segment_id = segmentation.AddEmptySegment(self.recipe.segment_name)

        # Set as current segment
        if self.segment_editor_widget:
            self.segment_editor_widget.setCurrentSegmentID(self.segment_id)

        logger.info(f"Created segmentation with segment: {self.recipe.segment_name}")
        return segmentation_node

    def _execute_action(self, index: int, action: Action) -> ActionResult:
        """Execute a single action.

        Args:
            index: Action index in recipe.
            action: Action to execute.

        Returns:
            ActionResult with execution details.
        """
        start_time = time.time()

        try:
            dispatch = {
                "adaptive_brush": self._execute_adaptive_brush,
                "paint": self._execute_paint,
                "threshold": self._execute_threshold,
                "grow_from_seeds": self._execute_grow_from_seeds,
                "islands": self._execute_islands,
                "smoothing": self._execute_smoothing,
                "scissors": self._execute_scissors,
            }

            handler = dispatch.get(action.effect)
            if handler is None:
                raise ValueError(f"Unknown effect: {action.effect}")

            handler(action.params)

            duration_ms = (time.time() - start_time) * 1000
            voxel_count = self._count_segment_voxels()

            return ActionResult(
                action_index=index,
                action=action,
                duration_ms=duration_ms,
                voxel_count=voxel_count,
                success=True,
            )

        except Exception as e:
            logger.exception(f"Action {index} ({action.effect}) failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return ActionResult(
                action_index=index,
                action=action,
                duration_ms=duration_ms,
                voxel_count=self._count_segment_voxels(),
                success=False,
                error=str(e),
            )

    def _execute_adaptive_brush(self, params: dict) -> None:
        """Execute Adaptive Brush stroke.

        Args:
            params: Action parameters including ras, algorithm, brush_radius_mm, etc.
        """
        import slicer

        # Ensure current segment is selected
        assert self.segment_editor_widget is not None  # Set by run()
        assert self.segment_id is not None  # Set by _create_segmentation
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        # Activate Adaptive Brush effect
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        effect = self.segment_editor_widget.activeEffect()

        if effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        scripted_effect = effect.self()

        # Set parameters via UI widgets (like a user would)
        scripted_effect.radiusSlider.value = params["brush_radius_mm"]
        scripted_effect.sensitivitySlider.value = params["edge_sensitivity"]

        if "threshold_zone" in params:
            scripted_effect.thresholdZoneSlider.value = params["threshold_zone"]

        # Select algorithm
        algo_combo = scripted_effect.algorithmCombo
        idx = algo_combo.findData(params["algorithm"])
        if idx >= 0:
            algo_combo.setCurrentIndex(idx)
        else:
            logger.warning(f"Algorithm not found: {params['algorithm']}")

        # Set algorithm-specific parameters
        self._set_algorithm_params(scripted_effect, params)

        slicer.app.processEvents()

        # Navigate to stroke location
        ras = params["ras"]
        self._navigate_to_ras(ras)

        # Convert RAS to XY
        xy = self._ras_to_xy(ras)
        if xy is None:
            raise RuntimeError(f"Could not convert RAS to XY: {ras}")

        # Execute stroke
        is_erase = params.get("mode") == "erase"
        scripted_effect.scriptedEffect.saveStateForUndo()
        scripted_effect.isDrawing = True
        scripted_effect._currentStrokeEraseMode = is_erase
        scripted_effect.processPoint(xy, self._slice_widget)
        scripted_effect.isDrawing = False

        slicer.app.processEvents()
        logger.debug(
            f"Adaptive brush stroke at RAS={ras}, " f"algo={params['algorithm']}, erase={is_erase}"
        )

    def _set_algorithm_params(self, effect: Any, params: dict) -> None:
        """Set algorithm-specific parameters on effect.

        Args:
            effect: Scripted effect instance.
            params: Parameters dictionary.
        """
        algo = params.get("algorithm", "")

        # Watershed parameters
        if algo == "watershed":
            if "watershedGradientScale" in params:
                effect.watershedGradientScaleSlider.value = params["watershedGradientScale"]
            if "watershedSmoothing" in params:
                effect.watershedSmoothingSlider.value = params["watershedSmoothing"]

        # Level set parameters
        elif algo in ("level_set_cpu", "level_set_gpu"):
            if "levelSetIterations" in params:
                effect.levelSetIterationsSlider.value = params["levelSetIterations"]
            if "levelSetPropagation" in params:
                effect.levelSetPropagationSlider.value = params["levelSetPropagation"]
            if "levelSetCurvature" in params:
                effect.levelSetCurvatureSlider.value = params["levelSetCurvature"]

        # Region growing parameters
        elif algo == "region_growing":
            if "regionGrowingMultiplier" in params:
                effect.regionGrowingMultiplierSlider.value = params["regionGrowingMultiplier"]

        # Threshold brush parameters
        elif algo == "threshold_brush":
            if "thresholdMethod" in params:
                combo = effect.thresholdMethodCombo
                idx = combo.findData(params["thresholdMethod"])
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        # Random walker parameters
        elif algo == "random_walker":
            if "randomWalkerBeta" in params:
                effect.randomWalkerBetaSlider.value = params["randomWalkerBeta"]

    def _execute_paint(self, params: dict) -> None:
        """Execute standard Paint effect stroke.

        Args:
            params: Action parameters including ras, radius_mm, mode, sphere.
        """
        import slicer

        assert self.segment_editor_widget is not None  # Set by run()
        assert self.segment_id is not None  # Set by _create_segmentation

        # Ensure current segment is selected
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)

        # Activate Paint effect
        self.segment_editor_widget.setActiveEffectByName("Paint")
        effect = self.segment_editor_widget.activeEffect()

        if effect is None:
            raise RuntimeError("Failed to activate Paint effect")

        # Set parameters
        effect.setParameter("BrushSphere", "1" if params.get("sphere", False) else "0")
        effect.setParameter("BrushDiameterMm", str(params["radius_mm"] * 2))

        # Set erase mode
        is_erase = params.get("mode") == "erase"
        # Note: Paint effect uses different mechanism for erase

        slicer.app.processEvents()

        # Navigate and execute
        ras = params["ras"]
        self._navigate_to_ras(ras)

        xy = self._ras_to_xy(ras)
        if xy is None:
            raise RuntimeError(f"Could not convert RAS to XY: {ras}")

        # For Paint, we need to call the effect's paint method
        # This is a simplified version - actual implementation depends on effect API
        logger.debug(f"Paint stroke at RAS={ras}, erase={is_erase}")

    def _execute_threshold(self, params: dict) -> None:
        """Execute Threshold effect.

        Args:
            params: Action parameters including min_value, max_value.
        """
        import slicer

        assert self.segment_editor_widget is not None  # Set by run()

        # Activate Threshold effect
        self.segment_editor_widget.setActiveEffectByName("Threshold")
        effect = self.segment_editor_widget.activeEffect()

        if effect is None:
            raise RuntimeError("Failed to activate Threshold effect")

        # Set threshold range
        effect.setParameter("MinimumThreshold", str(params["min_value"]))
        effect.setParameter("MaximumThreshold", str(params["max_value"]))

        # Apply
        effect.self().onApply()
        slicer.app.processEvents()

        logger.debug(f"Threshold applied: [{params['min_value']}, {params['max_value']}]")

    def _execute_grow_from_seeds(self, params: dict) -> None:
        """Execute Grow from Seeds effect.

        Args:
            params: Action parameters (currently unused).
        """
        import slicer

        assert self.segment_editor_widget is not None  # Set by run()

        # Activate Grow from Seeds effect
        self.segment_editor_widget.setActiveEffectByName("Grow from seeds")
        effect = self.segment_editor_widget.activeEffect()

        if effect is None:
            raise RuntimeError("Failed to activate Grow from Seeds effect")

        # Apply
        effect.self().onApply()
        slicer.app.processEvents()

        logger.debug("Grow from Seeds applied")

    def _execute_islands(self, params: dict) -> None:
        """Execute Islands effect.

        Args:
            params: Action parameters including operation, min_size.
        """
        import slicer

        assert self.segment_editor_widget is not None  # Set by run()

        # Activate Islands effect
        self.segment_editor_widget.setActiveEffectByName("Islands")
        effect = self.segment_editor_widget.activeEffect()

        if effect is None:
            raise RuntimeError("Failed to activate Islands effect")

        # Set operation
        operation = params.get("operation", "KEEP_LARGEST")
        effect.setParameter("Operation", operation)

        if "min_size" in params:
            effect.setParameter("MinimumSize", str(params["min_size"]))

        # Apply
        effect.self().onApply()
        slicer.app.processEvents()

        logger.debug(f"Islands applied: {operation}")

    def _execute_smoothing(self, params: dict) -> None:
        """Execute Smoothing effect.

        Args:
            params: Action parameters including method, kernel_size_mm.
        """
        import slicer

        assert self.segment_editor_widget is not None  # Set by run()

        # Activate Smoothing effect
        self.segment_editor_widget.setActiveEffectByName("Smoothing")
        effect = self.segment_editor_widget.activeEffect()

        if effect is None:
            raise RuntimeError("Failed to activate Smoothing effect")

        # Set method
        method = params.get("method", "MEDIAN")
        effect.setParameter("SmoothingMethod", method)

        if "kernel_size_mm" in params:
            effect.setParameter("KernelSizeMm", str(params["kernel_size_mm"]))

        # Apply
        effect.self().onApply()
        slicer.app.processEvents()

        logger.debug(f"Smoothing applied: {method}")

    def _execute_scissors(self, params: dict) -> None:
        """Execute Scissors effect.

        Args:
            params: Action parameters including ras_points, operation.
        """
        logger.warning("Scissors effect execution not fully implemented")
        # Scissors requires drawing a path which is complex to automate

    def _navigate_to_ras(self, ras: tuple[float, float, float]) -> None:
        """Navigate slice view to RAS coordinates.

        Args:
            ras: RAS coordinates.
        """
        import slicer

        if self._slice_widget:
            slice_logic = self._slice_widget.sliceLogic()
            slice_logic.SetSliceOffset(ras[2])  # Navigate to Z coordinate
            slicer.app.processEvents()

    def _ras_to_xy(self, ras: tuple[float, float, float]) -> tuple[int, int] | None:
        """Convert RAS coordinates to screen XY.

        Args:
            ras: RAS coordinates.

        Returns:
            Screen XY coordinates or None if conversion failed.
        """
        import vtk

        if self._slice_widget is None:
            return None

        slice_logic = self._slice_widget.sliceLogic()
        slice_node = slice_logic.GetSliceNode()

        # Get XY to RAS matrix and invert it
        xy_to_ras = slice_node.GetXYToRAS()
        ras_to_xy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)

        ras_point = [ras[0], ras[1], ras[2], 1]
        xy_point = [0, 0, 0, 1]
        ras_to_xy.MultiplyPoint(ras_point, xy_point)

        return (int(xy_point[0]), int(xy_point[1]))

    def _count_segment_voxels(self) -> int:
        """Count voxels in current segment.

        Returns:
            Number of voxels in segment.
        """
        try:
            import numpy as np
            import slicer

            labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(
                self.segmentation_node, self.segment_id, self.volume_node
            )
            return int(np.sum(labelmap > 0))
        except Exception as e:
            logger.warning(f"Could not count voxels: {e}")
            return 0

    def save_segmentation(self, output_path: Path | str) -> None:
        """Save current segmentation to file.

        Args:
            output_path: Path where to save segmentation.
        """
        import slicer

        output_path = Path(output_path)
        if self.segmentation_node:
            slicer.util.saveNode(self.segmentation_node, str(output_path))
            logger.info(f"Saved segmentation to {output_path}")
