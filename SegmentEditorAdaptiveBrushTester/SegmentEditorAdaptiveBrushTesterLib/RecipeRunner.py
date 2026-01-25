"""Recipe execution engine.

Runs recipe scripts in Slicer with the Adaptive Brush effect.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .Recipe import Recipe

logger = logging.getLogger(__name__)


@dataclass
class RecipeResult:
    """Result of executing a recipe."""

    recipe_name: str
    success: bool
    duration_ms: float = 0.0
    voxel_count: int = 0
    segmentation_node: Any = None
    volume_node: Any = None
    segment_id: str = ""
    error: str | None = None


class RecipeRunner:
    """Execute segmentation recipes in Slicer.

    Example:
        recipe = Recipe.load("recipes/brain_tumor_1.py")
        runner = RecipeRunner(recipe)
        result = runner.run()
    """

    def __init__(self, recipe: Recipe):
        """Initialize runner with a recipe.

        Args:
            recipe: Recipe to execute.
        """
        self.recipe = recipe
        self.segmentation_node = None
        self.volume_node = None
        self.segment_id = ""

    def run(self, clear_scene: bool = True) -> RecipeResult:
        """Execute the recipe.

        Args:
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
            if self.recipe.sample_data:
                import SampleData

                self.volume_node = SampleData.downloadSample(self.recipe.sample_data)
                result.volume_node = self.volume_node
                logger.info(f"Loaded: {self.recipe.sample_data}")

            # Create segmentation
            self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            if self.segmentation_node is None:
                raise RuntimeError("Failed to create segmentation node")
            self.segmentation_node.CreateDefaultDisplayNodes()
            if self.volume_node:
                self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                    self.volume_node
                )

            # Add segment
            segmentation = self.segmentation_node.GetSegmentation()
            if segmentation is None:
                raise RuntimeError("Failed to get segmentation")
            self.segment_id = segmentation.AddEmptySegment(self.recipe.segment_name)
            result.segmentation_node = self.segmentation_node
            result.segment_id = self.segment_id

            # Switch to Segment Editor and set up
            slicer.util.selectModule("SegmentEditor")
            slicer.app.processEvents()

            editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            editor_widget.setSegmentationNode(self.segmentation_node)
            if self.volume_node:
                editor_widget.setSourceVolumeNode(self.volume_node)
            editor_widget.setCurrentSegmentID(self.segment_id)
            slicer.app.processEvents()

            # Activate Adaptive Brush
            editor_widget.setActiveEffectByName("Adaptive Brush")
            effect = editor_widget.activeEffect()
            if effect is None:
                raise RuntimeError("Failed to activate Adaptive Brush effect")

            scripted_effect = effect.self()
            slicer.app.processEvents()

            # Execute the recipe
            self.recipe.execute(scripted_effect)
            slicer.app.processEvents()

            # Get final voxel count
            result.voxel_count = self._count_segment_voxels()
            result.success = True

        except Exception as e:
            logger.exception(f"Recipe failed: {e}")
            result.error = str(e)

        result.duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Recipe '{self.recipe.name}': {result.voxel_count:,} voxels "
            f"in {result.duration_ms:.0f}ms"
        )

        return result

    def _count_segment_voxels(self) -> int:
        """Count voxels in the current segment."""
        import numpy as np
        import slicer

        if not self.segmentation_node or not self.segment_id:
            return 0

        # Get labelmap
        labelmap = slicer.vtkOrientedImageData()
        self.segmentation_node.GetBinaryLabelmapRepresentation(self.segment_id, labelmap)

        # Count non-zero voxels
        from vtk.util.numpy_support import vtk_to_numpy

        scalars = labelmap.GetPointData().GetScalars()
        if scalars is None:
            return 0

        arr = vtk_to_numpy(scalars)
        return int(np.count_nonzero(arr))


def run_recipe(recipe_path: Path | str, clear_scene: bool = True) -> RecipeResult:
    """Convenience function to load and run a recipe.

    Args:
        recipe_path: Path to recipe file.
        clear_scene: Whether to clear scene first.

    Returns:
        RecipeResult with execution details.
    """
    recipe = Recipe.load(recipe_path)
    runner = RecipeRunner(recipe)
    return runner.run(clear_scene=clear_scene)
