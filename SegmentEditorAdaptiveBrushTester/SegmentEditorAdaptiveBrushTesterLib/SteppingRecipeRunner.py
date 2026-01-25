"""Step-by-step recipe execution with checkpointing.

Enables stepping through recipes one action at a time, with the ability
to rewind to previous states and branch to create new variations.

See ADR-014 for architecture decisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from .ActionRecipe import ActionRecipe, RecipeAction

logger = logging.getLogger(__name__)


@dataclass
class SegmentationCheckpoint:
    """A saved state of the segmentation at a specific step.

    Stores the labelmap as a numpy array for fast restoration.

    Attributes:
        step_index: The step index this checkpoint represents (after executing that step).
        labelmap_array: The segmentation labelmap as a numpy array.
        timestamp: When the checkpoint was created.
    """

    step_index: int
    labelmap_array: np.ndarray
    timestamp: float = field(default_factory=time.time)

    @property
    def size_bytes(self) -> int:
        """Return the size of the labelmap in bytes."""
        return int(self.labelmap_array.nbytes)


class SteppingRecipeRunner:
    """Execute recipes step-by-step with checkpoint and rewind support.

    Usage:
        runner = SteppingRecipeRunner(action_recipe)

        # Set up the scene
        runner.setup()

        # Step through the recipe
        while runner.step_forward():
            action = runner.get_current_action()
            print(f"Executed step {runner.current_step}: {action.type}")

        # Rewind to step 2
        runner.goto_step(2)

        # Branch from here
        runner.start_branch()
        # ... user performs manual actions ...
        runner.add_manual_action(RecipeAction.paint(...))
        branched_recipe = runner.save_branch()

    Attributes:
        recipe: The ActionRecipe being executed.
        current_step: Current step index (-1 = before any steps).
        checkpoints: List of saved segmentation states.
    """

    def __init__(self, action_recipe: ActionRecipe) -> None:
        """Initialize the stepping runner.

        Args:
            action_recipe: The ActionRecipe to execute.
        """
        self.recipe = action_recipe
        self.current_step: int = -1  # -1 = not started
        self.checkpoints: list[SegmentationCheckpoint] = []

        # Branching state
        self._recording_branch: bool = False
        self._branch_actions: list[RecipeAction] = []
        self._branch_start_step: int = -1

        # Slicer objects (set during setup)
        self._volume_node: Any = None
        self._segmentation_node: Any = None
        self._segment_id: str | None = None
        self._effect: Any = None

        # Callbacks
        self._step_callback: Callable[[int, RecipeAction], None] | None = None
        self._checkpoint_callback: Callable[[SegmentationCheckpoint], None] | None = None

    @property
    def total_steps(self) -> int:
        """Return total number of steps in the recipe."""
        return len(self.recipe.actions)

    @property
    def is_at_start(self) -> bool:
        """Return True if at the start (before any steps)."""
        return self.current_step < 0

    @property
    def is_at_end(self) -> bool:
        """Return True if at the end (all steps executed)."""
        return self.current_step >= self.total_steps - 1

    @property
    def is_branching(self) -> bool:
        """Return True if currently recording a branch."""
        return self._recording_branch

    def set_step_callback(self, callback: Callable[[int, RecipeAction], None] | None) -> None:
        """Set callback to be called after each step.

        Args:
            callback: Function taking (step_index, action) or None to clear.
        """
        self._step_callback = callback

    def set_checkpoint_callback(
        self, callback: Callable[[SegmentationCheckpoint], None] | None
    ) -> None:
        """Set callback to be called when a checkpoint is created.

        Args:
            callback: Function taking (checkpoint) or None to clear.
        """
        self._checkpoint_callback = callback

    def setup(self) -> bool:
        """Set up the scene for recipe execution.

        Loads sample data, creates segmentation, and activates the effect.

        Returns:
            True if setup succeeded, False otherwise.
        """
        try:
            import slicer

            # Load sample data
            if self.recipe.sample_data:
                import SampleData

                self._volume_node = SampleData.downloadSample(self.recipe.sample_data)
                if not self._volume_node:
                    logger.error(f"Failed to load sample data: {self.recipe.sample_data}")
                    return False

            # Create segmentation node
            self._segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            self._segmentation_node.SetName(f"{self.recipe.name}_segmentation")
            self._segmentation_node.CreateDefaultDisplayNodes()
            self._segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                self._volume_node
            )

            # Create segment
            segmentation = self._segmentation_node.GetSegmentation()
            self._segment_id = segmentation.AddEmptySegment(self.recipe.segment_name)

            # Set up segment editor
            segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            segment_editor_node.SetAndObserveSourceVolumeNode(self._volume_node)
            segment_editor_node.SetAndObserveSegmentationNode(self._segmentation_node)
            segment_editor_node.SetSelectedSegmentID(self._segment_id)

            # Get segment editor widget and activate Adaptive Brush
            segment_editor_widget = (
                slicer.modules.segmenteditor.widgetRepresentation().self().editor
            )
            segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
            segment_editor_widget.setActiveEffectByName("Adaptive Brush")

            active_effect = segment_editor_widget.activeEffect()
            if active_effect:
                self._effect = active_effect.self()

            # Save initial checkpoint (empty segmentation)
            self._save_checkpoint(-1)

            self.current_step = -1
            logger.info(f"Setup complete for recipe: {self.recipe.name}")
            return True

        except Exception as e:
            logger.exception(f"Setup failed: {e}")
            return False

    def step_forward(self) -> bool:
        """Execute the next step in the recipe.

        Returns:
            True if a step was executed, False if already at the end.
        """
        if self.is_at_end:
            return False

        next_step = self.current_step + 1
        action = self.recipe.actions[next_step]

        # Execute the action
        self._execute_action(action)

        # Save checkpoint after execution
        self._save_checkpoint(next_step)

        self.current_step = next_step

        # Call step callback
        if self._step_callback:
            self._step_callback(next_step, action)

        logger.debug(f"Executed step {next_step}: {action.type}")
        return True

    def step_backward(self) -> bool:
        """Rewind to the previous step by restoring checkpoint.

        Returns:
            True if rewound successfully, False if already at start.
        """
        if self.is_at_start:
            return False

        return self.goto_step(self.current_step - 1)

    def goto_step(self, step_index: int) -> bool:
        """Go to a specific step by restoring checkpoint.

        Args:
            step_index: Target step index (-1 for initial state).

        Returns:
            True if successful, False otherwise.
        """
        if step_index < -1 or step_index >= self.total_steps:
            logger.warning(f"Invalid step index: {step_index}")
            return False

        # Find checkpoint for this step
        checkpoint = self._find_checkpoint(step_index)
        if checkpoint:
            self._restore_checkpoint(checkpoint)
            self.current_step = step_index
            logger.debug(f"Restored to step {step_index}")
            return True

        # No checkpoint found - need to replay from nearest earlier checkpoint
        nearest_checkpoint = self._find_nearest_checkpoint_before(step_index)
        if nearest_checkpoint:
            self._restore_checkpoint(nearest_checkpoint)
            self.current_step = nearest_checkpoint.step_index

            # Replay steps up to target
            while self.current_step < step_index:
                if not self.step_forward():
                    break

            return self.current_step == step_index

        logger.warning(f"Could not find checkpoint for step {step_index}")
        return False

    def run_to_end(self) -> int:
        """Execute all remaining steps.

        Returns:
            Number of steps executed.
        """
        steps_executed = 0
        while self.step_forward():
            steps_executed += 1
        return steps_executed

    def get_current_action(self) -> RecipeAction | None:
        """Get the action for the current step.

        Returns:
            The current action, or None if at start.
        """
        if self.is_at_start:
            return None
        return self.recipe.actions[self.current_step]

    def get_next_action(self) -> RecipeAction | None:
        """Get the action for the next step.

        Returns:
            The next action, or None if at end.
        """
        if self.is_at_end:
            return None
        return self.recipe.actions[self.current_step + 1]

    # --- Branching methods ---

    def start_branch(self) -> None:
        """Start recording a branch from the current step.

        After calling this, manual actions can be added and saved as a new recipe.
        """
        if self._recording_branch:
            logger.warning("Already recording a branch")
            return

        self._recording_branch = True
        self._branch_start_step = self.current_step
        self._branch_actions = []
        logger.info(f"Started branch recording from step {self.current_step}")

    def stop_branch(self) -> None:
        """Stop recording the branch without saving."""
        self._recording_branch = False
        self._branch_actions = []
        self._branch_start_step = -1
        logger.info("Stopped branch recording")

    def add_manual_action(self, action: RecipeAction) -> None:
        """Add a manually recorded action to the branch.

        Args:
            action: The action to add.
        """
        if not self._recording_branch:
            logger.warning("Not recording a branch - call start_branch() first")
            return

        self._branch_actions.append(action)
        logger.debug(f"Added branch action: {action.type}")

    def save_branch(self, name: str | None = None) -> ActionRecipe:
        """Save the current branch as a new ActionRecipe.

        Args:
            name: Name for the branched recipe. If None, auto-generated.

        Returns:
            The branched ActionRecipe.
        """
        from .ActionRecipe import ActionRecipe

        if not self._recording_branch:
            logger.warning("Not recording a branch")
            # Return a copy of the current recipe up to current step
            return self.recipe.slice_to(self.current_step)

        if name is None:
            name = f"{self.recipe.name}_branch_{int(time.time())}"

        # Take actions up to branch start + branch actions
        base_actions = self.recipe.actions[: self._branch_start_step + 1]
        all_actions = base_actions + self._branch_actions

        branched = ActionRecipe(
            name=name,
            sample_data=self.recipe.sample_data,
            segment_name=self.recipe.segment_name,
            actions=all_actions,
            gold_standard=None,  # Branched recipe doesn't have a gold standard yet
            description=f"Branched from {self.recipe.name} at step {self._branch_start_step}",
            metadata={
                "source_recipe": self.recipe.name,
                "branch_start_step": self._branch_start_step,
                "branch_actions_count": len(self._branch_actions),
                "created_at": time.time(),
            },
        )

        logger.info(f"Created branch recipe: {name} ({len(all_actions)} actions)")

        # Stop recording
        self.stop_branch()

        return branched

    # --- Private methods ---

    def _execute_action(self, action: RecipeAction) -> None:
        """Execute a single action.

        Args:
            action: The action to execute.
        """
        try:
            import slicer

            if action.type == "paint":
                self._execute_paint(action)
            elif action.type == "erase":
                self._execute_erase(action)
            elif action.type == "set_param":
                self._execute_set_param(action)
            elif action.type == "set_algorithm":
                self._execute_set_algorithm(action)
            else:
                logger.warning(f"Unknown action type: {action.type}")

            # Process events to update UI
            slicer.app.processEvents()

        except Exception as e:
            logger.exception(f"Failed to execute action: {e}")

    def _execute_paint(self, action: RecipeAction) -> None:
        """Execute a paint action."""
        if not self._effect or not action.ras:
            return

        # Apply parameters from action
        params = action.params
        if "algorithm" in params:
            self._effect.algorithm = params["algorithm"]
        if "brush_radius_mm" in params:
            self._effect.brushRadiusMm = params["brush_radius_mm"]
        if "edge_sensitivity" in params:
            self._effect.edgeSensitivity = params["edge_sensitivity"]

        # Execute paint
        r, a, s = action.ras
        self._effect.paintAt(r, a, s)

    def _execute_erase(self, action: RecipeAction) -> None:
        """Execute an erase action."""
        if not self._effect or not action.ras:
            return

        # Apply parameters from action
        params = action.params
        if "algorithm" in params:
            self._effect.algorithm = params["algorithm"]
        if "brush_radius_mm" in params:
            self._effect.brushRadiusMm = params["brush_radius_mm"]
        if "edge_sensitivity" in params:
            self._effect.edgeSensitivity = params["edge_sensitivity"]

        # Execute erase
        r, a, s = action.ras
        self._effect.eraseAt(r, a, s)

    def _execute_set_param(self, action: RecipeAction) -> None:
        """Execute a parameter change action."""
        if not self._effect:
            return

        name = action.params.get("name")
        value = action.params.get("value")

        if name and value is not None:
            setattr(self._effect, name, value)

    def _execute_set_algorithm(self, action: RecipeAction) -> None:
        """Execute an algorithm change action."""
        if not self._effect:
            return

        algorithm = action.params.get("algorithm")
        if algorithm:
            self._effect.algorithm = algorithm

    def _save_checkpoint(self, step_index: int) -> SegmentationCheckpoint | None:
        """Save a checkpoint of the current segmentation state.

        Args:
            step_index: The step index this checkpoint represents.

        Returns:
            The created checkpoint, or None if failed.
        """
        try:
            import slicer

            if not self._segmentation_node or not self._segment_id:
                return None

            # Get the labelmap array
            labelmap_array = slicer.util.arrayFromSegmentBinaryLabelmap(
                self._segmentation_node,
                self._segment_id,
                self._volume_node,
            )

            if labelmap_array is None:
                # Empty segmentation - create empty array
                dims = self._volume_node.GetImageData().GetDimensions()
                labelmap_array = np.zeros(dims[::-1], dtype=np.uint8)

            checkpoint = SegmentationCheckpoint(
                step_index=step_index,
                labelmap_array=labelmap_array.copy(),
            )

            self.checkpoints.append(checkpoint)

            if self._checkpoint_callback:
                self._checkpoint_callback(checkpoint)

            logger.debug(f"Saved checkpoint at step {step_index} ({checkpoint.size_bytes} bytes)")
            return checkpoint

        except Exception as e:
            logger.exception(f"Failed to save checkpoint: {e}")
            return None

    def _restore_checkpoint(self, checkpoint: SegmentationCheckpoint) -> bool:
        """Restore segmentation from a checkpoint.

        Args:
            checkpoint: The checkpoint to restore.

        Returns:
            True if successful, False otherwise.
        """
        try:
            import slicer

            if not self._segmentation_node or not self._segment_id:
                return False

            # Get the labelmap node for the segment
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                checkpoint.labelmap_array,
                self._segmentation_node,
                self._segment_id,
                self._volume_node,
            )

            logger.debug(f"Restored checkpoint from step {checkpoint.step_index}")
            return True

        except Exception as e:
            logger.exception(f"Failed to restore checkpoint: {e}")
            return False

    def _find_checkpoint(self, step_index: int) -> SegmentationCheckpoint | None:
        """Find checkpoint for a specific step.

        Args:
            step_index: The step index to find.

        Returns:
            The checkpoint if found, None otherwise.
        """
        for checkpoint in self.checkpoints:
            if checkpoint.step_index == step_index:
                return checkpoint
        return None

    def _find_nearest_checkpoint_before(self, step_index: int) -> SegmentationCheckpoint | None:
        """Find the nearest checkpoint at or before a step.

        Args:
            step_index: The step index.

        Returns:
            The nearest checkpoint, or None if none found.
        """
        nearest: SegmentationCheckpoint | None = None
        for checkpoint in self.checkpoints:
            if checkpoint.step_index <= step_index:
                if nearest is None or checkpoint.step_index > nearest.step_index:
                    nearest = checkpoint
        return nearest

    def get_checkpoint_stats(self) -> dict[str, Any]:
        """Get statistics about stored checkpoints.

        Returns:
            Dictionary with checkpoint statistics.
        """
        if not self.checkpoints:
            return {"count": 0, "total_bytes": 0, "steps_covered": []}

        total_bytes = sum(cp.size_bytes for cp in self.checkpoints)
        steps = [cp.step_index for cp in self.checkpoints]

        return {
            "count": len(self.checkpoints),
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "steps_covered": sorted(steps),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.checkpoints.clear()
        self._effect = None
        self._segmentation_node = None
        self._volume_node = None
        self._segment_id = None
        self.current_step = -1
        self.stop_branch()
