"""Recipe recording for capturing manual segmentation sessions.

Records user actions in Slicer and generates recipe files that can
be replayed, edited, and optimized.

See ADR-013 for architecture decisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from .Recipe import Action, Recipe

logger = logging.getLogger(__name__)


@dataclass
class RecordedAction:
    """An action recorded during a manual session."""

    action: Action
    timestamp: float
    view_name: str = "Red"
    notes: str = ""


class RecipeRecorder:
    """Record manual Slicer sessions into recipe files.

    Captures user actions including:
    - Adaptive Brush strokes with all parameters
    - Other segment editor effects
    - Parameter changes

    Example:
        recorder = RecipeRecorder()
        recorder.start("MRBrainTumor1", "Tumor")

        # ... user performs manual segmentation ...

        recipe = recorder.stop()
        recipe.save("recipes/my_segmentation.py")
    """

    def __init__(self) -> None:
        """Initialize recorder."""
        self.recorded_actions: list[RecordedAction] = []
        self.recording: bool = False
        self.sample_data: str = ""
        self.segment_name: str = ""
        self.start_time: float = 0.0
        self._observers: list[tuple[Any, int]] = []
        self._effect_observers: dict[str, Any] = {}

    def start(self, sample_data: str, segment_name: str) -> None:
        """Start recording a session.

        Args:
            sample_data: Slicer SampleData name being segmented.
            segment_name: Name of the segment being created.
        """
        if self.recording:
            logger.warning("Already recording. Call stop() first.")
            return

        self.sample_data = sample_data
        self.segment_name = segment_name
        self.recorded_actions = []
        self.start_time = time.time()
        self.recording = True

        self._install_hooks()
        logger.info(f"Started recording: {sample_data} / {segment_name}")

    def stop(self) -> Recipe:
        """Stop recording and return Recipe.

        Returns:
            Recipe object with recorded actions.
        """
        self.recording = False
        self._remove_hooks()

        recipe = Recipe(
            name=f"recorded_{datetime.now():%Y%m%d_%H%M%S}",
            description=f"Recorded from manual session ({len(self.recorded_actions)} actions)",
            sample_data=self.sample_data,
            segment_name=self.segment_name,
            actions=[ra.action for ra in self.recorded_actions],
            optimization_hints={
                "vary_globally": ["edge_sensitivity", "threshold_zone"],
                "vary_per_action": ["brush_radius_mm"],
            },
            metadata={
                "recorded_at": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.start_time,
                "action_count": len(self.recorded_actions),
            },
        )

        logger.info(f"Stopped recording: {len(self.recorded_actions)} actions captured")
        return recipe

    def save(self, output_path: Path | str) -> Recipe:
        """Stop recording and save recipe to file.

        Args:
            output_path: Path where to save recipe file.

        Returns:
            The saved Recipe object.
        """
        recipe = self.stop()
        recipe.save(output_path)
        return recipe

    def add_note(self, note: str) -> None:
        """Add a note to the most recent action.

        Args:
            note: Note text to add.
        """
        if self.recorded_actions:
            self.recorded_actions[-1].notes = note
            logger.debug(f"Added note to action: {note}")

    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            True if recording is active.
        """
        return self.recording

    def get_action_count(self) -> int:
        """Get number of recorded actions.

        Returns:
            Number of actions recorded so far.
        """
        return len(self.recorded_actions)

    def _install_hooks(self) -> None:
        """Install Slicer event hooks for recording."""
        try:
            import slicer

            # Hook into segment editor widget
            segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation()
            if segment_editor_module:
                # We can't directly observe effect events, but we can poll
                # or use the action recorder from the tester module
                # Editor reference available via: segment_editor_module.self().editor
                logger.debug("Installed Slicer hooks for recording")

        except Exception as e:
            logger.warning(f"Could not install Slicer hooks: {e}")

    def _remove_hooks(self) -> None:
        """Remove Slicer event hooks."""
        for obj, tag in self._observers:
            try:
                obj.RemoveObserver(tag)
            except Exception:
                pass
        self._observers.clear()
        self._effect_observers.clear()
        logger.debug("Removed Slicer hooks")

    def record_adaptive_brush_stroke(
        self,
        ras: tuple[float, float, float],
        algorithm: str,
        brush_radius_mm: float,
        edge_sensitivity: int,
        threshold_zone: int = 50,
        mode: str = "add",
        view_name: str = "Red",
        **kwargs: Any,
    ) -> None:
        """Record an Adaptive Brush stroke.

        This method is called by the Adaptive Brush effect when
        recording is active.

        Args:
            ras: RAS coordinates of the stroke.
            algorithm: Algorithm used.
            brush_radius_mm: Brush radius in mm.
            edge_sensitivity: Edge sensitivity (0-100).
            threshold_zone: Threshold zone percentage.
            mode: "add" or "erase".
            view_name: Name of the slice view.
            **kwargs: Additional algorithm-specific parameters.
        """
        if not self.recording:
            return

        action = Action.adaptive_brush(
            ras=ras,
            algorithm=algorithm,
            brush_radius_mm=brush_radius_mm,
            edge_sensitivity=edge_sensitivity,
            threshold_zone=threshold_zone,
            mode=mode,
            **kwargs,
        )

        recorded = RecordedAction(
            action=action,
            timestamp=time.time() - self.start_time,
            view_name=view_name,
        )
        self.recorded_actions.append(recorded)

        logger.debug(
            f"Recorded adaptive brush: RAS={ras}, algo={algorithm}, "
            f"radius={brush_radius_mm}mm, mode={mode}"
        )

    def record_paint_stroke(
        self,
        ras: tuple[float, float, float],
        radius_mm: float,
        mode: str = "add",
        sphere: bool = False,
        view_name: str = "Red",
    ) -> None:
        """Record a Paint stroke.

        Args:
            ras: RAS coordinates of the stroke.
            radius_mm: Paint radius in mm.
            mode: "add" or "erase".
            sphere: Whether sphere brush was used.
            view_name: Name of the slice view.
        """
        if not self.recording:
            return

        action = Action.paint(
            ras=ras,
            radius_mm=radius_mm,
            mode=mode,
            sphere=sphere,
        )

        recorded = RecordedAction(
            action=action,
            timestamp=time.time() - self.start_time,
            view_name=view_name,
        )
        self.recorded_actions.append(recorded)

        logger.debug(f"Recorded paint: RAS={ras}, radius={radius_mm}mm, mode={mode}")

    def record_threshold(
        self,
        min_value: float,
        max_value: float,
    ) -> None:
        """Record a Threshold application.

        Args:
            min_value: Minimum threshold value.
            max_value: Maximum threshold value.
        """
        if not self.recording:
            return

        action = Action.threshold(min_value=min_value, max_value=max_value)

        recorded = RecordedAction(
            action=action,
            timestamp=time.time() - self.start_time,
        )
        self.recorded_actions.append(recorded)

        logger.debug(f"Recorded threshold: [{min_value}, {max_value}]")

    def record_grow_from_seeds(self) -> None:
        """Record a Grow from Seeds application."""
        if not self.recording:
            return

        action = Action.grow_from_seeds()

        recorded = RecordedAction(
            action=action,
            timestamp=time.time() - self.start_time,
        )
        self.recorded_actions.append(recorded)

        logger.debug("Recorded grow from seeds")

    def record_islands(
        self,
        operation: str = "KEEP_LARGEST",
        min_size: int = 1000,
    ) -> None:
        """Record an Islands operation.

        Args:
            operation: Islands operation type.
            min_size: Minimum island size.
        """
        if not self.recording:
            return

        action = Action.islands(operation=operation, min_size=min_size)

        recorded = RecordedAction(
            action=action,
            timestamp=time.time() - self.start_time,
        )
        self.recorded_actions.append(recorded)

        logger.debug(f"Recorded islands: {operation}")

    def record_smoothing(
        self,
        method: str = "MEDIAN",
        kernel_size_mm: float = 3.0,
    ) -> None:
        """Record a Smoothing operation.

        Args:
            method: Smoothing method.
            kernel_size_mm: Kernel size in mm.
        """
        if not self.recording:
            return

        action = Action.smoothing(method=method, kernel_size_mm=kernel_size_mm)

        recorded = RecordedAction(
            action=action,
            timestamp=time.time() - self.start_time,
        )
        self.recorded_actions.append(recorded)

        logger.debug(f"Recorded smoothing: {method}")

    def get_current_adaptive_brush_params(self) -> dict[str, Any] | None:
        """Get current parameters from Adaptive Brush effect.

        Returns:
            Dictionary of current parameters or None if not available.
        """
        try:
            import slicer

            segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation()
            if not segment_editor_module:
                return None

            editor = segment_editor_module.self().editor
            effect = editor.activeEffect()

            if effect is None or effect.name != "Adaptive Brush":
                return None

            scripted_effect = effect.self()

            params = {
                "algorithm": scripted_effect.algorithmCombo.currentData,
                "brush_radius_mm": scripted_effect.radiusSlider.value,
                "edge_sensitivity": int(scripted_effect.sensitivitySlider.value),
                "threshold_zone": int(scripted_effect.thresholdZoneSlider.value),
            }

            # Add algorithm-specific params
            algo = params["algorithm"]

            if algo == "watershed":
                params["watershedGradientScale"] = (
                    scripted_effect.watershedGradientScaleSlider.value
                )
                params["watershedSmoothing"] = scripted_effect.watershedSmoothingSlider.value

            elif algo in ("level_set_cpu", "level_set_gpu"):
                params["levelSetIterations"] = int(scripted_effect.levelSetIterationsSlider.value)
                params["levelSetPropagation"] = scripted_effect.levelSetPropagationSlider.value
                params["levelSetCurvature"] = scripted_effect.levelSetCurvatureSlider.value

            elif algo == "region_growing":
                params["regionGrowingMultiplier"] = (
                    scripted_effect.regionGrowingMultiplierSlider.value
                )

            elif algo == "threshold_brush":
                params["thresholdMethod"] = scripted_effect.thresholdMethodCombo.currentData

            elif algo == "random_walker":
                params["randomWalkerBeta"] = int(scripted_effect.randomWalkerBetaSlider.value)

            return params

        except Exception as e:
            logger.warning(f"Could not get Adaptive Brush params: {e}")
            return None


# Global recorder instance for use by effects
_global_recorder: RecipeRecorder | None = None


def get_global_recorder() -> RecipeRecorder:
    """Get or create the global recipe recorder.

    Returns:
        Global RecipeRecorder instance.
    """
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = RecipeRecorder()
    return _global_recorder


def start_recording(sample_data: str, segment_name: str) -> None:
    """Start global recording session.

    Args:
        sample_data: Slicer SampleData name.
        segment_name: Segment name being created.
    """
    get_global_recorder().start(sample_data, segment_name)


def stop_recording() -> Recipe:
    """Stop global recording and return recipe.

    Returns:
        Recipe with recorded actions.
    """
    return get_global_recorder().stop()


def is_recording() -> bool:
    """Check if global recording is active.

    Returns:
        True if recording.
    """
    return get_global_recorder().is_recording()
