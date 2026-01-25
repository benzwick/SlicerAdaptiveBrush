"""Recipe recording for capturing manual segmentation sessions.

Records user actions and generates Python recipe files.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RecordedStroke:
    """A recorded brush stroke."""

    ras: tuple[float, float, float]
    erase: bool = False
    timestamp: float = 0.0


@dataclass
class RecordedParamChange:
    """A recorded parameter change."""

    name: str
    value: Any
    timestamp: float = 0.0


class RecipeRecorder:
    """Record manual Slicer sessions into Python recipe files.

    Example:
        recorder = RecipeRecorder()
        recorder.start("MRBrainTumor1", "Tumor")

        # ... user performs manual segmentation ...

        recorder.stop()
        recorder.save("recipes/my_segmentation.py")
    """

    def __init__(self) -> None:
        """Initialize recorder."""
        self.strokes: list[RecordedStroke] = []
        self.param_changes: list[RecordedParamChange] = []
        self.recording: bool = False
        self.sample_data: str = ""
        self.segment_name: str = ""
        self.start_time: float = 0.0
        self.initial_preset: str = ""

    def start(self, sample_data: str, segment_name: str) -> None:
        """Start recording a session.

        Args:
            sample_data: Slicer SampleData name.
            segment_name: Name of the segment being created.
        """
        if self.recording:
            logger.warning("Already recording. Call stop() first.")
            return

        self.sample_data = sample_data
        self.segment_name = segment_name
        self.strokes = []
        self.param_changes = []
        self.start_time = time.time()
        self.recording = True
        self.initial_preset = self._get_current_preset()

        logger.info(f"Started recording: {sample_data} / {segment_name}")

    def stop(self) -> None:
        """Stop recording."""
        self.recording = False
        logger.info(
            f"Stopped recording: {len(self.strokes)} strokes, "
            f"{len(self.param_changes)} param changes"
        )

    def record_stroke(
        self,
        ras: tuple[float, float, float],
        erase: bool = False,
    ) -> None:
        """Record a brush stroke.

        Args:
            ras: RAS coordinates.
            erase: Whether this was an erase stroke.
        """
        if not self.recording:
            return

        stroke = RecordedStroke(
            ras=ras,
            erase=erase,
            timestamp=time.time() - self.start_time,
        )
        self.strokes.append(stroke)
        logger.debug(f"Recorded stroke at {ras}, erase={erase}")

    def record_param_change(self, name: str, value: Any) -> None:
        """Record a parameter change.

        Args:
            name: Parameter name.
            value: New value.
        """
        if not self.recording:
            return

        change = RecordedParamChange(
            name=name,
            value=value,
            timestamp=time.time() - self.start_time,
        )
        self.param_changes.append(change)
        logger.debug(f"Recorded param change: {name}={value}")

    def save(self, output_path: Path | str) -> None:
        """Save recording as a Python recipe file.

        Args:
            output_path: Path for the output file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        code = self._generate_code()

        with open(output_path, "w") as f:
            f.write(code)

        logger.info(f"Saved recipe to: {output_path}")

    def _generate_code(self) -> str:
        """Generate Python recipe code from recording."""
        lines = [
            '"""',
            f"Recipe: {self.segment_name}",
            "",
            f"Recorded: {datetime.now():%Y-%m-%d %H:%M}",
            f"Sample Data: {self.sample_data}",
            f"Strokes: {len(self.strokes)}",
            '"""',
            "",
            f'sample_data = "{self.sample_data}"',
            f'segment_name = "{self.segment_name}"',
            "",
            "",
            "def run(effect):",
        ]

        # Add preset if set
        if self.initial_preset:
            lines.append(f'    effect.applyPreset("{self.initial_preset}")')
            lines.append("")

        # Merge strokes and param changes by timestamp
        events: list[tuple[str, float, RecordedStroke | RecordedParamChange]] = []
        for stroke in self.strokes:
            events.append(("stroke", stroke.timestamp, stroke))
        for change in self.param_changes:
            events.append(("param", change.timestamp, change))

        events.sort(key=lambda x: x[1])

        # Generate code for each event
        for event_type, _, data in events:
            if event_type == "stroke" and isinstance(data, RecordedStroke):
                r, a, s = data.ras
                if data.erase:
                    lines.append(f"    effect.paintAt({r:.2f}, {a:.2f}, {s:.2f}, erase=True)")
                else:
                    lines.append(f"    effect.paintAt({r:.2f}, {a:.2f}, {s:.2f})")
            elif event_type == "param" and isinstance(data, RecordedParamChange):
                if isinstance(data.value, str):
                    lines.append(f'    effect.{data.name} = "{data.value}"')
                else:
                    lines.append(f"    effect.{data.name} = {data.value}")

        if len(self.strokes) == 0:
            lines.append("    pass  # No strokes recorded")

        lines.append("")

        return "\n".join(lines)

    def _get_current_preset(self) -> str:
        """Get the current preset from the effect."""
        try:
            import slicer

            editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            effect = editor.activeEffect()
            if effect and effect.name == "Adaptive Brush":
                return getattr(effect.self(), "_currentPreset", "")
        except Exception:
            pass
        return ""

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording


# Global recorder instance
_global_recorder: RecipeRecorder | None = None


def get_global_recorder() -> RecipeRecorder:
    """Get or create the global recipe recorder."""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = RecipeRecorder()
    return _global_recorder


def start_recording(sample_data: str, segment_name: str) -> None:
    """Start global recording session."""
    get_global_recorder().start(sample_data, segment_name)


def stop_recording() -> None:
    """Stop global recording."""
    get_global_recorder().stop()


def is_recording() -> bool:
    """Check if global recording is active."""
    return get_global_recorder().is_recording()
