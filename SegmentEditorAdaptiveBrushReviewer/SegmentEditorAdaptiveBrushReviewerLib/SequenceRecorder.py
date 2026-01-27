"""Sequence-based workflow recording using Slicer Sequences.

Records full segmentation workflows for step-by-step playback review:
- Segmentation state at each brush stroke
- Slice positions (Red/Yellow/Green)
- 3D camera position
- Effect parameter node
- Text notes for reviewer annotations

Integrates with existing ActionRecorder infrastructure to provide visual
state capture alongside stroke data.

See ADR-016 for architecture context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SequenceRecorder:
    """Record full segmentation workflow using Slicer Sequences.

    This class captures visual state (how the segmentation looks) at each step,
    complementing ActionRecorder which captures action data (what happened).

    Usage:
        recorder = SequenceRecorder()

        # Start recording with a segmentation node
        recorder.start_recording(segmentation_node, reference_volume)

        # Record a step after each brush stroke
        recorder.record_step("Geodesic at (128, 100, 45)")

        # Add standalone notes
        recorder.add_note("Reviewer noted boundary uncertainty")

        # Navigate to a specific step
        recorder.goto_step(5)

        # Clean up
        recorder.cleanup()

    Attributes:
        step_count: Number of recorded steps.
        is_recording: True if currently recording.
    """

    def __init__(self) -> None:
        """Initialize the sequence recorder."""
        self._browser_node: Any = None
        self._sequences: dict[str, Any] = {}  # name -> sequenceNode
        self._step_index: int = 0
        self._is_recording: bool = False
        self._segmentation_node: Any = None
        self._reference_volume: Any = None

    @property
    def step_count(self) -> int:
        """Return number of recorded steps."""
        return self._step_index

    @property
    def is_recording(self) -> bool:
        """Return True if currently recording."""
        return self._is_recording

    def start_recording(
        self,
        segmentation_node: Any,
        reference_volume: Any | None = None,
    ) -> bool:
        """Initialize sequences for recording.

        Args:
            segmentation_node: The segmentation node to record.
            reference_volume: Optional reference volume for geometry.

        Returns:
            True if recording started successfully, False otherwise.
        """
        if self._is_recording:
            logger.warning("Already recording - stop first")
            return False

        try:
            import slicer

            self._segmentation_node = segmentation_node
            self._reference_volume = reference_volume
            self._step_index = 0

            # Create browser for synchronized playback
            self._browser_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
            self._browser_node.SetName("WorkflowRecording")

            # Enable recording mode on browser
            self._browser_node.SetRecordingActive(True)

            # Create sequence for segmentation state
            self._sequences["segmentation"] = self._create_sequence(
                "SegmentationHistory",
                segmentation_node,
            )

            # Create sequences for all slice views
            for slice_node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
                name = slice_node.GetName()
                self._sequences[f"slice_{name}"] = self._create_sequence(
                    f"SliceState_{name}",
                    slice_node,
                )

            # Create sequence for 3D camera if available
            camera_nodes = slicer.util.getNodesByClass("vtkMRMLCameraNode")
            if camera_nodes:
                camera = camera_nodes[0]
                self._sequences["camera"] = self._create_sequence(
                    "CameraState",
                    camera,
                )

            # Create sequence for notes (will store text nodes)
            notes_sequence = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
            notes_sequence.SetName("ReviewerNotes")
            self._browser_node.AddSynchronizedSequenceNodeID(notes_sequence.GetID())
            self._sequences["notes"] = notes_sequence

            self._is_recording = True
            logger.info(f"Started sequence recording for {segmentation_node.GetName()}")
            return True

        except ImportError:
            logger.error("Slicer not available - cannot start recording")
            return False
        except Exception as e:
            logger.exception(f"Failed to start sequence recording: {e}")
            self.cleanup()
            return False

    def _create_sequence(self, name: str, proxy_node: Any) -> Any:
        """Create a sequence and add to browser.

        Args:
            name: Name for the sequence.
            proxy_node: The node to use as proxy for this sequence.

        Returns:
            The created sequence node.
        """
        import slicer

        seq = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
        seq.SetName(name)

        # Add to browser for synchronized playback
        self._browser_node.AddSynchronizedSequenceNodeID(seq.GetID())

        # Set up proxy node (the node shown during playback)
        if proxy_node:
            self._browser_node.AddProxyNode(proxy_node, seq, False)

        return seq

    def record_step(self, action_description: str = "") -> int:
        """Record current state as a new step.

        Args:
            action_description: Optional description of the action.

        Returns:
            The step index recorded, or -1 if not recording.
        """
        if not self._is_recording:
            logger.warning("Not recording - call start_recording() first")
            return -1

        try:
            import slicer

            index_value = str(self._step_index)

            # Record segmentation state
            seg_seq = self._sequences.get("segmentation")
            if seg_seq and self._segmentation_node:
                # Save deep copy of segmentation at this step
                seg_seq.SetDataNodeAtValue(self._segmentation_node, index_value)

            # Record all slice states
            for slice_node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
                name = slice_node.GetName()
                slice_seq = self._sequences.get(f"slice_{name}")
                if slice_seq:
                    slice_seq.SetDataNodeAtValue(slice_node, index_value)

            # Record camera state
            camera_seq = self._sequences.get("camera")
            if camera_seq:
                camera_nodes = slicer.util.getNodesByClass("vtkMRMLCameraNode")
                if camera_nodes:
                    camera_seq.SetDataNodeAtValue(camera_nodes[0], index_value)

            # Add note if provided
            if action_description:
                self._add_note_at_step(action_description, index_value)

            current_step = self._step_index
            self._step_index += 1

            logger.debug(
                f"Recorded step {current_step}: {action_description or '(no description)'}"
            )
            return current_step

        except Exception as e:
            logger.exception(f"Failed to record step: {e}")
            return -1

    def _add_note_at_step(self, text: str, index_value: str) -> None:
        """Add a text note at a specific step.

        Args:
            text: The note text.
            index_value: The step index as string.
        """
        import slicer

        notes_seq = self._sequences.get("notes")
        if not notes_seq:
            return

        # Create text node for the note
        text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode")
        text_node.SetName(f"Note_step{index_value}")
        text_node.SetText(text)

        notes_seq.SetDataNodeAtValue(text_node, index_value)

    def add_note(self, text: str) -> None:
        """Add a text note at the current step.

        Args:
            text: The note text.
        """
        if not self._is_recording:
            logger.warning("Not recording - cannot add note")
            return

        current_index = str(max(0, self._step_index - 1))
        self._add_note_at_step(text, current_index)
        logger.debug(f"Added note at step {current_index}: {text[:50]}...")

    def goto_step(self, step: int) -> bool:
        """Navigate to a specific step in the recording.

        Args:
            step: The step index to navigate to.

        Returns:
            True if navigation succeeded, False otherwise.
        """
        if not self._browser_node:
            logger.warning("No recording to navigate")
            return False

        if step < 0 or step >= self._step_index:
            logger.warning(f"Step {step} out of range (0-{self._step_index - 1})")
            return False

        try:
            self._browser_node.SetSelectedItemNumber(step)
            logger.debug(f"Navigated to step {step}")
            return True

        except Exception as e:
            logger.exception(f"Failed to navigate to step {step}: {e}")
            return False

    def get_note_at_step(self, step: int) -> str | None:
        """Get the note text at a specific step.

        Args:
            step: The step index.

        Returns:
            The note text, or None if no note at that step.
        """
        notes_seq = self._sequences.get("notes")
        if not notes_seq:
            return None

        try:
            index_value = str(step)
            data_node = notes_seq.GetDataNodeAtValue(index_value)
            if data_node:
                text: str = data_node.GetText()
                return text
            return None

        except Exception:
            return None

    def stop_recording(self) -> None:
        """Stop recording but keep sequences for playback."""
        if self._browser_node:
            self._browser_node.SetRecordingActive(False)
        self._is_recording = False
        logger.info(f"Stopped recording at step {self._step_index}")

    def cleanup(self) -> None:
        """Clean up all sequences and browser node."""
        try:
            import slicer

            # Remove all sequences
            for seq in self._sequences.values():
                if seq:
                    slicer.mrmlScene.RemoveNode(seq)
            self._sequences.clear()

            # Remove browser node
            if self._browser_node:
                slicer.mrmlScene.RemoveNode(self._browser_node)
                self._browser_node = None

            self._step_index = 0
            self._is_recording = False
            self._segmentation_node = None
            self._reference_volume = None

            logger.info("Sequence recorder cleaned up")

        except Exception as e:
            logger.exception(f"Error during cleanup: {e}")

    def get_browser_node(self) -> Any:
        """Get the sequence browser node for direct manipulation.

        Returns:
            The vtkMRMLSequenceBrowserNode, or None if not recording.
        """
        return self._browser_node

    def get_sequence(self, name: str) -> Any | None:
        """Get a specific sequence by name.

        Args:
            name: Sequence name (e.g., "segmentation", "notes", "slice_Red").

        Returns:
            The sequence node, or None if not found.
        """
        return self._sequences.get(name)


class ViewGroupManager:
    """Manage Slicer View Groups for synchronized slice navigation.

    Provides native Slicer view linking using ViewGroups and LinkedControl,
    replacing manual slice synchronization.

    Usage:
        manager = ViewGroupManager()

        # Enable linked navigation
        manager.enable_linking()

        # Views now sync automatically

        # Disable if needed
        manager.disable_linking()

        # Clean up observers on module close
        manager.cleanup()
    """

    def __init__(self) -> None:
        """Initialize the view group manager."""
        self._observers: list[tuple[Any, int]] = []
        self._linked: bool = False
        self._view_group: int = 0  # Default view group

    @property
    def is_linked(self) -> bool:
        """Return True if view linking is enabled."""
        return self._linked

    def enable_linking(self, view_group: int = 0) -> None:
        """Enable linked control for all slice views.

        This enables Slicer's native view synchronization:
        - Slice offset (position)
        - Zoom/field of view
        - Pan
        - Orientation changes

        Args:
            view_group: View group ID to use (default 0).
        """
        try:
            import slicer

            self._view_group = view_group

            # Set all slice nodes to same view group
            for slice_node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
                slice_node.SetViewGroup(view_group)

            # Enable linked control on all composite nodes
            for composite_node in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
                composite_node.SetLinkedControl(True)
                composite_node.SetHotLinkedControl(True)  # Real-time during drag

            self._linked = True
            logger.info(f"Enabled view linking (group {view_group})")

        except ImportError:
            logger.error("Slicer not available")
        except Exception as e:
            logger.exception(f"Failed to enable view linking: {e}")

    def disable_linking(self) -> None:
        """Disable linked control for all slice views."""
        try:
            import slicer

            for composite_node in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
                composite_node.SetLinkedControl(False)
                composite_node.SetHotLinkedControl(False)

            self._linked = False
            logger.info("Disabled view linking")

        except ImportError:
            logger.error("Slicer not available")
        except Exception as e:
            logger.exception(f"Failed to disable view linking: {e}")

    def set_linked(self, linked: bool) -> None:
        """Set the linked state.

        Args:
            linked: True to enable linking, False to disable.
        """
        if linked:
            self.enable_linking(self._view_group)
        else:
            self.disable_linking()

    def setup_slice_observer(
        self,
        callback: Any,
        slice_name: str = "Red",
    ) -> bool:
        """Set up observer on a slice node for bidirectional sync.

        This allows the UI (e.g., slider) to update when the user
        navigates by other means (dragging in slice view).

        Args:
            callback: Function to call when slice changes: (caller, event) -> None
            slice_name: Name of slice node to observe (default "Red").

        Returns:
            True if observer was set up, False otherwise.
        """
        try:
            import slicer
            import vtk

            slice_node = slicer.util.getNode(f"vtkMRMLSliceNode{slice_name}")
            if slice_node:
                tag = slice_node.AddObserver(
                    vtk.vtkCommand.ModifiedEvent,
                    callback,
                )
                self._observers.append((slice_node, tag))
                logger.debug(f"Added observer on {slice_name} slice")
                return True

            logger.warning(f"Slice node {slice_name} not found")
            return False

        except Exception as e:
            logger.exception(f"Failed to set up slice observer: {e}")
            return False

    def get_slice_offset(self, slice_name: str = "Red") -> float | None:
        """Get the current slice offset for a view.

        Args:
            slice_name: Name of the slice view.

        Returns:
            The slice offset, or None if not available.
        """
        try:
            import slicer

            layout_manager = slicer.app.layoutManager()
            if not layout_manager:
                return None

            slice_widget = layout_manager.sliceWidget(slice_name)
            if slice_widget:
                slice_logic = slice_widget.sliceLogic()
                offset: float = slice_logic.GetSliceOffset()
                return offset

            return None

        except Exception:
            return None

    def get_slice_range(self, slice_name: str = "Red") -> tuple[float, float] | None:
        """Get the slice offset range for a view.

        Args:
            slice_name: Name of the slice view.

        Returns:
            Tuple of (min_offset, max_offset), or None if not available.
        """
        try:
            import slicer

            layout_manager = slicer.app.layoutManager()
            if not layout_manager:
                return None

            slice_widget = layout_manager.sliceWidget(slice_name)
            if slice_widget:
                slice_logic = slice_widget.sliceLogic()
                bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                slice_logic.GetLowestVolumeSliceBounds(bounds)
                return (bounds[4], bounds[5])

            return None

        except Exception:
            return None

    def set_slice_offset(self, offset: float, slice_name: str = "Red") -> bool:
        """Set the slice offset for a view.

        Note: If view linking is enabled, this will affect all linked views.

        Args:
            offset: The slice offset value.
            slice_name: Name of the slice view to set.

        Returns:
            True if set successfully, False otherwise.
        """
        try:
            import slicer

            layout_manager = slicer.app.layoutManager()
            if not layout_manager:
                return False

            slice_widget = layout_manager.sliceWidget(slice_name)
            if slice_widget:
                slice_logic = slice_widget.sliceLogic()
                slice_logic.SetSliceOffset(offset)
                return True

            return False

        except Exception as e:
            logger.exception(f"Failed to set slice offset: {e}")
            return False

    def cleanup(self) -> None:
        """Remove all observers."""
        for node, observer_id in self._observers:
            try:
                node.RemoveObserver(observer_id)
            except Exception as e:
                logger.debug(f"Error removing observer: {e}")
        self._observers.clear()
        logger.debug("View group manager cleaned up")


class SceneViewBookmarks:
    """Manage Scene View bookmarks for interesting slices.

    Scene Views capture complete visualization state:
    - All slice positions
    - 3D camera position
    - Node visibility
    - Display properties
    - Window/level

    Usage:
        bookmarks = SceneViewBookmarks()

        # Create a bookmark
        idx = bookmarks.add_bookmark("Interesting boundary at slice 45")

        # List bookmarks
        for name, desc in bookmarks.list_bookmarks():
            print(f"{name}: {desc}")

        # Restore a bookmark
        bookmarks.restore_bookmark(0)

        # Clean up
        bookmarks.cleanup()
    """

    def __init__(self) -> None:
        """Initialize the bookmarks manager."""
        self._bookmarks: list[Any] = []

    @property
    def count(self) -> int:
        """Return number of bookmarks."""
        return len(self._bookmarks)

    def add_bookmark(
        self,
        description: str = "",
        name: str | None = None,
    ) -> int:
        """Save current view as a Scene View bookmark.

        Args:
            description: Description of what's interesting.
            name: Optional name for the bookmark.

        Returns:
            Index of the created bookmark, or -1 on failure.
        """
        try:
            import slicer

            if name is None:
                name = f"Bookmark_{len(self._bookmarks) + 1}"

            scene_view = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSceneViewNode")
            scene_view.SetName(name)
            scene_view.SetSceneViewDescription(description)
            scene_view.StoreScene()

            self._bookmarks.append(scene_view)
            idx = len(self._bookmarks) - 1

            logger.info(f"Created bookmark {idx}: {name}")
            return idx

        except ImportError:
            logger.error("Slicer not available")
            return -1
        except Exception as e:
            logger.exception(f"Failed to create bookmark: {e}")
            return -1

    def restore_bookmark(self, index: int) -> bool:
        """Restore a saved Scene View bookmark.

        Args:
            index: Index of the bookmark to restore.

        Returns:
            True if restored successfully, False otherwise.
        """
        if index < 0 or index >= len(self._bookmarks):
            logger.warning(f"Bookmark index {index} out of range")
            return False

        try:
            scene_view = self._bookmarks[index]
            scene_view.RestoreScene()
            logger.debug(f"Restored bookmark {index}: {scene_view.GetName()}")
            return True

        except Exception as e:
            logger.exception(f"Failed to restore bookmark: {e}")
            return False

    def list_bookmarks(self) -> list[tuple[str, str]]:
        """List all bookmarks.

        Returns:
            List of (name, description) tuples.
        """
        result = []
        for bookmark in self._bookmarks:
            try:
                name = bookmark.GetName() or ""
                desc = bookmark.GetSceneViewDescription() or ""
                result.append((name, desc))
            except Exception:
                result.append(("(invalid)", ""))
        return result

    def get_bookmark_name(self, index: int) -> str | None:
        """Get the name of a bookmark.

        Args:
            index: Bookmark index.

        Returns:
            The bookmark name, or None if invalid.
        """
        if 0 <= index < len(self._bookmarks):
            name: str = self._bookmarks[index].GetName()
            return name
        return None

    def remove_bookmark(self, index: int) -> bool:
        """Remove a bookmark.

        Args:
            index: Index of the bookmark to remove.

        Returns:
            True if removed successfully, False otherwise.
        """
        if index < 0 or index >= len(self._bookmarks):
            return False

        try:
            import slicer

            scene_view = self._bookmarks.pop(index)
            slicer.mrmlScene.RemoveNode(scene_view)
            logger.debug(f"Removed bookmark {index}")
            return True

        except Exception as e:
            logger.exception(f"Failed to remove bookmark: {e}")
            return False

    def clear_all(self) -> None:
        """Remove all bookmarks."""
        try:
            import slicer

            for scene_view in self._bookmarks:
                slicer.mrmlScene.RemoveNode(scene_view)
            self._bookmarks.clear()
            logger.info("Cleared all bookmarks")

        except Exception as e:
            logger.exception(f"Error clearing bookmarks: {e}")
            self._bookmarks.clear()

    def cleanup(self) -> None:
        """Clean up all bookmarks."""
        self.clear_all()
