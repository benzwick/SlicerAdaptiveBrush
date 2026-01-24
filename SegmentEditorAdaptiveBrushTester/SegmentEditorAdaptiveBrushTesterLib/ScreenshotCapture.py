"""Screenshot capture utilities for Slicer views.

Captures screenshots of slice views, 3D views, layouts, and widgets.
Screenshots are organized into groups (subdirectories) with auto-incrementing numbers.
Descriptions are stored in a manifest file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotInfo:
    """Information about a captured screenshot."""

    filename: str
    path: Path
    group: str
    number: int
    description: str
    timestamp: str
    view_type: str  # "slice", "3d", "layout", "widget"
    view_name: str | None = None  # For slice views: "Red", "Yellow", "Green"
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "path": str(self.path),
            "group": self.group,
            "number": self.number,
            "description": self.description,
            "timestamp": self.timestamp,
            "view_type": self.view_type,
            "view_name": self.view_name,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ScreenshotGroup:
    """A group of screenshots (stored in a subdirectory)."""

    name: str
    path: Path
    counter: int = 0
    screenshots: list[ScreenshotInfo] = field(default_factory=list)

    def next_number(self) -> int:
        """Get the next screenshot number and increment counter."""
        self.counter += 1
        return self.counter


class ScreenshotCapture:
    """Capture screenshots organized into groups with auto-numbering.

    Screenshots are organized into subdirectories (groups). Each group
    has its own auto-incrementing counter. Descriptions are stored in
    a manifest.json file.

    Usage:
        capture = ScreenshotCapture(base_folder=Path("./screenshots"))

        # Start a group for a test
        capture.set_group("workflow_basic")

        # Take screenshots (auto-numbered)
        info = capture.screenshot("Initial state")  # -> 001.png
        info = capture.screenshot("After painting")  # -> 002.png

        # Start a new manual test group
        capture.new_group("manual_tumor_test")
        info = capture.screenshot("Found tumor region")  # -> 001.png

        # Save manifest with all descriptions
        capture.save_manifest()
    """

    def __init__(self, base_folder: Path | None = None) -> None:
        """Initialize screenshot capture.

        Args:
            base_folder: Base folder for screenshots. Groups are subdirectories.
        """
        self._base_folder = base_folder
        self._groups: dict[str, ScreenshotGroup] = {}
        self._current_group: ScreenshotGroup | None = None
        self._all_screenshots: list[ScreenshotInfo] = []

    def set_base_folder(self, base_folder: Path) -> None:
        """Set the base folder for screenshots."""
        self._base_folder = base_folder
        base_folder.mkdir(parents=True, exist_ok=True)

    def set_group(self, name: str) -> ScreenshotGroup:
        """Set the current screenshot group (creates if needed).

        Args:
            name: Group name (used as subdirectory name).

        Returns:
            The ScreenshotGroup object.
        """
        if self._base_folder is None:
            raise RuntimeError("Base folder not set. Call set_base_folder first.")

        if name not in self._groups:
            group_path = self._base_folder / name
            group_path.mkdir(parents=True, exist_ok=True)
            self._groups[name] = ScreenshotGroup(name=name, path=group_path)
            logger.info(f"Created screenshot group: {name}")

        self._current_group = self._groups[name]
        return self._current_group

    def new_group(self, name: str) -> ScreenshotGroup:
        """Create a new screenshot group and make it current.

        Args:
            name: Group name (used as subdirectory name).

        Returns:
            The new ScreenshotGroup object.
        """
        # Force creation of new group even if name exists (add suffix)
        if self._base_folder is None:
            raise RuntimeError("Base folder not set. Call set_base_folder first.")

        original_name = name
        counter = 1
        while name in self._groups:
            name = f"{original_name}_{counter}"
            counter += 1

        return self.set_group(name)

    @property
    def current_group(self) -> ScreenshotGroup | None:
        """Get the current screenshot group."""
        return self._current_group

    def screenshot(self, description: str = "") -> ScreenshotInfo:
        """Take a screenshot of the current layout (auto-numbered).

        Args:
            description: Description of what the screenshot shows.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        return self.capture_layout(description)

    def capture_layout(self, description: str = "") -> ScreenshotInfo:
        """Capture screenshot of the entire Slicer layout.

        Args:
            description: Human-readable description.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        import slicer

        group = self._ensure_group()
        number = group.next_number()
        filename = f"{number:03d}.png"
        filepath = group.path / filename

        # Capture the main window
        widget = slicer.util.mainWindow()
        pixmap = widget.grab()
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            group=group.name,
            number=number,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="layout",
            width=pixmap.width(),
            height=pixmap.height(),
        )

        group.screenshots.append(info)
        self._all_screenshots.append(info)
        logger.info(f"Screenshot {group.name}/{filename}: {description}")
        return info

    def capture_slice_view(self, view: str, description: str = "") -> ScreenshotInfo:
        """Capture screenshot of a specific slice view.

        Args:
            view: View name ("Red", "Yellow", "Green").
            description: Human-readable description.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        import slicer

        group = self._ensure_group()
        number = group.next_number()
        filename = f"{number:03d}_{view.lower()}.png"
        filepath = group.path / filename

        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(view)

        if sliceWidget is None:
            raise ValueError(f"Slice view not found: {view}")

        pixmap = sliceWidget.grab()
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            group=group.name,
            number=number,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="slice",
            view_name=view,
            width=pixmap.width(),
            height=pixmap.height(),
        )

        group.screenshots.append(info)
        self._all_screenshots.append(info)
        logger.info(f"Screenshot {group.name}/{filename}: {description}")
        return info

    def capture_3d_view(self, description: str = "", view_node_index: int = 0) -> ScreenshotInfo:
        """Capture screenshot of the 3D view.

        Args:
            description: Human-readable description.
            view_node_index: Index of 3D view node (default 0 for first).

        Returns:
            ScreenshotInfo with path and metadata.
        """
        import slicer

        group = self._ensure_group()
        number = group.next_number()
        filename = f"{number:03d}_3d.png"
        filepath = group.path / filename

        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(view_node_index)

        if threeDWidget is None:
            raise ValueError(f"3D view not found at index: {view_node_index}")

        pixmap = threeDWidget.grab()
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            group=group.name,
            number=number,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="3d",
            width=pixmap.width(),
            height=pixmap.height(),
        )

        group.screenshots.append(info)
        self._all_screenshots.append(info)
        logger.info(f"Screenshot {group.name}/{filename}: {description}")
        return info

    def capture_widget(self, widget, description: str = "") -> ScreenshotInfo:
        """Capture screenshot of a specific Qt widget.

        Args:
            widget: Qt widget to capture.
            description: Human-readable description.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        group = self._ensure_group()
        number = group.next_number()
        filename = f"{number:03d}_widget.png"
        filepath = group.path / filename

        pixmap = widget.grab()
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            group=group.name,
            number=number,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="widget",
            width=pixmap.width(),
            height=pixmap.height(),
        )

        group.screenshots.append(info)
        self._all_screenshots.append(info)
        logger.info(f"Screenshot {group.name}/{filename}: {description}")
        return info

    def _ensure_group(self) -> ScreenshotGroup:
        """Ensure there's a current group, create default if needed."""
        if self._current_group is None:
            if self._base_folder is None:
                raise RuntimeError("Base folder not set. Call set_base_folder first.")
            self.set_group("default")
        return self._current_group  # type: ignore

    def save_manifest(self) -> Path:
        """Save manifest with all screenshot descriptions.

        Returns:
            Path to the manifest file.
        """
        if self._base_folder is None:
            raise RuntimeError("Base folder not set.")

        manifest_path = self._base_folder / "manifest.json"

        # Organize by group
        groups_dict: dict[str, dict] = {}
        for group_name, group in self._groups.items():
            groups_dict[group_name] = {
                "count": len(group.screenshots),
                "screenshots": [s.to_dict() for s in group.screenshots],
            }

        manifest = {
            "generated": datetime.now().isoformat(),
            "total_screenshots": len(self._all_screenshots),
            "groups": groups_dict,
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved screenshot manifest: {manifest_path}")
        return manifest_path

    def update_description(self, group: str, number: int, description: str) -> bool:
        """Update the description for a screenshot.

        Args:
            group: Group name.
            number: Screenshot number within the group.
            description: New description.

        Returns:
            True if found and updated, False otherwise.
        """
        if group not in self._groups:
            return False

        for screenshot in self._groups[group].screenshots:
            if screenshot.number == number:
                screenshot.description = description
                logger.info(f"Updated description for {group}/{number:03d}")
                return True

        return False

    def get_all_screenshots(self) -> list[ScreenshotInfo]:
        """Get all screenshots taken."""
        return self._all_screenshots.copy()

    def get_group_screenshots(self, group: str) -> list[ScreenshotInfo]:
        """Get screenshots for a specific group."""
        if group not in self._groups:
            return []
        return self._groups[group].screenshots.copy()
