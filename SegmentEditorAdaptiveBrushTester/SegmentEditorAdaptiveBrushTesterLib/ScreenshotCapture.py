"""Screenshot capture utilities for Slicer views.

Captures screenshots of slice views, 3D views, layouts, and widgets.
Screenshots are saved with metadata for documentation generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotInfo:
    """Information about a captured screenshot."""

    filename: str
    path: Path
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
            "description": self.description,
            "timestamp": self.timestamp,
            "view_type": self.view_type,
            "view_name": self.view_name,
            "width": self.width,
            "height": self.height,
        }


class ScreenshotCapture:
    """Capture screenshots of Slicer views.

    Usage:
        capture = ScreenshotCapture()

        # Capture current layout
        info = capture.capture_layout(
            screenshot_id="001_initial",
            description="Initial state before painting",
            output_folder=Path("./screenshots"),
        )

        # Capture specific slice view
        info = capture.capture_slice_view(
            view="Red",
            screenshot_id="002_axial",
            description="Axial view after painting",
            output_folder=Path("./screenshots"),
        )
    """

    def capture_layout(
        self,
        screenshot_id: str,
        description: str,
        output_folder: Path,
    ) -> ScreenshotInfo:
        """Capture screenshot of the entire Slicer layout.

        Args:
            screenshot_id: Unique identifier for the screenshot.
            description: Human-readable description.
            output_folder: Directory to save screenshot.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        import slicer

        output_folder.mkdir(parents=True, exist_ok=True)
        filename = f"{screenshot_id}.png"
        filepath = output_folder / filename

        # Capture the main window
        widget = slicer.util.mainWindow()
        pixmap = widget.grab()

        # Save
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="layout",
            width=pixmap.width(),
            height=pixmap.height(),
        )

        logger.info(f"Captured layout screenshot: {filepath}")
        return info

    def capture_slice_view(
        self,
        view: str,
        screenshot_id: str,
        description: str,
        output_folder: Path,
    ) -> ScreenshotInfo:
        """Capture screenshot of a specific slice view.

        Args:
            view: View name ("Red", "Yellow", "Green").
            screenshot_id: Unique identifier for the screenshot.
            description: Human-readable description.
            output_folder: Directory to save screenshot.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        import slicer

        output_folder.mkdir(parents=True, exist_ok=True)
        filename = f"{screenshot_id}_{view.lower()}.png"
        filepath = output_folder / filename

        # Get the slice view widget
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(view)

        if sliceWidget is None:
            raise ValueError(f"Slice view not found: {view}")

        # Capture
        pixmap = sliceWidget.grab()
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="slice",
            view_name=view,
            width=pixmap.width(),
            height=pixmap.height(),
        )

        logger.info(f"Captured {view} slice screenshot: {filepath}")
        return info

    def capture_3d_view(
        self,
        screenshot_id: str,
        description: str,
        output_folder: Path,
        view_node_index: int = 0,
    ) -> ScreenshotInfo:
        """Capture screenshot of the 3D view.

        Args:
            screenshot_id: Unique identifier for the screenshot.
            description: Human-readable description.
            output_folder: Directory to save screenshot.
            view_node_index: Index of 3D view node (default 0 for first).

        Returns:
            ScreenshotInfo with path and metadata.
        """
        import slicer

        output_folder.mkdir(parents=True, exist_ok=True)
        filename = f"{screenshot_id}_3d.png"
        filepath = output_folder / filename

        # Get the 3D view widget
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(view_node_index)

        if threeDWidget is None:
            raise ValueError(f"3D view not found at index: {view_node_index}")

        # Capture
        pixmap = threeDWidget.grab()
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="3d",
            width=pixmap.width(),
            height=pixmap.height(),
        )

        logger.info(f"Captured 3D screenshot: {filepath}")
        return info

    def capture_widget(
        self,
        widget,
        screenshot_id: str,
        description: str,
        output_folder: Path,
    ) -> ScreenshotInfo:
        """Capture screenshot of a specific Qt widget.

        Args:
            widget: Qt widget to capture.
            screenshot_id: Unique identifier for the screenshot.
            description: Human-readable description.
            output_folder: Directory to save screenshot.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        filename = f"{screenshot_id}_widget.png"
        filepath = output_folder / filename

        # Capture
        pixmap = widget.grab()
        pixmap.save(str(filepath))

        info = ScreenshotInfo(
            filename=filename,
            path=filepath,
            description=description,
            timestamp=datetime.now().isoformat(),
            view_type="widget",
            width=pixmap.width(),
            height=pixmap.height(),
        )

        logger.info(f"Captured widget screenshot: {filepath}")
        return info

    def capture_segment_editor_panel(
        self,
        screenshot_id: str,
        description: str,
        output_folder: Path,
    ) -> ScreenshotInfo:
        """Capture screenshot of the Segment Editor module panel.

        Args:
            screenshot_id: Unique identifier for the screenshot.
            description: Human-readable description.
            output_folder: Directory to save screenshot.

        Returns:
            ScreenshotInfo with path and metadata.
        """
        import slicer

        # Get the segment editor module widget
        module_widget = slicer.modules.segmenteditor.widgetRepresentation()

        if module_widget is None:
            raise ValueError("Segment Editor module widget not found")

        return self.capture_widget(
            widget=module_widget,
            screenshot_id=screenshot_id,
            description=description,
            output_folder=output_folder,
        )
