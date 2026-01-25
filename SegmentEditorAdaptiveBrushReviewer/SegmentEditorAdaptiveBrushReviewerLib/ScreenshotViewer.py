"""Screenshot thumbnail viewer for review UI.

Displays screenshot thumbnails with selection and actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotInfo:
    """Information about a screenshot."""

    path: Path
    name: str
    description: str = ""
    trial_number: int = 0
    step: int = 0


class ScreenshotViewer:
    """Screenshot gallery with thumbnails.

    Manages a collection of screenshots with thumbnail display,
    selection, and actions (view full, copy path).
    """

    def __init__(self):
        """Initialize viewer."""
        self.screenshots: list[ScreenshotInfo] = []
        self.selected_index: int = -1

    def set_screenshots(self, screenshots: list[ScreenshotInfo]) -> None:
        """Update gallery with new screenshots.

        Args:
            screenshots: List of screenshot info objects.
        """
        self.screenshots = screenshots
        self.selected_index = 0 if screenshots else -1
        logger.debug(f"Set {len(screenshots)} screenshots")

    def load_from_directory(self, directory: Path | str) -> list[ScreenshotInfo]:
        """Load screenshots from a directory.

        Args:
            directory: Directory containing PNG files.

        Returns:
            List of ScreenshotInfo objects.
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        screenshots = []
        for i, png_path in enumerate(sorted(directory.glob("*.png"))):
            info = ScreenshotInfo(
                path=png_path,
                name=png_path.stem,
                description=f"Screenshot {i + 1}",
                step=i,
            )
            screenshots.append(info)

        self.set_screenshots(screenshots)
        return screenshots

    def select(self, index: int) -> ScreenshotInfo | None:
        """Select a screenshot by index.

        Args:
            index: Screenshot index.

        Returns:
            Selected ScreenshotInfo or None.
        """
        if 0 <= index < len(self.screenshots):
            self.selected_index = index
            return self.screenshots[index]
        return None

    def get_selected(self) -> ScreenshotInfo | None:
        """Get currently selected screenshot.

        Returns:
            Selected ScreenshotInfo or None.
        """
        if 0 <= self.selected_index < len(self.screenshots):
            return self.screenshots[self.selected_index]
        return None

    def select_next(self) -> ScreenshotInfo | None:
        """Select next screenshot.

        Returns:
            Selected ScreenshotInfo or None.
        """
        if self.selected_index < len(self.screenshots) - 1:
            return self.select(self.selected_index + 1)
        return self.get_selected()

    def select_previous(self) -> ScreenshotInfo | None:
        """Select previous screenshot.

        Returns:
            Selected ScreenshotInfo or None.
        """
        if self.selected_index > 0:
            return self.select(self.selected_index - 1)
        return self.get_selected()

    def view_full_size(self) -> bool:
        """Open selected screenshot in system viewer.

        Returns:
            True if opened successfully.
        """
        selected = self.get_selected()
        if not selected:
            return False

        try:
            import qt

            qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(selected.path)))
            return True
        except Exception as e:
            logger.error(f"Could not open screenshot: {e}")
            return False

    def copy_path_to_clipboard(self) -> bool:
        """Copy selected screenshot path to clipboard.

        Returns:
            True if copied successfully.
        """
        selected = self.get_selected()
        if not selected:
            return False

        try:
            import qt

            clipboard = qt.QApplication.clipboard()
            clipboard.setText(str(selected.path))
            logger.debug(f"Copied path: {selected.path}")
            return True
        except Exception as e:
            logger.error(f"Could not copy path: {e}")
            return False

    def get_count(self) -> int:
        """Get number of screenshots."""
        return len(self.screenshots)

    def clear(self) -> None:
        """Clear all screenshots."""
        self.screenshots = []
        self.selected_index = -1
