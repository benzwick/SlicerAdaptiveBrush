#!/usr/bin/env python3
"""Generate UI reference documentation from screenshots.

This script creates UI reference documentation by:
1. Reading screenshot manifest with doc_tags
2. Organizing screenshots by UI section
3. Generating documentation pages with screenshots
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# UI sections and their descriptions
UI_SECTIONS = {
    "options_panel": {
        "title": "Options Panel",
        "description": "The main options panel contains all brush settings and algorithm controls.",
    },
    "brush_settings": {
        "title": "Brush Settings",
        "description": "Configure brush radius, mode (2D/3D), and visualization options.",
    },
    "algorithm_selection": {
        "title": "Algorithm Selection",
        "description": "Choose from 7 different segmentation algorithms based on your needs.",
    },
    "threshold_settings": {
        "title": "Threshold Settings",
        "description": "Configure automatic or manual threshold modes for intensity-based segmentation.",
    },
    "post_processing": {
        "title": "Post-Processing",
        "description": "Apply post-processing operations like hole filling and morphological closing.",
    },
    "advanced_settings": {
        "title": "Advanced Settings",
        "description": "Fine-tune algorithm parameters for optimal results.",
    },
    "brush_mode": {
        "title": "Brush Modes",
        "description": "Switch between Add (paint) and Erase modes, or use keyboard modifiers.",
    },
}


def load_screenshot_index(screenshots_dir: Path) -> dict[str, object]:
    """Load the screenshot index JSON file.

    Args:
        screenshots_dir: Directory containing screenshot_index.json.

    Returns:
        Screenshot index dictionary.
    """
    index_path = screenshots_dir / "screenshot_index.json"
    if not index_path.exists():
        logger.warning(f"Screenshot index not found: {index_path}")
        return {"categories": {}}

    with open(index_path) as f:
        result: dict[str, object] = json.load(f)
        return result


def find_ui_screenshots(screenshots_dir: Path, section_id: str) -> list[dict]:
    """Find screenshots for a UI section.

    Args:
        screenshots_dir: Base screenshots directory.
        section_id: UI section identifier.

    Returns:
        List of screenshot info dictionaries.
    """
    screenshots: list[dict[str, object]] = []

    # Load screenshot index
    index = load_screenshot_index(screenshots_dir)

    # Look in UI category
    categories = index.get("categories", {})
    if isinstance(categories, dict):
        ui_category = categories.get("ui", {})
        if isinstance(ui_category, dict):
            ui_screenshots = ui_category.get("screenshots", [])
        else:
            ui_screenshots = []
    else:
        ui_screenshots = []

    if not isinstance(ui_screenshots, list):
        ui_screenshots = []

    for screenshot in ui_screenshots:
        tags = screenshot.get("doc_tags", [])
        if section_id in tags:
            screenshots.append(screenshot)

    # Also look for files directly
    ui_dir = screenshots_dir / "ui"
    if ui_dir.exists():
        for img_file in ui_dir.glob(f"*{section_id}*.png"):
            # Check if already in list
            already_found = any(s.get("docs_filename") == img_file.name for s in screenshots)
            if not already_found:
                screenshots.append(
                    {
                        "docs_path": f"screenshots/ui/{img_file.name}",
                        "docs_filename": img_file.name,
                        "description": img_file.stem.replace("_", " ").title(),
                    }
                )

    return screenshots


def generate_ui_section_page(section_id: str, metadata: dict, screenshots: list[dict]) -> str:
    """Generate markdown documentation for a UI section.

    Args:
        section_id: Section identifier.
        metadata: Section metadata dictionary.
        screenshots: List of screenshot info.

    Returns:
        Markdown content for the section page.
    """
    title = metadata["title"]
    description = metadata["description"]

    lines = [
        f"# {title}",
        "",
        description,
        "",
    ]

    # Add screenshots
    if screenshots:
        for screenshot in screenshots:
            desc = screenshot.get("description", "UI Screenshot")
            path = screenshot.get(
                "docs_path", f"screenshots/ui/{screenshot.get('docs_filename', '')}"
            )

            lines.append(f"## {desc}")
            lines.append("")
            lines.append(f"![{desc}](/_static/{path})")
            lines.append("")
    else:
        lines.append("*Screenshots will be generated automatically during CI builds.*")
        lines.append("")

    return "\n".join(lines)


def generate_keyboard_shortcuts_page() -> str:
    """Generate keyboard shortcuts reference page."""
    lines = [
        "# Keyboard Shortcuts",
        "",
        "AdaptiveBrush uses keyboard modifiers for quick access to common operations.",
        "",
        "## Brush Control",
        "",
        "| Shortcut | Action |",
        "|----------|--------|",
        "| `Shift` + Scroll Wheel | Adjust brush radius |",
        "| `Ctrl` + `Shift` + Scroll Wheel | Adjust threshold zone |",
        "| `Ctrl` + Left Click | Erase mode (while held) |",
        "| Middle + Left Click | Erase mode (while held) |",
        "",
        "## Standard Slicer Shortcuts",
        "",
        "| Shortcut | Action |",
        "|----------|--------|",
        "| Right-click + drag | Zoom |",
        "| Middle-click + drag | Pan |",
        "| `Shift` + Left-click + drag | Pan |",
        "| Scroll Wheel | Scroll through slices |",
        "",
        "## Brush Visualization",
        "",
        "- **Yellow outline**: Add mode (painting voxels)",
        "- **Red outline**: Erase mode (removing voxels)",
        "- **Inner circle**: Threshold sampling zone",
        "",
    ]

    return "\n".join(lines)


def generate_ui_index() -> str:
    """Generate the UI reference index page."""
    lines = [
        "# UI Reference",
        "",
        "This section documents all user interface elements of AdaptiveBrush.",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
        "keyboard_shortcuts",
    ]

    for section_id in sorted(UI_SECTIONS.keys()):
        lines.append(section_id)

    lines.extend(
        [
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate UI documentation")
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("docs/source/_static/screenshots"),
        help="Directory containing screenshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/source/generated/ui"),
        help="Output directory for UI docs",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual section pages
    for section_id, metadata in UI_SECTIONS.items():
        screenshots = find_ui_screenshots(args.screenshots_dir, section_id)
        content = generate_ui_section_page(section_id, metadata, screenshots)

        output_path = args.output_dir / f"{section_id}.md"
        output_path.write_text(content)
        logger.info(f"Generated: {output_path}")

    # Generate keyboard shortcuts page
    shortcuts_content = generate_keyboard_shortcuts_page()
    shortcuts_path = args.output_dir / "keyboard_shortcuts.md"
    shortcuts_path.write_text(shortcuts_content)
    logger.info(f"Generated: {shortcuts_path}")

    # Generate index
    index_content = generate_ui_index()
    index_path = args.output_dir / "index.md"
    index_path.write_text(index_content)
    logger.info(f"Generated: {index_path}")

    logger.info(f"Generated UI docs for {len(UI_SECTIONS)} sections")
    return 0


if __name__ == "__main__":
    exit(main())
