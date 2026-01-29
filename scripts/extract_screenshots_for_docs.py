#!/usr/bin/env python3
"""Extract screenshots from test runs for documentation generation.

This script reads the screenshot manifest from test runs, filters screenshots
by doc_tags, and copies them to the docs/_static/screenshots/ directory
organized by tag categories.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_latest_manifest(screenshots_dir: Path) -> Path | None:
    """Find the most recent manifest.json in the screenshots directory."""
    manifests = list(screenshots_dir.glob("**/manifest.json"))
    if not manifests:
        return None
    # Sort by modification time, newest first
    manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0]


def load_manifest(manifest_path: Path) -> dict[str, object]:
    """Load and parse the manifest JSON file."""
    with open(manifest_path) as f:
        result: dict[str, object] = json.load(f)
        return result


def organize_by_tags(screenshots: list[dict]) -> dict[str, list[dict]]:
    """Organize screenshots by their primary doc_tag.

    Screenshots are organized into categories based on their first tag:
    - algorithm/* -> algorithms/
    - ui/* -> ui/
    - workflow/* -> workflows/
    - reviewer/* -> reviewer/
    """
    categories: dict[str, list[dict]] = {
        "algorithms": [],
        "ui": [],
        "workflows": [],
        "reviewer": [],
        "other": [],
    }

    for screenshot in screenshots:
        tags = screenshot.get("doc_tags", [])
        if not tags:
            continue

        # Categorize by first tag
        first_tag = tags[0].lower()
        if first_tag.startswith("algorithm") or first_tag in [
            "geodesic",
            "watershed",
            "random_walker",
            "level_set",
            "connected_threshold",
            "region_growing",
            "threshold_brush",
        ]:
            categories["algorithms"].append(screenshot)
        elif first_tag.startswith("ui") or first_tag in [
            "options_panel",
            "brush_settings",
            "threshold_settings",
            "parameter_wizard",
        ]:
            categories["ui"].append(screenshot)
        elif first_tag.startswith("workflow") or first_tag in [
            "getting_started",
            "tutorial",
        ]:
            categories["workflows"].append(screenshot)
        elif first_tag.startswith("reviewer"):
            categories["reviewer"].append(screenshot)
        else:
            categories["other"].append(screenshot)

    return categories


def copy_screenshots(
    screenshots: list[dict],
    source_dir: Path,
    output_dir: Path,
    category: str,
) -> list[dict]:
    """Copy screenshots to output directory and return updated metadata.

    Args:
        screenshots: List of screenshot metadata dictionaries.
        source_dir: Directory containing the original screenshots.
        output_dir: Base output directory for docs screenshots.
        category: Category name for subdirectory.

    Returns:
        List of updated screenshot metadata with new paths.
    """
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for screenshot in screenshots:
        original_path = Path(screenshot["path"])

        # Handle relative paths from manifest
        if not original_path.is_absolute():
            original_path = source_dir / original_path

        if not original_path.exists():
            # Try finding in the source_dir tree
            possible = list(source_dir.glob(f"**/{screenshot['filename']}"))
            if possible:
                original_path = possible[0]
            else:
                logger.warning(f"Screenshot not found: {screenshot['path']}")
                continue

        # Create descriptive filename from tags and description
        tags = screenshot.get("doc_tags", [])
        base_name = "_".join(tags[:3]) if tags else "screenshot"
        base_name = base_name.replace("/", "-").replace(" ", "_")
        new_filename = f"{base_name}_{screenshot['number']:03d}.png"

        dest_path = category_dir / new_filename
        shutil.copy2(original_path, dest_path)

        # Update metadata with new path
        updated = screenshot.copy()
        updated["docs_path"] = str(dest_path.relative_to(output_dir.parent))
        updated["docs_filename"] = new_filename
        copied.append(updated)

        logger.info(f"Copied: {original_path.name} -> {category}/{new_filename}")

    return copied


def generate_screenshot_index(categorized: dict[str, list[dict]], output_dir: Path) -> Path:
    """Generate an index JSON file for all documentation screenshots.

    Args:
        categorized: Dictionary of category -> screenshot list.
        output_dir: Output directory.

    Returns:
        Path to the generated index file.
    """
    categories_dict: dict[str, dict[str, object]] = {}
    total = 0

    for category, screenshots in categorized.items():
        if screenshots:
            categories_dict[category] = {
                "count": len(screenshots),
                "screenshots": screenshots,
            }
            total += len(screenshots)

    index: dict[str, object] = {
        "categories": categories_dict,
        "total_screenshots": total,
    }

    index_path = output_dir / "screenshot_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Generated screenshot index: {index_path}")
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Extract screenshots for documentation")
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("downloaded-screenshots"),
        help="Directory containing test run screenshots",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("downloaded-manifest"),
        help="Directory containing manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/source/_static/screenshots"),
        help="Output directory for documentation screenshots",
    )
    args = parser.parse_args()

    # Find manifest
    manifest_path = find_latest_manifest(args.manifest_dir)
    if manifest_path is None:
        manifest_path = find_latest_manifest(args.screenshots_dir)

    if manifest_path is None:
        logger.error("No manifest.json found in search directories")
        return 1

    logger.info(f"Using manifest: {manifest_path}")

    # Load manifest
    manifest = load_manifest(manifest_path)
    screenshots = manifest.get("screenshots", [])

    if not screenshots:
        logger.warning("No screenshots found in manifest")
        return 0

    # Filter to only screenshots with doc_tags
    doc_screenshots = [s for s in screenshots if s.get("doc_tags")]
    logger.info(
        f"Found {len(doc_screenshots)} screenshots with doc_tags "
        f"(out of {len(screenshots)} total)"
    )

    if not doc_screenshots:
        logger.warning("No screenshots with doc_tags found")
        return 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Organize by category
    categorized = organize_by_tags(doc_screenshots)

    # Copy screenshots to output directory
    result_categories: dict[str, list[dict]] = {}
    for category, category_screenshots in categorized.items():
        if category_screenshots:
            copied = copy_screenshots(
                category_screenshots,
                args.screenshots_dir,
                args.output_dir,
                category,
            )
            result_categories[category] = copied

    # Generate index
    generate_screenshot_index(result_categories, args.output_dir)

    # Summary
    total = sum(len(s) for s in result_categories.values())
    logger.info(f"Extracted {total} screenshots for documentation")
    for category, screenshots in result_categories.items():
        if screenshots:
            logger.info(f"  {category}: {len(screenshots)} screenshots")

    return 0


if __name__ == "__main__":
    exit(main())
