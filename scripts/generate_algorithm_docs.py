#!/usr/bin/env python3
"""Generate algorithm documentation pages from metadata and screenshots.

This script creates comprehensive algorithm documentation pages by:
1. Reading algorithm metadata from source code
2. Including auto-captured screenshots
3. Generating comparison tables and usage examples
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Algorithm metadata (could be extracted from source code in future)
ALGORITHMS = {
    "geodesic": {
        "name": "Geodesic Distance",
        "description": "Uses Fast Marching to compute geodesic distances from the seed point, combining image gradient and intensity similarity to find natural boundaries.",
        "speed": "Fast",
        "precision": "High",
        "best_for": "General use, most tissue types",
        "parameters": ["Sensitivity", "Brush Radius"],
        "pros": ["Fast computation", "Good boundary detection", "Works well on most images"],
        "cons": ["May leak through weak boundaries"],
    },
    "watershed": {
        "name": "Watershed",
        "description": "Marker-based watershed segmentation that treats image gradients as a topographic surface and finds catchment basins.",
        "speed": "Medium",
        "precision": "High",
        "best_for": "Marker-based segmentation, clear boundaries",
        "parameters": ["Edge Sensitivity", "Brush Radius"],
        "pros": ["Good at finding natural boundaries", "Handles complex shapes"],
        "cons": ["Can over-segment in noisy images"],
    },
    "random_walker": {
        "name": "Random Walker",
        "description": "Probabilistic algorithm using random walks to determine pixel labels based on similarity to seed point.",
        "speed": "Medium",
        "precision": "High",
        "best_for": "Noisy images, probabilistic segmentation",
        "parameters": ["Sensitivity", "Brush Radius"],
        "pros": ["Robust to noise", "Smooth boundaries"],
        "cons": ["Slower on large regions", "Requires scikit-image"],
    },
    "level_set": {
        "name": "Level Set",
        "description": "Geodesic Active Contour using level set methods to evolve a contour toward object boundaries.",
        "speed": "Slow",
        "precision": "Very High",
        "best_for": "Irregular boundaries, high precision needed",
        "parameters": ["Sensitivity", "Iterations", "Brush Radius"],
        "pros": ["Handles complex topology", "Very precise boundaries"],
        "cons": ["Slowest algorithm", "Requires parameter tuning"],
    },
    "connected_threshold": {
        "name": "Connected Threshold",
        "description": "Simple region growing based on intensity thresholds. Fast but less adaptive.",
        "speed": "Very Fast",
        "precision": "Low",
        "best_for": "Quick rough segmentation, uniform regions",
        "parameters": ["Threshold Range", "Brush Radius"],
        "pros": ["Very fast", "Simple to understand"],
        "cons": ["Not adaptive", "Requires manual threshold setting"],
    },
    "region_growing": {
        "name": "Region Growing",
        "description": "Confidence-connected region growing that uses local statistics to determine region membership.",
        "speed": "Fast",
        "precision": "Medium",
        "best_for": "Homogeneous regions with clear boundaries",
        "parameters": ["Sensitivity", "Brush Radius"],
        "pros": ["Fast", "Adapts to local statistics"],
        "cons": ["Sensitive to seed point placement"],
    },
    "threshold_brush": {
        "name": "Threshold Brush",
        "description": "Simple threshold-based painting with automatic threshold methods (Otsu, Huang, etc.).",
        "speed": "Very Fast",
        "precision": "Variable",
        "best_for": "Simple threshold painting, when boundaries are clear",
        "parameters": ["Auto Method", "Threshold Range", "Brush Radius"],
        "pros": ["Very fast", "Multiple auto-threshold methods"],
        "cons": ["No boundary awareness", "Simple threshold only"],
    },
}


def find_algorithm_screenshots(screenshots_dir: Path, algorithm_id: str) -> list[dict]:
    """Find screenshots for a specific algorithm.

    Args:
        screenshots_dir: Directory containing screenshots.
        algorithm_id: Algorithm identifier.

    Returns:
        List of screenshot info dictionaries.
    """
    screenshots: list[dict[str, str]] = []

    # Look for screenshots with algorithm tag
    algorithms_dir = screenshots_dir / "algorithms"
    if not algorithms_dir.exists():
        return screenshots

    # Find matching files
    for img_file in algorithms_dir.glob(f"*{algorithm_id}*.png"):
        screenshots.append(
            {
                "path": str(img_file.relative_to(screenshots_dir.parent.parent)),
                "filename": img_file.name,
            }
        )

    return screenshots


def generate_algorithm_page(algorithm_id: str, metadata: dict, screenshots: list[dict]) -> str:
    """Generate markdown documentation for an algorithm.

    Args:
        algorithm_id: Algorithm identifier.
        metadata: Algorithm metadata dictionary.
        screenshots: List of screenshot info.

    Returns:
        Markdown content for the algorithm page.
    """
    name = metadata["name"]
    description = metadata["description"]
    speed = metadata["speed"]
    precision = metadata["precision"]
    best_for = metadata["best_for"]
    parameters = metadata["parameters"]
    pros = metadata["pros"]
    cons = metadata["cons"]

    lines = [
        f"# {name}",
        "",
        description,
        "",
        "## Overview",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| **Speed** | {speed} |",
        f"| **Precision** | {precision} |",
        f"| **Best For** | {best_for} |",
        "",
        "## Parameters",
        "",
    ]

    for param in parameters:
        lines.append(f"- **{param}**")
    lines.append("")

    lines.extend(
        [
            "## Strengths",
            "",
        ]
    )
    for pro in pros:
        lines.append(f"- {pro}")
    lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
        ]
    )
    for con in cons:
        lines.append(f"- {con}")
    lines.append("")

    # Add screenshots if available
    if screenshots:
        lines.extend(
            [
                "## Screenshots",
                "",
            ]
        )
        for i, screenshot in enumerate(screenshots):
            caption = f"{name} - Example {i + 1}"
            lines.append(f"![{caption}](/_static/{screenshot['path']})")
            lines.append("")

    return "\n".join(lines)


def generate_comparison_table() -> str:
    """Generate algorithm comparison table."""
    lines = [
        "# Algorithm Comparison",
        "",
        "This table compares all available algorithms to help you choose the right one for your use case.",
        "",
        "| Algorithm | Speed | Precision | Best For |",
        "|-----------|-------|-----------|----------|",
    ]

    for algo_id, metadata in ALGORITHMS.items():
        name = metadata["name"]
        speed = metadata["speed"]
        precision = metadata["precision"]
        best_for = metadata["best_for"]
        lines.append(f"| [{name}]({algo_id}.md) | {speed} | {precision} | {best_for} |")

    lines.extend(
        [
            "",
            "## When to Use Each Algorithm",
            "",
            "### For Speed",
            "If you need fast results, use **Threshold Brush** or **Connected Threshold**.",
            "",
            "### For Precision",
            "If you need precise boundaries, use **Level Set** or **Geodesic Distance**.",
            "",
            "### For Noisy Images",
            "If your image has noise, use **Random Walker** which is robust to noise.",
            "",
            "### For General Use",
            "**Geodesic Distance** is the recommended default for most use cases.",
        ]
    )

    return "\n".join(lines)


def generate_algorithms_index(algorithms: dict) -> str:
    """Generate the algorithms index page."""
    lines = [
        "# Algorithms",
        "",
        "AdaptiveBrush provides multiple segmentation algorithms, each with different strengths.",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
        "comparison",
    ]

    for algo_id in sorted(algorithms.keys()):
        lines.append(algo_id)

    lines.extend(
        [
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate algorithm documentation")
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("docs/source/_static/screenshots"),
        help="Directory containing screenshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/source/generated/algorithms"),
        help="Output directory for algorithm docs",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual algorithm pages
    for algo_id, metadata in ALGORITHMS.items():
        screenshots = find_algorithm_screenshots(args.screenshots_dir, algo_id)
        content = generate_algorithm_page(algo_id, metadata, screenshots)

        output_path = args.output_dir / f"{algo_id}.md"
        output_path.write_text(content)
        logger.info(f"Generated: {output_path}")

    # Generate comparison table
    comparison_content = generate_comparison_table()
    comparison_path = args.output_dir / "comparison.md"
    comparison_path.write_text(comparison_content)
    logger.info(f"Generated: {comparison_path}")

    # Generate index
    index_content = generate_algorithms_index(ALGORITHMS)
    index_path = args.output_dir / "index.md"
    index_path.write_text(index_content)
    logger.info(f"Generated: {index_path}")

    logger.info(f"Generated algorithm docs for {len(ALGORITHMS)} algorithms")
    return 0


if __name__ == "__main__":
    exit(main())
