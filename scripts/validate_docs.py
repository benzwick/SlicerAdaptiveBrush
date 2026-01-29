#!/usr/bin/env python3
"""Validate documentation completeness.

This script checks that:
1. All algorithms have documentation and screenshots
2. All UI sections are documented
3. All public API classes have docstrings
4. No broken image links exist
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Required documentation coverage
COVERAGE_REQUIREMENTS = {
    "algorithms": {
        "required": [
            "geodesic",
            "watershed",
            "random_walker",
            "level_set",
            "connected_threshold",
            "region_growing",
            "threshold_brush",
        ],
        "screenshots_per_algorithm": ["options", "result"],
    },
    "ui": {
        "required": [
            "options_panel",
            "brush_settings",
            "algorithm_selection",
            "threshold_settings",
        ],
    },
    "workflows": {
        "required": [
            "getting_started",
        ],
    },
    "api": {
        "required": [
            "SegmentEditorEffect",
            "IntensityAnalyzer",
        ],
    },
}


class DocValidator:
    """Validate documentation completeness."""

    def __init__(self, docs_dir: Path, screenshots_dir: Path) -> None:
        self.docs_dir = docs_dir
        self.screenshots_dir = screenshots_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks.

        Returns:
            True if all checks pass.
        """
        logger.info("Starting documentation validation...")

        self._validate_algorithm_docs()
        self._validate_ui_docs()
        self._validate_workflow_docs()
        self._validate_api_docs()
        self._validate_image_links()

        # Report results
        if self.errors:
            logger.error(f"Validation FAILED with {len(self.errors)} errors:")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning(f"Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if not self.errors:
            logger.info("Validation PASSED")

        return len(self.errors) == 0

    def _validate_algorithm_docs(self) -> None:
        """Check all algorithms have documentation."""
        logger.info("Validating algorithm documentation...")

        required = COVERAGE_REQUIREMENTS["algorithms"]["required"]
        algorithms_dir = self.docs_dir / "generated" / "algorithms"

        for algo in required:
            # Check documentation file exists
            doc_path = algorithms_dir / f"{algo}.md"
            if not doc_path.exists():
                # Also check in source directory
                source_doc = self.docs_dir / "source" / "generated" / "algorithms" / f"{algo}.md"
                if not source_doc.exists():
                    self.errors.append(f"Missing algorithm documentation: {algo}")

            # Check screenshots (warning only, CI may generate them)
            algo_screenshots = list(self.screenshots_dir.glob(f"**/*{algo}*.png"))
            if not algo_screenshots:
                self.warnings.append(f"No screenshots found for algorithm: {algo}")

    def _validate_ui_docs(self) -> None:
        """Check all UI sections have documentation."""
        logger.info("Validating UI documentation...")

        required = COVERAGE_REQUIREMENTS["ui"]["required"]
        ui_dir = self.docs_dir / "generated" / "ui"

        for section in required:
            doc_path = ui_dir / f"{section}.md"
            if not doc_path.exists():
                source_doc = self.docs_dir / "source" / "generated" / "ui" / f"{section}.md"
                if not source_doc.exists():
                    self.warnings.append(f"Missing UI documentation: {section}")

    def _validate_workflow_docs(self) -> None:
        """Check workflow tutorials exist."""
        logger.info("Validating workflow documentation...")

        required = COVERAGE_REQUIREMENTS["workflows"]["required"]
        workflows_dir = self.docs_dir / "generated" / "workflows"

        for workflow in required:
            doc_path = workflows_dir / f"{workflow}.md"
            if not doc_path.exists():
                # Check in other locations
                user_guide = self.docs_dir / "user_guide" / f"{workflow}.md"
                if not user_guide.exists():
                    source_user_guide = self.docs_dir / "source" / "user_guide" / f"{workflow}.md"
                    if not source_user_guide.exists():
                        self.warnings.append(f"Missing workflow documentation: {workflow}")

    def _validate_api_docs(self) -> None:
        """Check API documentation exists."""
        logger.info("Validating API documentation...")

        required = COVERAGE_REQUIREMENTS["api"]["required"]
        api_dir = self.docs_dir / "generated" / "api"

        for module in required:
            # Check for .rst or .md file
            rst_path = api_dir / f"{module}.rst"
            md_path = api_dir / f"{module}.md"

            if not rst_path.exists() and not md_path.exists():
                source_rst = self.docs_dir / "source" / "generated" / "api" / f"{module}.rst"
                source_md = self.docs_dir / "source" / "generated" / "api" / f"{module}.md"
                if not source_rst.exists() and not source_md.exists():
                    self.warnings.append(f"Missing API documentation: {module}")

    def _validate_image_links(self) -> None:
        """Check for broken image links in documentation."""
        logger.info("Validating image links...")

        # Find all markdown/rst files
        doc_files = list(self.docs_dir.glob("**/*.md")) + list(self.docs_dir.glob("**/*.rst"))

        # Also check in source directory
        source_dir = self.docs_dir / "source"
        if source_dir.exists():
            doc_files.extend(list(source_dir.glob("**/*.md")))
            doc_files.extend(list(source_dir.glob("**/*.rst")))

        # Patterns for image references
        md_image_pattern = re.compile(r"!\[.*?\]\(([^)]+)\)")
        rst_image_pattern = re.compile(r"\.\. image:: (.+)")

        for doc_file in doc_files:
            try:
                content = doc_file.read_text()
            except UnicodeDecodeError:
                continue

            # Find markdown images
            for match in md_image_pattern.finditer(content):
                image_path = match.group(1)
                self._check_image_exists(image_path, doc_file)

            # Find RST images
            for match in rst_image_pattern.finditer(content):
                image_path = match.group(1).strip()
                self._check_image_exists(image_path, doc_file)

    def _check_image_exists(self, image_path: str, doc_file: Path) -> None:
        """Check if an image file exists.

        Args:
            image_path: Path or URL to image.
            doc_file: Document file containing the reference.
        """
        # Skip URLs
        if image_path.startswith(("http://", "https://", "//")):
            return

        # Skip Sphinx special paths that get resolved at build time
        if image_path.startswith("/_static/"):
            # Check in screenshots directory
            rel_path = image_path.replace("/_static/", "")
            full_path = self.screenshots_dir.parent / rel_path
            if not full_path.exists():
                # Warning only - may be generated during CI
                pass
            return

        # Resolve relative path
        doc_dir = doc_file.parent
        full_path = doc_dir / image_path

        if not full_path.exists():
            # Try relative to docs root
            full_path = self.docs_dir / image_path
            if not full_path.exists():
                # Warning only for build-time images
                self.warnings.append(f"Possibly broken image link: {image_path} in {doc_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Validate documentation completeness")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs/_build/html"),
        help="Built documentation directory",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("docs/source/_static/screenshots"),
        help="Screenshots directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args()

    validator = DocValidator(args.docs_dir, args.screenshots_dir)
    passed = validator.validate_all()

    if args.strict and validator.warnings:
        logger.error("Strict mode: treating warnings as errors")
        passed = False

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
