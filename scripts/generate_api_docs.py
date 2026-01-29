#!/usr/bin/env python3
"""Generate API documentation from Python docstrings.

This script extracts docstrings from Python modules and generates
Sphinx-compatible RST files for API documentation.
"""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DocstringExtractor(ast.NodeVisitor):
    """Extract docstrings from Python AST."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.classes: list[dict] = []
        self.functions: list[dict] = []
        self._current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class docstring and methods."""
        docstring = ast.get_docstring(node) or ""
        bases = [self._get_base_name(b) for b in node.bases]

        # Collect methods
        methods: list[dict[str, object]] = []
        self._current_class = node.name
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = ast.get_docstring(item) or ""
                # Skip private methods except __init__
                if not item.name.startswith("_") or item.name == "__init__":
                    methods.append(
                        {
                            "name": item.name,
                            "docstring": method_doc,
                            "lineno": item.lineno,
                        }
                    )
        self._current_class = None

        class_info: dict[str, object] = {
            "name": node.name,
            "docstring": docstring,
            "bases": bases,
            "methods": methods,
            "lineno": node.lineno,
        }

        self.classes.append(class_info)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function docstrings (module-level only)."""
        if self._current_class is None and not node.name.startswith("_"):
            docstring = ast.get_docstring(node) or ""
            self.functions.append(
                {
                    "name": node.name,
                    "docstring": docstring,
                    "lineno": node.lineno,
                }
            )
        self.generic_visit(node)

    def _get_base_name(self, base: ast.expr) -> str:
        """Get the name of a base class."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._get_base_name(base.value)}.{base.attr}"
        return "?"


def extract_module_docs(file_path: Path) -> dict:
    """Extract documentation from a Python module.

    Args:
        file_path: Path to the Python file.

    Returns:
        Dictionary with module documentation info.
    """
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.warning(f"Syntax error in {file_path}: {e}")
        return {}

    module_name = file_path.stem
    module_docstring = ast.get_docstring(tree) or ""

    extractor = DocstringExtractor(module_name)
    extractor.visit(tree)

    return {
        "name": module_name,
        "path": str(file_path),
        "docstring": module_docstring,
        "classes": extractor.classes,
        "functions": extractor.functions,
    }


def generate_class_rst(class_info: dict, module_name: str) -> str:
    """Generate RST documentation for a class."""
    lines = []
    name = class_info["name"]
    docstring = class_info["docstring"]
    bases = class_info["bases"]
    methods = class_info["methods"]

    # Class header
    lines.append(f".. py:class:: {name}")
    if bases:
        lines.append(f"   :bases: {', '.join(bases)}")
    lines.append("")

    # Docstring
    if docstring:
        for line in docstring.split("\n"):
            lines.append(f"   {line}")
        lines.append("")

    # Methods
    if methods:
        lines.append("   **Methods:**")
        lines.append("")
        for method in methods:
            method_name = method["name"]
            method_doc = method["docstring"]

            lines.append(f"   .. py:method:: {method_name}()")
            lines.append("")
            if method_doc:
                first_line = method_doc.split("\n")[0]
                lines.append(f"      {first_line}")
                lines.append("")

    return "\n".join(lines)


def generate_module_rst(module_info: dict) -> str:
    """Generate RST documentation for a module."""
    lines = []
    name = module_info["name"]
    docstring = module_info["docstring"]
    classes = module_info["classes"]
    functions = module_info["functions"]

    # Module header
    lines.append(name)
    lines.append("=" * len(name))
    lines.append("")

    # Module docstring
    if docstring:
        lines.append(docstring)
        lines.append("")

    # Classes
    if classes:
        lines.append("Classes")
        lines.append("-" * 7)
        lines.append("")

        for class_info in classes:
            lines.append(generate_class_rst(class_info, name))
            lines.append("")

    # Functions
    if functions:
        lines.append("Functions")
        lines.append("-" * 9)
        lines.append("")

        for func_info in functions:
            func_name = func_info["name"]
            func_doc = func_info["docstring"]

            lines.append(f".. py:function:: {func_name}()")
            lines.append("")
            if func_doc:
                for line in func_doc.split("\n"):
                    lines.append(f"   {line}")
                lines.append("")

    return "\n".join(lines)


def generate_api_index(modules: list[dict], output_dir: Path) -> Path:
    """Generate the API index page.

    Args:
        modules: List of module info dictionaries.
        output_dir: Output directory.

    Returns:
        Path to the generated index file.
    """
    lines = [
        "API Reference",
        "=============",
        "",
        "This section contains auto-generated API documentation from Python docstrings.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]

    for module_info in sorted(modules, key=lambda m: m["name"]):
        name = module_info["name"]
        lines.append(f"   {name}")

    index_path = output_dir / "index.rst"
    index_path.write_text("\n".join(lines))
    logger.info(f"Generated API index: {index_path}")
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument(
        "--source-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("SegmentEditorAdaptiveBrush/SegmentEditorAdaptiveBrushLib"),
            Path("SegmentEditorAdaptiveBrushTester/SegmentEditorAdaptiveBrushTesterLib"),
            Path("SegmentEditorAdaptiveBrushReviewer/SegmentEditorAdaptiveBrushReviewerLib"),
        ],
        help="Source directories to scan",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/source/generated/api"),
        help="Output directory for API docs",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract documentation from all modules
    all_modules = []

    for source_dir in args.source_dirs:
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue

        logger.info(f"Scanning: {source_dir}")

        for py_file in source_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            logger.info(f"Processing: {py_file.name}")
            module_info = extract_module_docs(py_file)

            if module_info and (module_info.get("classes") or module_info.get("functions")):
                all_modules.append(module_info)

                # Generate module RST
                rst_content = generate_module_rst(module_info)
                rst_path = args.output_dir / f"{module_info['name']}.rst"
                rst_path.write_text(rst_content)
                logger.info(f"Generated: {rst_path}")

    # Generate index
    if all_modules:
        generate_api_index(all_modules, args.output_dir)

    logger.info(f"Generated API docs for {len(all_modules)} modules")
    return 0


if __name__ == "__main__":
    exit(main())
