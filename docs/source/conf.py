# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "SlicerAdaptiveBrush"
copyright = "2026, SlicerAdaptiveBrush Contributors"
author = "SlicerAdaptiveBrush Contributors"
release = "0.16.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

# MyST parser configuration (allows Markdown)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = "bysource"
