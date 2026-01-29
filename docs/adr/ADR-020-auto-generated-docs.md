# ADR-020: Auto-Generated Documentation System

## Status

Accepted

## Context

Manual documentation has several inherent problems:

1. **Staleness**: Documentation quickly becomes outdated as code evolves
2. **Inconsistency**: Screenshots don't match current UI after changes
3. **Incompleteness**: New features often ship without documentation
4. **Maintenance burden**: Keeping docs in sync with code is tedious
5. **Duplication**: Same information exists in code, tests, and docs

The project already has:
- Comprehensive test suite with screenshot capture (ADR-010)
- Sphinx documentation infrastructure (ADR-009)
- CI/CD pipeline (ADR-008)

We need a system where ALL documentation is generated from code and tests, ensuring it's always accurate and complete.

## Decision

Implement a **100% auto-generated documentation system** where no documentation is written manually (except structural configuration).

### Core Principles

1. **No Manual Writing**: All content derives from code, docstrings, and tests
2. **Tests Drive Documentation**: If it's not tested with screenshots, it's not documented
3. **100% Coverage**: Every UI state, algorithm, and feature has documentation
4. **Always Current**: Documentation regenerates on every merge to main

### Documentation Sources

| Source | Generates |
|--------|-----------|
| Test screenshots (doc_tags) | Algorithm examples, UI reference, tutorials |
| Python docstrings | API reference |
| Algorithm metadata | Algorithm comparison pages |
| Test descriptions | Feature documentation |
| Configuration files | Parameter reference |

### Screenshot Tagging System

Tests produce documentation-ready screenshots using `doc_tags`:

```python
def test_watershed_algorithm(ctx: TestContext):
    """Test watershed algorithm - generates documentation screenshots."""
    # Setup screenshot
    ctx.screenshot("Initial state before watershed",
                   doc_tags=["algorithm", "watershed", "before"])

    # Execute algorithm
    effect.paint_at(128, 100, 90)

    # Result screenshot
    ctx.screenshot("Watershed segmentation result",
                   doc_tags=["algorithm", "watershed", "after", "result"])
```

Tag conventions:
- `algorithm/<name>` - Algorithm-specific screenshots
- `ui/<section>` - UI element screenshots
- `workflow/<name>` - Tutorial workflow screenshots
- `before/after` - State change comparisons
- `result` - Final output screenshots

### Documentation Test Categories

| Category | Purpose | Output |
|----------|---------|--------|
| `test_docs_algorithms.py` | All 7 algorithms with variations | Algorithm reference pages |
| `test_docs_ui_reference.py` | All UI panels and widgets | UI reference guide |
| `test_docs_workflows.py` | Step-by-step tutorials | Getting started, tutorials |
| `test_docs_reviewer.py` | Reviewer module UI | Reviewer module docs |

### Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. generate-screenshots job                                        │
│     ┌───────────────┐    ┌──────────────────┐                      │
│     │ Run Slicer    │───▶│ test_docs_*.py   │                      │
│     │ with Xvfb     │    │ (doc_tags)       │                      │
│     └───────────────┘    └──────────────────┘                      │
│            │                      │                                 │
│            ▼                      ▼                                 │
│     ┌───────────────┐    ┌──────────────────┐                      │
│     │ Screenshots   │    │ manifest.json    │                      │
│     │ (artifact)    │    │ (with doc_tags)  │                      │
│     └───────────────┘    └──────────────────┘                      │
│                                                                     │
│  2. build-docs job                                                  │
│     ┌───────────────────────────────────────┐                      │
│     │ extract_screenshots_for_docs.py       │                      │
│     │ - Filter by doc_tags                  │                      │
│     │ - Copy to docs/_static/screenshots/   │                      │
│     └───────────────────────────────────────┘                      │
│            │                                                        │
│            ▼                                                        │
│     ┌───────────────────────────────────────┐                      │
│     │ generate_algorithm_docs.py            │                      │
│     │ - Read algorithm metadata from source │                      │
│     │ - Generate algorithm pages            │                      │
│     │ - Include tagged screenshots          │                      │
│     └───────────────────────────────────────┘                      │
│            │                                                        │
│            ▼                                                        │
│     ┌───────────────────────────────────────┐                      │
│     │ generate_api_docs.py                  │                      │
│     │ - Extract from docstrings             │                      │
│     │ - Generate API reference              │                      │
│     └───────────────────────────────────────┘                      │
│            │                                                        │
│            ▼                                                        │
│     ┌───────────────────────────────────────┐                      │
│     │ Sphinx build                          │                      │
│     └───────────────────────────────────────┘                      │
│            │                                                        │
│            ▼                                                        │
│     ┌───────────────────────────────────────┐                      │
│     │ validate_docs.py                      │                      │
│     │ - Check completeness                  │                      │
│     │ - Verify no broken links              │                      │
│     │ - Ensure coverage requirements met    │                      │
│     └───────────────────────────────────────┘                      │
│                                                                     │
│  3. deploy job (main branch only)                                   │
│     ┌───────────────────────────────────────┐                      │
│     │ GitHub Pages deployment               │                      │
│     └───────────────────────────────────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Coverage Requirements

Documentation validation enforces completeness:

```python
# scripts/validate_docs.py
COVERAGE_REQUIREMENTS = {
    "algorithms": {
        "required": ["geodesic", "watershed", "random_walker", "level_set",
                     "connected_threshold", "region_growing", "threshold_brush"],
        "screenshots_per_algorithm": ["options", "result"],
    },
    "ui": {
        "required": ["options_panel", "brush_settings", "threshold_settings",
                     "post_processing", "parameter_wizard"],
    },
    "workflows": {
        "required": ["getting_started", "brain_tumor"],
    },
    "api": {
        "required": ["SegmentEditorEffect", "IntensityAnalyzer"],
    },
}
```

Validation fails the CI build if:
- Any required algorithm lacks screenshots
- Any UI section is undocumented
- Any public API class lacks docstrings
- Broken image links exist

### Existing Documentation Handling

All existing manual documentation will be replaced:
- `docs/source/user_guide/*.md` - Regenerated from tests
- `docs/source/developer_guide/*.md` - Regenerated from docstrings and tests
- Manual screenshots deleted, replaced with auto-captured versions

This ensures 100% consistency between code and documentation.

## Consequences

### Positive

- Documentation is **always accurate** - generated from tested code
- Screenshots **always match** current UI - captured automatically
- **No maintenance burden** - docs update automatically
- **Complete coverage** - validation enforces documentation
- **Single source of truth** - code and tests are canonical
- **Faster feature delivery** - no separate doc writing step
- **CI catches gaps** - missing docs fail the build

### Negative

- Initial setup complexity (one-time cost)
- CI pipeline runs longer (~10 min for full docs build)
- Less flexibility in prose (template-driven)
- Developers must write good docstrings and test descriptions
- Screenshot quality depends on test design

### Mitigations

- Cache Slicer installation in CI for faster runs
- Run docs job only on main, not on every PR
- Provide templates for common documentation patterns
- Claude Code skills assist with test/docstring writing

## Alternatives Considered

### Hybrid manual + generated

**Rejected**: Creates maintenance burden for manual portions, leads to inconsistency.

### Wiki-based documentation

**Rejected**: Disconnected from code, no automation, no validation.

### Readme-only documentation

**Rejected**: Insufficient for comprehensive extension documentation.

### MkDocs instead of Sphinx

**Rejected**: Sphinx has better autodoc support for Python API documentation.

## Implementation

See ADR-008 for CI/CD integration and ADR-009 for the living documentation foundation.

### Files Created

| File | Purpose |
|------|---------|
| `.github/workflows/docs.yml` | Documentation CI workflow |
| `scripts/extract_screenshots_for_docs.py` | Screenshot extraction |
| `scripts/generate_api_docs.py` | API doc generation |
| `scripts/generate_algorithm_docs.py` | Algorithm page generation |
| `scripts/generate_ui_docs.py` | UI reference generation |
| `scripts/validate_docs.py` | Completeness validation |
| `TestCases/test_docs_*.py` | Documentation test suite |

### Claude Code Integration

Skills for documentation workflow:
- `/generate-docs` - Run full documentation generation
- `/review-docs` - Validate documentation completeness

Agents for documentation tasks:
- `documentation-generator` - Generates docs from code
- `docs-validator` - Validates completeness and accuracy

## References

- [ADR-008: CI/CD Pipeline](ADR-008-ci-cd-pipeline.md)
- [ADR-009: Living Documentation](ADR-009-living-documentation.md)
- [ADR-010: Testing Framework](ADR-010-testing-framework.md)
- [Sphinx Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Living Documentation (Gojko Adzic)](https://gojko.net/books/specification-by-example/)
