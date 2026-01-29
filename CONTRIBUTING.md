# Contributing to SlicerAdaptiveBrush

Thank you for your interest in contributing! This document provides guidelines for development and contribution.

## Quick Links

- [Full Development Guide](CLAUDE.md) - Comprehensive development guidelines
- [Roadmap](ROADMAP.md) - Planned features and milestones
- [Architecture Decisions](docs/adr/) - Design decisions and rationale
- [Documentation Site](https://benzwick.github.io/SlicerAdaptiveBrush/) - User guide and API reference

## Getting Started

### Prerequisites

- [3D Slicer 5.10+](https://www.slicer.org/)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Git

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/benzwick/SlicerAdaptiveBrush.git
   cd SlicerAdaptiveBrush
   ```

2. **Install development dependencies:**
   ```bash
   uv sync --extra dev
   ```

3. **Set up pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

4. **Install in Slicer:**
   - Drag-and-drop the `SlicerAdaptiveBrush` folder onto Slicer
   - Or add to **Edit > Application Settings > Modules > Additional module paths**

### Running Tests

```bash
# Local tests (outside Slicer)
uv run pytest -v

# In Slicer Python console
import SegmentEditorAdaptiveBrush
SegmentEditorAdaptiveBrush.SegmentEditorAdaptiveBrushTest().runTest()

# Full test suite in Slicer
Slicer --python-script scripts/run_tests.py --exit
```

## Code Style

### Formatting and Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check .    # Check for issues
uv run ruff format .   # Format code
uv run mypy .          # Type checking
```

### Style Guidelines

- **Type hints** for all public functions
- **Docstrings** (Google style) for public classes and functions
- **Maximum line length:** 100 characters
- **Logging** instead of print statements

### Example Function

```python
def calculate_threshold(
    image: sitk.Image,
    seed_point: tuple[int, int, int],
    radius_mm: float,
) -> tuple[float, float]:
    """Calculate intensity thresholds for adaptive segmentation.

    Args:
        image: Input image in SimpleITK format.
        seed_point: Seed location in IJK coordinates.
        radius_mm: Brush radius in millimeters.

    Returns:
        Tuple of (lower_threshold, upper_threshold).

    Raises:
        ValueError: If seed_point is outside image bounds.
    """
    ...
```

## Development Workflow

### Test-Driven Development (TDD)

We follow TDD for all features:

1. **Write a failing test** describing expected behavior
2. **Run the test** to confirm it fails
3. **Implement minimal code** to pass the test
4. **Run the test** to confirm it passes
5. **Refactor** while keeping tests green
6. **Commit** with both test and implementation

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build, config, or tooling

**Example:**
```
feat(algorithm): Add GMM-based intensity analysis

Implements automatic threshold estimation using Gaussian Mixture Model
to eliminate manual parameter tuning. Falls back to simple statistics
when sklearn is not available.

Tests: test_intensity_analyzer.py
```

## Pull Request Guidelines

### Before Submitting

1. **Run all tests** and ensure they pass
2. **Run linting** (`uv run ruff check .`)
3. **Update documentation** if needed
4. **Add tests** for new functionality

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Docstrings added/updated
- [ ] CLAUDE.md updated if needed
- [ ] No unrelated changes included

### Keeping PRs Focused

- One feature/fix per PR
- Keep changes minimal and focused
- Avoid "drive-by" refactoring
- Don't add features beyond what was requested

## Reporting Issues

When reporting bugs:

1. **Check existing issues** first
2. **Include Slicer version** and platform
3. **Provide steps to reproduce**
4. **Include error messages** from the Python console
5. **Attach screenshots** if relevant

## Architecture

### Key Files

```
SegmentEditorAdaptiveBrush/
├── SegmentEditorAdaptiveBrush.py      # Module entry point
└── SegmentEditorAdaptiveBrushLib/
    ├── SegmentEditorEffect.py         # Main effect implementation
    ├── IntensityAnalyzer.py           # GMM-based threshold estimation
    └── PerformanceCache.py            # Caching for drag operations
```

### Adding a New Algorithm

1. Add algorithm to `AVAILABLE_ALGORITHMS` dict in `SegmentEditorEffect.py`
2. Implement `_yourAlgorithmName(self, ...)` method
3. Add case in `_runAlgorithm()` switch
4. Add tests in `Testing/Python/test_adaptive_algorithm.py`
5. Document in CLAUDE.md algorithm table

## Questions?

- Open a [GitHub Issue](https://github.com/benzwick/SlicerAdaptiveBrush/issues)
- Check the [Slicer Discourse](https://discourse.slicer.org/) for general Slicer questions

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
