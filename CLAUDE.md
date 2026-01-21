# CLAUDE.md - AI Development Instructions

## Project Overview

SlicerAdaptiveBrush is a 3D Slicer extension providing an adaptive brush segment editor effect. The brush automatically segments regions based on image intensity similarity, adapting to image features (edges, boundaries) rather than using a fixed geometric shape.

**License:** Apache 2.0
**Target Platform:** 3D Slicer 5.10+
**Development Approach:** Test-Driven Development (TDD)

## Build & Test Commands

```bash
# Local development with uv (outside Slicer)
uv sync --dev              # Install dev dependencies
uv run pre-commit install  # Set up git hooks (first time only)
uv run pytest -v           # Run tests
uv run ruff check .        # Lint code
uv run ruff format .       # Format code
uv run mypy .              # Type check

# Run tests within Slicer Python console
import SegmentEditorAdaptiveBrush
SegmentEditorAdaptiveBrush.SegmentEditorAdaptiveBrushTest().runTest()

# Run with verbose output in Slicer
slicer.util.runTests(module='SegmentEditorAdaptiveBrush')
```

## Code Architecture

### Directory Structure

```
SlicerAdaptiveBrush/
├── CLAUDE.md                              # This file
├── ROADMAP.md                             # Development milestones
├── LICENSE                                # Apache 2.0
├── README.md                              # Project overview
├── CMakeLists.txt                         # Extension build config
├── SlicerAdaptiveBrush.png                # Extension icon
├── .claude/
│   ├── settings.local.json
│   └── skills/                            # Claude skill files
├── docs/
│   └── adr/                               # Architecture Decision Records
└── SegmentEditorAdaptiveBrush/
    ├── CMakeLists.txt
    ├── SegmentEditorAdaptiveBrush.py      # Module entry point
    ├── SegmentEditorAdaptiveBrushLib/
    │   ├── __init__.py
    │   ├── SegmentEditorEffect.py         # Main effect (AbstractScriptedSegmentEditorEffect)
    │   ├── AdaptiveBrushAlgorithm.py      # Core multi-stage algorithm
    │   ├── IntensityAnalyzer.py           # GMM-based threshold estimation
    │   └── PerformanceCache.py            # Caching for drag operations
    ├── Resources/
    │   └── Icons/
    │       └── SegmentEditorAdaptiveBrush.png
    └── Testing/
        └── Python/
            ├── conftest.py                # pytest fixtures
            ├── test_intensity_analyzer.py
            ├── test_adaptive_algorithm.py
            └── test_fixtures/
                └── synthetic_image.py     # Test data generators
```

### Key Classes

- **SegmentEditorEffect**: Main effect class inheriting from `AbstractScriptedSegmentEditorEffect`. Handles UI creation and mouse event processing.
- **AdaptiveBrushAlgorithm**: Multi-stage segmentation algorithm combining connected threshold, watershed, and optional geodesic active contours.
- **IntensityAnalyzer**: Automatic threshold estimation using Gaussian Mixture Model (GMM).
- **PerformanceCache**: Caches intermediate computations during drag operations for smooth painting.

### Algorithm Overview

The adaptive brush provides **multiple user-selectable algorithms** with both CPU and GPU backends:

**Available Algorithms:**
| Algorithm | Backend | Speed | Precision | Best For |
|-----------|---------|-------|-----------|----------|
| Watershed | CPU | Medium | High | General use (default) |
| Level Set | GPU | Fast | Very High | Users with GPU |
| Level Set | CPU | Slow | Very High | Precision without GPU |
| Connected Threshold | CPU | Very Fast | Low | Quick rough segmentation |
| Region Growing | CPU | Fast | Medium | Homogeneous regions |

**Shared Pipeline (all algorithms):**
1. ROI Extraction around cursor
2. Intensity Analysis (GMM-based threshold estimation)
3. Algorithm-specific segmentation
4. Post-processing (mask to brush radius)
5. Apply to segment

## Development Guidelines

### Test-Driven Development (TDD)

**Always follow the TDD cycle:**

1. Write failing test FIRST describing expected behavior
2. Run test to confirm it fails
3. Implement minimal code to pass the test
4. Run test to confirm it passes
5. Refactor while keeping tests green
6. Commit with both test and implementation

**Never commit code without corresponding tests.**

### Python Best Practices

- Type hints for all public functions
- Docstrings (Google style) for all public classes and functions
- Use `ruff` for linting and formatting (replaces flake8, black, isort)
- Maximum line length: 100 characters
- Use `logging` module, not print statements

### Segment Editor Effect Pattern

Effects inherit from `AbstractScriptedSegmentEditorEffect` and must implement:

```python
def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Adaptive Brush'
    scriptedEffect.perSegment = True

def clone(self)              # Return new instance
def icon(self)               # Return QIcon
def helpText(self)           # Return HTML help string
def setupOptionsFrame(self)  # Create UI widgets
def processInteractionEvents(self, callerInteractor, eventId, viewWidget)  # Handle mouse
```

### Performance Considerations

- Profile before optimizing
- Cache expensive computations during drag operations
- Use SimpleITK for CPU image processing (bundled with Slicer)
- Use OpenCL/CUDA for GPU acceleration
- Support both CPU and GPU backends with automatic fallback

**Performance Targets:**
| Operation | CPU Target | GPU Target |
|-----------|-----------|------------|
| 2D brush (10mm) | < 50ms | < 10ms |
| 3D brush (10mm) | < 200ms | < 50ms |
| Drag operation | < 30ms | < 10ms |

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors with `logging.error()` or `logging.exception()`
- Graceful degradation for optional features (e.g., sklearn GMM)

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding/updating tests
- `refactor`: Code refactoring (no functional change)
- `perf`: Performance improvement
- `chore`: Build, config, or tooling changes

**Examples:**

```
feat(algorithm): Add GMM-based intensity analysis

Implements automatic threshold estimation using Gaussian Mixture Model
to eliminate manual parameter tuning. Falls back to simple statistics
when sklearn is not available.

Tests: test_intensity_analyzer.py
```

## Architecture Decision Records

Key decisions are documented in `docs/adr/`:

- **ADR-001**: Algorithm selection (multi-stage hybrid)
- **ADR-002**: Python vs C++ implementation boundaries
- **ADR-003**: Testing strategy
- **ADR-004**: Caching strategy for performance

## Dependencies

**Bundled with 3D Slicer 5.x (required):**
- SimpleITK - Image processing
- VTK - Visualization and data structures
- NumPy - Array operations
- Qt/PythonQt - UI framework

**Optional (for advanced features):**
- scikit-learn - GMM fitting (fallback available without it)

## Slicer API Quick Reference

### Getting the Source Volume
```python
sourceVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
```

### Converting Coordinates
```python
# Screen (xy) to volume (ijk)
ijk = self.xyToIjk(xy, viewWidget)

# World (RAS) to volume (IJK)
rasToIjk = vtk.vtkMatrix4x4()
sourceVolumeNode.GetRASToIJKMatrix(rasToIjk)
```

### SimpleITK Integration
```python
import sitkUtils
sitkImage = sitkUtils.PullVolumeFromSlicer(volumeNode)
sitkUtils.PushVolumeToSlicer(sitkImage, volumeNode)
```

### Applying Mask to Segment
```python
modifierLabelmap = self.scriptedEffect.defaultModifierLabelmap()
# ... fill modifierLabelmap ...
self.scriptedEffect.modifySelectedSegmentByLabelmap(
    modifierLabelmap,
    slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
)
```

## References

- [3D Slicer Developer Guide](https://slicer.readthedocs.io/en/latest/developer_guide/)
- [Segment Editor Documentation](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
- [SlicerSegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects) - Example effects
- [ITK-SNAP Adaptive Brush](https://github.com/pyushkevich/itksnap) - Reference implementation
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
