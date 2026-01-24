# CLAUDE.md - AI Development Instructions

## Project Overview

SlicerAdaptiveBrush is a 3D Slicer extension providing an adaptive brush segment editor effect. The brush automatically segments regions based on image intensity similarity, adapting to image features (edges, boundaries) rather than using a fixed geometric shape.

**License:** Apache 2.0
**Target Platform:** 3D Slicer 5.10+
**Development Approach:** Test-Driven Development (TDD)

## Build & Test Commands

```bash
# Local development with uv (outside Slicer)
uv sync --extra dev        # Install dev dependencies
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
    │   ├── SegmentEditorEffect.py         # Main effect + all algorithm implementations
    │   ├── IntensityAnalyzer.py           # GMM-based threshold estimation
    │   └── PerformanceCache.py            # Caching structure for drag operations
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

- **SegmentEditorEffect**: Main effect class inheriting from `AbstractScriptedSegmentEditorEffect`. Contains UI creation, mouse event processing, and all algorithm implementations (`_watershed`, `_levelSet`, `_connectedThreshold`, `_regionGrowing`, `_thresholdBrush`).
- **BrushOutlinePipeline**: VTK 2D pipeline for brush outline visualization in slice views.
- **IntensityAnalyzer**: Automatic threshold estimation using Gaussian Mixture Model (GMM) with simple statistics fallback.
- **PerformanceCache**: Cache structure for drag operations (infrastructure ready, optimization pending).

### Algorithm Overview

The adaptive brush provides **multiple user-selectable algorithms**, all implemented using CPU-based SimpleITK:

**Available Algorithms:**
| Algorithm | Speed | Precision | Best For |
|-----------|-------|-----------|----------|
| Watershed | Medium | High | General use (default) |
| Level Set | Slow | Very High | High precision needs |
| Connected Threshold | Very Fast | Low | Quick rough segmentation |
| Region Growing | Fast | Medium | Homogeneous regions |
| Threshold Brush | Very Fast | Variable | Simple threshold painting |

**Threshold Brush Auto-Methods:**
- Otsu, Huang, Triangle, Maximum Entropy, IsoData, Li
- Auto-detects whether to segment above or below threshold based on seed intensity

**Shared Pipeline (all algorithms):**
1. ROI Extraction around cursor (1.2x brush radius margin)
2. Intensity Analysis (GMM-based threshold estimation)
3. Algorithm-specific segmentation
4. Post-processing (apply circular/spherical brush mask)
5. Apply to segment via OR operation

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
- Use SimpleITK for all image processing (bundled with Slicer)
- ROI extraction limits computation to brush region
- Cache structure ready for optimization when needed
- GPU acceleration planned for Phase 2

**Current Performance (CPU):**
| Operation | Typical Time |
|-----------|-------------|
| 2D brush (10mm) | 30-100ms |
| 3D brush (10mm) | 100-500ms |

**Target Performance (with GPU - Phase 2):**
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
- Algorithm methods catch exceptions and fall back to simpler algorithms

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

- **ADR-001**: Algorithm selection (multiple user-selectable options)
- **ADR-002**: Implementation strategy (CPU foundation with GPU roadmap)
- **ADR-003**: Testing strategy
- **ADR-004**: Caching strategy for performance
- **ADR-005**: Mouse and keyboard controls
- **ADR-010**: Slicer testing framework architecture

## Slicer Testing Framework

The **SegmentEditorAdaptiveBrushTester** module provides comprehensive testing inside Slicer.

### Running Slicer Tests

```bash
# Configure Slicer path (copy .env.example to .env)
cp .env.example .env
# Edit .env to set SLICER_PATH

# Use the run-slicer-tests skill
/run-slicer-tests                    # Run all tests
/run-slicer-tests algorithms         # Run algorithm tests only
/run-slicer-tests ui                 # Run UI tests only
```

### Test Output

Test runs are saved to `test_runs/` (git-ignored):

```
test_runs/2026-01-24_143025_algorithms/
├── metadata.json          # Run config, summary
├── results.json           # Test results
├── metrics.json           # Performance metrics
├── manual_actions.jsonl   # Recorded manual testing
├── screenshots/
│   ├── manifest.json      # Screenshot descriptions
│   └── *.png              # Captured screenshots
└── logs/
    ├── test_run.log       # Test execution log
    └── slicer_session.log # Slicer log copy
```

### Claude Code Skills

- **/run-slicer-tests**: Launch Slicer, run test suite, leave open for manual testing
- **/review-test-results**: Analyze test output with specialized agent
- **/add-test-case**: Create new test case from template

### Claude Code Agents

- **test-reviewer**: Reviews results, suggests improvements
- **bug-fixer**: Analyzes failures, proposes fixes
- **algorithm-improver**: Reviews metrics, suggests optimizations
- **ui-improver**: Reviews screenshots, suggests UI improvements

### Writing Test Cases

Test cases inherit from `TestCase` and use Slicer API directly:

```python
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

@register_test(category="algorithm")
class TestAlgorithmWatershed(TestCase):
    name = "algorithm_watershed"
    description = "Test watershed algorithm on brain tissue"

    def setup(self, ctx: TestContext):
        import SampleData
        self.volume = SampleData.downloadSample("MRHead")
        self.segmentation = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )

    def run(self, ctx: TestContext):
        ctx.screenshot("001_before", "Before painting")
        with ctx.timing("watershed_stroke"):
            # Call effect methods directly
            self._paint_at_ijk(128, 100, 90)
        ctx.screenshot("002_after", "After painting")

    def verify(self, ctx: TestContext):
        voxel_count = self._count_voxels()
        ctx.assert_greater(voxel_count, 100, "Watershed should segment tissue")
```

See `docs/adr/ADR-010-testing-framework.md` for full architecture details.

## Dependencies

**Bundled with 3D Slicer 5.x (required):**
- SimpleITK - Image processing (all algorithms)
- VTK - Visualization and brush outline
- NumPy - Array operations
- Qt/PythonQt - UI framework

**Optional (for advanced features):**
- scikit-learn - GMM fitting (fallback to simple statistics without it)

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

## Slicer Mouse & Keyboard Shortcuts

Reference: https://slicer.readthedocs.io/en/latest/user_guide/user_interface.html#mouse-keyboard-shortcuts

### Slice Views (Conflicts to Avoid)

| Action | Operation |
|--------|-----------|
| Right-click + drag | Zoom |
| Middle-click + drag | Pan |
| Shift + left-click + drag | Pan |
| Ctrl + mouse wheel | Zoom |
| Shift + mouse wheel | Brush size (segment editor) |

### Safe for Effects

| Action | Status |
|--------|--------|
| Left-click | Captured by effects |
| Ctrl + left-click | Available |
| Middle-click | Can override |

## Manual Testing Requirements

All features MUST include documented manual testing steps. Include:

1. **Prerequisites** - What data to load, what state to set up
2. **Test steps** - Numbered actions to perform
3. **Expected results** - What should happen at each step
4. **Edge cases** - Boundary conditions to verify

### Example: Testing Erase Mode

**Prerequisites:**
1. Open 3D Slicer
2. Load sample data: `File > Download Sample Data > MRHead`
3. Open Segment Editor module
4. Create a new segment
5. Select Adaptive Brush effect
6. Paint some voxels with Add mode

**Test: Erase Mode via UI**
1. Click "Erase" radio button
2. Verify brush outline turns red/orange
3. Paint over existing segmentation
4. **Expected:** Voxels are removed from segment

**Test: Ctrl Modifier**
1. Set UI to "Add" mode
2. Hold Ctrl key
3. Verify brush outline turns red/orange while Ctrl held
4. Paint while holding Ctrl
5. **Expected:** Voxels are removed (mode inverted)
6. Release Ctrl, verify brush returns to yellow

**Test: Middle + Left-Click Modifier**
1. Start in "Add" mode
2. Hold middle button (scroll wheel button)
3. While holding middle, left-click and drag
4. Verify brush outline turns red/orange while painting
5. **Expected:** Voxels are erased (mode inverted by middle button)
6. Release middle button, verify brush returns to yellow

**Edge cases:**
- Mode does NOT change mid-stroke (locked at stroke start)
- All algorithms work in both add and erase modes
- Standard Slicer navigation (right-click zoom, middle-drag pan, shift+drag pan) still works
- Middle button pan works when not also left-clicking

## References

- [3D Slicer Developer Guide](https://slicer.readthedocs.io/en/latest/developer_guide/)
- [Segment Editor Documentation](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
- [SlicerSegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects) - Example effects
- [ITK-SNAP Adaptive Brush](https://github.com/pyushkevich/itksnap) - Reference implementation
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
