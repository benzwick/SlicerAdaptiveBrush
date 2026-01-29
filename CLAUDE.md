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

- **SegmentEditorEffect**: Main effect class inheriting from `AbstractScriptedSegmentEditorEffect`. Contains UI creation, mouse event processing, and all algorithm implementations (`_geodesicDistance`, `_watershed`, `_randomWalker`, `_levelSet`, `_connectedThreshold`, `_regionGrowing`, `_thresholdBrush`).
- **BrushOutlinePipeline**: VTK 2D pipeline for brush outline visualization in slice views.
- **IntensityAnalyzer**: Automatic threshold estimation using Gaussian Mixture Model (GMM) with simple statistics fallback.
- **PerformanceCache**: Cache structure for drag operations (infrastructure ready, optimization pending).

### Algorithm Overview

The adaptive brush provides **multiple user-selectable algorithms**, all implemented using CPU-based SimpleITK:

**Available Algorithms:**
| Algorithm | Speed | Precision | Best For |
|-----------|-------|-----------|----------|
| Geodesic Distance | Fast | High | General use (default) |
| Watershed | Medium | High | Marker-based segmentation |
| Random Walker | Medium | High | Probabilistic diffusion |
| Level Set | Slow | Very High | Irregular boundaries |
| Connected Threshold | Very Fast | Low | Quick rough segmentation |
| Region Growing | Fast | Medium | Homogeneous regions |
| Threshold Brush | Very Fast | Variable | Simple threshold painting |

**Threshold Brush Auto-Methods:**
- Otsu, Huang, Triangle, Maximum Entropy, IsoData, Li
- Auto-detects whether to segment above or below threshold based on seed intensity

**Zone-Based Threshold System:**

The brush uses a zone-based threshold override that samples intensities from a configurable inner zone around the seed point, overriding the IntensityAnalyzer thresholds:

- `threshold_zone` - Inner radius as percentage of brush radius (0.0-1.0)
- `sampling_method` - How intensities are sampled:
  - `mean_std` - Weighted mean ± std_multiplier * std
  - `percentile` - Weighted percentiles (default 5th-95th)
  - `gaussian_weighting` - Gaussian falloff from seed
  - `histogram_peak` - Mode detection

**Shared Pipeline (all algorithms):**
1. ROI Extraction around cursor (1.2x brush radius margin)
2. Local seed calculation within ROI
3. Zone-based threshold computation (if enabled, overrides analyzer)
4. Intensity Analysis (GMM-based threshold estimation)
5. Algorithm-specific segmentation
6. Post-processing:
   - Binary fill holes (if `fill_holes=True`, default)
   - Morphological closing (if `closing_radius > 0`)
7. Inner zone inclusion (if "Guarantee inner zone" enabled; OFF by default)
8. Apply circular/spherical brush mask
9. Apply to segment via OR operation

### Reproducibility

Algorithm reproducibility for identical inputs:

| Algorithm | Deterministic | Notes |
|-----------|---------------|-------|
| Geodesic Distance | ✓ Yes | |
| Watershed | ✓ Yes | |
| Connected Threshold | ✓ Yes | |
| Region Growing | ✓ Yes | |
| Threshold Brush | ✓ Yes | |
| Level Set | ✓ Yes | |
| Random Walker | ✓ Yes | Uses scikit-image defaults |

**GMM Threshold Estimation:** Deterministic (uses `random_state=42` and seeded subsampling with `np.random.default_rng(42)`).

## Development Guidelines

### AI Assistant Rules

- **Never make unverified claims** - Do not assert behavior, features, or facts without first verifying them in the code or documentation. If uncertain, say so.
- **Read before writing** - Always read relevant files before making changes or claims about how they work.
- **Test claims** - If you claim something works a certain way, verify it by reading the implementation or running a test.

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

### Exception Handling Philosophy: Fail Fast

This codebase follows **fail-fast, test-driven development** with detailed logging:

1. **Errors should surface immediately** - Never hide or swallow errors
2. **Catch specific exceptions** - Never use bare `except:` or `except Exception:`
3. **All exceptions must be logged** - For human and LLM debugging
4. **Document why suppression is valid** - If catching is necessary, explain why
5. **Tests catch bugs early** - Not error suppression in production

#### Bad Practices (NEVER do these)

```python
# BAD - Swallows ALL errors silently
try:
    do_something()
except:
    pass

# BAD - Too broad, catches keyboard interrupts
try:
    do_something()
except Exception:
    pass

# BAD - Logs but continues as if nothing happened
try:
    do_something()
except Exception as e:
    logger.error(e)
    pass
```

#### Good Practices (DO these)

```python
# GOOD - Specific exception, documented reason, appropriate log level
try:
    self.viewWidget.scheduleRender()
except RuntimeError as e:
    # Widget deleted during cleanup - expected, non-critical
    logging.debug(f"Render skipped (widget deleted): {e}")

# GOOD - Specific exception, re-raise after logging
try:
    result = process_data()
except ValueError as e:
    logging.exception("Data processing failed")
    raise

# GOOD - Specific exception, valid fallback with documentation
try:
    from sklearn.mixture import GaussianMixture
    HAS_GMM = True
except ImportError:
    # sklearn is optional - fall back to simple statistics
    logging.info("sklearn not available, using simple threshold estimation")
    HAS_GMM = False
```

#### Valid Exception Handling Cases

Not all exception handling is bad. Valid cases include:

1. **Optional feature degradation** - Import errors for optional dependencies
2. **Widget lifecycle** - RuntimeError when widgets deleted during cleanup
3. **Post-processing failures** - SimpleITK filter failures where result is still valid
4. **User input validation** - Catching parse errors to show user-friendly messages

For each valid case, document:
- The specific exception type being caught
- Why this exception can occur (root cause)
- What happens if this exception occurs (consequence)
- Why catching is valid (rather than letting it propagate)

#### Exception Handling Decision Tree

```
Is this a programming error (bug)?
├─ Yes → Let it crash (don't catch)
└─ No → Can user/system recover?
         ├─ Yes → Catch, log, handle gracefully
         └─ No → Catch, log with context, re-raise or wrap

When catching:
├─ Can you name the specific exception?
│   ├─ Yes → Catch that specific type
│   └─ No → Research what exceptions can occur
└─ Is logging added?
    ├─ Yes → Good
    └─ No → Add logger.exception() call
```

#### Skills for Exception Handling

- **/audit-code-quality** - Find exception handling issues
- **/fix-bad-practices** - Systematically fix bad patterns
- **/autonomous-code-review** - Full automated review cycle

## Git Workflow

- **Never push** - Only the user pushes to remote. Claude creates commits but never pushes.
- **Never run ad-hoc code** - Create proper scripts instead of inline Python/bash. Add utilities to `scripts/` or update existing tools.
- **Commit when asked** - Don't commit automatically; wait for user to request it.
- **New scripts need permission approval** - After creating a new script, ask the user to approve adding it to `.claude/settings.local.json` permissions before running it.

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
- **ADR-006**: Iconography
- **ADR-007**: Dependency management
- **ADR-008**: CI/CD pipeline - *implemented*
- **ADR-009**: Living documentation - *implemented*
- **ADR-010**: Slicer testing framework architecture
- **ADR-011**: Smart optimization framework (Optuna integration) - *implemented*
- **ADR-012**: Results review module - *implemented*
- **ADR-013**: Segmentation recipes - *implemented*
- **ADR-014**: Enhanced testing with recipe replay
- **ADR-015**: Parameter wizard - *implemented*
- **ADR-020**: Auto-generated documentation system - *implemented*

## Parameter Optimization (Optuna)

The project includes an Optuna-based parameter optimization framework for tuning algorithm parameters against gold standard segmentations.

### Running Optimization

```bash
# Run optimization with a config file
Slicer --python-script scripts/run_optimization.py configs/tumor_optimization.yaml

# Check progress
python scripts/check_optimization_progress.py
```

### Gold Standard Limitations (IMPORTANT)

**The optimizer can only be as good as the gold standard.** Key limitations:

1. **Dice score is bounded by gold standard quality** - If the gold standard has errors, the optimizer will learn those errors. A Dice of 0.99 against a flawed gold standard does NOT mean good segmentation.

2. **Segmentations require expert review** - Never trust metrics alone. The output segmentations should be reviewed by:
   - Domain experts (radiologists, pathologists) for medical imaging
   - An AI model specifically trained for segmentation quality assessment
   - At minimum, visual inspection by someone familiar with the anatomy

3. **Gold standards should be expert-created** - Before optimization, ensure gold standards are created or validated by domain experts.

4. **High-scoring trials may exceed gold standard** - The gold_candidates/ directory captures trials with Dice > 0.999, which could indicate the optimizer found parameters producing segmentations that are actually *better* than the gold standard. These warrant careful expert review.

5. **Metrics can mislead** - A segmentation could have high Dice but miss clinically important regions, or include spurious regions that happen to overlap the gold standard.

### Output Files

```
optimization_results/<timestamp>_<name>/
├── config.yaml              # Copy of optimization config
├── lab_notebook.md          # Human-readable results summary
├── results.json             # Full results in JSON
├── parameter_importance.json # Parameter importance scores
├── optuna_study.db          # SQLite database for Optuna
├── screenshots/             # Screenshots per trial
├── segmentations/           # Segmentation outputs per trial
├── gold_candidates/         # Trials exceeding Dice threshold
└── logs/
    └── slicer_session.log   # Slicer log copy
```

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
/run-slicer-tests reviewer           # Run Reviewer module tests
/run-slicer-tests reviewer_unit      # Run Reviewer unit tests only
/run-slicer-tests reviewer_ui        # Run Reviewer UI tests only
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

### Review Output

Reviews are saved to `reviews/` (git-ignored):

```
reviews/
├── reports/
│   └── <timestamp>_<type>/
│       ├── report.json    # Machine-readable findings
│       └── report.md      # Human-readable summary
├── history/
│   └── issues.jsonl       # Issue tracking over time
└── baselines/
    └── <date>_baseline.json  # For comparison
```

**Issue severity levels:**
| Level | Criteria |
|-------|----------|
| critical | Security risk, data loss, patient safety |
| high | Hides bugs, blocks audit trail |
| medium | Style violation, maintainability |
| low | Minor inconsistency |

### Claude Code Skills

**Testing Skills:**
- **/run-slicer-tests**: Launch Slicer, run test suite, leave open for manual testing
- **/run-reviewer-tests**: Run Reviewer module UI tests specifically
- **/review-test-results**: Analyze test output with specialized agent
- **/add-test-case**: Create new test case from template

**Reviewer Testing Skills:**
- **/run-reviewer-tests**: Run all Reviewer module tests
- **/run-reviewer-tests reviewer_unit**: Unit tests (SequenceRecorder, ViewGroupManager, Bookmarks)
- **/run-reviewer-tests reviewer_ui**: UI tests (navigation, bookmarks, playback, visualization, rating, keyboard)
- **/run-reviewer-tests reviewer_integration**: Full workflow integration test

**Review Skills:**
- **/review-code-quality**: Analyze exception handling (fail-fast), logging, type hints, dead code
- **/review-documentation**: Verify ADRs, README, CLAUDE.md against implementation
- **/review-tests**: Find coverage gaps, stale tests, quality issues
- **/review-algorithms**: Verify algorithm implementations match documentation
- **/review-medical-compliance**: Check audit logging, input validation, error handling
- **/review-full-audit**: Run all reviews, generate aggregate report with action items

**Fix Skills:**
- **/fix-bad-practices**: Systematically fix exception handling and other bad patterns

**Documentation Skills:**
- **/generate-docs**: Generate documentation from code and test screenshots
- **/review-docs**: Validate documentation completeness and report coverage gaps

### Claude Code Agents

**Testing Agents:**
- **test-reviewer**: Reviews results, suggests improvements
- **bug-fixer**: Analyzes failures, proposes fixes
- **algorithm-improver**: Reviews metrics, suggests optimizations
- **ui-improver**: Reviews screenshots, suggests UI improvements

**Reviewer Module Agents:**
- **reviewer-bug-fixer**: Fixes bugs in Reviewer module (specializes in common patterns)
- **reviewer-test-analyst**: Analyzes Reviewer test screenshots and results

**Review Agents:**
- **code-quality-reviewer**: Deep code analysis for quality issues
- **documentation-auditor**: Cross-references docs with implementation
- **medical-compliance-reviewer**: Medical imaging software best practices

**Documentation Agents:**
- **documentation-generator**: Generates docs from code, screenshots, and docstrings
- **docs-validator**: Validates documentation completeness and accuracy

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

See `docs/adr/ADR-010-testing-framework.md` for architecture details.
See `SegmentEditorAdaptiveBrushTester/README.md` for manual testing workflow.

## Reviewer Module Testing

The **SegmentEditorAdaptiveBrushReviewer** module has comprehensive Playwright-style UI tests.

### Reviewer Test Structure

```
SegmentEditorAdaptiveBrushTester/TestCases/
├── fixtures/
│   ├── mock_optimization_run.py    # Create fake optimization runs
│   └── mock_segmentations.py       # Create test segmentation nodes
├── test_reviewer_unit_sequence.py      # SequenceRecorder class
├── test_reviewer_unit_viewgroup.py     # ViewGroupManager class
├── test_reviewer_unit_bookmarks.py     # SceneViewBookmarks class
├── test_reviewer_ui_slice_navigation.py # Slice slider, buttons, linking
├── test_reviewer_ui_bookmarks.py       # Bookmark add/restore/delete
├── test_reviewer_ui_workflow_playback.py # Recording start/stop/step
├── test_reviewer_ui_visualization.py   # Layout, view modes, toggles
├── test_reviewer_ui_rating.py          # Rating buttons, save, export
├── test_reviewer_ui_keyboard.py        # All keyboard shortcuts
└── test_reviewer_integration.py        # Full workflow test
```

### Running Reviewer Tests

```bash
# Run all reviewer tests
"$SLICER_PATH" --python-script scripts/run_tests.py reviewer

# Run unit tests only (fastest)
"$SLICER_PATH" --python-script scripts/run_tests.py --exit reviewer_unit

# Run UI tests only
"$SLICER_PATH" --python-script scripts/run_tests.py --exit reviewer_ui

# Run integration test only
"$SLICER_PATH" --python-script scripts/run_tests.py --exit reviewer_integration
```

### Reviewer Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| `reviewer_unit` | 3 | Core class unit tests |
| `reviewer_ui` | 6 | UI widget interaction tests |
| `reviewer_integration` | 1 | Full 9-phase workflow test |

### Test Fixtures

**MockOptimizationRunFactory**: Creates fake optimization run directories
```python
factory = MockOptimizationRunFactory()
run_path = factory.create_run("test_run", num_trials=5)
# Creates: results.json, config.yaml, parameter_importance.json
factory.cleanup()
```

**MockSegmentationFactory**: Creates test segmentation nodes
```python
factory = MockSegmentationFactory()
seg_node = factory.create_sphere_segmentation(center=(128,128,64), radius=20, volume_node=vol)
gold_node, test_node = factory.create_gold_test_pair(volume_node, overlap_ratio=0.8)
factory.cleanup()
```

### Expected Bugs to Find

The tests are designed to catch these common issues:

1. **`'list' object has no attribute 'get'`** in `_on_run_selected`
   - Cause: ResultsLoader returns list for test format runs

2. **Bidirectional sync race conditions**
   - Cause: Slider updates triggering callbacks during programmatic changes

3. **Bookmark restoration with changed scene**
   - Cause: SceneView references deleted nodes

4. **Metrics with empty segments**
   - Cause: Division by zero or empty labelmap

5. **Recording without loaded segmentation**
   - Cause: Start recording clicked before loading data

6. **Keyboard shortcuts when dialogs open**
   - Cause: Shortcuts fire even during modal dialogs

### Reviewer Bug Fix Workflow

1. Run tests:
   ```bash
   /run-reviewer-tests --exit reviewer_unit
   ```

2. Review failures in `test_runs/<timestamp>/results.json`

3. Check screenshots for visual context

4. Use `reviewer-bug-fixer` agent to fix identified issues

5. Re-run tests to verify fix

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

## Local Resources

Paths configured in `.env` (see `.env.example` for setup):

- **SLICER_PATH**: Path to Slicer executable for running scripts
- **SLICER_SOURCE**: Path to Slicer source code for reference

### Slicer Source Code

When you need to understand Slicer internals, check the source at `$SLICER_SOURCE`:

| Path | Contents |
|------|----------|
| `Base/Python/slicer/` | Python API (`slicer` module implementation) |
| `Modules/Scripted/` | Built-in scripted modules (good examples) |
| `Modules/Scripted/SegmentEditor/` | Segment Editor module |
| `Modules/Loadable/Segmentations/` | Segmentation infrastructure |
| `Libs/MRML/Core/` | MRML node classes (C++) |
| `Libs/vtkSegmentationCore/` | Segmentation core (C++) |

**To verify claims about Slicer behavior**, read the source code rather than guessing.

## References

- [3D Slicer Developer Guide](https://slicer.readthedocs.io/en/latest/developer_guide/)
- [Segment Editor Documentation](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
- [SlicerSegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects) - Example effects
- [ITK-SNAP Adaptive Brush](https://github.com/pyushkevich/itksnap) - Reference implementation
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
- [Slicer Source Code](https://github.com/Slicer/Slicer) - Local clone at `$SLICER_SOURCE`
