# ADR-010: Slicer Testing Framework Architecture

## Status

Accepted

## Context

The project needs a comprehensive testing framework that:

1. Runs tests inside 3D Slicer where the effect actually operates
2. Captures screenshots for visual verification and documentation
3. Collects performance metrics to ensure targets are met
4. Enables AI-assisted review and improvement workflows via Claude Code
5. Records manual testing actions for reproducibility

### Challenges

- Slicer tests run in a different Python environment than development
- Visual verification requires screenshot capture and organization
- Performance regression detection needs consistent metric collection
- Manual testing insights are lost without recording mechanisms
- Test case generation should be systematic, not ad-hoc

### Previous Approach (ADR-008)

ADR-008 outlined CI/CD integration but focused on GitHub Actions. This ADR extends that work with a **local-first** testing framework that runs before any CI integration.

## Decision

Implement a **SegmentEditorAdaptiveBrushTester** Slicer module providing:

### Core Architecture

```
SegmentEditorAdaptiveBrushTester/
├── SegmentEditorAdaptiveBrushTester.py      # Module entry + interactive panel
├── SegmentEditorAdaptiveBrushTesterLib/
│   ├── TestRunner.py                        # Core test execution
│   ├── TestCase.py                          # Base test case class
│   ├── TestContext.py                       # Test-specific utilities
│   ├── TestRegistry.py                      # Registered test cases
│   ├── ScreenshotCapture.py                 # Screenshot utilities
│   ├── MetricsCollector.py                  # Performance/quality metrics
│   ├── TestRunFolder.py                     # Output organization
│   ├── ActionRecorder.py                    # Manual testing recorder
│   └── ReportGenerator.py                   # Generate review reports
└── TestCases/
    ├── test_algorithm_watershed.py
    ├── test_algorithm_threshold.py
    ├── test_ui_options_panel.py
    └── test_workflow_basic.py
```

### Design Principles

1. **No arbitrary code execution**: All functionality lives in documented Python modules. Skills invoke registered TestRunner methods, not generated code.

2. **Git-tracked tests**: All TestCase files are version-controlled. New tests require review.

3. **Direct Slicer API usage**: Tests call Slicer functions directly (no wrappers):
   ```python
   def setup(self, ctx: TestContext):
       import SampleData
       self.volume = SampleData.downloadSample("MRHead")
   ```

4. **TestContext for test-specific utilities only**:
   ```python
   class TestContext:
       output_folder: Path
       screenshot_capture: ScreenshotCapture
       metrics: MetricsCollector

       def screenshot(id: str, description: str) -> ScreenshotInfo
       def timing(operation: str) -> TimingContext
       def assert_greater(actual, expected, msg: str) -> Assertion
   ```

5. **Verbose logging**: Log enough that Claude can reconstruct what happened:
   ```python
   logging.info(f"paint_at_ijk: i={i}, j={j}, k={k}, radius_mm={radius_mm}")
   logging.debug(f"  algorithm={self.algorithm}, sensitivity={self.sensitivity}")
   ```

6. **No hidden errors**: Exceptions propagate unless there's a specific recovery action.

### Test Run Output Structure

```
test_runs/
└── 2026-01-24_143025_algorithms/
    ├── metadata.json          # Run config, summary
    ├── results.json           # Test results
    ├── metrics.json           # Performance metrics
    ├── manual_actions.jsonl   # Recorded manual testing actions
    ├── screenshots/
    │   ├── manifest.json      # Screenshot descriptions
    │   └── *.png              # Captured screenshots
    └── logs/
        ├── test_run.log       # Test execution log
        └── slicer_session.log # Copy of Slicer log
```

### Interactive Testing Panel

The module provides a Slicer panel for manual testing:

```
┌─────────────────────────────────────────┐
│ Adaptive Brush Tester                   │
├─────────────────────────────────────────┤
│ Test Run: 2026-01-24_143025             │
│ Status: ● Recording                     │
├─────────────────────────────────────────┤
│ [📷 Screenshot] [⏹ Stop Recording]      │
│ [📝 Add Note]   [✓ Mark Pass/Fail]      │
├─────────────────────────────────────────┤
│ Recent Actions:                         │
│ • Painted at (128, 100, 90) watershed   │
│ • Changed algorithm to level_set        │
│ • Screenshot: ui_after_change.png       │
└─────────────────────────────────────────┘
```

### Claude Code Integration

**Skills** invoke TestRunner methods:
- `run-slicer-tests`: Launch Slicer, run test suite, leave open for manual testing
- `review-test-results`: Analyze test run output with specialized agent
- `add-test-case`: Create new test case from template

**Agents** provide specialized analysis:
- `test-reviewer`: Reviews results, suggests improvements
- `bug-fixer`: Analyzes failures, proposes fixes
- `algorithm-improver`: Reviews metrics, suggests optimizations
- `ui-improver`: Reviews screenshots, suggests UI improvements

### Configuration

`.env` file (git-ignored) stores local configuration:
```bash
SLICER_PATH=/path/to/Slicer
```

`.env.example` (git-tracked) provides template.

## Consequences

### Positive

- **Local-first**: Fast iteration without waiting for CI
- **Visual verification**: Screenshots capture actual Slicer state
- **Performance tracking**: Metrics enable regression detection
- **Manual testing preserved**: ActionRecorder captures human testing
- **AI-assisted review**: Agents analyze output systematically
- **Reproducibility**: All tests are versioned Python code

### Negative

- Requires Slicer installation for testing
- Screenshot comparison is fragile (font rendering varies)
- Manual action recording adds complexity
- Multiple agents may produce redundant suggestions

### Trade-offs

| Aspect | Alternative | Chosen | Reason |
|--------|-------------|--------|--------|
| Test location | CI only | Local + CI | Faster iteration |
| Slicer API | Wrapper classes | Direct calls | Less indirection, clearer tests |
| Error handling | Catch all | Propagate | Easier debugging |

## Alternatives Considered

### Wrapper Classes for Slicer API

**Rejected**: Adds indirection without benefit. Tests should call Slicer directly so they match documentation and are easier to understand.

### GUI Record-and-Replay (QtTesting)

**Rejected**: QtTesting is experimental and brittle. Python scripts manipulating MRML nodes are more reliable.

### CI-Only Testing

**Rejected**: Too slow for development iteration. Local testing enables rapid TDD.

### Catch-All Error Handling

**Rejected**: Hides problems. Let exceptions propagate with full tracebacks for debugging.

## Implementation Notes

### Adding DEBUG Logging to Main Module

The main SegmentEditorEffect.py should have DEBUG logging added:
- Log algorithm execution with parameters
- Log timing for operations
- Log results and voxel counts

### Test Case Registration

Test cases register via decorators:
```python
@register_test(category="algorithm")
class TestAlgorithmWatershed(TestCase):
    name = "algorithm_watershed"
    description = "Test watershed algorithm on brain tissue"
```

### Slicer Log File Location

Copy Slicer session log from:
- Linux: `~/.config/Slicer/Slicer.log`
- macOS: `~/Library/Application Support/Slicer/Slicer.log`
- Windows: `%LOCALAPPDATA%/Slicer/Slicer.log`

## Phase 2: Optimization & Regression Testing

Phase 2 extends the framework with ground truth comparison, metrics computation, parameter optimization, and regression testing capabilities.

### New Components

```
SegmentEditorAdaptiveBrushTester/
├── SegmentEditorAdaptiveBrushTesterLib/
│   ├── ... (Phase 1 files)
│   ├── SegmentationMetrics.py       # Dice, Hausdorff computation
│   ├── GoldStandardManager.py       # Save/load gold standards
│   └── OptunaOptimizer.py           # Smart parameter optimization (see ADR-011)
├── GoldStandards/                    # Git-tracked gold files
│   ├── README.md
│   └── <name>/
│       ├── gold.seg.nrrd
│       ├── metadata.json
│       └── reference_screenshots/
├── LabNotebooks/                     # Documentation from optimization
│   └── *.md
└── TestCases/
    └── test_regression_gold.py       # Gold standard comparison
```

### Segmentation Metrics

`SegmentationMetrics` class computes:
- **Dice coefficient**: Overlap accuracy (0-1)
- **Hausdorff distance**: Maximum surface distance (mm)
- **Hausdorff 95%**: 95th percentile surface distance
- **Volume similarity**: Size comparison
- **False positive/negative rates**: Over/under-segmentation

```python
from SegmentEditorAdaptiveBrushTesterLib import SegmentationMetrics

metrics = SegmentationMetrics.compute(
    test_seg, test_id,
    gold_seg, gold_id,
    volume_node
)
print(f"Dice: {metrics.dice:.3f}")
print(f"Hausdorff 95%: {metrics.hausdorff_95:.1f}mm")
```

### Stroke Metrics Tracker

`StrokeMetricsTracker` monitors per-stroke improvement:

```python
tracker = StrokeMetricsTracker(gold_seg, gold_id, volume)

for stroke_params in strokes:
    # Apply stroke...
    record = tracker.record_stroke(test_seg, test_id, stroke_params)
    print(f"Stroke {record.stroke}: Dice={record.dice:.3f}")

summary = tracker.get_summary()
print(f"Strokes to 90%: {summary['strokes_to_90pct']}")
```

### Gold Standards

Gold standards provide ground truth for regression testing:

```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()

# Save
manager.save_as_gold(
    segmentation_node=seg,
    volume_node=vol,
    segment_id="Segment_1",
    name="MRBrainTumor1_tumor",
    click_locations=clicks,
    algorithm="watershed",
    parameters={"edge_sensitivity": 40}
)

# Load
gold_seg, metadata = manager.load_gold("MRBrainTumor1_tumor")
```

### Parameter Optimization

Parameter optimization is now handled by `OptunaOptimizer` (see ADR-011 for details):

```python
from SegmentEditorAdaptiveBrushTesterLib import OptunaOptimizer, OptimizationConfig

config = OptimizationConfig.load("configs/tumor_optimization.yaml")
optimizer = OptunaOptimizer(config, output_dir)
results = optimizer.optimize()

print(f"Best Dice: {results.best_trial.value:.3f}")
print(f"Best params: {results.best_trial.params}")
```

Key features:
- TPE (Tree-structured Parzen Estimator) for smart parameter suggestion
- HyperbandPruner for early stopping of poor trials
- FAnova for parameter importance analysis
- SQLite persistence for study resumption

### Lab Notebooks

The `LabNotebooks/` folder stores documentation from optimization runs.
These are generated manually or by optimization scripts to document
findings, parameter choices, and recommendations.

### Enhanced Debug Logging

SegmentEditorEffect.py now includes structured action logging:

```python
# Logged actions (to AdaptiveBrush.Actions logger)
{"action": "algorithm_changed", "params": {"old": "watershed", "new": "level_set_cpu"}, "state": {...}}
{"action": "radius_changed", "params": {"old": 5.0, "new": 25.0}, "state": {...}}
{"action": "paint_stroke", "params": {"xy": [512, 384], "ijk": [128, 100, 45], "ras": [...], "view": "Red"}, "state": {...}}
```

This enables:
- Automated test generation from manual sessions
- Debugging algorithm behavior
- Reproducing user-reported issues

### New Skills

- `create-gold-standard`: Save gold standard from current Slicer session
- `run-optimization`: Run parameter optimization loop
- `run-regression`: Test algorithms against gold standards

### New Agents

- `metrics-optimizer`: Analyze optimization trials, suggest parameters
- `gold-standard-curator`: Maintain gold standards, propose updates

### Regression Testing

Regression tests compare algorithm results against gold standards:

```python
# test_regression_gold.py
@register_test(category="regression")
class TestRegressionGold(TestCase):
    def run(self, ctx):
        for gold in manager.list_gold_standards():
            # Reproduce segmentation
            # Compare metrics
            # Flag regressions
```

Regression thresholds:
- **Dice:** >= 0.80
- **Hausdorff 95%:** <= 10.0mm

## References

- [ADR-008](ADR-008-ci-cd-pipeline.md): CI/CD Pipeline Strategy
- [ADR-003](ADR-003-testing-strategy.md): Testing Strategy
- [Slicer Testing Documentation](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html#testing)
