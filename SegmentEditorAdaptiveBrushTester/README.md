# Adaptive Brush Tester

Testing framework for the Adaptive Brush segment editor effect.

## Quick Start

```bash
# 1. Configure Slicer path
cp .env.example .env
# Edit .env to set SLICER_PATH

# 2. Run tests (via Claude Code)
/run-slicer-tests           # Run all tests
/run-slicer-tests algorithm # Algorithm tests only
/run-slicer-tests ui        # UI tests only
```

## Manual Testing

After automated tests complete, Slicer stays open for manual testing.

### Recording Actions

1. Open the **Adaptive Brush Tester** module in Slicer
2. Click **Start Recording**
3. Paint with the Adaptive Brush - actions are captured automatically
4. Use the panel buttons:
   - **Take Screenshot** - capture current view
   - **Add Note** - record observations
   - **Mark Pass/Fail** - record test outcomes
5. Click **Stop Recording**

### What Gets Recorded

Paint events are automatically captured to `manual_actions.jsonl`:

```json
{
  "type": "paint",
  "timestamp": "2026-01-24T18:30:00",
  "algorithm": "watershed",
  "mode": "add",
  "radius_mm": 5.0,
  "edge_sensitivity": 50,
  "cursor_ijk": [128, 100, 90],
  "full_state": {...}
}
```

### Converting Recordings to Tests

Ask Claude to review recordings and generate test code:

> "Review the manual_actions.jsonl from the latest test run and generate a Python test case based on what I tested."

## Test Output

Test runs are saved to `test_runs/` (git-ignored):

```
test_runs/2026-01-24_143025_all/
├── metadata.json          # Run config, summary
├── results.json           # Test results
├── metrics.json           # Performance metrics
├── manual_actions.jsonl   # Recorded actions
├── screenshots/           # Captured images
└── logs/
    ├── test_run.log       # Test execution log
    └── slicer_session.log # Slicer log copy
```

## Writing Test Cases

Create a new file in `TestCases/`:

```python
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

@register_test(category="algorithm")
class TestMyFeature(TestCase):
    name = "my_feature"
    description = "Test description"

    def setup(self, ctx: TestContext):
        # Load data, set up scene
        import SampleData
        self.volume = SampleData.downloadSample("MRHead")

    def run(self, ctx: TestContext):
        # Perform test actions
        ctx.screenshot("001_before", "Before painting")
        # ... test logic ...
        ctx.screenshot("002_after", "After painting")

    def verify(self, ctx: TestContext):
        # Check results
        ctx.assert_true(some_condition, "Expected outcome")

    def teardown(self, ctx: TestContext):
        # Clean up (optional)
        pass
```

## Architecture

See `docs/adr/ADR-010-testing-framework.md` for design details.
