# Test Cases

This directory contains test cases for the AdaptiveBrush testing framework.

## Test Case Index

| Test | Category | Description |
|------|----------|-------------|
| test_workflow_basic | workflow | Basic workflow: load data, paint, verify |
| test_algorithm_watershed | algorithm | Watershed algorithm tests |
| test_ui_options_panel | ui | Options panel UI verification |

## Writing Test Cases

Test cases inherit from `TestCase` and use the `@register_test` decorator:

```python
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

@register_test(category="algorithm")
class TestAlgorithmWatershed(TestCase):
    name = "algorithm_watershed"
    description = "Test watershed algorithm on brain tissue"

    def setup(self, ctx: TestContext):
        # Call Slicer API directly
        import SampleData
        self.volume = SampleData.downloadSample("MRHead")

    def run(self, ctx: TestContext):
        ctx.screenshot("001_before", "Before painting")
        # Execute test actions
        ctx.screenshot("002_after", "After painting")

    def verify(self, ctx: TestContext):
        # Check results with assertions
        ctx.assert_greater(voxel_count, 100, "Should segment tissue")

    def teardown(self, ctx: TestContext):
        # Optional cleanup
        pass
```

## Test Categories

- **workflow**: End-to-end workflow tests
- **algorithm**: Algorithm-specific tests
- **ui**: UI and widget tests
- **performance**: Performance benchmarks
