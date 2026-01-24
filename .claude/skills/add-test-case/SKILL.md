# add-test-case

Add a new test case to the SegmentEditorAdaptiveBrushTester module.

## Usage

```
/add-test-case <name> <category> <description>
```

Arguments:
- `name` - Test case name (e.g., "algorithm_level_set")
- `category` - Test category: "algorithm", "ui", "workflow", "performance"
- `description` - Brief description of what the test verifies

## What This Skill Does

1. Validates the test name is unique
2. Creates a new test case file from template
3. Registers the test in TestCases/__init__.py
4. Opens the file for editing

## Execution Steps

### Step 1: Validate Name

Check that the test name doesn't already exist:

```bash
ls SegmentEditorAdaptiveBrushTester/TestCases/test_*.py
```

### Step 2: Create Test File

Create a new file at `SegmentEditorAdaptiveBrushTester/TestCases/test_<name>.py`:

```python
"""<Description>"""

from __future__ import annotations

import logging

import slicer

from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


@register_test(category="<category>")
class Test<ClassName>(TestCase):
    """<Description>"""

    name = "<name>"
    description = "<description>"

    def __init__(self) -> None:
        super().__init__()
        # TODO: Add instance variables

    def setup(self, ctx: TestContext) -> None:
        """Set up test environment."""
        logger.info("Setting up <name> test")

        # TODO: Load data, create segmentation, etc.
        # Call Slicer API directly

        ctx.screenshot("001_setup", "Initial state")

    def run(self, ctx: TestContext) -> None:
        """Execute test actions."""
        logger.info("Running <name> test")

        # TODO: Perform test actions

        ctx.screenshot("002_after_action", "After test action")

    def verify(self, ctx: TestContext) -> None:
        """Verify test results."""
        logger.info("Verifying <name> test")

        # TODO: Add assertions
        # ctx.assert_greater(value, threshold, "Description")

        ctx.screenshot("003_verified", "Verification complete")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down <name> test")
        ctx.log("Teardown complete")
```

### Step 3: Register Test

Add import to `SegmentEditorAdaptiveBrushTester/TestCases/__init__.py`:

```python
from . import test_<name>
```

### Step 4: Update CMakeLists.txt

Add the new file to `SegmentEditorAdaptiveBrushTester/CMakeLists.txt`:

```cmake
set(MODULE_PYTHON_SCRIPTS
  ...
  TestCases/test_<name>.py
  )
```

## Template Variables

- `<name>` - Snake case test name (e.g., "algorithm_level_set")
- `<ClassName>` - Pascal case class name (e.g., "AlgorithmLevelSet")
- `<category>` - Test category
- `<description>` - Test description

## Test Case Patterns

### Algorithm Test

```python
def setup(self, ctx: TestContext):
    import SampleData
    self.volume = SampleData.downloadSample("MRHead")
    # Set up segmentation and effect

def run(self, ctx: TestContext):
    # Configure algorithm parameters
    # Simulate painting
    with ctx.timing("operation"):
        self._paint_at_ijk(128, 100, 90)

def verify(self, ctx: TestContext):
    ctx.assert_greater(voxel_count, 100, "Should segment tissue")
```

### UI Test

```python
def run(self, ctx: TestContext):
    # Switch between algorithms
    # Capture screenshots of panel state
    ctx.screenshot("panel_state", "Options panel")

def verify(self, ctx: TestContext):
    # Verify widget visibility
    ctx.assert_true(widget.isVisible(), "Widget should be visible")
```

### Performance Test

```python
def run(self, ctx: TestContext):
    for _ in range(10):
        with ctx.timing("operation"):
            self._perform_operation()

def verify(self, ctx: TestContext):
    timing = ctx.metrics.get("timings", [])
    avg_ms = sum(t["duration_ms"] for t in timing) / len(timing)
    ctx.assert_less(avg_ms, 50, "Average should be < 50ms")
```

## Verification

After creating the test, verify it's registered:

```python
# In Slicer Python console
from SegmentEditorAdaptiveBrushTesterLib import TestRegistry
print([t.name for t in TestRegistry.list_tests()])
```
