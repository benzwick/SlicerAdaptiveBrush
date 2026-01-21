# Skill: Run Tests

Run the test suite for SlicerAdaptiveBrush.

## In Slicer Python Console

```python
# Run all tests
import SegmentEditorAdaptiveBrush
SegmentEditorAdaptiveBrush.SegmentEditorAdaptiveBrushTest().runTest()

# Run with verbose output
slicer.util.runTests(module='SegmentEditorAdaptiveBrush')
```

## Standalone pytest (for algorithm tests)

```bash
cd SegmentEditorAdaptiveBrush/Testing/Python
python -m pytest -v

# Run specific test file
python -m pytest test_intensity_analyzer.py -v

# Run with coverage
python -m pytest --cov=SegmentEditorAdaptiveBrushLib -v
```

## Common Test Patterns

### Running a single test
```bash
python -m pytest test_intensity_analyzer.py::TestIntensityAnalyzer::test_gmm_fitting -v
```

### Running tests matching a pattern
```bash
python -m pytest -k "watershed" -v
```

### Running with debug output
```bash
python -m pytest -v -s  # -s shows print statements
```
