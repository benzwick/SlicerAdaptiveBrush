# ADR-003: Testing Strategy

## Status

Accepted (Infrastructure Complete, Test Implementations Pending)

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Test infrastructure (pytest) | Complete | conftest.py, fixtures ready |
| Synthetic image generators | Complete | test_fixtures/synthetic_image.py |
| Test file scaffolding | Complete | All test files created |
| IntensityAnalyzer tests | Complete | 17 tests (GMM, thresholds, edge cases) |
| Algorithm tests | Complete | 10 tests (all algorithms + brush mask) |
| Cache tests | Complete | 7 tests (PerformanceCache + CacheStats) |
| Auto-threshold tests | Complete | 14 tests (all threshold methods) |
| Integration tests | Not started | Requires Slicer environment |

### Current Test Output

```
$ uv run pytest -v
51 tests collected
32 passed, 19 skipped (SimpleITK-dependent tests skip outside Slicer)
```

### Next Steps

1. Add Slicer integration tests (full effect testing with UI)
2. Add visual/manual test documentation

## Context

Need comprehensive testing for medical imaging software where correctness is critical. Must support both:
- Slicer-integrated tests (full effect testing with UI)
- Standalone tests (algorithm unit tests without Slicer)

## Decision

### Test Levels

1. **Unit Tests** - Algorithm components in isolation
   - IntensityAnalyzer GMM fitting
   - Watershed refinement
   - Connected threshold
   - Coordinate conversions
   - Caching logic

2. **Integration Tests** - Full pipeline with sample data
   - End-to-end brush application
   - Different imaging modalities (CT, MRI)
   - Various parameter combinations

3. **Visual/Manual Tests** - UI verification
   - Effect appears in Segment Editor
   - Sliders work correctly
   - Preview updates properly

### Test Frameworks

| Framework | Purpose | Location |
|-----------|---------|----------|
| unittest | Slicer integration tests | `Testing/Python/` |
| pytest | Standalone algorithm tests | `Testing/Python/` |
| numpy.testing | Numerical assertions | Used in both |

### Test Data Strategy

1. **Synthetic Images** - Known ground truth
   ```python
   def create_bimodal_image(size, mean1, mean2, std):
       """Create image with two distinct intensity regions."""
       image = np.zeros(size)
       image[:size[0]//2] = np.random.normal(mean1, std, ...)
       image[size[0]//2:] = np.random.normal(mean2, std, ...)
       return image
   ```

2. **Sample Medical Images** - Real-world testing
   - Use Slicer Sample Data (MRHead, CTChest)
   - Download on first test run

3. **Edge Case Images** - Robustness testing
   - Uniform intensity (no edges)
   - Very noisy images
   - Low contrast boundaries
   - Anisotropic voxels

### TDD Process

```
1. Write test describing expected behavior
   └── test_gmm_estimates_threshold_for_bimodal_image()

2. Run test (RED - should fail)
   └── pytest test_intensity_analyzer.py -v

3. Implement minimal code to pass
   └── IntensityAnalyzer.analyze()

4. Run test (GREEN - should pass)
   └── pytest test_intensity_analyzer.py -v

5. Refactor while tests pass
   └── Clean up, add docstrings

6. Commit both test and implementation
   └── git commit -m "feat: Add GMM intensity analysis"
```

### Test Coverage Targets

| Component | Target Coverage |
|-----------|----------------|
| IntensityAnalyzer | > 90% |
| SegmentEditorEffect (algorithms) | > 90% |
| PerformanceCache | > 85% |
| SegmentEditorEffect (UI/integration) | > 70% |
| Overall | > 80% |

## Consequences

### Positive

- **Confidence in correctness** through comprehensive testing
- **TDD prevents regressions** when refactoring
- **Synthetic data provides ground truth** for validation
- **Standalone tests run fast** without Slicer startup
- **Integration tests verify real usage** scenarios

### Negative

- **Test maintenance overhead** as code evolves
- **Synthetic data may not capture all real-world cases**
- **Slicer tests require full application** (slow startup)
- **Coverage metrics can be misleading** without quality tests

## Test File Organization

```
Testing/
└── Python/
    ├── conftest.py                    # pytest fixtures
    ├── test_intensity_analyzer.py     # Unit tests
    ├── test_adaptive_algorithm.py     # Unit tests
    ├── test_performance_cache.py      # Unit tests
    ├── test_segment_editor_effect.py  # Integration tests
    └── test_fixtures/
        └── synthetic_image.py         # Test data generators
```

## Example Test

```python
class TestIntensityAnalyzer(unittest.TestCase):
    def test_gmm_identifies_seed_component(self):
        """GMM should identify the component containing the seed point."""
        # Arrange
        image = create_bimodal_image(
            size=(100, 100, 1),
            mean1=100, mean2=200, std=10
        )
        seed_in_region1 = (25, 25, 0)

        # Act
        analyzer = IntensityAnalyzer()
        thresholds = analyzer.analyze(image, seed_in_region1)

        # Assert
        self.assertGreater(thresholds['lower'], 60)
        self.assertLess(thresholds['upper'], 140)
        self.assertAlmostEqual(thresholds['mean'], 100, delta=15)
```

## References

- [Slicer Testing Guide](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#run-tests)
- [pytest Documentation](https://docs.pytest.org/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
