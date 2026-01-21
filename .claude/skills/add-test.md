# Skill: Add Test (TDD)

When adding a new feature, ALWAYS write the test first.

## Test File Location

`SegmentEditorAdaptiveBrush/Testing/Python/test_<feature>.py`

## Test Template

```python
"""Tests for <feature> functionality."""

import unittest
import numpy as np


class Test<Feature>(unittest.TestCase):
    """Tests for <feature> functionality."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_<specific_behavior>(self):
        """Test that <specific behavior> works correctly."""
        # Arrange
        expected = ...

        # Act
        result = ...

        # Assert
        self.assertEqual(result, expected)

    def test_<edge_case>(self):
        """Test handling of <edge case>."""
        # Arrange
        ...

        # Act & Assert
        with self.assertRaises(ValueError):
            ...


if __name__ == '__main__':
    unittest.main()
```

## TDD Workflow

1. **Write the test first**
   ```python
   def test_gmm_estimates_threshold_for_bimodal_image(self):
       """GMM should identify two components in bimodal intensity distribution."""
       # Create bimodal test image
       image = create_bimodal_image(mean1=100, mean2=200, std=10)
       seed_point = (50, 50, 0)  # In region with mean=100

       analyzer = IntensityAnalyzer()
       thresholds = analyzer.analyze(image, seed_point)

       # Should estimate thresholds around the seed region
       self.assertGreater(thresholds['lower'], 70)
       self.assertLess(thresholds['upper'], 130)
   ```

2. **Run the test (should fail)**
   ```bash
   python -m pytest test_intensity_analyzer.py -v
   ```

3. **Implement minimal code to pass**

4. **Run the test (should pass)**

5. **Refactor if needed**

6. **Commit both test and implementation**

## Running the New Test

```bash
# Run just the new test
python -m pytest test_<feature>.py -v

# Run with verbose output
python -m pytest test_<feature>.py -v -s
```

## Test Fixtures

Use fixtures from `conftest.py` for common test data:

```python
def test_with_synthetic_image(self, synthetic_ct_image):
    """Test using the synthetic CT image fixture."""
    result = process(synthetic_ct_image)
    ...
```
