# review-tests

Analyzes test coverage and quality.

## Usage

```
/review-tests [path]
```

Where `path` is:
- Path to a specific test file or directory
- Or omit to analyze all tests

## What This Skill Does

1. Identifies untested code paths
2. Finds stale tests that don't match implementation
3. Evaluates test quality
4. Suggests new test cases

## Focus Areas

### Missing Coverage

Identify code without tests:
- Classes with no corresponding test file
- Public methods with no test cases
- Error handling paths not exercised

**Known gaps to check:**
- PerformanceCache.py
- BrushOutlinePipeline class
- EmbeddedWizardUI class
- Each algorithm's error paths

### Stale Tests

Tests that no longer match implementation:
- Tests for removed features
- Outdated mocks/fixtures
- Wrong expected values
- API changes not reflected

### Test Quality

Evaluate existing tests:
- Assertions are meaningful (not just `assertTrue(True)`)
- Tests are isolated (don't depend on order)
- Fixtures are appropriate
- Edge cases covered

### Test Suggestions

Propose new test cases for:
- Boundary conditions
- Error scenarios
- Integration points
- Performance regressions

## Execution Steps

### Step 1: Map Code to Tests

```bash
# Find all source files
find SegmentEditorAdaptiveBrush -name "*.py" -not -name "test_*" -not -path "*/Testing/*"

# Find all test files
find . -name "test_*.py"
```

### Step 2: Coverage Analysis

For each source file, check:
1. Corresponding test file exists
2. Public classes have test classes
3. Public methods have test methods

### Step 3: Quality Analysis

For each test file:
1. Read test implementations
2. Check assertion quality
3. Verify fixtures are current
4. Check for test isolation issues

### Step 4: Generate Report

Create report in `reviews/reports/<timestamp>_tests/`:
- `report.json` - Machine-readable findings
- `report.md` - Human-readable summary

## Coverage Categories

| Category | Description |
|----------|-------------|
| NO_TEST | Source file has no corresponding test |
| PARTIAL_COVERAGE | Some public methods untested |
| STALE_TEST | Test doesn't match current implementation |
| WEAK_ASSERTION | Test has poor assertion quality |
| NOT_ISOLATED | Test depends on external state |

## Output Example

```markdown
## Test Coverage Review

**Date:** 2026-01-26T14:30:00
**Source Files:** 8
**Test Files:** 4

### Summary
- Full Coverage: 2 files
- Partial Coverage: 3 files
- No Coverage: 3 files

### Missing Test Coverage

#### PerformanceCache.py
**Coverage:** 0%
**Untested:**
- `PerformanceCache.__init__`
- `PerformanceCache.get`
- `PerformanceCache.set`
- `PerformanceCache.invalidate`

**Suggested Test Cases:**
1. Test cache initialization with default values
2. Test cache hit returns stored value
3. Test cache miss returns None
4. Test invalidation clears specific keys
5. Test TTL expiration

#### BrushOutlinePipeline (in SegmentEditorEffect.py)
**Coverage:** 0%
**Untested:**
- `createBrushOutline`
- `updateBrushOutline`
- `removeBrushOutline`

**Suggested Test Cases:**
1. Test outline creation with valid view
2. Test outline updates position correctly
3. Test cleanup removes VTK actors

### Stale Tests

#### test_intensity_analyzer.py::test_gmm_fallback
- **Issue:** Tests sklearn fallback but sklearn is now always available
- **Suggestion:** Update to test actual fallback conditions

### Quality Issues

#### test_adaptive_algorithm.py::test_watershed_basic
- **Issue:** Only checks result is not None
- **Suggestion:** Add assertions for voxel count, spatial accuracy
```

## Follow-up Actions

Based on the review, you may want to:
- Add test files for uncovered modules
- Add test cases for uncovered methods
- Update stale tests to match implementation
- Strengthen weak assertions
