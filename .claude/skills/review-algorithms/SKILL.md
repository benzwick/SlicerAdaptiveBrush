# review-algorithms

Verifies algorithm implementations match documentation and claims.

## Usage

```
/review-algorithms [algorithm]
```

Where `algorithm` is:
- `watershed` - Review watershed algorithm only
- `level_set` - Review level set algorithm only
- `connected_threshold` - Review connected threshold only
- `region_growing` - Review region growing only
- `threshold_brush` - Review threshold brush only
- Or omit to review all algorithms

## What This Skill Does

1. Verifies algorithm descriptions match implementation
2. Checks performance claims have evidence
3. Validates metrics computation
4. Confirms default values are documented correctly

## Focus Areas

### Implementation Verification

For each algorithm:
- Code exists and matches described behavior
- Parameters match documentation
- Return values are correct
- Error handling is appropriate

### Performance Claims

Verify performance metrics:
- Claims have supporting benchmark data
- Metrics are from recent runs
- Hardware/conditions documented
- Variability noted

### Metrics Computation

Check metric calculations:
- Dice coefficient formula correct
- Hausdorff distance computed correctly
- Volume overlap calculated correctly
- Edge cases handled

### Default Values

Verify defaults:
- Documented defaults match code
- Defaults are reasonable for typical use
- Auto-tuning behavior documented

## Execution Steps

### Step 1: Read Algorithm Implementations

Find and read all algorithm methods in SegmentEditorEffect.py:
- `_watershed`
- `_levelSet`
- `_connectedThreshold`
- `_regionGrowing`
- `_thresholdBrush`

### Step 2: Compare with Documentation

For each algorithm:
1. Read CLAUDE.md description
2. Read any ADR decisions
3. Compare to actual implementation
4. Note discrepancies

### Step 3: Check Performance Data

Look for evidence of performance claims:
- Benchmark results in test_runs/
- Optimization results in optimization_results/
- Performance profiles in docs/

### Step 4: Verify Metrics

Read metric computation code:
- Find metric calculation functions
- Verify formulas are correct
- Check edge case handling

### Step 5: Generate Report

Create report in `reviews/reports/<timestamp>_algorithms/`:
- `report.json` - Machine-readable findings
- `report.md` - Human-readable summary

## Verification Categories

| Category | Description |
|----------|-------------|
| IMPL_MISMATCH | Implementation differs from description |
| PERF_UNVERIFIED | Performance claim lacks evidence |
| METRIC_ERROR | Metric computation may be incorrect |
| DEFAULT_MISMATCH | Default value doesn't match docs |
| PARAM_UNDOCUMENTED | Parameter not documented |

## Output Example

```markdown
## Algorithm Review

**Date:** 2026-01-26T14:30:00
**Algorithms Reviewed:** 5

### Summary
- Verified: 3
- Issues: 2

### Watershed Algorithm
**Status:** Verified

- Implementation matches description
- Default parameters documented correctly
- Performance: ~50ms (verified in optimization_results/)

### Level Set Algorithm
**Status:** Issues Found

#### Issue: PERF_UNVERIFIED
- **Claim:** "Very High" precision
- **Issue:** No quantitative precision metrics found
- **Suggestion:** Add benchmark comparing to ground truth

#### Issue: DEFAULT_MISMATCH
- **Documented:** propagationScaling=1.0
- **Actual:** propagationScaling=0.8
- **Location:** SegmentEditorEffect.py:2450

### Threshold Brush Auto-Methods

#### Issue: IMPL_MISMATCH
- **Documented:** "IsoData" method
- **Actual:** Implementation uses "Intermodes"
- **Location:** SegmentEditorEffect.py:3200

### Metrics Computation
**Status:** Verified

- Dice coefficient: Correct (2*|A∩B|/(|A|+|B|))
- Hausdorff distance: Correct (max min distance)
- Volume overlap: Correct
```

## Follow-up Actions

Based on the review, you may want to:
- Update documentation to match implementation
- Add benchmark tests for unverified claims
- Fix metric computation bugs
- Run optimization to get current performance data
