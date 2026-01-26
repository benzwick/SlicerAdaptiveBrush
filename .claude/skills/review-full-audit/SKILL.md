# review-full-audit

Orchestrates all review skills for comprehensive project audit.

## Usage

```
/review-full-audit [options]
```

Options:
- `--baseline` - Compare against previous baseline
- `--save-baseline` - Save results as new baseline
- `--quick` - Run only high-priority checks

## What This Skill Does

1. Runs all 5 specialized review skills
2. Aggregates findings into unified summary
3. Compares against baseline (optional)
4. Generates prioritized action items
5. Updates issue tracking

## Review Skills Executed

1. `/review-code-quality` - Exception handling, logging, type hints
2. `/review-documentation` - ADRs, README, CLAUDE.md accuracy
3. `/review-tests` - Coverage gaps, stale tests
4. `/review-algorithms` - Implementation verification
5. `/review-medical-compliance` - Audit logging, validation

## Execution Steps

### Step 1: Create Audit Directory

```bash
mkdir -p reviews/reports/$(date +%Y%m%d_%H%M%S)_full-audit
```

### Step 2: Run Individual Reviews

Execute each review skill and capture results:

1. Run code quality review
2. Run documentation review
3. Run tests review
4. Run algorithms review
5. Run medical compliance review

### Step 3: Aggregate Findings

Combine all findings into unified report:
- Total issues by severity
- Issues by category
- Cross-cutting concerns

### Step 4: Compare to Baseline (Optional)

If `--baseline` specified:
1. Load baseline from `reviews/baselines/`
2. Compare issue counts
3. Identify new issues
4. Identify resolved issues
5. Calculate delta

### Step 5: Generate Summary Report

Create files in `reviews/reports/<timestamp>_full-audit/`:
- `summary.json` - Machine-readable aggregate
- `summary.md` - Human-readable summary
- `action_items.md` - Prioritized task list
- Individual review reports linked

### Step 6: Update Issue Tracking

Append new/changed issues to `reviews/history/issues.jsonl`

### Step 7: Save Baseline (Optional)

If `--save-baseline` specified:
```bash
cp reviews/reports/<timestamp>_full-audit/summary.json \
   reviews/baselines/$(date +%Y%m%d)_baseline.json
```

## Report Structure

```
reviews/reports/<timestamp>_full-audit/
├── summary.json           # Aggregate machine-readable
├── summary.md             # Aggregate human-readable
├── action_items.md        # Prioritized tasks
├── code_quality.json      # Individual review
├── code_quality.md
├── documentation.json
├── documentation.md
├── tests.json
├── tests.md
├── algorithms.json
├── algorithms.md
├── medical_compliance.json
└── medical_compliance.md
```

## Output Example

```markdown
## Full Project Audit

**Date:** 2026-01-26T14:30:00
**Baseline:** 2026-01-20 (6 days ago)

### Executive Summary

| Category | Critical | High | Medium | Low | Total | Δ |
|----------|----------|------|--------|-----|-------|---|
| Code Quality | 0 | 5 | 22 | 7 | 34 | +3 |
| Documentation | 0 | 2 | 5 | 3 | 10 | -2 |
| Tests | 0 | 4 | 6 | 2 | 12 | +1 |
| Algorithms | 0 | 1 | 3 | 1 | 5 | 0 |
| Medical Compliance | 0 | 3 | 8 | 5 | 16 | +2 |
| **Total** | **0** | **15** | **44** | **18** | **77** | **+4** |

### Changes Since Baseline

**New Issues:** 8
- 3 new exception handling issues (new code)
- 2 new test coverage gaps
- 2 new logging gaps
- 1 new documentation mismatch

**Resolved Issues:** 4
- 2 type hint issues fixed
- 2 docstring issues fixed

### Top Priority Action Items

1. **[High] Fix 5 generic exception handlers**
   - SegmentEditorEffect.py: lines 3501, 3550, 3600, ...
   - Replace with specific exception types

2. **[High] Add audit logging to user actions**
   - Algorithm selection changes
   - Parameter modifications
   - Error conditions

3. **[High] Add tests for PerformanceCache**
   - Currently 0% coverage
   - Critical for drag performance

4. **[High] Fix validation for sigma parameter**
   - Add range checking
   - Could cause algorithm failure

5. **[Medium] Update ADR-012 status**
   - Currently "proposed" but implemented
   - Update to "accepted"

### Quick Wins (Low Effort, High Impact)

1. Add logging to `setParameter` calls (2 lines each)
2. Update ADR status fields (5 minutes)
3. Add basic PerformanceCache tests (30 minutes)

### Detailed Reports

- [Code Quality](code_quality.md)
- [Documentation](documentation.md)
- [Tests](tests.md)
- [Algorithms](algorithms.md)
- [Medical Compliance](medical_compliance.md)
```

## Issue Tracking

Issues are tracked in `reviews/history/issues.jsonl`:

```jsonl
{"id":"EH-001","first_seen":"2026-01-20","status":"open","severity":"medium",...}
{"id":"EH-001","timestamp":"2026-01-26","status":"resolved",...}
```

Query issue history:
```bash
# Count open issues
grep '"status":"open"' reviews/history/issues.jsonl | wc -l

# Find issues resolved this week
grep '"status":"resolved"' reviews/history/issues.jsonl | grep "2026-01-2"
```

## Follow-up Actions

Based on the audit, you may want to:
- Create tasks for high-priority issues
- Schedule medium-priority issues for next sprint
- Track progress with weekly audits
- Set up baseline comparison
