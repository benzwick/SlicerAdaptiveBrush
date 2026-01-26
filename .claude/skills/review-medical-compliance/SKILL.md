# review-medical-compliance

Checks medical software best practices using the medical-compliance-reviewer agent.

## Usage

```
/review-medical-compliance [area]
```

Where `area` is:
- `logging` - Review audit trail logging only
- `validation` - Review input validation only
- `errors` - Review error handling only
- `uncertainty` - Review uncertainty communication only
- Or omit to review all areas

## What This Skill Does

1. Checks for proper audit trail logging
2. Verifies input validation
3. Reviews error handling for safety
4. Evaluates uncertainty communication

## Context

Medical imaging software has special requirements:
- Actions must be auditable
- Errors must not cause silent failures
- User inputs must be validated
- Uncertainty should be communicated clearly

While this extension is not FDA-regulated, following best practices:
- Builds good habits for medical software development
- Reduces risk of errors in clinical use
- Improves overall code quality

## Focus Areas

### Audit Trail Logging

Check that user actions are logged:
- Parameter changes (algorithm, threshold, etc.)
- Painting actions (add, erase)
- Configuration changes
- Error conditions

### Input Validation

Check parameter validation:
- Range checking (min/max values)
- Type checking (int vs float)
- Bounds checking (brush radius > 0)
- Reasonable defaults

### Error Handling for Safety

Check error handling:
- No silent failures
- Errors reported to user
- Graceful degradation
- Recovery options

### Uncertainty Communication

Check uncertainty handling:
- Algorithm confidence not overstated
- Limitations documented
- User warned of edge cases
- Results marked as suggestions, not diagnoses

## Execution Steps

### Step 1: Audit Logging Analysis

Search for logging in user action handlers:

```bash
# Find event handlers
grep -rn "def.*Event\|def on\|def handle" --include="*.py"

# Check for logging in those functions
# Each should have logging.info/debug calls
```

### Step 2: Input Validation Analysis

Search for parameter setting and validation:

```bash
# Find parameter setters
grep -rn "setParameter\|self\.\w\+ =" --include="*.py"

# Check for validation
grep -rn "if.*<\|if.*>\|raise ValueError\|raise TypeError" --include="*.py"
```

### Step 3: Error Handling Analysis

Check error handling patterns:

```bash
# Find exception handlers
grep -rn "except\|try:" --include="*.py"

# Check for user notification
grep -rn "slicer.util.errorDisplay\|QMessageBox\|logging.error" --include="*.py"
```

### Step 4: Uncertainty Analysis

Review user-facing text:
- Help text
- Status messages
- Result descriptions

### Step 5: Generate Report

Create report in `reviews/reports/<timestamp>_medical-compliance/`:
- `report.json` - Machine-readable findings
- `report.md` - Human-readable summary

## Compliance Categories

| Category | Description |
|----------|-------------|
| AUDIT_MISSING | User action not logged |
| VALIDATION_MISSING | Parameter not validated |
| SILENT_FAILURE | Error swallowed without notification |
| UNCERTAINTY_UNCLEAR | Results presented without limitations |
| RECOVERY_MISSING | No recovery option for error |

## Severity for Medical Context

| Level | Criteria |
|-------|----------|
| critical | Could affect patient care decisions |
| high | Could hide errors affecting results |
| medium | Reduces auditability |
| low | Best practice not followed |

## Output Example

```markdown
## Medical Compliance Review

**Date:** 2026-01-26T14:30:00
**Focus:** Medical imaging software best practices

### Summary
- Critical: 0
- High: 3
- Medium: 8
- Low: 5

### Audit Trail Logging

#### AUDIT_MISSING [high] SegmentEditorEffect.py:1250
- **Action:** Algorithm change by user
- **Issue:** No logging when user changes algorithm selection
- **Impact:** Cannot audit what algorithm was used
- **Suggestion:** Add `logging.info(f"Algorithm changed to {algorithm}")`

#### AUDIT_MISSING [medium] SegmentEditorEffect.py:890
- **Action:** Brush size change
- **Issue:** Brush size changes not logged
- **Suggestion:** Log parameter changes at debug level

### Input Validation

#### VALIDATION_MISSING [high] SegmentEditorEffect.py:1500
- **Parameter:** sigma value for level set
- **Issue:** No range validation, could cause algorithm failure
- **Suggestion:** Add validation: `if sigma <= 0: raise ValueError("sigma must be positive")`

### Error Handling

#### SILENT_FAILURE [high] SegmentEditorEffect.py:3501
- **Context:** Generic exception in algorithm
- **Issue:** Error is logged but processing continues silently
- **Impact:** User may not know segmentation failed
- **Suggestion:** Show user error dialog or clear visual indication

### Uncertainty Communication

#### UNCERTAINTY_UNCLEAR [medium] Help text
- **Text:** "Segments based on image intensity"
- **Issue:** No mention of limitations or typical accuracy
- **Suggestion:** Add note about algorithm suitability for different tissue types
```

## Follow-up Actions

Based on the review, you may want to:
- Add logging to user action handlers
- Add parameter validation
- Improve error notifications
- Add uncertainty notes to help text
