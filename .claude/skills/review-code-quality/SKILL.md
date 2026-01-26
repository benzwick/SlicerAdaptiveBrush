# review-code-quality

Analyzes codebase for code quality issues following **fail-fast principles**.

## Philosophy: Fail Fast

This codebase follows **fail-fast, test-driven development** with detailed logging:

1. **Errors should surface immediately** - Never hide or swallow errors
2. **Catch specific exceptions** - Never use bare `except:` or `except Exception:`
3. **All exceptions must be logged** - For human and LLM debugging
4. **Document why suppression is valid** - If catching is necessary, explain why
5. **Tests catch bugs early** - Not error suppression in production

## Usage

```
/review-code-quality [path]
```

Where `path` is:
- Path to a specific file or directory to analyze
- Or omit to analyze the entire codebase

## What This Skill Does

1. Scans Python files for fail-fast violations
2. Categorizes findings by severity (critical first)
3. Generates machine-readable and human-readable reports
4. Tracks issues over time for progress monitoring

## Focus Areas (Priority Order)

### 1. Exception Swallowing (CRITICAL)

```python
# BAD - Swallows ALL errors silently
try:
    do_something()
except:
    pass

# BAD - Catches everything, continues as if nothing happened
try:
    do_something()
except Exception as e:
    logger.error(e)
    pass
```

**Detection:**
```bash
# Bare except:pass
grep -rn "except.*:$" --include="*.py" -A1 | grep -B1 "pass$"

# except Exception with pass
grep -rn "except.*Exception" --include="*.py" -A2 | grep -B2 "pass$"
```

### 2. Overly Broad Exception Handling (HIGH)

```python
# BAD - Too broad, catches SystemExit, KeyboardInterrupt
try:
    do_something()
except Exception:
    handle_error()

# BAD - Catches literally everything
try:
    do_something()
except:
    handle_error()
```

**Detection:**
```bash
grep -rn "except Exception\b" --include="*.py"
grep -rn "except:$" --include="*.py"
```

### 3. Missing Exception Logging (HIGH)

```python
# BAD - Handles error but doesn't log
try:
    do_something()
except SpecificError:
    return default_value  # No log!
```

### 4. Silent Continuation (MEDIUM)

```python
# BAD - Silently skips errors
for item in items:
    if not item.is_valid():
        continue  # Error hidden!
    process(item)

# BAD - Returns None on error without indication
def get_data():
    if error_condition:
        return None  # Caller can't distinguish from valid None
```

### 5. Logging Gaps (MEDIUM)

- Functions without any logging
- Print statements that should be logging calls
- Missing debug/info log points

### 6. Type Hints (LOW)

- Public functions missing type hints
- Incomplete return type annotations

## Valid Exception Handling

Not all exception handling is bad. Valid cases include:

1. **Optional imports** - ImportError for optional dependencies
2. **Widget lifecycle** - RuntimeError when widgets deleted during cleanup
3. **Post-processing** - SimpleITK filter failures where result is still valid
4. **User input** - Parse errors to show user-friendly messages

**Each valid case must document:**
- The specific exception type being caught
- Why this exception can occur
- What happens if this exception occurs
- Why catching is valid

## Execution Steps

### Step 1: Find Critical Issues First

```bash
# CRITICAL: Exception swallowing
echo "=== CRITICAL: Exception Swallowing ==="
grep -rn "except.*:$" --include="*.py" -A1 | grep -B1 "^\s*pass$"

# HIGH: Broad exceptions
echo "=== HIGH: Broad Exceptions ==="
grep -rn "except Exception\b\|except:$" --include="*.py"
```

### Step 2: Analyze Each Finding

For each issue, check:
1. Is there a comment explaining why this is valid?
2. Is the exception type specific enough?
3. Is there logging at appropriate level?
4. What happens downstream if this exception occurs?

### Step 3: Generate Report

Create report in `reviews/reports/<timestamp>_code-quality/`:
- `report.json` - Machine-readable findings
- `report.md` - Human-readable summary

## Issue JSON Schema

```json
{
  "id": "EH-001",
  "severity": "critical",
  "category": "exception_swallowing",
  "file": "SegmentEditorEffect.py",
  "line": 3501,
  "pattern": "except Exception: pass",
  "description": "Swallows all errors silently",
  "suggestion": "Catch specific exception, add logging, document why",
  "context": "except Exception as e:\n    pass"
}
```

## Severity Levels

| Level | Criteria | Examples |
|-------|----------|----------|
| critical | Hides bugs completely | `except: pass`, `except Exception: pass` |
| high | May hide bugs | `except Exception:`, missing logging in catch |
| medium | Maintainability | Silent returns, missing type hints |
| low | Minor style | Minor inconsistency |

## Category Prefixes

| Prefix | Category |
|--------|----------|
| ES | Exception Swallowing (critical) |
| BE | Broad Exception (high) |
| ML | Missing Logging (high) |
| SC | Silent Continuation (medium) |
| LG | Logging Gaps (medium) |
| TH | Type Hints (low) |

## Follow-up Actions

Based on the review:
- **Critical issues**: Run `/fix-bad-practices` immediately
- **High issues**: Schedule fixes before next release
- **Medium issues**: Add to technical debt backlog
- Run tests after fixes to verify no regressions
