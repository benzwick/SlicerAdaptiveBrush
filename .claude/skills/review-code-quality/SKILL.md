# review-code-quality

Analyzes codebase for code quality issues using the code-quality-reviewer agent.

## Usage

```
/review-code-quality [path]
```

Where `path` is:
- Path to a specific file or directory to analyze
- Or omit to analyze the entire codebase

## What This Skill Does

1. Scans Python files for common quality issues
2. Categorizes findings by severity
3. Generates machine-readable and human-readable reports
4. Tracks issues over time for progress monitoring

## Focus Areas

### Exception Handling
- Generic `except Exception:` blocks that hide bugs
- Swallowed exceptions (`except: pass`)
- Missing exception logging

### Logging Gaps
- Functions without any logging
- Print statements that should be logging calls
- Missing debug/info log points

### Dead Code
- Unused imports
- Unreachable code paths
- Commented-out code blocks

### Code Duplication
- Similar code blocks >10 lines
- Copy-pasted logic that should be refactored

### Type Hint Completeness
- Public functions missing type hints
- Incomplete return type annotations
- Missing Optional[] for nullable parameters

### Docstring Coverage
- Public classes without docstrings
- Public functions without docstrings
- Missing parameter/return documentation

## Execution Steps

### Step 1: Find Files to Analyze

```bash
# Find all Python files in the codebase
find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*"
```

### Step 2: Pattern Detection

Use Grep to find issues:

```bash
# Generic exceptions
grep -rn "except\s\+Exception" --include="*.py"
grep -rn "except.*:\s*pass" --include="*.py"

# Print statements
grep -rn "\bprint\s*(" --include="*.py"

# Missing type hints (functions without -> )
grep -rn "def\s\+\w\+([^)]*):$" --include="*.py"
```

### Step 3: Generate Report

Create report in `reviews/reports/<timestamp>_code-quality/`:
- `report.json` - Machine-readable findings
- `report.md` - Human-readable summary

### Step 4: Update Issue Tracking

Append new issues to `reviews/history/issues.jsonl`

## Issue JSON Schema

```json
{
  "id": "EH-001",
  "severity": "medium",
  "category": "exception_handling",
  "file": "SegmentEditorEffect.py",
  "line": 3501,
  "description": "Generic exception catch",
  "suggestion": "Use specific: RuntimeError, ValueError",
  "context": "except Exception as e:"
}
```

## Severity Levels

| Level | Criteria |
|-------|----------|
| critical | Security risk, data loss, patient safety |
| high | Hides bugs, blocks audit trail |
| medium | Style violation, maintainability |
| low | Minor inconsistency |

## Category Prefixes

| Prefix | Category |
|--------|----------|
| EH | Exception Handling |
| LG | Logging |
| DC | Dead Code |
| CD | Code Duplication |
| TH | Type Hints |
| DS | Docstrings |

## Output Example

```markdown
## Code Quality Review

**Date:** 2026-01-26T14:30:00
**Files Analyzed:** 15
**Issues Found:** 34

### Summary by Severity
- Critical: 0
- High: 5
- Medium: 22
- Low: 7

### Exception Handling (22 issues)
#### EH-001 [medium] SegmentEditorEffect.py:3501
Generic exception catch hides specific errors
```python
except Exception as e:
```
**Suggestion:** Catch specific exceptions: RuntimeError, ValueError, sitk.RuntimeError

...
```

## Follow-up Actions

Based on the review, you may want to:
- Fix high-severity issues immediately
- Create tasks for medium-severity issues
- Track progress with `/review-full-audit`
