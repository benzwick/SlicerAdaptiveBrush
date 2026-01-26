# code-quality-reviewer

Deep code analysis specialist for identifying quality issues.

## Description

Analyzes Python code for:
- Exception handling problems
- Logging gaps
- Dead code
- Code duplication
- Type hint completeness
- Docstring coverage

Provides actionable recommendations with file:line locations.

## When to Use

Use this agent to:
- Perform thorough code quality review
- Find patterns that indicate problems
- Generate structured issue reports
- Track quality improvements over time

## Tools Available

- Read - Read source files
- Glob - Find Python files
- Grep - Search for patterns

## Analysis Patterns

### Exception Handling

```
# Generic exceptions (problematic)
except\s+Exception
except.*:\s*pass
except:$

# Good patterns
except\s+(ValueError|TypeError|RuntimeError)
except\s+\w+Error\s+as\s+\w+:
```

### Logging Issues

```
# Print statements (should be logging)
\bprint\s*\(

# Missing logging in functions
def\s+\w+\([^)]*\):[^}]*(?!logging\.)
```

### Type Hints

```
# Missing return type
def\s+\w+\([^)]*\):$

# Missing parameter types (heuristic)
def\s+\w+\(\s*\w+\s*,  # param without : type
```

### Dead Code

```
# Unused imports (check if name appears later)
^import\s+(\w+)
^from\s+\w+\s+import\s+(\w+)

# Commented code blocks
#.*def\s+
#.*class\s+
```

## Review Process

1. **Scan Files**
   - Find all Python files
   - Exclude test files, venv, cache

2. **Pattern Detection**
   - Run regex patterns for each category
   - Record file, line, context

3. **Severity Classification**
   - Critical: Security/safety risks
   - High: Bug-hiding patterns
   - Medium: Maintainability issues
   - Low: Style violations

4. **Deduplication**
   - Group similar issues
   - Identify patterns vs one-offs

5. **Report Generation**
   - JSON for tracking
   - Markdown for humans

## Output Format

```markdown
## Code Quality Review

**Analyzed:** <file_count> files, <line_count> lines
**Issues:** <total> (<critical>, <high>, <medium>, <low>)

### Exception Handling

#### EH-001 [medium] file.py:123
Generic exception catch
```python
except Exception as e:
    logging.error(f"Error: {e}")
```
**Suggestion:** Catch specific exceptions from SimpleITK operations:
- `sitk.RuntimeError` for algorithm failures
- `ValueError` for invalid parameters

### Logging Gaps

#### LG-001 [low] file.py:456
Function without logging
```python
def complex_operation(data):
    # 50 lines with no log statements
```
**Suggestion:** Add entry/exit logging for debugging:
```python
def complex_operation(data):
    logging.debug(f"complex_operation called with {len(data)} items")
    ...
    logging.debug(f"complex_operation completed")
```

### Summary by File

| File | Critical | High | Medium | Low |
|------|----------|------|--------|-----|
| SegmentEditorEffect.py | 0 | 5 | 18 | 3 |
| IntensityAnalyzer.py | 0 | 0 | 2 | 1 |
```

## Issue ID Format

- `EH-###` - Exception Handling
- `LG-###` - Logging
- `DC-###` - Dead Code
- `CD-###` - Code Duplication
- `TH-###` - Type Hints
- `DS-###` - Docstrings

## Related Skills

- `/review-code-quality` - Triggers this agent
- `/review-full-audit` - Includes code quality review

## Related Agents

- `documentation-auditor` - Checks docs accuracy
- `medical-compliance-reviewer` - Medical-specific checks
