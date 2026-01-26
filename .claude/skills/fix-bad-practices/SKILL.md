# fix-bad-practices

Systematically fix bad coding practices following **fail-fast principles**.

## When to Use

Use this skill:
- After running `/review-code-quality`
- When fixing specific bad patterns
- During refactoring for code quality

## Philosophy: Fail Fast Fixes

When fixing bad practices:

1. **Make errors visible** - Replace silent failures with loud ones
2. **Be specific** - Catch only exceptions you can handle
3. **Always log** - Enable debugging by humans and LLMs
4. **Document why** - Explain why catching is valid (if it is)
5. **Test after fixing** - Ensure fixes don't break functionality

---

## Fix Patterns

### Fix 1: Bare `except: pass` → Specific Exception + Logging

**Before (BAD)**:
```python
try:
    do_something()
except:
    pass
```

**After (GOOD) - If error should propagate:**
```python
do_something()  # Let it fail - no try/except needed
```

**After (GOOD) - If suppression is valid:**
```python
try:
    do_something()
except SpecificError as e:
    # Document why catching is valid:
    # - When this can occur (e.g., "widget deleted during cleanup")
    # - What happens if it fails (e.g., "render skipped, no functional impact")
    logging.debug(f"do_something skipped (widget deleted): {e}")
```

---

### Fix 2: `except Exception` → Specific Exceptions

**Before (BAD)**:
```python
try:
    data = parse_json(text)
except Exception as e:
    logger.error(e)
    return None
```

**After (GOOD)**:
```python
try:
    data = parse_json(text)
except json.JSONDecodeError as e:
    logger.exception("Invalid JSON")
    raise ValueError(f"Cannot parse JSON: {e}") from e
except FileNotFoundError as e:
    logger.exception("File not found")
    raise
```

---

### Fix 3: Silent `continue` → Explicit Error Handling

**Before (BAD)**:
```python
for item in items:
    if not item.is_valid():
        continue
    process(item)
```

**After (GOOD) - Option A: Fail on first error:**
```python
for item in items:
    if not item.is_valid():
        raise ValueError(f"Invalid item: {item}")
    process(item)
```

**After (GOOD) - Option B: Log warning if skip is intentional:**
```python
for item in items:
    if not item.is_valid():
        logger.warning("Skipping invalid item: %s (reason: %s)", item, item.validation_error)
        continue
    process(item)
```

---

### Fix 4: `return None` on Error → Raise Exception

**Before (BAD)**:
```python
def load_config(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
```

**After (GOOD)**:
```python
def load_config(path: Path) -> dict:
    """Load configuration from path.

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config is not valid JSON
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        return json.load(f)
```

---

### Fix 5: Post-Processing Failures (Valid Suppression Pattern)

**Before (BAD)**:
```python
try:
    result = sitk.BinaryFillhole(mask)
except Exception:
    pass  # Silent failure
```

**After (GOOD) - Documented valid suppression:**
```python
try:
    result = sitk.BinaryFillhole(mask)
except RuntimeError as e:
    # SimpleITK filter failure (e.g., degenerate mask, all zeros)
    # Result: segmentation still valid, just without hole filling
    # Valid suppression: post-processing is optional enhancement
    logging.warning(f"Binary fillhole failed (result may have holes): {e}")
```

---

### Fix 6: Widget Lifecycle (Valid Suppression Pattern)

**Before (BAD)**:
```python
try:
    widget.scheduleRender()
except Exception:
    pass
```

**After (GOOD) - Documented valid suppression:**
```python
try:
    widget.scheduleRender()
except RuntimeError as e:
    # Widget deleted between check and use - expected during cleanup
    # Result: render not scheduled, no functional impact
    # Valid suppression: widget lifecycle is outside our control
    logging.debug(f"Render skipped (widget deleted): {e}")
```

---

## Adding Proper Logging

### Logger Setup

Ensure each module has a logger:

```python
import logging

logger = logging.getLogger(__name__)
```

### Logging Levels Guide

| Level | Use For |
|-------|---------|
| `logger.debug()` | Expected non-critical failures (widget deleted, optional post-processing) |
| `logger.info()` | Normal operation milestones |
| `logger.warning()` | Recoverable issues, degraded operation |
| `logger.error()` | Errors that don't stop execution |
| `logger.exception()` | Errors with stack trace (use in except blocks) |

### Logging Best Practices

```python
# GOOD - Use exception() in except blocks (includes stack trace)
except SomeError as e:
    logger.exception("Operation failed: %s", e)

# GOOD - Use lazy formatting
logger.debug("Processing item %s of %s", i, total)

# BAD - Eager string formatting
logger.debug(f"Processing item {i} of {total}")  # Formats even if debug disabled

# GOOD - Include relevant context
logger.error("Failed to load preset '%s' from %s: %s", preset_id, path, e)

# BAD - Vague messages
logger.error("Error occurred")
```

---

## Fix Workflow

### Step 1: Run Quality Review

```
/review-code-quality
```

### Step 2: Categorize Issues by Priority

1. `except: pass` or `except Exception: pass` (CRITICAL)
2. Overly broad `except Exception` (HIGH)
3. Missing logging in except blocks (HIGH)
4. Silent continuation patterns (MEDIUM)
5. Return None on error (MEDIUM)

### Step 3: Fix Each Issue

For each issue:

1. **Understand the intent** - Why was error suppressed?
2. **Determine correct handling**:
   - Should it fail fast? → Remove try/except or re-raise
   - Is recovery possible? → Handle specifically with logging
   - Is it truly optional? → Document and log at DEBUG level
3. **Apply fix pattern** from above
4. **Add/update tests** for error conditions
5. **Run tests** to verify fix

### Step 4: Verify Fixes

```bash
# Run tests
uv run pytest -v

# Run linting
uv run ruff check .

# Re-run review to confirm
/review-code-quality
```

---

## Exception Handling Decision Tree

```
Is this a programming error (bug)?
├─ Yes → Let it crash (don't catch)
└─ No → Can user/system recover?
         ├─ Yes → Catch, log, handle gracefully
         └─ No → Catch, log with context, re-raise or wrap

When catching:
├─ Can you name the specific exception?
│   ├─ Yes → Catch that specific type
│   └─ No → Research what exceptions can occur
└─ Is logging added?
    ├─ Yes → Good
    └─ No → Add logger.exception() call
```

---

## Valid Exception Handling Reference

### 1. Optional Feature Degradation

```python
try:
    from sklearn.mixture import GaussianMixture
    HAS_GMM = True
except ImportError:
    # sklearn is optional - fall back to simple statistics
    logger.info("sklearn not available, using simple threshold estimation")
    HAS_GMM = False
```

### 2. Widget Lifecycle

```python
try:
    self.viewWidget.scheduleRender()
except RuntimeError as e:
    # Widget deleted during cleanup - expected, non-critical
    logging.debug(f"Render skipped (widget deleted): {e}")
```

### 3. Post-Processing Fallback

```python
try:
    result = sitk.BinaryFillhole(mask)
except RuntimeError as e:
    # SimpleITK filter failure - result still valid without enhancement
    logging.warning(f"Fill holes failed (result may have holes): {e}")
```

### 4. User Input Validation

```python
try:
    value = int(user_input)
except ValueError:
    logger.warning("Invalid input '%s', prompting user", user_input)
    show_error_to_user("Please enter a valid number")
```

---

## Testing Error Paths

After fixing, add tests for error conditions:

```python
def test_load_config_missing_file():
    """Test that missing config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(Path("/nonexistent/path"))

def test_process_invalid_data():
    """Test that invalid data raises ValueError with details."""
    with pytest.raises(ValueError, match="Invalid data"):
        process_data({"bad": "data"})
```

---

## Commit Message Template

After fixes:

```
fix: replace exception swallowing with proper error handling

- Remove except:pass in module_name.py
- Add specific exception types with logging
- Document why suppression is valid for optional operations
- Add error path tests

Follows fail-fast principle: errors now surface immediately
with full context for debugging.
```
