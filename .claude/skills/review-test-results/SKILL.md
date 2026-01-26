# review-test-results

Review test results from a test run folder with the test-reviewer agent.

## Usage

```
/review-test-results [folder]
```

Where `folder` is:
- Path to a specific test run folder
- Or omit to review the most recent test run

## What This Skill Does

1. Finds the test run folder (specified or most recent)
2. Reads test results, screenshots, and logs
3. Analyzes with the test-reviewer agent
4. Provides structured feedback and improvement suggestions

## Execution Steps

### Step 1: Find Test Run Folder

If no folder specified, find the most recent:

```bash
ls -td test_runs/*/ | head -1
```

### Step 2: Read Results

Read the following files from the test run folder:
- `results.json` - Test pass/fail status
- `metadata.json` - Run configuration
- `metrics.json` - Performance metrics
- `manual_actions.jsonl` - Manual testing actions (if present)
- `logs/test_run.log` - Test execution log
- `logs/slicer_session.log` - Slicer application log (if present)

### Step 3: Check Logs for Errors and Warnings

Search for errors and warnings in the log files:

```bash
# Find all errors
grep -iE "error|exception|traceback|failed|failure" logs/*.log 2>/dev/null

# Find all warnings
grep -iE "warning|warn|deprecated" logs/*.log 2>/dev/null
```

**Error categories:**
- **Exceptions** - Python tracebacks, RuntimeError, AttributeError, etc.
- **Test failures** - Assertion failures, test errors
- **Algorithm failures** - SimpleITK filter errors, segmentation failures
- **UI errors** - Qt/widget errors, rendering failures

**Warning categories:**
- **Deprecation warnings** - APIs removed in future Slicer versions
  - `DeprecationWarning`, `FutureWarning`, `PendingDeprecationWarning`
- **Qt warnings** - UI framework issues
  - `QFont::fromString`, `qt.qpa.plugin`, widget lifecycle
- **VTK warnings** - Rendering and data processing
  - `vtkOutputWindow`, mapper warnings, rendering issues
- **SimpleITK warnings** - Algorithm processing
  - Filter warnings, image type issues, parameter warnings
- **Python warnings** - Runtime issues
  - `RuntimeWarning`, `UserWarning`, resource warnings

**Prioritization:**
1. Errors that caused test failures (fix immediately)
2. Errors in logs that didn't cause failures (investigate)
3. Deprecation warnings (fix before Slicer upgrade)
4. Recurring warnings (fix to reduce noise)
5. One-off warnings (document or suppress if harmless)

### Step 4: Analyze Screenshots

View the screenshots in `screenshots/` folder, using `manifest.json` for descriptions.

### Step 5: Generate Report

Provide a structured report covering:

1. **Test Summary**
   - Pass/fail counts
   - Duration
   - Any errors

2. **Failed Tests Analysis**
   - Root cause for each failure
   - Suggested fixes

3. **Log Analysis (Errors & Warnings)**
   - Errors and exceptions from logs (with line numbers)
   - Count of warnings by category (deprecation, Qt, VTK, SimpleITK, Python)
   - Most frequent warnings (top 10 recurring)
   - Deprecation warnings to address before next Slicer version
   - Suggested fixes for each category

4. **Performance Issues**
   - Operations exceeding targets (see CLAUDE.md)
   - Optimization suggestions

5. **UI Issues** (from screenshots)
   - Layout problems
   - Widget visibility issues
   - Alignment concerns

6. **Suggested Improvements**
   - New test cases to add
   - Algorithm improvements
   - UI enhancements
   - Warning fixes to implement

## Output

The skill produces a markdown report with actionable recommendations.

## Follow-up Actions

Based on the review, you may want to:
- `/add-test-case` - Add new test cases for coverage gaps
- Use `bug-fixer` agent - Fix identified issues
- Use `algorithm-improver` agent - Optimize slow operations
- Use `ui-improver` agent - Address UI issues
