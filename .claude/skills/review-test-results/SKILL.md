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

### Step 3: Analyze Screenshots

View the screenshots in `screenshots/` folder, using `manifest.json` for descriptions.

### Step 4: Generate Report

Provide a structured report covering:

1. **Test Summary**
   - Pass/fail counts
   - Duration
   - Any errors

2. **Failed Tests Analysis**
   - Root cause for each failure
   - Suggested fixes

3. **Performance Issues**
   - Operations exceeding targets (see CLAUDE.md)
   - Optimization suggestions

4. **UI Issues** (from screenshots)
   - Layout problems
   - Widget visibility issues
   - Alignment concerns

5. **Suggested Improvements**
   - New test cases to add
   - Algorithm improvements
   - UI enhancements

## Output

The skill produces a markdown report with actionable recommendations.

## Follow-up Actions

Based on the review, you may want to:
- `/add-test-case` - Add new test cases for coverage gaps
- Use `bug-fixer` agent - Fix identified issues
- Use `algorithm-improver` agent - Optimize slow operations
- Use `ui-improver` agent - Address UI issues
