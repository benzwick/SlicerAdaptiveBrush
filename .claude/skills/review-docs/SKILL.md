# review-docs

Validate documentation completeness and report coverage gaps.

## Usage

```
/review-docs [--strict]
```

Options:
- `--strict` - Treat warnings as errors (fails if any warnings)

## What This Skill Does

1. Validates all algorithms have documentation
2. Validates all UI sections are documented
3. Checks API documentation exists for public classes
4. Detects broken image links
5. Reports coverage gaps

## Execution Steps

### Step 1: Run Documentation Validation

```bash
python scripts/validate_docs.py \
    --docs-dir docs/build/html/ \
    --screenshots-dir docs/source/_static/screenshots/
```

### Step 2: Review Results

The script reports:
- **Errors**: Missing required documentation (fails validation)
- **Warnings**: Missing recommended documentation (informational)

### Step 3: Report Coverage

Report which documentation is present and which is missing:

| Category | Required | Status |
|----------|----------|--------|
| Algorithms | 7 | See details |
| UI Sections | 4+ | See details |
| Workflows | 1+ | See details |
| API Reference | 2+ | See details |

## Expected Coverage

### Algorithms (Required)

All 7 algorithms should have documentation:
- Geodesic Distance
- Watershed
- Random Walker
- Level Set
- Connected Threshold
- Region Growing
- Threshold Brush

### UI Sections (Required)

Core UI elements:
- Options panel
- Brush settings
- Algorithm selection
- Threshold settings

### Workflows (Recommended)

At least one tutorial:
- Getting started

### API Reference (Recommended)

Key classes:
- SegmentEditorEffect
- IntensityAnalyzer

## Fixing Coverage Gaps

If validation fails:

1. **Missing algorithm docs**: Check scripts/generate_algorithm_docs.py ran
2. **Missing screenshots**: Run docs test suite with `/generate-docs screenshots`
3. **Missing API docs**: Ensure docstrings exist in source files
4. **Broken image links**: Check screenshot paths in markdown files

## Notes

- CI runs this automatically on every PR
- Build fails if required documentation is missing
- Screenshots may be generated during CI, so warnings about missing screenshots are normal locally
