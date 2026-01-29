# ADR-008: CI/CD Pipeline Strategy

## Status

Accepted

## Context

The project needs automated testing and deployment to:

1. Ensure code quality on every commit
2. Verify the extension installs and works in actual Slicer environment
3. Package the extension for Slicer Extensions Index submission
4. Provide confidence for contributors and users
5. Generate documentation automatically with screenshots

### Testing Levels

| Level | What | Where | Speed |
|-------|------|-------|-------|
| Unit tests | Algorithm logic, caching, utilities | pytest (no Slicer) | Fast |
| Integration tests | Effect in Slicer, UI, mouse events | Slicer Python | Slow |
| End-to-end tests | Full workflow on sample data | Slicer headless | Slowest |

## Decision

Implement a **multi-stage GitHub Actions pipeline**:

### Stage 1: Fast Checks (on every commit)

```yaml
- name: Lint and format
  run: |
    uv run ruff check .
    uv run ruff format --check .

- name: Unit tests
  run: uv run pytest -v
```

### Stage 2: Slicer Integration (on PR and main)

```yaml
- name: Install Slicer
  uses: Slicer/slicer-action@v1

- name: Install extension
  run: |
    Slicer --python-script scripts/install_extension.py

- name: Run Slicer tests
  run: |
    Slicer --testing --python-code "slicer.util.runTests(module='SegmentEditorAdaptiveBrush')"
```

### Stage 3: End-to-End Tests (on release)

```yaml
- name: Test on sample data
  run: |
    Slicer --python-script scripts/e2e_test.py
```

### Stage 4: Documentation Generation (on main)

```yaml
name: Documentation

jobs:
  generate-screenshots:
    runs-on: ubuntu-latest
    steps:
      - name: Set up Xvfb
        run: |
          sudo apt-get update
          sudo apt-get install -y xvfb
          Xvfb :99 -screen 0 1920x1080x24 &
          export DISPLAY=:99

      - name: Install Slicer
        uses: Slicer/slicer-action@v1

      - name: Run documentation tests
        run: |
          Slicer --python-script scripts/run_tests.py --exit docs

      - name: Upload screenshots
        uses: actions/upload-artifact@v4
        with:
          name: doc-screenshots
          path: test_runs/*/screenshots/

  build-docs:
    needs: generate-screenshots
    steps:
      - name: Download screenshots
        uses: actions/download-artifact@v4

      - name: Extract screenshots for docs
        run: python scripts/extract_screenshots_for_docs.py

      - name: Build Sphinx docs
        run: |
          cd docs
          make html

      - name: Validate documentation
        run: python scripts/validate_docs.py

  deploy:
    needs: build-docs
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/build/html
```

### Record-and-Replay Testing

Slicer has a **QtTesting** framework for recording GUI interactions to XML files, but it's
considered experimental. The recommended approach is Python scripts operating on MRML nodes.

**Workflow:**
1. User describes or records actions in Slicer (QtTesting XML or manual description)
2. Claude converts to Python test scripts that manipulate MRML nodes directly
3. Tests replay actions programmatically (more reliable than GUI replay)
4. Screenshots captured at key points for documentation

**Example test from user description:**
```
User: "I loaded MRHead, selected Adaptive Brush, set preset to Brain,
       clicked on white matter at roughly (128, 100, 90), verified it
       segmented the surrounding tissue"

→ Converted to Python test that:
   - Loads MRHead via SampleData
   - Activates Adaptive Brush effect
   - Applies Brain preset
   - Simulates click at specified coordinates
   - Verifies segment volume > threshold
   - Captures screenshot
```

### CI Status Badge

Add badge to README:
```markdown
![CI](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/ci.yml/badge.svg)
```

## Consequences

### Positive

- Automatic quality checks on every commit
- Catches regressions before merge
- Verified installation process
- Builds confidence for extension submission
- Screenshots generated automatically for documentation
- Documentation always up-to-date with code
- Single source of truth (tests drive docs)

### Negative

- CI runs add time to PR process
- Slicer installation in CI is complex
- Headless testing has limitations (no real GPU)
- Documentation build adds CI time (~5 min)

## Alternatives Considered

### Local testing only

**Rejected**: No automated verification, relies on developer discipline.

### Test only in Slicer

**Rejected**: Too slow for every commit, unit tests provide faster feedback.

### Skip integration tests

**Rejected**: Would miss installation and UI issues that only appear in Slicer.

## Implementation Notes

### GitHub Actions Workflow Structure

```
.github/
  workflows/
    ci.yml           # Fast checks on every commit
    integration.yml  # Slicer tests on PR/main
    release.yml      # Full tests + packaging on release
    docs.yml         # Documentation generation and deployment
  scripts/
    install_extension.py
    e2e_test.py
    generate_screenshots.py
scripts/
  extract_screenshots_for_docs.py  # Copy screenshots to docs
  generate_api_docs.py             # Auto-generate API reference
  generate_algorithm_docs.py       # Auto-generate algorithm pages
  generate_ui_docs.py              # Auto-generate UI reference
  validate_docs.py                 # Verify documentation completeness
```

### Slicer Action Reference

See [Slicer GitHub Actions](https://github.com/Slicer/SlicerGitHubActions) for official CI setup patterns.

## References

- [Slicer Extension Testing](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html#testing)
- [SlicerGitHubActions](https://github.com/Slicer/SlicerGitHubActions)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
