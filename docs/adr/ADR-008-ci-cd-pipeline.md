# ADR-008: CI/CD Pipeline Strategy

## Status

Proposed

## Context

The project needs automated testing and deployment to:

1. Ensure code quality on every commit
2. Verify the extension installs and works in actual Slicer environment
3. Package the extension for Slicer Extensions Index submission
4. Provide confidence for contributors and users

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

### Record-and-Replay Testing

Users can record Slicer sessions, and test scripts are generated from recordings:

1. User records actions in Slicer (File > Record Macro or custom recorder)
2. Recording saved as JSON/Python script
3. Claude converts recording to pytest-compatible test
4. Test replays actions and verifies expected outcomes
5. Screenshots captured at key points for documentation

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
- Screenshots generated for documentation

### Negative

- CI runs add time to PR process
- Slicer installation in CI is complex
- Headless testing has limitations (no real GPU)

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
  scripts/
    install_extension.py
    e2e_test.py
    generate_screenshots.py
```

### Slicer Action Reference

See [Slicer GitHub Actions](https://github.com/Slicer/SlicerGitHubActions) for official CI setup patterns.

## References

- [Slicer Extension Testing](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html#testing)
- [SlicerGitHubActions](https://github.com/Slicer/SlicerGitHubActions)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
