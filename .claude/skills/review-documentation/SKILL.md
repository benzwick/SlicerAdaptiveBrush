# review-documentation

Verifies documentation accuracy against implementation using the documentation-auditor agent.

## Usage

```
/review-documentation [scope]
```

Where `scope` is:
- `adr` - Review Architecture Decision Records only
- `readme` - Review README.md only
- `claude` - Review CLAUDE.md only
- Or omit to review all documentation

## What This Skill Does

1. Cross-references documentation with code
2. Verifies claims are accurate
3. Checks examples are runnable
4. Identifies outdated information

## Focus Areas

### ADR Verification
- Each ADR file exists as listed in index
- Status fields match current implementation
- Decisions are still relevant
- Superseded ADRs are marked

### README Claims
- Feature claims have supporting code
- Installation instructions work
- Usage examples are accurate
- Screenshots match current UI

### CLAUDE.md Accuracy
- Directory structure matches reality
- Commands actually work
- Architecture descriptions are current
- Performance claims have evidence

### Skills/Agents Documentation
- All skills have SKILL.md files
- All agents have .md files
- Documentation matches functionality
- Examples are accurate

## Execution Steps

### Step 1: Inventory Documentation

Find all documentation files:

```bash
find . -name "*.md" -not -path "./.venv/*"
find docs/adr -name "*.md"
ls .claude/skills/
ls .claude/agents/
```

### Step 2: ADR Verification

For each ADR:
1. Read the ADR file
2. Check status field
3. Grep for implementation artifacts mentioned
4. Verify decisions are still in effect

### Step 3: README Verification

1. Read README.md
2. For each feature claim:
   - Search for supporting code
   - Verify functionality exists
3. Test installation commands (if applicable)

### Step 4: CLAUDE.md Verification

1. Verify directory structure:
```bash
# Compare documented structure to actual
ls -la SegmentEditorAdaptiveBrush/
ls -la .claude/
```

2. Verify commands work:
```bash
# Test documented commands
uv run pytest --collect-only  # Check test collection
uv run ruff check . --select E  # Quick lint check
```

3. Search for referenced classes/functions

### Step 5: Generate Report

Create report in `reviews/reports/<timestamp>_documentation/`:
- `report.json` - Machine-readable findings
- `report.md` - Human-readable summary

## Issue Categories

| Category | Description |
|----------|-------------|
| MISSING_DOC | Documented item doesn't exist |
| STALE_DOC | Documentation outdated vs implementation |
| INCORRECT_CLAIM | Claim doesn't match code behavior |
| BROKEN_LINK | Link or reference doesn't resolve |
| INCOMPLETE | Missing required documentation |

## Output Example

```markdown
## Documentation Review

**Date:** 2026-01-26T14:30:00
**Files Reviewed:** 25

### Summary
- Verified: 18
- Issues: 7

### ADR Issues
#### docs/adr/ADR-012-results-review.md
- **Status:** Listed as "proposed" but implementation exists
- **Suggestion:** Update status to "accepted"

### CLAUDE.md Issues
#### Performance Claims
- **Claim:** "2D brush (10mm): 30-100ms"
- **Issue:** No benchmark data to support this
- **Suggestion:** Add benchmark tests or note as "estimated"

### Missing Documentation
- PerformanceCache.py has no docstrings
- EmbeddedWizardUI not mentioned in CLAUDE.md
```

## Follow-up Actions

Based on the review, you may want to:
- Update outdated ADR statuses
- Add missing documentation
- Correct inaccurate claims
- Add verification tests for claims
