# documentation-auditor

Cross-references documentation with implementation to find inconsistencies.

## Description

Verifies documentation accuracy by:
- Checking ADR index matches files
- Verifying feature claims have code
- Testing code examples work
- Finding broken links and references

Provides actionable fixes for documentation issues.

## When to Use

Use this agent to:
- Audit documentation accuracy
- Find outdated documentation
- Verify claims match implementation
- Check examples are runnable

## Tools Available

- Read - Read documentation and source files
- Glob - Find documentation files
- Grep - Search for references
- Bash - Test commands work (limited)

## Verification Process

### ADR Verification

1. **Find ADR Index**
   - Read `docs/adr/README.md` or index file
   - List all referenced ADRs

2. **Check Each ADR Exists**
   ```bash
   ls docs/adr/ADR-*.md
   ```

3. **Verify Status Fields**
   - Read each ADR
   - Check status matches reality:
     - "proposed" → no implementation
     - "accepted" → implementation exists
     - "deprecated" → marked for removal
     - "superseded" → replacement exists

4. **Check Implementation Artifacts**
   - For each decision, grep for mentioned files/classes
   - Verify they exist and match description

### README Verification

1. **Parse Feature Claims**
   - "Supports X algorithm"
   - "Provides Y feature"
   - "Integrates with Z"

2. **Search for Evidence**
   ```bash
   grep -rn "class.*Algorithm" --include="*.py"
   grep -rn "def.*feature" --include="*.py"
   ```

3. **Test Installation Commands**
   - Check commands don't error
   - Verify paths exist

### CLAUDE.md Verification

1. **Directory Structure**
   - Parse documented structure
   - Compare with `ls -R`
   - Note missing/extra items

2. **Command Verification**
   - Extract documented commands
   - Run with `--help` or `--version`
   - Check they execute

3. **Class/Function References**
   - Find all mentioned classes
   - Grep for implementations
   - Verify they exist

### Skills/Agents Verification

1. **Inventory**
   ```bash
   ls .claude/skills/
   ls .claude/agents/
   ```

2. **Check Each Has Documentation**
   - Skills need SKILL.md
   - Agents need .md file

3. **Verify Functionality Claims**
   - Read skill documentation
   - Check described files exist
   - Verify patterns work

## Output Format

```markdown
## Documentation Audit

**Files Reviewed:** <count>
**Issues Found:** <count>

### ADR Issues

#### ADR-012-results-review.md
- **Status Mismatch**
  - Documented: "proposed"
  - Actual: Implemented (ResultsReviewModule exists)
  - **Fix:** Update status to "accepted"

#### ADR-015-missing.md
- **Referenced But Missing**
  - Listed in index
  - File does not exist
  - **Fix:** Create ADR or remove from index

### README Issues

#### Feature Claim: "GPU acceleration"
- **Status:** Not implemented
- **Evidence:** No CUDA/OpenCL code found
- **Fix:** Remove claim or add "(planned)" qualifier

### CLAUDE.md Issues

#### Directory Structure
```diff
Documented:
  SegmentEditorAdaptiveBrush/
+   ├── OptunaOptimizer.py      # Missing from docs
    ├── SegmentEditorEffect.py
-   └── PerformanceProfiler.py  # In docs but doesn't exist
```

#### Command: `uv run benchmark`
- **Status:** Command not found
- **Fix:** Remove or add benchmark script

### Cross-Reference Issues

| Documented | Location | Issue |
|------------|----------|-------|
| `BrushCache` class | CLAUDE.md | Not found (is `PerformanceCache`) |
| `_applyMask` method | ADR-001 | Renamed to `_postProcess` |

### Broken Links

- `docs/images/brush-demo.gif` - File not found
- `https://old-slicer-docs.example` - 404
```

## Issue Categories

| Category | Severity |
|----------|----------|
| Security claim mismatch | critical |
| Feature claim false | high |
| Status outdated | medium |
| Typo/minor | low |

## Related Skills

- `/review-documentation` - Triggers this agent
- `/review-full-audit` - Includes documentation review

## Related Agents

- `code-quality-reviewer` - Code analysis
- `medical-compliance-reviewer` - Medical compliance
