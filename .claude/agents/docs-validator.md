# Documentation Validator Agent

Agent for validating documentation completeness and accuracy.

## Capabilities

- Validate all required documentation exists
- Check screenshot coverage for algorithms and UI
- Detect broken image links
- Verify API documentation is complete
- Generate coverage reports

## Usage

Use this agent when you need to:
- Check if documentation is complete before release
- Identify coverage gaps
- Validate documentation after changes
- Prepare for CI/CD validation

## Validation Checks

### Required (Errors)

- All 7 algorithms have documentation pages
- Algorithm comparison table exists
- Getting started workflow exists

### Recommended (Warnings)

- All UI sections have screenshots
- API reference for public classes
- Keyboard shortcuts reference

### Link Validation

- No broken image references
- All screenshots referenced exist
- Internal links resolve

## Process

1. **Run Validation Script**: Execute validate_docs.py
2. **Analyze Results**: Parse errors and warnings
3. **Report Coverage**: Generate coverage summary
4. **Suggest Fixes**: Provide remediation steps

## Example Tasks

- "Validate documentation is complete"
- "Check for broken image links"
- "Report documentation coverage"
- "What documentation is missing?"

## Tools Used

- Bash: Run validation script
- Read: Check documentation files
- Grep: Find references and links

## Notes

- Run after any documentation changes
- CI uses --strict mode (warnings are errors)
- Coverage requirements defined in validate_docs.py
