# Documentation Generator Agent

Agent for generating documentation from code and tests.

## Capabilities

- Run documentation test suite to capture screenshots
- Extract and organize screenshots by doc_tags
- Generate algorithm documentation pages
- Generate UI reference pages
- Generate API documentation from docstrings
- Build Sphinx documentation

## Usage

Use this agent when you need to:
- Generate complete documentation from scratch
- Update documentation after code changes
- Add screenshots for new features
- Build the documentation website

## Process

1. **Analyze Changes**: Determine what documentation needs updating
2. **Run Tests**: Execute docs test suite if screenshots needed
3. **Extract Screenshots**: Copy tagged screenshots to docs folder
4. **Generate Pages**: Create markdown/RST from templates and metadata
5. **Build Docs**: Run Sphinx to build HTML
6. **Validate**: Run validation to check completeness

## Example Tasks

- "Generate documentation for the new algorithm"
- "Update screenshots after UI changes"
- "Rebuild documentation website"
- "Add documentation for the Parameter Wizard"

## Tools Used

- Bash: Run Python scripts and Sphinx
- Read: Check existing documentation
- Write: Create new documentation files
- Grep: Find documentation references

## Notes

- Always run validation after generating docs
- Screenshots require Slicer to be installed
- Generated docs go in docs/source/generated/
- Built HTML goes in docs/build/html/
