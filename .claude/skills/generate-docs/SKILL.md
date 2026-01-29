# generate-docs

Generate documentation from code and test screenshots.

## Usage

```
/generate-docs [component]
```

Where `component` is optionally one of:
- `all` (default) - Generate all documentation
- `api` - Generate API reference only
- `algorithms` - Generate algorithm pages only
- `ui` - Generate UI reference only
- `screenshots` - Run documentation tests to capture screenshots

## What This Skill Does

1. Runs documentation tests to capture screenshots (if needed)
2. Extracts screenshots by doc_tags
3. Generates algorithm documentation pages
4. Generates UI reference pages
5. Generates API documentation from docstrings
6. Builds Sphinx documentation

## Prerequisites

1. Slicer must be installed and SLICER_PATH set in `.env`
2. Sphinx and dependencies installed: `uv pip install sphinx sphinx-rtd-theme myst-parser`

## Execution Steps

### Step 1: Run Documentation Tests (if screenshots needed)

Check if screenshots exist. If not, run the docs test suite:

```bash
source .env
"$SLICER_PATH" --python-script scripts/run_tests.py --exit docs
```

### Step 2: Extract Screenshots

```bash
python scripts/extract_screenshots_for_docs.py \
    --screenshots-dir test_runs/*/screenshots/ \
    --output-dir docs/source/_static/screenshots/
```

### Step 3: Generate Algorithm Documentation

```bash
python scripts/generate_algorithm_docs.py \
    --screenshots-dir docs/source/_static/screenshots/ \
    --output-dir docs/source/generated/algorithms/
```

### Step 4: Generate UI Documentation

```bash
python scripts/generate_ui_docs.py \
    --screenshots-dir docs/source/_static/screenshots/ \
    --output-dir docs/source/generated/ui/
```

### Step 5: Generate API Documentation

```bash
python scripts/generate_api_docs.py \
    --output-dir docs/source/generated/api/
```

### Step 6: Build Sphinx Documentation

```bash
cd docs && make html
```

### Step 7: Report Results

Report the generated documentation location:
- Built docs: `docs/build/html/`
- Index: `docs/build/html/index.html`

## Output

Generated documentation is in `docs/build/html/`. Open `index.html` in a browser to view.

## Notes

- Screenshot generation requires Slicer and takes several minutes
- API documentation is generated from Python docstrings
- Algorithm pages include auto-captured screenshots when available
- CI runs this automatically on merge to main
