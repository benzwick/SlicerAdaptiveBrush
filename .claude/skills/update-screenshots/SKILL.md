---
name: update-screenshots
description: Update Screenshots folder from test runs for extension submission
allowed-tools:
  - Bash
  - Read
  - Write
context: manual
---

# Update Screenshots Skill

Update the Screenshots/ folder with images from test runs for extension submission and documentation.

## When to Use

- Before submitting extension to Extension Index
- When updating documentation screenshots
- After running UI tests that capture good screenshots

## How to Run

```bash
./scripts/update_screenshots.sh
```

## What It Does

1. Finds the most recent test run with screenshots
2. Copies the main UI screenshot to `Screenshots/main-ui.png`
3. Lists available screenshots for manual selection

## After Running

1. Review the copied screenshots
2. Stage and commit:
   ```bash
   git add Screenshots/
   git commit -m "docs: Add screenshots for extension submission"
   git push
   ```
3. Update `CMakeLists.txt` with the screenshot URL:
   ```cmake
   set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/benzwick/SlicerAdaptiveBrush/main/Screenshots/main-ui.png")
   ```

## Manual Screenshot Selection

If you need different screenshots, browse the test run folders:

```bash
ls test_runs/*/screenshots/
```

Then copy manually:

```bash
cp test_runs/2026-01-24_192147_ui/screenshots/001.png Screenshots/main-ui.png
```

## Screenshot Requirements for Extension Index

- At least one informative screenshot showing the extension in use
- Raw GitHub URL format: `https://raw.githubusercontent.com/OWNER/REPO/BRANCH/Screenshots/filename.png`
- Should show the main UI/functionality
- Good contrast and readable text
