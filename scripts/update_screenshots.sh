#!/bin/bash
#
# Update Screenshots folder from test runs
#
# Usage:
#   ./scripts/update_screenshots.sh
#
# This script copies selected screenshots from test runs to the Screenshots/
# folder for use in the extension submission and documentation.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Updating Screenshots folder..."

# Find test runs with screenshots (exclude reviewer runs for main UI screenshot)
# Prefer: _ui runs > _all runs > _algorithm runs
# Exclude: _reviewer_ui, _reviewer_unit, _reviewer_integration runs
LATEST_UI_RUN=$(ls -td test_runs/*_ui 2>/dev/null | grep -v reviewer | head -1)
LATEST_ALL_RUN=$(ls -td test_runs/*_all 2>/dev/null | grep -v reviewer | head -1)
LATEST_ALG_RUN=$(ls -td test_runs/*_algorithm 2>/dev/null | head -1)

if [ -z "$LATEST_UI_RUN" ] && [ -z "$LATEST_ALL_RUN" ] && [ -z "$LATEST_ALG_RUN" ]; then
    echo "Error: No test runs found with screenshots"
    echo "Run tests first: ./scripts/run_tests.py ui"
    exit 1
fi

# Prefer UI test run, then all, then algorithm
SOURCE_RUN="${LATEST_UI_RUN:-${LATEST_ALL_RUN:-$LATEST_ALG_RUN}}"
echo "Using test run: $SOURCE_RUN"

# Ensure Screenshots directory exists
mkdir -p Screenshots

# Copy main UI screenshot (first screenshot from UI test)
if [ -f "$SOURCE_RUN/screenshots/001.png" ]; then
    cp "$SOURCE_RUN/screenshots/001.png" Screenshots/main-ui.png
    echo "Copied: Screenshots/main-ui.png"

    # Compress to meet git pre-commit hook limit (1000 KB)
    if ! command -v pngquant &> /dev/null; then
        echo "Error: pngquant not found"
        echo "Install with: sudo apt install pngquant"
        exit 1
    fi
    ORIG_SIZE=$(du -k Screenshots/main-ui.png | cut -f1)
    pngquant --force --quality=65-80 --output Screenshots/main-ui.png Screenshots/main-ui.png
    NEW_SIZE=$(du -k Screenshots/main-ui.png | cut -f1)
    echo "Compressed: ${ORIG_SIZE}KB -> ${NEW_SIZE}KB"
else
    echo "Error: No screenshot found at $SOURCE_RUN/screenshots/001.png"
    exit 1
fi

# List available screenshots
echo ""
echo "Available screenshots in $SOURCE_RUN/screenshots/:"
ls -la "$SOURCE_RUN/screenshots/"*.png 2>/dev/null | head -10

echo ""
echo "Screenshots folder contents:"
ls -la Screenshots/

echo ""
echo "Done. Remember to:"
echo "  1. git add Screenshots/"
echo "  2. git commit -m 'docs: Add screenshots for extension submission'"
echo "  3. git push"
echo "  4. Update CMakeLists.txt EXTENSION_SCREENSHOTURLS"
