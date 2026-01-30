#!/bin/bash
#
# Validate extension for Slicer Extension Index submission
#
# Usage:
#   ./scripts/validate_extension.sh
#
# Checks:
#   - Repository structure (CMakeLists.txt, LICENSE, README.md)
#   - CMakeLists.txt metadata fields
#   - URL accessibility (homepage, icon, screenshots)
#   - Extension JSON file (if present)
#   - GitHub repository settings
#   - License type
#

# Don't exit on error - we handle errors ourselves
set +e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

PASS=0
FAIL=0
WARN=0

pass() {
    echo "[PASS] $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "[FAIL] $1"
    FAIL=$((FAIL + 1))
}

warn() {
    echo "[WARN] $1"
    WARN=$((WARN + 1))
}

info() {
    echo "[INFO] $1"
}

echo "============================================"
echo "Slicer Extension Submission Validator"
echo "============================================"
echo ""

# Step 1: Repository Structure
echo "=== Repository Structure ==="

[ -f CMakeLists.txt ] && pass "CMakeLists.txt exists" || fail "CMakeLists.txt missing"

if [ -f LICENSE.txt ] || [ -f LICENSE ]; then
    pass "LICENSE file exists"
else
    fail "LICENSE file missing"
fi

[ -f README.md ] && pass "README.md exists" || fail "README.md missing"

echo ""

# Step 2: CMakeLists.txt Metadata
echo "=== CMakeLists.txt Metadata ==="

HOMEPAGE=$(grep -oP 'EXTENSION_HOMEPAGE\s+"[^"]*"' CMakeLists.txt 2>/dev/null | cut -d'"' -f2 || echo "")
CONTRIBUTORS=$(grep -oP 'EXTENSION_CONTRIBUTORS\s+"[^"]*"' CMakeLists.txt 2>/dev/null | cut -d'"' -f2 || echo "")
DESCRIPTION=$(grep -oP 'EXTENSION_DESCRIPTION\s+"[^"]*"' CMakeLists.txt 2>/dev/null | cut -d'"' -f2 || echo "")
ICONURL=$(grep -oP 'EXTENSION_ICONURL\s+"[^"]*"' CMakeLists.txt 2>/dev/null | cut -d'"' -f2 || echo "")
SCREENSHOTURLS=$(grep -oP 'EXTENSION_SCREENSHOTURLS\s+"[^"]*"' CMakeLists.txt 2>/dev/null | cut -d'"' -f2 || echo "")

info "HOMEPAGE: $HOMEPAGE"
[ -n "$HOMEPAGE" ] && [[ "$HOMEPAGE" != *"example.com"* ]] && pass "Homepage set" || fail "Homepage missing or placeholder"

info "CONTRIBUTORS: $CONTRIBUTORS"
[ -n "$CONTRIBUTORS" ] && [[ "$CONTRIBUTORS" != *"John Doe"* ]] && pass "Contributors set" || fail "Contributors missing or placeholder"

info "DESCRIPTION: $DESCRIPTION"
[ -n "$DESCRIPTION" ] && pass "Description set" || fail "Description missing"

info "ICONURL: $ICONURL"
[ -n "$ICONURL" ] && [[ "$ICONURL" == https://raw.githubusercontent.com/* ]] && pass "Icon URL is raw GitHub" || fail "Icon URL should be raw.githubusercontent.com"

info "SCREENSHOTURLS: $SCREENSHOTURLS"
[ -n "$SCREENSHOTURLS" ] && [[ "$SCREENSHOTURLS" == https://raw.githubusercontent.com/* ]] && pass "Screenshot URL is raw GitHub" || fail "Screenshot URL should be raw.githubusercontent.com"

echo ""

# Step 3: URL Accessibility
echo "=== URL Accessibility ==="

if [ -n "$HOMEPAGE" ]; then
    HTTP_CODE=$(curl -sIL -o /dev/null -w "%{http_code}" "$HOMEPAGE" 2>/dev/null || echo "000")
    [ "$HTTP_CODE" = "200" ] && pass "Homepage accessible (HTTP $HTTP_CODE)" || fail "Homepage returned HTTP $HTTP_CODE"
fi

if [ -n "$ICONURL" ]; then
    HTTP_CODE=$(curl -sI -o /dev/null -w "%{http_code}" "$ICONURL" 2>/dev/null || echo "000")
    [ "$HTTP_CODE" = "200" ] && pass "Icon URL accessible (HTTP $HTTP_CODE)" || fail "Icon URL returned HTTP $HTTP_CODE"
fi

if [ -n "$SCREENSHOTURLS" ]; then
    FIRST_SCREENSHOT=$(echo "$SCREENSHOTURLS" | awk '{print $1}')
    HTTP_CODE=$(curl -sI -o /dev/null -w "%{http_code}" "$FIRST_SCREENSHOT" 2>/dev/null || echo "000")
    [ "$HTTP_CODE" = "200" ] && pass "Screenshot URL accessible (HTTP $HTTP_CODE)" || fail "Screenshot URL returned HTTP $HTTP_CODE"
fi

echo ""

# Step 4: Extension JSON
echo "=== Extension Index JSON ==="

JSON_FILE=$(find . -maxdepth 1 -name "*.json" -type f 2>/dev/null | grep -v package | grep -v tsconfig | head -1 || echo "")

if [ -n "$JSON_FILE" ] && [ -f "$JSON_FILE" ]; then
    info "Found: $JSON_FILE"

    python3 -c "import json; json.load(open('$JSON_FILE'))" 2>&1 && pass "Valid JSON syntax" || fail "Invalid JSON syntax"

    python3 << EOF
import json
import sys

with open('$JSON_FILE') as f:
    data = json.load(f)

errors = 0

# Check required fields
for field in ['\$schema', 'category', 'scm_url']:
    if field in data:
        print(f"[PASS] Required field '{field}' present")
    else:
        print(f"[FAIL] Required field '{field}' missing")
        errors += 1

# Check schema URL
if data.get('\$schema', '').startswith('https://raw.githubusercontent.com/Slicer/Slicer/main/Schemas/'):
    print("[PASS] Schema URL is official Slicer schema")
else:
    print("[FAIL] Schema URL should reference official Slicer schema")
    errors += 1

sys.exit(errors)
EOF

    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        PASS=$((PASS + 4))
    else
        FAIL=$((FAIL + RESULT))
    fi
else
    warn "No Extension Index JSON file found"
    info "Run /prepare-extension-metadata skill to generate one"
fi

echo ""

# Step 5: GitHub Repository
echo "=== GitHub Repository ==="

if command -v gh &> /dev/null; then
    REPO_PATH=$(gh repo view --json nameWithOwner -q '.nameWithOwner' 2>/dev/null || echo "")

    if [ -n "$REPO_PATH" ]; then
        info "Repository: $REPO_PATH"

        TOPICS=$(gh repo view --json repositoryTopics -q '.repositoryTopics[].name' 2>/dev/null || echo "")
        if echo "$TOPICS" | grep -q "3d-slicer-extension"; then
            pass "Repository has '3d-slicer-extension' topic"
        else
            fail "Repository missing '3d-slicer-extension' topic"
            info "Add via: gh repo edit --add-topic 3d-slicer-extension"
        fi
    fi
else
    warn "gh CLI not available - skipping GitHub checks"
fi

echo ""

# Step 6: License
echo "=== License ==="

LICENSE_FILE=""
[ -f LICENSE.txt ] && LICENSE_FILE="LICENSE.txt"
[ -f LICENSE ] && LICENSE_FILE="LICENSE"
[ -f LICENSE.md ] && LICENSE_FILE="LICENSE.md"

if [ -n "$LICENSE_FILE" ]; then
    info "License file: $LICENSE_FILE"

    if grep -qi "BSD" "$LICENSE_FILE"; then
        pass "BSD License detected (Slicer-compatible)"
    elif grep -qi "MIT License" "$LICENSE_FILE"; then
        pass "MIT License detected (recommended)"
    elif grep -qi "Apache License" "$LICENSE_FILE"; then
        pass "Apache License detected (acceptable)"
    else
        warn "License type not auto-detected - verify manually"
    fi
else
    fail "No LICENSE file found"
fi

echo ""

# Summary
echo "============================================"
echo "VALIDATION SUMMARY"
echo "============================================"
echo ""
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Warnings: $WARN"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "Ready for submission!"
    echo ""
    echo "Next steps:"
    echo "  1. Fork https://github.com/Slicer/ExtensionsIndex"
    echo "  2. Add your JSON file to the repository"
    echo "  3. Create a pull request"
    exit 0
else
    echo "Fix $FAIL issue(s) before submission."
    exit 1
fi
