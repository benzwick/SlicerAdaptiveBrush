#!/bin/bash
# Run a Python script inside 3D Slicer
#
# Usage:
#   ./scripts/run_in_slicer.sh <script.py> [--background]
#
# Examples:
#   ./scripts/run_in_slicer.sh scripts/comprehensive_optimization.py
#   ./scripts/run_in_slicer.sh scripts/create_gold_standard.py --background

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env if it exists
if [[ -f "$PROJECT_DIR/.env" ]]; then
    source "$PROJECT_DIR/.env"
fi

# Check SLICER_PATH
if [[ -z "$SLICER_PATH" ]]; then
    echo "Error: SLICER_PATH not set. Configure it in .env file."
    exit 1
fi

if [[ ! -x "$SLICER_PATH" ]]; then
    echo "Error: Slicer not found at: $SLICER_PATH"
    exit 1
fi

# Parse arguments
SCRIPT_TO_RUN=""
BACKGROUND=false

show_help() {
    echo "Run a Python script inside 3D Slicer"
    echo ""
    echo "Usage: $0 <script.py> [--background]"
    echo ""
    echo "Options:"
    echo "  --background    Run Slicer in background, don't block terminal"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 scripts/comprehensive_optimization.py"
    echo "  $0 scripts/create_gold_standard.py --background"
}

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            exit 0
            ;;
        --background)
            BACKGROUND=true
            ;;
        *)
            SCRIPT_TO_RUN="$arg"
            ;;
    esac
done

if [[ -z "$SCRIPT_TO_RUN" ]]; then
    show_help
    exit 1
fi

# Resolve script path
if [[ ! -f "$SCRIPT_TO_RUN" ]]; then
    # Try relative to project dir
    if [[ -f "$PROJECT_DIR/$SCRIPT_TO_RUN" ]]; then
        SCRIPT_TO_RUN="$PROJECT_DIR/$SCRIPT_TO_RUN"
    else
        echo "Error: Script not found: $SCRIPT_TO_RUN"
        exit 1
    fi
fi

# Create log directory
LOG_DIR="$PROJECT_DIR/test_runs"
mkdir -p "$LOG_DIR"

# Generate log filename from script name and timestamp
SCRIPT_NAME="$(basename "$SCRIPT_TO_RUN" .py)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log"

echo "Running: $SCRIPT_TO_RUN"
echo "Log: $LOG_FILE"

cd "$PROJECT_DIR"

if [[ "$BACKGROUND" == true ]]; then
    "$SLICER_PATH" --python-script "$SCRIPT_TO_RUN" > "$LOG_FILE" 2>&1 &
    echo "Started in background, PID: $!"
    echo "Monitor with: tail -f $LOG_FILE"
else
    "$SLICER_PATH" --python-script "$SCRIPT_TO_RUN" 2>&1 | tee "$LOG_FILE"
fi
