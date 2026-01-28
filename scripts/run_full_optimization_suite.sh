#!/bin/bash
# Run comprehensive optimization suite
# Two optimization runs: original clicks and alternative clicks

set -e

cd "$(dirname "$0")/.."
source .env

echo "========================================"
echo "FULL OPTIMIZATION SUITE"
echo "========================================"
echo "Started: $(date)"
echo ""

# Run 1: Comprehensive with original clicks (150 trials)
echo "========================================"
echo "RUN 1: Comprehensive All Algorithms (Original Clicks)"
echo "Config: comprehensive_all_algorithms.yaml"
echo "Trials: 150"
echo "Started: $(date)"
echo "========================================"

"$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/comprehensive_all_algorithms.yaml

echo ""
echo "Run 1 completed: $(date)"
echo ""

# Run 2: Comprehensive with alternative clicks (100 trials)
echo "========================================"
echo "RUN 2: Comprehensive All Algorithms (Alternative Clicks)"
echo "Config: comprehensive_alt_clicks.yaml"
echo "Trials: 100"
echo "Started: $(date)"
echo "========================================"

"$SLICER_PATH" --python-script scripts/run_optimization.py \
    SegmentEditorAdaptiveBrushTester/configs/comprehensive_alt_clicks.yaml

echo ""
echo "========================================"
echo "FULL OPTIMIZATION SUITE COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Results saved to optimization_results/"
ls -lt optimization_results/ | head -5
