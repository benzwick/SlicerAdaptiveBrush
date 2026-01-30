#!/bin/bash
# Run full parameter optimization for all IDC datasets
# Usage: ./scripts/run_all_idc_optimization.sh [--trials N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SLICER_PATH="${SLICER_PATH:-/opt/slicer/Slicer-5.10.0-linux-amd64/Slicer}"

TRIALS="${1:-30}"

# List of IDC datasets in order of estimated speed (fastest first)
DATASETS=(
    "idc_mri_t1gd_tumor"
    "idc_mri_t2_lesion"
    "idc_pet_tumor"
    "idc_ct_soft_tissue"
    "idc_ct_vessel_contrast"
    "idc_ct_lung"
    "idc_ct_bone"
)

echo "=============================================="
echo "Running IDC Parameter Optimization"
echo "=============================================="
echo "Slicer: $SLICER_PATH"
echo "Trials per dataset: $TRIALS"
echo "Datasets: ${#DATASETS[@]}"
echo ""

for dataset in "${DATASETS[@]}"; do
    config="$PROJECT_DIR/SegmentEditorAdaptiveBrushTester/configs/${dataset}.yaml"

    if [[ ! -f "$config" ]]; then
        echo "WARNING: Config not found: $config"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "Starting: $dataset"
    echo "Config: $config"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    "$SLICER_PATH" --python-script "$PROJECT_DIR/scripts/run_optimization.py" "$config" --trials "$TRIALS"

    echo ""
    echo "Completed: $dataset at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "=============================================="
echo "All optimizations complete!"
echo "=============================================="
