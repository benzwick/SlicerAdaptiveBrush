#!/usr/bin/env python
"""Run batch optimization across all algorithms and recipes.

This script runs optimization for each algorithm independently, allowing
comparison of algorithm performance on the same gold standards.

Usage:
    Slicer --python-script scripts/batch_optimization.py [--trials N] [--algorithms A1,A2,...]

Example:
    # Run all algorithms with 20 trials each
    Slicer --python-script scripts/batch_optimization.py --trials 20

    # Run specific algorithms
    Slicer --python-script scripts/batch_optimization.py --algorithms watershed,level_set
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_optimization")


def get_project_root() -> Path:
    """Get the project root directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


PROJECT_ROOT = get_project_root()

# All available algorithms
ALL_ALGORITHMS = [
    "watershed",
    "connected_threshold",
    "region_growing",
    "threshold_brush",
    "geodesic_distance",
    "level_set",
    "random_walker",
]

# Algorithm-specific parameter spaces
ALGORITHM_PARAMS: dict[str, dict[str, dict[str, float]]] = {
    "watershed": {
        "watershed_gradient_scale": {"low": 0.5, "high": 2.0},
        "watershed_smoothing": {"low": 0.1, "high": 1.0},
    },
    "connected_threshold": {},
    "region_growing": {},
    "threshold_brush": {},
    "geodesic_distance": {
        "geodesic_edge_weight": {"low": 1.0, "high": 15.0},
        "geodesic_smoothing": {"low": 0.1, "high": 1.0},
    },
    "level_set": {
        "level_set_iterations": {"low": 30, "high": 150},
    },
    "random_walker": {
        "random_walker_beta": {"low": 20, "high": 200},
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch optimization across algorithms")
    parser.add_argument("--trials", type=int, default=20, help="Trials per algorithm")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(ALL_ALGORITHMS),
        help="Comma-separated list of algorithms",
    )
    parser.add_argument("--gold-standard", type=str, default="MRBrainTumor1_tumor")
    parser.add_argument("--no-exit", action="store_true", help="Don't exit Slicer")
    return parser.parse_args()


def setup_slicer_scene(sample_data_name: str):
    """Load sample data and set up Slicer scene."""
    import SampleData
    import slicer

    logger.info(f"Loading sample data: {sample_data_name}")
    slicer.mrmlScene.Clear(0)

    volume_node = SampleData.downloadSample(sample_data_name)
    if volume_node is None:
        raise RuntimeError(f"Failed to load sample data: {sample_data_name}")

    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

    segmentation = segmentation_node.GetSegmentation()
    segment_id = segmentation.AddEmptySegment("Optimization")

    return volume_node, segmentation_node, segment_id


def setup_effect(segmentation_node, volume_node):
    """Set up the Adaptive Brush effect."""
    import slicer

    slicer.util.selectModule("SegmentEditor")
    slicer.app.processEvents()

    segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
    segment_editor_widget = segment_editor_module.editor

    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setSourceVolumeNode(volume_node)
    slicer.app.processEvents()

    segment_editor_widget.setActiveEffectByName("Adaptive Brush")
    effect = segment_editor_widget.activeEffect()

    if effect is None:
        raise RuntimeError("Failed to activate Adaptive Brush effect")

    return effect.self(), segment_editor_widget


def load_gold_standard(gold_name: str):
    """Load gold standard segmentation and metadata."""
    import slicer

    tester_path = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester"
    gold_path = tester_path / "GoldStandards" / gold_name

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold standard not found: {gold_path}")

    seg_file = gold_path / "gold.seg.nrrd"
    gold_seg_node = slicer.util.loadSegmentation(str(seg_file))

    metadata_file = gold_path / "metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    gold_segmentation = gold_seg_node.GetSegmentation()
    gold_segment_id = gold_segmentation.GetNthSegmentID(0)

    click_locations = metadata.get("clicks", [])
    return gold_seg_node, gold_segment_id, click_locations, metadata


def clear_segment(segmentation_node, segment_id: str) -> str:
    """Clear all voxels from a segment."""
    segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
    if segment:
        segmentation = segmentation_node.GetSegmentation()
        segment_name = segment.GetName()
        color = segment.GetColor()
        segmentation.RemoveSegment(segment_id)
        new_segment_id = segmentation.AddEmptySegment(segment_name, segment_name, color)
        return str(new_segment_id)
    return segment_id


def compute_dice(
    test_seg_node, test_segment_id, gold_seg_node, gold_segment_id, volume_node
) -> float:
    """Compute Dice coefficient."""
    import numpy as np
    import slicer

    test_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
        test_seg_node, test_segment_id, volume_node
    )
    gold_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
        gold_seg_node, gold_segment_id, volume_node
    )

    test_binary = test_arr > 0
    gold_binary = gold_arr > 0

    intersection = int(np.sum(test_binary & gold_binary))
    union = int(np.sum(test_binary)) + int(np.sum(gold_binary))

    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


def apply_params_to_effect(effect, params: dict[str, Any]) -> None:
    """Apply optimization parameters to effect."""
    import slicer

    if "algorithm" in params:
        effect.setAlgorithm(params["algorithm"])

    if "brush_radius_mm" in params:
        effect.brushRadiusMm = params["brush_radius_mm"]

    if "edge_sensitivity" in params:
        effect.edgeSensitivityValue = params["edge_sensitivity"]

    if "threshold_zone" in params:
        effect.thresholdZone = params["threshold_zone"]
        effect.zoneSlider.value = params["threshold_zone"]

    if "watershed_gradient_scale" in params:
        effect.watershedGradientScale = params["watershed_gradient_scale"]
        effect.watershedGradientScaleSlider.value = params["watershed_gradient_scale"]
    if "geodesic_edge_weight" in params:
        effect.geodesicEdgeWeight = params["geodesic_edge_weight"]
        effect.geodesicEdgeWeightSlider.value = params["geodesic_edge_weight"]
    if "level_set_iterations" in params:
        effect.levelSetIterations = int(params["level_set_iterations"])
        effect.levelSetIterationsSlider.value = params["level_set_iterations"]
    if "random_walker_beta" in params:
        effect.randomWalkerBeta = params["random_walker_beta"]
        if hasattr(effect, "randomWalkerBetaSlider"):
            effect.randomWalkerBetaSlider.value = params["random_walker_beta"]

    slicer.app.processEvents()


def run_single_algorithm_optimization(
    algorithm: str,
    n_trials: int,
    gold_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run optimization for a single algorithm."""
    import slicer

    sys.path.insert(0, str(PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester"))

    from SegmentEditorAdaptiveBrushTesterLib import OPTUNA_AVAILABLE

    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available")
        return {"error": "Optuna not installed"}

    import optuna

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALGORITHM: {algorithm.upper()}")
    logger.info("=" * 60)

    # Load gold standard metadata to get sample data name
    tester_path = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester"
    gold_path = tester_path / "GoldStandards" / gold_name / "metadata.json"
    with open(gold_path) as f:
        gold_metadata = json.load(f)

    sample_data = gold_metadata.get("sample_data", "MRBrainTumor1")

    # Set up scene
    volume_node, segmentation_node, segment_id = setup_slicer_scene(sample_data)

    # Load gold standard after scene setup
    gold_seg_node, gold_segment_id, click_locations, gold_metadata = load_gold_standard(gold_name)

    # Set up effect
    effect, segment_editor_widget = setup_effect(segmentation_node, volume_node)

    # Create Optuna study
    algo_output_dir = output_dir / algorithm
    algo_output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=len(click_locations), reduction_factor=2
        ),
    )

    # Get algorithm-specific params
    algo_params = ALGORITHM_PARAMS.get(algorithm, {})

    best_dice = 0.0
    best_params: dict[str, Any] = {}
    all_trials: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal segment_id, best_dice, best_params

        # Sample common parameters
        params: dict[str, Any] = {
            "algorithm": algorithm,
            "edge_sensitivity": trial.suggest_int("edge_sensitivity", 30, 80),
            "threshold_zone": trial.suggest_int("threshold_zone", 30, 80),
            "brush_radius_mm": trial.suggest_float("brush_radius_mm", 15.0, 35.0, step=5.0),
        }

        # Sample algorithm-specific parameters
        for param_name, bounds in algo_params.items():
            if "iterations" in param_name or "beta" in param_name:
                params[param_name] = trial.suggest_int(param_name, bounds["low"], bounds["high"])
            else:
                params[param_name] = trial.suggest_float(param_name, bounds["low"], bounds["high"])

        logger.info(f"  Trial {trial.number}: {params}")

        # Clear segment
        segment_id = clear_segment(segmentation_node, segment_id)
        segment_editor_widget.setCurrentSegmentID(segment_id)
        slicer.app.processEvents()

        # Apply parameters
        apply_params_to_effect(effect, params)

        # Run clicks with intermediate reporting
        start_time = time.time()
        for i, click in enumerate(click_locations):
            ras = click["ras"]
            effect.paintAt(ras[0], ras[1], ras[2])
            slicer.app.processEvents()

            current_dice = compute_dice(
                segmentation_node, segment_id, gold_seg_node, gold_segment_id, volume_node
            )
            trial.report(current_dice, i)

            if trial.should_prune():
                logger.info(f"    Pruned at click {i + 1}")
                raise optuna.TrialPruned()

        final_dice = compute_dice(
            segmentation_node, segment_id, gold_seg_node, gold_segment_id, volume_node
        )
        elapsed = time.time() - start_time

        logger.info(f"    Dice: {final_dice:.4f} ({elapsed * 1000:.0f}ms)")

        # Track best
        if final_dice > best_dice:
            best_dice = final_dice
            best_params = params.copy()

        all_trials.append(
            {
                "trial": trial.number,
                "params": params,
                "dice": final_dice,
                "duration_ms": elapsed * 1000,
                "pruned": False,
            }
        )

        return final_dice

    # Run optimization
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    # Compute parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        importance = {}

    # Save results
    results = {
        "algorithm": algorithm,
        "gold_standard": gold_name,
        "n_trials": n_trials,
        "best_dice": best_dice,
        "best_params": best_params,
        "parameter_importance": importance,
        "trials": all_trials,
        "completed_at": datetime.now().isoformat(),
    }

    results_path = algo_output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"  Best Dice: {best_dice:.4f}")
    logger.info(f"  Results saved to: {results_path}")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",")]
    invalid = set(algorithms) - set(ALL_ALGORITHMS)
    if invalid:
        logger.error(f"Invalid algorithms: {invalid}")
        logger.error(f"Valid algorithms: {ALL_ALGORITHMS}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = PROJECT_ROOT / "optimization_results" / f"{timestamp}_batch_all_algorithms"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Batch optimization: {len(algorithms)} algorithms, {args.trials} trials each")
    logger.info(f"Gold standard: {args.gold_standard}")
    logger.info(f"Output: {output_dir}")

    all_results = {}
    summary = []

    for algorithm in algorithms:
        try:
            results = run_single_algorithm_optimization(
                algorithm=algorithm,
                n_trials=args.trials,
                gold_name=args.gold_standard,
                output_dir=output_dir,
            )
            all_results[algorithm] = results
            summary.append(
                {
                    "algorithm": algorithm,
                    "best_dice": results.get("best_dice", 0),
                    "best_params": results.get("best_params", {}),
                }
            )
        except Exception as e:
            logger.exception(f"Failed optimization for {algorithm}: {e}")
            all_results[algorithm] = {"error": str(e)}
            summary.append({"algorithm": algorithm, "best_dice": 0, "error": str(e)})

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "gold_standard": args.gold_standard,
                "trials_per_algorithm": args.trials,
                "algorithms": summary,
                "completed_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("BATCH OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Algorithm':<25} {'Best Dice':>10}")
    logger.info("-" * 40)

    sorted_summary = sorted(summary, key=lambda x: x.get("best_dice", 0), reverse=True)
    for s in sorted_summary:
        dice = s.get("best_dice", 0)
        if "error" in s:
            logger.info(f"{s['algorithm']:<25} {'ERROR':>10}")
        else:
            logger.info(f"{s['algorithm']:<25} {dice:>10.4f}")

    logger.info("-" * 40)
    logger.info(f"Results saved to: {output_dir}")

    # Generate markdown report
    generate_batch_report(output_dir, sorted_summary, args)

    if not args.no_exit:
        import slicer

        slicer.app.quit()


def generate_batch_report(output_dir: Path, summary: list, args) -> None:
    """Generate markdown report for batch optimization."""
    report_path = output_dir / "batch_report.md"

    with open(report_path, "w") as f:
        f.write("# Batch Algorithm Optimization Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Gold Standard:** {args.gold_standard}\n")
        f.write(f"**Trials per Algorithm:** {args.trials}\n\n")

        f.write("## Algorithm Ranking\n\n")
        f.write("| Rank | Algorithm | Best Dice |\n")
        f.write("|------|-----------|----------|\n")
        for i, s in enumerate(summary, 1):
            dice = s.get("best_dice", 0)
            if "error" in s:
                f.write(f"| {i} | {s['algorithm']} | ERROR |\n")
            else:
                f.write(f"| {i} | {s['algorithm']} | {dice:.4f} |\n")

        f.write("\n## Best Parameters by Algorithm\n\n")
        for s in summary:
            if "error" not in s and s.get("best_params"):
                f.write(f"### {s['algorithm']}\n\n")
                f.write("```json\n")
                f.write(json.dumps(s["best_params"], indent=2))
                f.write("\n```\n\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
