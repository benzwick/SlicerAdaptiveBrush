#!/usr/bin/env python
"""Run Optuna parameter optimization inside Slicer.

This script is designed to be run via:
    Slicer --python-script scripts/run_optimization.py <config.yaml> [--trials N]

It loads a YAML configuration, sets up the adaptive brush effect,
and runs optimization trials using Optuna's TPE sampler with pruning.

Example:
    Slicer --python-script scripts/run_optimization.py configs/quick_test.yaml
    Slicer --python-script scripts/run_optimization.py configs/tumor_optimization.yaml --trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimization")


def get_project_root() -> Path:
    """Get the project root directory.

    The project root contains SegmentEditorAdaptiveBrushTester/.
    """
    # scripts/run_optimization.py -> parent is scripts/ -> parent is project root
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


PROJECT_ROOT = get_project_root()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Optuna optimization for Adaptive Brush")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--trials", type=int, help="Override number of trials")
    parser.add_argument("--timeout", type=int, help="Timeout in minutes")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from previous study")
    parser.add_argument("--no-exit", action="store_true", help="Don't exit Slicer after completion")
    return parser.parse_args()


def setup_slicer_scene(sample_data_name: str):
    """Load sample data and set up Slicer scene.

    Args:
        sample_data_name: Name of Slicer SampleData to load.

    Returns:
        Tuple of (volume_node, segmentation_node, segment_id).
    """
    import SampleData
    import slicer

    logger.info(f"Loading sample data: {sample_data_name}")
    slicer.mrmlScene.Clear(0)

    volume_node = SampleData.downloadSample(sample_data_name)
    if volume_node is None:
        raise RuntimeError(f"Failed to load sample data: {sample_data_name}")

    logger.info(f"Loaded volume: {volume_node.GetName()}")

    # Create segmentation
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

    # Add segment
    segmentation = segmentation_node.GetSegmentation()
    segment_id = segmentation.AddEmptySegment("Optimization")

    return volume_node, segmentation_node, segment_id


def setup_effect(segmentation_node, volume_node):
    """Set up the Adaptive Brush effect.

    Args:
        segmentation_node: Segmentation MRML node.
        volume_node: Volume MRML node.

    Returns:
        The scripted effect instance.
    """
    import slicer

    # Switch to Segment Editor module
    slicer.util.selectModule("SegmentEditor")
    slicer.app.processEvents()

    # Get segment editor widget
    segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
    segment_editor_widget = segment_editor_module.editor

    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setSourceVolumeNode(volume_node)
    slicer.app.processEvents()

    # Activate Adaptive Brush
    segment_editor_widget.setActiveEffectByName("Adaptive Brush")
    effect = segment_editor_widget.activeEffect()

    if effect is None:
        raise RuntimeError("Failed to activate Adaptive Brush effect")

    return effect.self(), segment_editor_widget


def apply_params_to_effect(effect, params: dict[str, Any]) -> None:
    """Apply optimization parameters to effect.

    Args:
        effect: The scripted Adaptive Brush effect.
        params: Dictionary of parameter name -> value.
    """
    # Algorithm selection
    if "algorithm" in params:
        effect.setAlgorithm(params["algorithm"])

    # Brush radius
    if "brush_radius_mm" in params:
        effect.brushRadiusMm = params["brush_radius_mm"]

    # Edge sensitivity
    if "edge_sensitivity" in params:
        effect.edgeSensitivityValue = params["edge_sensitivity"]

    # Threshold zone (inner radius as percentage 0-100)
    if "threshold_zone" in params:
        effect.thresholdZone = params["threshold_zone"]
        effect.zoneSlider.value = params["threshold_zone"]

    # Algorithm-specific parameters - also update sliders so UI reflects changes
    if "watershed_gradient_scale" in params:
        effect.watershedGradientScale = params["watershed_gradient_scale"]
        effect.watershedGradientScaleSlider.value = params["watershed_gradient_scale"]
    if "geodesic_edge_weight" in params:
        effect.geodesicEdgeWeight = params["geodesic_edge_weight"]
        effect.geodesicEdgeWeightSlider.value = params["geodesic_edge_weight"]
    if "level_set_iterations" in params:
        effect.levelSetIterations = int(params["level_set_iterations"])
        effect.levelSetIterationsSlider.value = params["level_set_iterations"]

    import slicer

    slicer.app.processEvents()


def run_clicks(effect, click_locations: list[dict], segment_id: str, segmentation_node) -> int:
    """Run a series of clicks and return total voxel count.

    Args:
        effect: The scripted effect.
        click_locations: List of click dicts with "ras" key.
        segment_id: Segment ID to paint into.
        segmentation_node: Segmentation node.

    Returns:
        Total voxels after all clicks.
    """
    import numpy as np
    import slicer

    for click in click_locations:
        ras = click["ras"]
        effect.paintAt(ras[0], ras[1], ras[2])
        slicer.app.processEvents()

    # Count voxels
    try:
        volume_node = effect.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        arr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id, volume_node)
        return int(np.sum(arr > 0))
    except Exception as e:
        logger.warning(f"Could not count voxels: {e}")
        return 0


def compute_dice(
    test_seg_node, test_segment_id: str, gold_seg_node, gold_segment_id: str, volume_node
) -> float:
    """Compute Dice coefficient between test and gold segmentations.

    Args:
        test_seg_node: Test segmentation node.
        test_segment_id: Test segment ID.
        gold_seg_node: Gold standard segmentation node.
        gold_segment_id: Gold segment ID.
        volume_node: Volume node providing geometry.

    Returns:
        Dice coefficient (0-1).
    """
    import numpy as np
    import slicer

    try:
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
            return 1.0  # Both empty

        return float(2.0 * intersection / union)
    except Exception as e:
        logger.error(f"Error computing Dice: {e}")
        return 0.0


def load_gold_standard(gold_name: str):
    """Load gold standard segmentation and metadata.

    Args:
        gold_name: Name of the gold standard.

    Returns:
        Tuple of (gold_seg_node, gold_segment_id, click_locations, metadata).
    """
    import slicer

    # Find gold standard path
    tester_path = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester"
    gold_path = tester_path / "GoldStandards" / gold_name

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold standard not found: {gold_path}")

    # Load segmentation
    seg_file = gold_path / "gold.seg.nrrd"
    gold_seg_node = slicer.util.loadSegmentation(str(seg_file))

    # Load metadata
    metadata_file = gold_path / "metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Get segment ID (first segment in the gold standard)
    gold_segmentation = gold_seg_node.GetSegmentation()
    gold_segment_id = gold_segmentation.GetNthSegmentID(0)

    click_locations = metadata.get("clicks", [])

    logger.info(f"Loaded gold standard: {gold_name} with {len(click_locations)} clicks")

    return gold_seg_node, gold_segment_id, click_locations, metadata


def clear_segment(segmentation_node, segment_id: str) -> str:
    """Clear all voxels from a segment by removing and re-adding it.

    Args:
        segmentation_node: Segmentation node.
        segment_id: Segment ID to clear.

    Returns:
        The new segment ID (may differ from input if segment was recreated).
    """
    try:
        # Get the segment's binary labelmap representation
        segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
        if segment:
            # Remove and re-add segment to clear it
            segmentation = segmentation_node.GetSegmentation()
            segment_name = segment.GetName()
            color = segment.GetColor()
            segmentation.RemoveSegment(segment_id)
            new_segment_id = segmentation.AddEmptySegment(segment_name, segment_name, color)
            return str(new_segment_id)
    except Exception as e:
        logger.warning(f"Could not clear segment: {e}")
    return segment_id


def run_optimization(
    config_path: Path,
    n_trials_override: int | None = None,
    timeout_override: int | None = None,
    output_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the optimization.

    Args:
        config_path: Path to YAML config file.
        n_trials_override: Override number of trials.
        timeout_override: Override timeout in minutes.
        output_dir: Output directory.
        resume: Whether to resume from previous study.

    Returns:
        Results dictionary.
    """
    import slicer

    # Add tester lib to path
    tester_path = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester"
    sys.path.insert(0, str(tester_path))

    from SegmentEditorAdaptiveBrushTesterLib import (
        OPTUNA_AVAILABLE,
        OptimizationConfig,
        OptunaOptimizer,
    )

    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available. Install with: pip install optuna")
        return {"error": "Optuna not installed"}

    import optuna

    # Load config
    config = OptimizationConfig.load(config_path)
    logger.info(f"Loaded config: {config.name}")

    # Apply overrides
    if n_trials_override is not None:
        config.n_trials = n_trials_override
    if timeout_override is not None:
        config.timeout_minutes = timeout_override
    logger.info(f"Running {config.n_trials} trials")

    # Get recipe info (first recipe)
    if not config.recipes:
        raise ValueError("Config must specify at least one recipe")

    recipe_spec = config.recipes[0]
    recipe_path = tester_path / recipe_spec.path

    # Load recipe to get sample data name
    from SegmentEditorAdaptiveBrushTesterLib import Recipe

    recipe = Recipe.load(recipe_path)
    logger.info(f"Recipe: {recipe.name}, sample_data={recipe.sample_data}")

    # Determine gold standard
    # recipe.gold_standard is just the name (e.g., "MRBrainTumor1_tumor")
    # recipe_spec.gold_standard from config is a full path like "GoldStandards/X/gold.seg.nrrd"
    gold_name = recipe.gold_standard
    if recipe_spec.gold_standard:
        # Extract directory name from path like "GoldStandards/MRBrainTumor1_tumor/gold.seg.nrrd"
        gold_path = recipe_spec.gold_standard
        if gold_path.name == "gold.seg.nrrd":
            gold_name = gold_path.parent.name
        else:
            # Might be just the directory name
            gold_name = str(gold_path.stem)

    if not gold_name:
        raise ValueError("No gold standard specified in recipe or config")

    # Set up scene FIRST (it clears the scene)
    logger.info(f"Loading sample data: {recipe.sample_data}")
    volume_node, segmentation_node, segment_id = setup_slicer_scene(recipe.sample_data)

    # Load gold standard AFTER scene setup (so it doesn't get cleared)
    logger.info(f"Loading gold standard: {gold_name}")
    gold_seg_node, gold_segment_id, click_locations, gold_metadata = load_gold_standard(gold_name)
    logger.info(f"Gold standard has {len(click_locations)} clicks")

    # Set up effect
    effect, segment_editor_widget = setup_effect(segmentation_node, volume_node)

    # Create optimizer
    optimizer = OptunaOptimizer(config, output_dir=output_dir)
    logger.info(f"Output directory: {optimizer.output_dir}")

    if resume:
        optimizer.resume()
    else:
        optimizer.create_study()

    # Get brush radius from gold standard metadata (default if not optimizing it)
    gold_brush_radius = gold_metadata.get("parameters", {}).get("brush_radius_mm", 25.0)

    # Define objective function
    def objective(trial: optuna.Trial, params: dict[str, Any]) -> float:
        """Objective function for optimization."""
        nonlocal segment_id

        trial_start = time.time()

        logger.info(f"Trial {trial.number}: {params}")

        # Clear segment for fresh start
        segment_id = clear_segment(segmentation_node, segment_id)
        segment_editor_widget.setCurrentSegmentID(segment_id)
        slicer.app.processEvents()

        # Apply brush radius from gold standard if not in params
        if "brush_radius_mm" not in params:
            effect.brushRadiusMm = gold_brush_radius

        # Apply parameters
        apply_params_to_effect(effect, params)

        # Run clicks one by one and report intermediate values
        for i, click in enumerate(click_locations):
            ras = click["ras"]
            effect.paintAt(ras[0], ras[1], ras[2])
            slicer.app.processEvents()

            # Compute intermediate Dice
            current_dice = compute_dice(
                segmentation_node, segment_id, gold_seg_node, gold_segment_id, volume_node
            )

            # Report intermediate value for pruning
            trial.report(current_dice, i)

            logger.info(f"  Click {i + 1}/{len(click_locations)}: Dice={current_dice:.4f}")

            # Check for pruning
            if trial.should_prune():
                logger.info(f"  Trial pruned at click {i + 1}")
                raise optuna.TrialPruned()

        # Final Dice
        final_dice = compute_dice(
            segmentation_node, segment_id, gold_seg_node, gold_segment_id, volume_node
        )

        elapsed = time.time() - trial_start
        logger.info(f"  Final Dice: {final_dice:.4f} ({elapsed * 1000:.0f}ms)")

        # Store additional info
        trial.set_user_attr("duration_ms", elapsed * 1000)
        trial.set_user_attr("n_clicks", len(click_locations))

        return final_dice

    # Run optimization
    logger.info(f"Starting optimization: {config.n_trials} trials")
    results = optimizer.optimize(objective, n_trials=config.n_trials)

    # Report results
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    if results.best_trial:
        logger.info(f"Best Dice: {results.best_trial.value:.4f}")
        logger.info(f"Best params: {results.best_trial.params}")
    logger.info(f"Total trials: {results.n_trials}")
    logger.info(f"Pruned trials: {sum(1 for t in results.trials if t.pruned)}")
    logger.info(f"Duration: {results.duration_seconds:.1f}s")
    logger.info(f"Results saved to: {optimizer.output_dir}")

    if results.parameter_importance:
        logger.info("\nParameter Importance:")
        for name, importance in sorted(results.parameter_importance.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: {importance:.3f}")

    # Generate lab notebook
    generate_lab_notebook(optimizer.output_dir, results, config, gold_metadata)

    return dict(results.to_dict())  # type: ignore[arg-type]


def generate_lab_notebook(output_dir: Path, results, config, gold_metadata: dict) -> None:
    """Generate human-readable lab notebook markdown.

    Args:
        output_dir: Output directory.
        results: OptimizationResults.
        config: OptimizationConfig.
        gold_metadata: Gold standard metadata.
    """
    notebook_path = output_dir / "lab_notebook.md"

    with open(notebook_path, "w") as f:
        f.write(f"# {config.name} - Optimization Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Config:** {config.source_path}\n")
        f.write(f"**Gold Standard:** {gold_metadata.get('description', 'N/A')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Best Dice:** {results.best_trial.value:.4f}\n")
        f.write(f"- **Total Trials:** {results.n_trials}\n")
        f.write(f"- **Pruned Trials:** {sum(1 for t in results.trials if t.pruned)}\n")
        f.write(f"- **Duration:** {results.duration_seconds:.1f}s\n\n")

        f.write("## Best Parameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(results.best_trial.params, indent=2))
        f.write("\n```\n\n")

        if results.parameter_importance:
            f.write("## Parameter Importance\n\n")
            f.write("| Parameter | Importance |\n")
            f.write("|-----------|------------|\n")
            for name, importance in sorted(
                results.parameter_importance.items(), key=lambda x: -x[1]
            ):
                f.write(f"| {name} | {importance:.3f} |\n")
            f.write("\n")

        f.write("## Top 5 Trials\n\n")
        sorted_trials = sorted(
            [t for t in results.trials if not t.pruned],
            key=lambda t: t.value,
            reverse=True,
        )[:5]

        f.write("| Trial | Dice | Parameters |\n")
        f.write("|-------|------|------------|\n")
        for t in sorted_trials:
            params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
            f.write(f"| {t.trial_number} | {t.value:.4f} | {params_str} |\n")

        f.write("\n## Trial History\n\n")
        f.write("| Trial | Dice | Pruned | Duration (ms) |\n")
        f.write("|-------|------|--------|---------------|\n")
        for t in results.trials:
            f.write(f"| {t.trial_number} | {t.value:.4f} | {t.pruned} | {t.duration_ms:.0f} |\n")

    logger.info(f"Lab notebook saved to: {notebook_path}")


def main():
    """Main entry point."""
    logger.info("Starting optimization script...")
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

    args = parse_args()

    # Resolve config path relative to PROJECT_ROOT if not absolute
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Config: {config_path}")
    try:
        run_optimization(
            config_path=config_path,
            n_trials_override=args.trials,
            timeout_override=args.timeout,
            output_dir=args.output,
            resume=args.resume,
        )

        if not args.no_exit:
            import slicer

            logger.info("Optimization complete. Exiting Slicer...")
            slicer.app.quit()
    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        if not args.no_exit:
            import slicer

            slicer.app.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
