#!/usr/bin/env python3
"""Generate segmentation review images for VLM review.

Simple usage - just pass the trial segmentation:
    Slicer --python-script scripts/run_review_screenshots.py trial_079

The script auto-detects:
- Gold standard from optimization config
- Output directory (review/ folder alongside segmentations/)
- Volume from sample data name
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OPTIMIZATION_RESULTS = PROJECT_ROOT / "optimization_results"
GOLD_STANDARDS = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester" / "GoldStandards"


def find_trial_segmentation(trial_id: str) -> Path | None:
    """Find trial segmentation file by ID (e.g., 'trial_079' or '079' or '79')."""
    # Normalize trial ID
    if trial_id.startswith("trial_"):
        trial_num = trial_id.replace("trial_", "")
    else:
        trial_num = trial_id

    trial_num = trial_num.zfill(3)  # Pad to 3 digits
    filename = f"trial_{trial_num}.seg.nrrd"

    # Search in most recent optimization results first
    results_dirs = sorted(OPTIMIZATION_RESULTS.glob("*"), reverse=True)
    for results_dir in results_dirs:
        seg_path = results_dir / "segmentations" / filename
        if seg_path.exists():
            return seg_path

    return None


def find_gold_standard(optimization_dir: Path) -> Path | None:
    """Find gold standard from optimization config."""
    import yaml  # type: ignore[import-untyped]

    config_path = optimization_dir / "config.yaml"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Try recipes[0].gold_standard (relative path in config)
    recipes = config.get("recipes", [])
    if recipes and "gold_standard" in recipes[0]:
        gold_rel_path: str = recipes[0]["gold_standard"]
        # Config paths are relative to SegmentEditorAdaptiveBrushTester/
        gold_path = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester" / gold_rel_path
        if gold_path.exists():
            return Path(gold_path)

    # Try gold_standard.name (old format)
    gold_name = config.get("gold_standard", {}).get("name")
    if gold_name:
        gold_path = GOLD_STANDARDS / gold_name / "gold.seg.nrrd"
        if gold_path.exists():
            return Path(gold_path)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation review images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review trial 79 from most recent optimization
  Slicer --python-script scripts/run_review_screenshots.py 79

  # Review specific trial file
  Slicer --python-script scripts/run_review_screenshots.py /path/to/trial_079.seg.nrrd
""",
    )
    parser.add_argument("trial", help="Trial ID (e.g., '79') or full path to segmentation")
    parser.add_argument("--no-gold", action="store_true", help="Don't include gold standard")

    args = parser.parse_args()

    # Find the segmentation file
    if Path(args.trial).exists():
        seg_path = Path(args.trial)
    else:
        seg_path = find_trial_segmentation(args.trial)
        if seg_path is None:
            logger.error(f"Could not find trial: {args.trial}")
            sys.exit(1)

    logger.info(f"Found segmentation: {seg_path}")

    # Determine optimization directory and output path
    optimization_dir = seg_path.parent.parent
    output_dir = optimization_dir / "review" / seg_path.stem

    # Find gold standard
    gold_path = None
    if not args.no_gold:
        gold_path = find_gold_standard(optimization_dir)
        if gold_path:
            logger.info(f"Found gold standard: {gold_path}")
        else:
            logger.warning("No gold standard found")

    # Import and run
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_segmentation_review_screenshots import generate_review_images

    logger.info(f"Output: {output_dir}")

    manifest = generate_review_images(
        segmentation_path=seg_path,
        output_dir=output_dir,
        gold_standard_path=gold_path,
    )

    print(f"\nImages saved to: {output_dir}")
    print(f"Manifest: {manifest}")

    # Exit Slicer
    import slicer

    slicer.app.exit()


if __name__ == "__main__":
    main()
