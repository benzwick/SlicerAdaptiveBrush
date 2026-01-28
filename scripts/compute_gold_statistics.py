#!/usr/bin/env python
"""Compute and update statistics for gold standard segmentations.

This script computes statistics (voxel count, volume, bounding box, centroid,
checksum) for gold standards and updates their metadata.json files.

Usage:
    Slicer --python-script scripts/compute_gold_statistics.py [gold_name]

If no gold_name is provided, updates all gold standards.

Examples:
    # Update specific gold standard
    Slicer --python-script scripts/compute_gold_statistics.py MRBrainTumor1_tumor

    # Update all gold standards
    Slicer --python-script scripts/compute_gold_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester"))

GOLD_DIR = PROJECT_ROOT / "SegmentEditorAdaptiveBrushTester" / "GoldStandards"


def log(msg: str) -> None:
    """Log message with immediate flush."""
    print(msg, flush=True)


def compute_statistics_for_gold(gold_name: str) -> bool:
    """Compute and save statistics for a gold standard.

    Args:
        gold_name: Name of the gold standard directory.

    Returns:
        True if successful.
    """
    import json

    import SampleData
    import slicer
    from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

    gold_path = GOLD_DIR / gold_name
    metadata_file = gold_path / "metadata.json"

    if not metadata_file.exists():
        log(f"Metadata not found: {metadata_file}")
        return False

    # Load metadata to get sample data name
    with open(metadata_file) as f:
        metadata = json.load(f)

    sample_name = metadata.get("sample_data", metadata.get("volume", {}).get("name"))
    if not sample_name:
        log(f"No sample_data in metadata for {gold_name}")
        return False

    # Load volume
    log(f"Loading sample data: {sample_name}")
    volume_node = SampleData.downloadSample(sample_name)
    if not volume_node:
        log(f"Could not load sample data: {sample_name}")
        return False

    # Load gold standard (without verification since we're computing stats)
    manager = GoldStandardManager()
    log(f"Loading gold standard: {gold_name}")
    seg_node, _ = manager.load_gold(gold_name, volume_node=None, verify=False)

    # Compute and save statistics
    log("Computing statistics...")
    stats = manager.update_statistics(gold_name, seg_node, volume_node)

    log(f"\nStatistics for {gold_name}:")
    log(manager.format_statistics(stats))
    log(f"Checksum: {stats['checksum_sha256'][:16]}...")

    # Cleanup
    slicer.mrmlScene.RemoveNode(seg_node)
    slicer.mrmlScene.RemoveNode(volume_node)

    return True


def main():
    """Main entry point."""
    import slicer

    args = sys.argv[1:]

    if args:
        gold_names = args
    else:
        # Find all gold standards
        gold_names = [
            p.name for p in GOLD_DIR.iterdir() if p.is_dir() and (p / "gold.seg.nrrd").exists()
        ]

    if not gold_names:
        log("No gold standards found")
        slicer.app.quit()
        return

    log(f"Computing statistics for {len(gold_names)} gold standard(s)\n")

    success = 0
    failed = 0

    for gold_name in gold_names:
        log(f"--- {gold_name} ---")
        try:
            if compute_statistics_for_gold(gold_name):
                success += 1
            else:
                failed += 1
        except Exception as e:
            log(f"Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        log("")

    log(f"Complete: {success} succeeded, {failed} failed")
    slicer.app.quit()


if __name__ == "__main__":
    main()
