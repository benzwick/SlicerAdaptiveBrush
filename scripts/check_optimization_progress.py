#!/usr/bin/env python3
"""Check optimization progress from Optuna database.

Usage:
    python scripts/check_optimization_progress.py [results_dir]

If no results_dir is specified, uses the most recent one in optimization_results/.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def get_latest_results_dir() -> Path | None:
    """Find the most recent optimization results directory."""
    results_root = Path(__file__).parent.parent / "optimization_results"
    if not results_root.exists():
        return None
    dirs = sorted(results_root.iterdir(), key=lambda d: d.name, reverse=True)
    return dirs[0] if dirs else None


def check_progress(results_dir: Path) -> dict:
    """Query optimization progress from Optuna database.

    Args:
        results_dir: Path to optimization results directory.

    Returns:
        Dictionary with progress info.
    """
    db_path = results_dir / "optuna_study.db"
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get trial counts by state
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END) as complete,
            SUM(CASE WHEN state='PRUNED' THEN 1 ELSE 0 END) as pruned,
            SUM(CASE WHEN state='RUNNING' THEN 1 ELSE 0 END) as running,
            SUM(CASE WHEN state='FAIL' THEN 1 ELSE 0 END) as failed
        FROM trials
    """)
    row = cursor.fetchone()
    total, complete, pruned, running, failed = row

    # Get best trial
    cursor.execute("""
        SELECT trial_id, value
        FROM trial_values
        WHERE value = (SELECT MAX(value) FROM trial_values)
        LIMIT 1
    """)
    best_row = cursor.fetchone()
    best_trial_id, best_value = best_row if best_row else (None, None)

    # Get best trial params if exists
    best_params = {}
    if best_trial_id is not None:
        cursor.execute(
            """
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """,
            (best_trial_id,),
        )
        best_params = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    return {
        "total": total,
        "complete": complete,
        "pruned": pruned,
        "running": running,
        "failed": failed,
        "best_trial": best_trial_id,
        "best_value": best_value,
        "best_params": best_params,
    }


def get_algorithm_names(results_dir: Path) -> dict[int, str]:
    """Load algorithm name mapping from config.yaml.

    Args:
        results_dir: Path to optimization results directory.

    Returns:
        Dict mapping algorithm index to name.
    """
    config_path = results_dir / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        import yaml  # type: ignore[import-untyped]

        with open(config_path) as f:
            config = yaml.safe_load(f)
        candidates = (
            config.get("parameter_space", {})
            .get("algorithm_substitution", {})
            .get("candidates", [])
        )
        return dict(enumerate(candidates))
    except Exception:
        return {}


def get_algorithm_breakdown(results_dir: Path) -> list[dict]:
    """Get performance breakdown by algorithm.

    Args:
        results_dir: Path to optimization results directory.

    Returns:
        List of dicts with per-algorithm stats.
    """
    db_path = results_dir / "optuna_study.db"
    if not db_path.exists():
        return []

    # Load algorithm name mapping
    algo_names = get_algorithm_names(results_dir)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            tp.param_value as algorithm,
            COUNT(*) as trials,
            SUM(CASE WHEN t.state='COMPLETE' THEN 1 ELSE 0 END) as complete,
            SUM(CASE WHEN t.state='PRUNED' THEN 1 ELSE 0 END) as pruned,
            MAX(tv.value) as best_dice,
            AVG(CASE WHEN t.state='COMPLETE' THEN tv.value END) as avg_dice
        FROM trial_params tp
        JOIN trials t ON tp.trial_id = t.trial_id
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE tp.param_name = 'algorithm'
        GROUP BY tp.param_value
        ORDER BY best_dice DESC
    """)

    results = []
    for row in cursor.fetchall():
        algo_idx, trials, complete, pruned, best, avg = row
        # Convert algorithm index to name
        algo_name = algo_names.get(int(float(algo_idx)), str(algo_idx))
        results.append(
            {
                "algorithm": algo_name,
                "trials": trials,
                "complete": complete,
                "pruned": pruned,
                "best_dice": best,
                "avg_dice": avg,
            }
        )

    conn.close()
    return results


def get_top_trials_for_review(results_dir: Path, top_n: int = 10) -> list[dict]:
    """Get top trials that should be reviewed for gold standard update.

    Args:
        results_dir: Path to optimization results directory.
        top_n: Number of top trials to return.

    Returns:
        List of dicts with trial info, sorted by Dice descending.
    """
    db_path = results_dir / "optuna_study.db"
    if not db_path.exists():
        return []

    algo_names = get_algorithm_names(results_dir)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get top completed trials by Dice score
    cursor.execute(
        """
        SELECT
            t.trial_id,
            tv.value as dice,
            GROUP_CONCAT(tp.param_name || '=' || tp.param_value, ', ') as params
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        JOIN trial_params tp ON t.trial_id = tp.trial_id
        WHERE t.state = 'COMPLETE'
        GROUP BY t.trial_id
        ORDER BY tv.value DESC
        LIMIT ?
    """,
        (top_n,),
    )

    results = []
    for row in cursor.fetchall():
        trial_id, dice, params_str = row

        # Parse params and convert algorithm index to name
        params = {}
        for param in params_str.split(", "):
            if "=" in param:
                k, v = param.split("=", 1)
                if k == "algorithm":
                    v = algo_names.get(int(float(v)), v)
                params[k] = v

        # Check if segmentation file exists
        seg_path = results_dir / "segmentations" / f"trial_{trial_id:03d}.seg.nrrd"

        results.append(
            {
                "trial_id": trial_id,
                "dice": dice,
                "params": params,
                "segmentation_path": str(seg_path) if seg_path.exists() else None,
            }
        )

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Check optimization progress")
    parser.add_argument("results_dir", nargs="?", type=Path, help="Results directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--by-algorithm", action="store_true", help="Show breakdown by algorithm")
    parser.add_argument(
        "--review",
        type=int,
        nargs="?",
        const=10,
        metavar="N",
        help="Show top N trials to review for gold standard update (default: 10)",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = get_latest_results_dir()
        if results_dir is None:
            print("No optimization results found")
            sys.exit(1)

    print(f"Results: {results_dir.name}")
    print("-" * 50)

    progress = check_progress(results_dir)

    if "error" in progress:
        print(f"Error: {progress['error']}")
        sys.exit(1)

    if args.json:
        import json

        print(json.dumps(progress, indent=2))
    else:
        total = progress["total"]
        complete = progress["complete"]
        pruned = progress["pruned"]
        running = progress["running"]
        failed = progress["failed"]

        print(f"Total trials:     {total}")
        print(f"  Complete:       {complete}")
        print(f"  Pruned:         {pruned}")
        print(f"  Running:        {running}")
        print(f"  Failed:         {failed}")
        print()

        if progress["best_value"] is not None:
            print(f"Best trial: #{progress['best_trial']}")
            print(f"Best Dice:  {progress['best_value']:.4f}")
            print("Best params:")
            for k, v in sorted(progress["best_params"].items()):
                print(f"  {k}: {v}")

    if args.by_algorithm:
        print()
        print("By Algorithm:")
        print("-" * 80)
        print(
            f"{'Algorithm':<20} {'Trials':>8} {'Complete':>10} "
            f"{'Pruned':>8} {'Best Dice':>12} {'Avg Dice':>12}"
        )
        print("-" * 80)

        for algo_stats in get_algorithm_breakdown(results_dir):
            algo = algo_stats["algorithm"]
            trials = algo_stats["trials"]
            complete = algo_stats["complete"]
            pruned = algo_stats["pruned"]
            best = algo_stats["best_dice"]
            avg = algo_stats["avg_dice"]
            avg_str = f"{avg:.4f}" if avg else "N/A"
            best_str = f"{best:.4f}" if best else "N/A"
            print(
                f"{algo:<20} {trials:>8} {complete:>10} "
                f"{pruned:>8} {best_str:>12} {avg_str:>12}"
            )

    if args.review:
        print()
        print("=" * 80)
        print(f"TOP {args.review} TRIALS FOR GOLD STANDARD REVIEW")
        print("=" * 80)
        print()
        print("These segmentations may be BETTER than the current gold standard.")
        print("Review them visually and consider updating the gold standard if appropriate.")
        print()

        top_trials = get_top_trials_for_review(results_dir, args.review)
        for i, trial in enumerate(top_trials, 1):
            print(f"{i}. Trial #{trial['trial_id']} - Dice: {trial['dice']:.4f}")
            print(f"   Algorithm: {trial['params'].get('algorithm', 'N/A')}")
            if trial["segmentation_path"]:
                print(f"   File: {trial['segmentation_path']}")
            print()


if __name__ == "__main__":
    main()
