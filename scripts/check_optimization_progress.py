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


def main():
    parser = argparse.ArgumentParser(description="Check optimization progress")
    parser.add_argument("results_dir", nargs="?", type=Path, help="Results directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
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


if __name__ == "__main__":
    main()
