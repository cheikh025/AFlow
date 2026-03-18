"""Reset an AFlow experiment to a clean state.

Keeps: workspace/{dataset}/workflows/template/
       workspace/{dataset}/workflows/round_1/
Deletes: all round_N folders (N >= 2)
         results.json
         processed_experience.json

Usage:
    python reset_experiment.py --dataset MMLU
    python reset_experiment.py --dataset MATH
    python reset_experiment.py --dataset MMLU --dry-run
"""

import argparse
import shutil
from pathlib import Path


def reset_experiment(dataset: str, dry_run: bool = False):
    workflows_path = Path("workspace") / dataset / "workflows"

    if not workflows_path.exists():
        print(f"[ERROR] Path not found: {workflows_path}")
        return

    deleted = []

    # Clean generated artifacts from round_1 (keep graph.py, prompt.py, __init__.py)
    ROUND1_KEEP = {"graph.py", "prompt.py", "__init__.py"}
    round_1_path = workflows_path / "round_1"
    if round_1_path.exists():
        for entry in round_1_path.iterdir():
            if entry.name not in ROUND1_KEEP:
                deleted.append(str(entry))
                if not dry_run:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()

    # Remove round_N directories (N >= 2)
    for entry in sorted(workflows_path.iterdir()):
        if entry.is_dir() and entry.name.startswith("round_"):
            try:
                round_num = int(entry.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if round_num >= 2:
                deleted.append(str(entry))
                if not dry_run:
                    shutil.rmtree(entry)

    # Remove optimizer state files
    for filename in ["results.json", "processed_experience.json"]:
        target = workflows_path / filename
        if target.exists():
            deleted.append(str(target))
            if not dry_run:
                target.unlink()

    if not deleted:
        print(f"[{dataset}] Nothing to clean — already at initial state.")
        return

    prefix = "[DRY RUN] Would delete" if dry_run else "Deleted"
    for path in deleted:
        print(f"  {prefix}: {path}")

    if not dry_run:
        print(f"\n[{dataset}] Reset complete. Ready to restart from round 1.")
    else:
        print(f"\n[{dataset}] Dry run done. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset an AFlow experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. MMLU, MATH)")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted without deleting")
    args = parser.parse_args()

    reset_experiment(dataset=args.dataset, dry_run=args.dry_run)
