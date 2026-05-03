"""Build MATH validation JSONL for AFlow experiments.

Uses the exact same subjects, level filter, seed, and sample count as MAS optimizer
so both systems evaluate on identical queries during the search phase.

MAS settings (from experiment_math.py + subject_math_benchmark.py):
  - Subjects : Prealgebra, Number Theory, Precalculus, Counting & Probability
  - Level    : Level 5 only
  - Seed     : 42
  - N/subject: 10
  - Split    : test

Output: data/datasets/math_validate.jsonl
"""

import json
import random
from pathlib import Path

# ── Must match MAS optimizer settings exactly ─────────────────────────────────
SUBJECTS = [
    "Number Theory",
    "Precalculus",
    "Counting & Probability",
]
LEVEL = "Level 5"
SEED = 42
N_PER_SUBJECT = 20
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_PATH = Path(__file__).parent / "datasets" / "math_validate.jsonl"
CACHE_DIR = Path(__file__).parent / "math_hf_cache"


def build_jsonl():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading hendrycks/competition_math from HuggingFace (test split)...")
    ds = load_dataset("hendrycks/competition_math", split="test", cache_dir=str(CACHE_DIR))
    all_data = list(ds)
    print(f"  Total test examples: {len(all_data)}")

    rng = random.Random(SEED)
    records = []

    for subject in SUBJECTS:
        subject_data = [
            row for row in all_data
            if row.get("type") == subject and row.get("level") == LEVEL
        ]
        sampled = rng.sample(subject_data, min(N_PER_SUBJECT, len(subject_data)))
        for row in sampled:
            records.append({
                "subject": subject,
                "problem": row["problem"],
                "solution": row["solution"],
                "level": row["level"],
                "type": row["type"],
            })
        print(f"  {subject} ({LEVEL}): {len(sampled)} samples")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWritten {len(records)} records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    build_jsonl()
