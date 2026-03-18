"""Build MMLU validation JSONL for AFlow experiments.

Uses the exact same subjects, seed, and sample count as MAS optimizer
so both systems evaluate on identical queries during the search phase.

Output: data/datasets/mmlu_validate.jsonl
"""

import json
import random
from pathlib import Path

# ── Must match MAS optimizer settings exactly ─────────────────────────────────
SUBJECTS = [
    "international_law",
    "anatomy",
    "business_ethics",
    "college_chemistry",
    "moral_scenarios",
    "econometrics",
]
SEED = 42
N_PER_SUBJECT = 20
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_PATH = Path(__file__).parent / "datasets" / "mmlu_validate.jsonl"
CACHE_DIR = Path(__file__).parent / "mmlu_hf_cache"

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_choices(choices):
    return "\n".join(f"{INDEX_TO_LETTER[i]}) {c}" for i, c in enumerate(choices))


def build_jsonl():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MMLU from HuggingFace (cais/mmlu, all)...")
    ds = load_dataset("cais/mmlu", "all", cache_dir=str(CACHE_DIR))
    test_split = list(ds["test"])

    rng = random.Random(SEED)
    records = []

    for subject in SUBJECTS:
        subject_data = [row for row in test_split if row["subject"] == subject]
        sampled = rng.sample(subject_data, min(N_PER_SUBJECT, len(subject_data)))
        for row in sampled:
            answer_idx = row["answer"]
            choices = list(row["choices"])
            records.append({
                "subject": subject,
                "question": row["question"],
                "choices": choices,
                "formatted_choices": format_choices(choices),
                "answer": INDEX_TO_LETTER[answer_idx],
            })
        print(f"  {subject}: {len(sampled)} samples")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWritten {len(records)} records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    build_jsonl()
