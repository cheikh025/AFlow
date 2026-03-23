"""Build MMLU-Pro validation JSONL for AFlow experiments.

Categories: law, history, philosophy, engineering
20 samples per category, seed=42, from the test split.

Output: data/datasets/mmlu_pro_validate.jsonl
Fields: subject (=category), question, choices, formatted_choices, answer (A-J)

Run:
    cd C:/Users/cheikh/Desktop/baseline/AFlow
    python data/build_mmlu_pro_validate.py
"""

import json
import random
from pathlib import Path

CATEGORIES = [
    "law",
    "history",
    "philosophy",
    "engineering",
]

SEED = 42
N_PER_CATEGORY = 20
OUTPUT_PATH = Path(__file__).parent / "datasets" / "mmlu_pro_validate.jsonl"
CACHE_DIR = Path(__file__).parent / "mmlu_pro_hf_cache"

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def format_choices(options):
    return "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def build_jsonl():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TIGER-Lab/MMLU-Pro from HuggingFace (test split)...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=str(CACHE_DIR), split="test")
    all_data = list(ds)
    print(f"  Total test examples: {len(all_data)}")

    rng = random.Random(SEED)
    records = []

    for category in CATEGORIES:
        pool = [ex for ex in all_data if ex["category"] == category]
        sampled = rng.sample(pool, min(N_PER_CATEGORY, len(pool)))
        print(f"  {category}: {len(pool)} available → {len(sampled)} sampled")

        for ex in sampled:
            options = list(ex["options"])
            records.append({
                "subject": category,  # "subject" key for benchmark compatibility
                "question": ex["question"],
                "choices": options,
                "formatted_choices": format_choices(options),
                "answer": str(ex["answer"]).upper(),
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWritten {len(records)} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    build_jsonl()
