"""
Build FullStackBench validate JSONL for AFlow.

Matches RobustMAS experiment_fullstack.py scenario setup:
- Categories: Advanced Programming, Scientific Computing, Data Analysis, Desktop and Web Development
- Stratified: 10 hard + 10 medium per category, no easy, Locale: en, Seed: 42
- Total: 80 examples

Run:
    cd C:/Users/cheikh/Desktop/AFlow
    python data/build_fullstack_validate.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

CATEGORIES = [
    "Advanced Programming",
    "Scientific Computing",
    "Data Analysis",
    "Desktop and Web Development",
]
LOCALE = "en"
SEED = 42
N_HARD = 10
N_MEDIUM = 10

OUTPUT_PATH = Path(__file__).parent / "datasets" / "fullstack_validate.jsonl"


def main():
    print("Loading FullStackBench from HuggingFace (ByteDance/FullStackBench)...")
    dataset = load_dataset("ByteDance/FullStackBench", LOCALE, split="test")
    data = list(dataset)
    print(f"Total examples: {len(data)}")

    records = []

    for category in CATEGORIES:
        rng = random.Random(SEED)
        filtered = [ex for ex in data if ex["labels"].get("category") == category]
        hard   = [ex for ex in filtered if ex["labels"].get("difficulty") == "hard"]
        medium = [ex for ex in filtered if ex["labels"].get("difficulty") == "medium"]

        sampled_hard   = rng.sample(hard,   min(N_HARD,   len(hard)))
        sampled_medium = rng.sample(medium, min(N_MEDIUM, len(medium)))
        sampled = sampled_hard + sampled_medium

        print(f"  {category}: {len(hard)} hard, {len(medium)} medium → sampled {len(sampled_hard)}h + {len(sampled_medium)}m")

        for ex in sampled:
            records.append({
                "id": ex["id"],
                "content": ex["content"],
                "category": ex["labels"]["category"],
                "difficulty": ex["labels"]["difficulty"],
                "programming_language": ex["labels"]["programming_language"],
                "raw_example": dict(ex),
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
