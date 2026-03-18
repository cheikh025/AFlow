"""
Build FullStackBench validate JSONL for AFlow.

Same subset as MAS optimizer:
- Categories: Advanced Programming, Operating System, Machine Learning
- Difficulty: hard, Locale: en, Seed: 42, N_PER_CATEGORY: 3
- Total: 9 examples

Run:
    cd C:/Users/cheikh/Desktop/AFlow
    python data/build_fullstack_validate.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

CATEGORIES = ["Advanced Programming", "Operating System", "Machine Learning"]
DIFFICULTY = "hard"
LOCALE = "en"
SEED = 42
N_PER_CATEGORY = 3

OUTPUT_PATH = Path(__file__).parent / "datasets" / "fullstack_validate.jsonl"


def main():
    print("Loading FullStackBench from HuggingFace (ByteDance/FullStackBench)...")
    dataset = load_dataset("ByteDance/FullStackBench", LOCALE, split="test")
    data = list(dataset)
    print(f"Total examples: {len(data)}")

    rng = random.Random(SEED)
    records = []

    for category in CATEGORIES:
        filtered = [
            ex for ex in data
            if ex["labels"].get("category") == category
            and ex["labels"].get("difficulty") == DIFFICULTY
        ]
        sampled = rng.sample(filtered, min(N_PER_CATEGORY, len(filtered)))
        print(f"  {category}: {len(filtered)} hard examples → sampled {len(sampled)}")

        for ex in sampled:
            record = {
                "id": ex["id"],
                "content": ex["content"],
                "category": ex["labels"]["category"],
                "difficulty": ex["labels"]["difficulty"],
                "programming_language": ex["labels"]["programming_language"],
                # Full row stored so benchmark can pass it as provided_data to SandboxFusion
                "raw_example": dict(ex),
            }
            records.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
