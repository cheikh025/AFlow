"""Build MATH validation JSONL for AFlow experiments.

Uses the exact same subjects, level filter, seed, and sample count as MAS optimizer
so both systems evaluate on identical queries during the search phase.

MAS settings (from experiment_math.py + subject_math_benchmark.py):
  - Subjects : Number Theory, Precalculus, Counting & Probability
  - Level    : Level 5 only
  - Seed     : 42
  - N/subject: 20
  - Split    : test

Output: data/datasets/math_validate.jsonl
"""

import json
import random
import zipfile
from pathlib import Path

import requests

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
MATH_URL = "https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip"


def download_math_data(save_dir: Path) -> None:
    zip_path = save_dir / "MATH.zip"
    save_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading MATH data from {MATH_URL} ...")
        response = requests.get(MATH_URL, stream=True)
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    print("Extracting MATH.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(save_dir)
    zip_path.unlink(missing_ok=True)


def load_math_split(split_folder: Path) -> list[dict]:
    records = []
    for subject_dir in split_folder.iterdir():
        if not subject_dir.is_dir():
            continue
        for json_file in subject_dir.glob("*.json"):
            with json_file.open("r", encoding="utf-8") as f:
                records.append(json.load(f))
    return records


def build_jsonl():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    math_root = CACHE_DIR / "MATH"
    test_root = math_root / "test"

    if not test_root.exists():
        download_math_data(CACHE_DIR)

    print("Loading MATH test split from raw JSON folders...")
    if not test_root.exists():
        raise FileNotFoundError(
            f"Expected MATH test folder not found after extraction: {test_root}"
        )
    all_data = load_math_split(test_root)
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
