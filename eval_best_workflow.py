"""
AFlow Held-out Evaluation of Best Workflow

Evaluates the best AFlow workflow (highest validation score) on held-out
queries that were NOT seen during training (validation phase).

Configuration:
    Set DATASET = "MATH", "MMLU", or "MMLUPro" below, then run from the AFlow directory:
        cd Baseline/AFlow
        python eval_best_workflow.py

Held-out query sampling:
    1. Load the validate.jsonl (training set) and fingerprint every problem.
    2. Load all available test-split data from the raw source.
    3. Filter out training fingerprints — zero overlap guaranteed.
    4. Randomly sample up to NUM_EVAL_QUERIES per subject (seed=99).
"""

import asyncio
import importlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# ─── make sure we can import AFlow modules regardless of CWD ─────────────────
_AFLOW_DIR = Path(__file__).parent.resolve()
if str(_AFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_AFLOW_DIR))
# ─────────────────────────────────────────────────────────────────────────────

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET          = "MMLUPro"   # "MATH", "MMLU", or "MMLUPro"
NUM_EVAL_QUERIES = 50      # held-out queries per subject
MAX_CONCURRENT   = 50       # concurrent evaluations
SEED             = 99       # sampling seed (training used 42)
# ──────────────────────────────────────────────────────────────────────────────

MATH_SUBJECTS = [
    "Number Theory",
    "Precalculus",
    "Counting & Probability",
]
MMLU_SUBJECTS = [
    "international_law",
    "anatomy",
    "business_ethics",
    "college_chemistry",
    "moral_scenarios",
    "econometrics",
]
MMLU_PRO_SUBJECTS = [
    "law",
    "history",
    "philosophy",
    "engineering",
]
MATH_LEVEL = "Level 5"

# ─── paths (relative to AFlow dir) ───────────────────────────────────────────
MATH_VALIDATE_JSONL     = _AFLOW_DIR / "data/datasets/math_validate.jsonl"
MMLU_VALIDATE_JSONL     = _AFLOW_DIR / "data/datasets/mmlu_validate.jsonl"
MMLU_PRO_VALIDATE_JSONL = _AFLOW_DIR / "data/datasets/mmlu_pro_validate.jsonl"
MATH_RAW_TEST_DIR       = _AFLOW_DIR / "data/math_hf_cache/MATH/test"
MMLU_HF_CACHE_DIR       = _AFLOW_DIR / "data/mmlu_hf_cache"
MMLU_PRO_HF_CACHE_DIR   = _AFLOW_DIR / "data/mmlu_pro_hf_cache"


# ─────────────────────────────────────────────────────────────────────────────
# Best-round discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_best_round(dataset: str) -> int:
    results_file = _AFLOW_DIR / f"workspace/{dataset}/workflows/results.json"
    with open(results_file) as f:
        results = json.load(f)
    best = max(results, key=lambda r: r["score"])
    print(f"Best round: {best['round']}  (validation score={best['score']:.4f})")
    return best["round"]


# ─────────────────────────────────────────────────────────────────────────────
# Graph loading (mirrors GraphUtils.load_graph)
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_class(dataset: str, round_n: int):
    module_name = f"workspace.{dataset}.workflows.round_{round_n}.graph"
    # Invalidate cached module so a fresh import is used
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = importlib.import_module(module_name)
    return getattr(mod, "Workflow")


# ─────────────────────────────────────────────────────────────────────────────
# LLM config loading
# ─────────────────────────────────────────────────────────────────────────────

def get_exec_llm_config():
    from scripts.async_llm import LLMsConfig
    models = LLMsConfig.default()
    return models.get("openai/gpt-4o-mini-2024-07-18")


# ─────────────────────────────────────────────────────────────────────────────
# Training fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def load_training_fingerprints(validate_jsonl: Path, key: str) -> set:
    """Return the set of field values (problem or question text) used in training."""
    if not validate_jsonl.exists():
        print(f"  [warn] validate.jsonl not found: {validate_jsonl}")
        return set()
    fingerprints = set()
    with open(validate_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            fingerprints.add(obj[key])
    print(f"  Training fingerprints loaded: {len(fingerprints)}")
    return fingerprints


# ─────────────────────────────────────────────────────────────────────────────
# Held-out data building
# ─────────────────────────────────────────────────────────────────────────────

def build_math_heldout(rng: random.Random) -> List[dict]:
    """
    Load MATH Level 5 test problems for the 4 subjects, excluding training
    fingerprints, then sample up to NUM_EVAL_QUERIES per subject.
    """
    training_fps = load_training_fingerprints(MATH_VALIDATE_JSONL, "problem")

    if not MATH_RAW_TEST_DIR.exists():
        raise FileNotFoundError(
            f"MATH raw test data not found: {MATH_RAW_TEST_DIR}\n"
            "Run: python data/build_math_validate.py  (it downloads the data)"
        )

    # Load all raw test problems
    raw = []
    for subject_dir in MATH_RAW_TEST_DIR.iterdir():
        if not subject_dir.is_dir():
            continue
        for json_file in subject_dir.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                raw.append(json.load(f))

    records = []
    for subject in MATH_SUBJECTS:
        pool = [
            r for r in raw
            if r.get("type") == subject
            and r.get("level") == MATH_LEVEL
            and r["problem"] not in training_fps
        ]
        n = min(NUM_EVAL_QUERIES, len(pool))
        if n < NUM_EVAL_QUERIES:
            print(f"  [warn] {subject}: only {n} held-out examples (requested {NUM_EVAL_QUERIES})")
        sampled = rng.sample(pool, n)
        for r in sampled:
            records.append({
                "subject": subject,
                "problem": r["problem"],
                "solution": r["solution"],
                "level": r.get("level", MATH_LEVEL),
                "type": r.get("type", subject),
            })
        print(f"  {subject}: {n} held-out queries")

    return records


def build_mmlu_heldout(rng: random.Random) -> List[dict]:
    """
    Load MMLU test examples for the 6 subjects, excluding training
    fingerprints, then sample up to NUM_EVAL_QUERIES per subject.
    """
    training_fps = load_training_fingerprints(MMLU_VALIDATE_JSONL, "question")
    INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    print("  Loading MMLU from HuggingFace cache...")
    ds = load_dataset("cais/mmlu", "all", cache_dir=str(MMLU_HF_CACHE_DIR))
    test_split = list(ds["test"])

    records = []
    for subject in MMLU_SUBJECTS:
        pool = [
            r for r in test_split
            if r["subject"] == subject
            and r["question"] not in training_fps
        ]
        n = min(NUM_EVAL_QUERIES, len(pool))
        if n < NUM_EVAL_QUERIES:
            print(f"  [warn] {subject}: only {n} held-out examples (requested {NUM_EVAL_QUERIES})")
        sampled = rng.sample(pool, n)
        for r in sampled:
            choices = list(r["choices"])
            formatted = "\n".join(f"{INDEX_TO_LETTER[i]}) {c}" for i, c in enumerate(choices))
            records.append({
                "subject": subject,
                "question": r["question"],
                "choices": choices,
                "formatted_choices": formatted,
                "answer": INDEX_TO_LETTER[r["answer"]],
            })
        print(f"  {subject}: {n} held-out queries")

    return records


def build_mmlu_pro_heldout(rng: random.Random) -> List[dict]:
    """
    Load MMLU-Pro test examples for the 4 categories, excluding training
    fingerprints, then sample up to NUM_EVAL_QUERIES per category.
    """
    training_fps = load_training_fingerprints(MMLU_PRO_VALIDATE_JSONL, "question")
    LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    print("  Loading MMLU-Pro from HuggingFace cache...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=str(MMLU_PRO_HF_CACHE_DIR), split="test")
    test_split = list(ds)

    records = []
    for category in MMLU_PRO_SUBJECTS:
        pool = [
            r for r in test_split
            if r["category"] == category
            and r["question"] not in training_fps
        ]
        n = min(NUM_EVAL_QUERIES, len(pool))
        if n < NUM_EVAL_QUERIES:
            print(f"  [warn] {category}: only {n} held-out examples (requested {NUM_EVAL_QUERIES})")
        sampled = rng.sample(pool, n)
        for r in sampled:
            options = list(r["options"])
            formatted = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
            records.append({
                "subject": category,
                "question": r["question"],
                "choices": options,
                "formatted_choices": formatted,
                "answer": str(r["answer"]).upper(),
            })
        print(f"  {category}: {n} held-out queries")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate(dataset: str, best_round: int, held_out: List[dict]) -> dict:
    """Run held-out evaluation and return per-subject score dict."""
    from scripts.async_llm import LLMsConfig

    llm_config = get_exec_llm_config()

    # Load the best workflow class
    WorkflowClass = load_graph_class(dataset, best_round)

    # Create a temp log dir so benchmark CSV / log.json go there
    log_dir = _AFLOW_DIR / f"workspace/{dataset}/workflows/heldout_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the benchmark (file_path unused — we call evaluate_all_problems directly)
    if dataset == "MATH":
        from benchmarks.math import MATHBenchmark
        benchmark = MATHBenchmark(name=dataset, file_path="", log_path=str(log_dir))
    elif dataset == "MMLUPro":
        from benchmarks.mmlu_pro import MMLUProBenchmark
        benchmark = MMLUProBenchmark(name=dataset, file_path="", log_path=str(log_dir))
    else:
        from benchmarks.mmlu import MMLUBenchmark
        benchmark = MMLUBenchmark(name=dataset, file_path="", log_path=str(log_dir))

    # Instantiate the workflow graph (new instance per run)
    graph = WorkflowClass(name=dataset, llm_config=llm_config, dataset=dataset)

    print(f"\nRunning evaluation on {len(held_out)} queries (max_concurrent={MAX_CONCURRENT}) …")
    results_raw = await benchmark.evaluate_all_problems(held_out, graph, MAX_CONCURRENT)

    columns = benchmark.get_result_columns()
    avg_score, _, _ = benchmark.save_results_to_csv(results_raw, columns)

    # Per-subject breakdown
    import pandas as pd
    df = pd.DataFrame(results_raw, columns=columns)
    per_subject = df.groupby("subject")["score"].mean().to_dict()
    per_subject["__average__"] = avg_score

    return per_subject


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, dataset: str, best_round: int):
    out_dir = _AFLOW_DIR / f"workspace/{dataset}/workflows"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"heldout_eval_{dataset}_{timestamp}.txt"

    if dataset == "MATH":
        subjects = MATH_SUBJECTS
    elif dataset == "MMLUPro":
        subjects = MMLU_PRO_SUBJECTS
    else:
        subjects = MMLU_SUBJECTS

    with open(out_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"AFLOW HELD-OUT EVALUATION — {dataset}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Best round:        {best_round}\n")
        f.write(f"Queries/subject:   {NUM_EVAL_QUERIES}\n")
        f.write(f"Sampling seed:     {SEED}\n")
        f.write(f"Date:              {timestamp}\n")
        f.write("-" * 70 + "\n\n")
        for subj in subjects:
            score = results.get(subj, float("nan"))
            f.write(f"  {subj:<35s}  {score:.4f}\n")
        f.write(f"\n  {'AVERAGE':<35s}  {results['__average__']:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    # Must run from AFlow directory for relative imports in graph.py
    os.chdir(_AFLOW_DIR)

    print("=" * 70)
    print(f"AFLOW HELD-OUT EVALUATION  —  {DATASET}")
    print(f"Queries/subject: {NUM_EVAL_QUERIES}  |  seed: {SEED}")
    print("=" * 70)

    best_round = find_best_round(DATASET)

    rng = random.Random(SEED)
    print(f"\nBuilding held-out queries (excluding training set) …")
    if DATASET == "MATH":
        held_out = build_math_heldout(rng)
    elif DATASET == "MMLUPro":
        held_out = build_mmlu_pro_heldout(rng)
    else:
        held_out = build_mmlu_heldout(rng)

    print(f"Total held-out examples: {len(held_out)}")

    results = await evaluate(DATASET, best_round, held_out)

    if DATASET == "MATH":
        subjects = MATH_SUBJECTS
    elif DATASET == "MMLUPro":
        subjects = MMLU_PRO_SUBJECTS
    else:
        subjects = MMLU_SUBJECTS
    print("\n" + "=" * 70)
    print(f"RESULTS  —  {DATASET}  (round {best_round})")
    print("=" * 70)
    for subj in subjects:
        print(f"  {subj:<35s}  {results.get(subj, float('nan')):.4f}")
    print(f"\n  {'AVERAGE':<35s}  {results['__average__']:.4f}")
    print("=" * 70)

    save_results(results, DATASET, best_round)


if __name__ == "__main__":
    asyncio.run(main())
