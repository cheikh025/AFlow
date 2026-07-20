"""
AFlow Held-out Evaluation of Best Workflow

Evaluates the best AFlow workflow (highest validation score) on held-out
queries that were NOT seen during training (validation phase).

Configuration:
    Set DATASET below, then run from the AFlow directory:
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
DATASET           = "MATH"   # "MATH", "MMLU", "MMLUPro", "FullStack", "SciCode", or "Mind2Web"
NUM_EVAL_QUERIES  = 100      # held-out queries per subject
MAX_CONCURRENT    = 50      # concurrent evaluations
SEED              = 99      # sampling seed (training used 42)
VALIDATION_ROUNDS = 3       # how many times to evaluate (scores are averaged)
SCICODE_VALIDATION_ROUNDS = 1  # ReMAS held-out default
MIND2WEB_VALIDATION_ROUNDS = 1  # ReMAS held-out default
QUERY_TIMEOUT     = 300     # seconds before a single query is abandoned (0 = no timeout)
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
    "history",
    "philosophy",
    "engineering",
    "law",
]
FULLSTACK_CATEGORIES = [
    "Mathematics",
    "DataBase",
    "Machine Learning",
    "Software Engineering",
]
SCICODE_FIELDS = ["mathematics", "physics", "material_science"]
MIND2WEB_DOMAINS = ["travel", "shopping", "entertainment"]
MATH_LEVEL = "Level 5"

# ─── paths (relative to AFlow dir) ───────────────────────────────────────────
MATH_VALIDATE_JSONL     = _AFLOW_DIR / "data/datasets/math_validate.jsonl"
MMLU_VALIDATE_JSONL     = _AFLOW_DIR / "data/datasets/mmlu_validate.jsonl"
MMLU_PRO_VALIDATE_JSONL = _AFLOW_DIR / "data/datasets/mmlu_pro_validate.jsonl"
MATH_RAW_TEST_DIR           = _AFLOW_DIR / "data/math_hf_cache/MATH/test"
MMLU_HF_CACHE_DIR           = _AFLOW_DIR / "data/mmlu_hf_cache"
MMLU_PRO_HF_CACHE_DIR       = _AFLOW_DIR / "data/mmlu_pro_hf_cache"
FULLSTACK_VALIDATE_JSONL    = _AFLOW_DIR / "data/datasets/fullstack_validate.jsonl"
FULLSTACK_HF_CACHE_DIR      = _AFLOW_DIR / "data/fullstack_hf_cache"
SCICODE_TEST_JSONL          = _AFLOW_DIR / "data/datasets/scicode_test.jsonl"
MIND2WEB_TEST_JSONL         = _AFLOW_DIR / "data/datasets/mind2web_test.jsonl"


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
    return models.get("deepseek/deepseek-v4-flash-noreason")


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


def build_fullstack_heldout(rng: random.Random) -> List[dict]:
    """
    Load FullStackBench test examples for the 4 categories, excluding training
    fingerprints (by id), then sample up to NUM_EVAL_QUERIES per category.
    Stratified: 50 hard + 50 medium, no easy (capped at available if fewer).
    """
    training_fps = load_training_fingerprints(FULLSTACK_VALIDATE_JSONL, "id")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    print("  Loading FullStackBench from HuggingFace cache...")
    ds = load_dataset("ByteDance/FullStackBench", "en", cache_dir=str(FULLSTACK_HF_CACHE_DIR), split="test")
    test_split = list(ds)

    records = []
    for category in FULLSTACK_CATEGORIES:
        pool = [
            ex for ex in test_split
            if ex["labels"].get("category") == category
            and ex["id"] not in training_fps
        ]
        hard   = [ex for ex in pool if ex["labels"].get("difficulty") == "hard"]
        medium = [ex for ex in pool if ex["labels"].get("difficulty") == "medium"]
        easy   = [ex for ex in pool if ex["labels"].get("difficulty") == "easy"]

        result = list(hard)
        remaining = NUM_EVAL_QUERIES - len(result)
        if remaining > 0:
            result += medium[:remaining]
        remaining = NUM_EVAL_QUERIES - len(result)
        if remaining > 0:
            result += easy[:remaining]

        for ex in result:
            records.append({
                "id": ex["id"],
                "content": ex["content"],
                "category": ex["labels"]["category"],
                "difficulty": ex["labels"]["difficulty"],
                "programming_language": ex["labels"]["programming_language"],
                "raw_example": dict(ex),
            })
        n = len(result)
        print(f"  {category}: {n} held-out queries")

    return records


def build_scicode_heldout() -> List[dict]:
    """Load the frozen ReMAS-compatible held-out set without resampling."""
    if not SCICODE_TEST_JSONL.is_file():
        raise FileNotFoundError(
            f"SciCode held-out data not found: {SCICODE_TEST_JSONL}\n"
            "Run: python data/build_scicode_data.py"
        )
    from benchmarks.scicode_spec import validate_manifest_records

    with SCICODE_TEST_JSONL.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    validate_manifest_records(
        records,
        "heldout",
        SCICODE_TEST_JSONL.with_name("scicode_manifest.json"),
    )
    return records


def build_mind2web_heldout() -> List[dict]:
    """Load the frozen ReMAS-compatible held-out set without resampling."""
    if not MIND2WEB_TEST_JSONL.is_file():
        raise FileNotFoundError(
            f"Mind2Web held-out data not found: {MIND2WEB_TEST_JSONL}\n"
            "Run: python data/build_mind2web_data.py"
        )
    from benchmarks.mind2web_spec import validate_manifest_records

    with MIND2WEB_TEST_JSONL.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    validate_manifest_records(
        records,
        "heldout",
        MIND2WEB_TEST_JSONL.with_name("mind2web_manifest.json"),
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate(dataset: str, best_round: int, held_out: List[dict]) -> dict:
    """Run held-out evaluation and return per-subject score dict."""
    import asyncio
    import pandas as pd

    llm_config = get_exec_llm_config()
    WorkflowClass = load_graph_class(dataset, best_round)

    log_dir = _AFLOW_DIR / f"workspace/{dataset}/workflows/heldout_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "MATH":
        from benchmarks.math import MATHBenchmark
        benchmark = MATHBenchmark(name=dataset, file_path="", log_path=str(log_dir))
    elif dataset == "MMLUPro":
        from benchmarks.mmlu_pro import MMLUProBenchmark
        benchmark = MMLUProBenchmark(name=dataset, file_path="", log_path=str(log_dir))
    elif dataset == "FullStack":
        from benchmarks.fullstack import FullStackBenchmark
        benchmark = FullStackBenchmark(name=dataset, file_path="", log_path=str(log_dir))
    elif dataset == "SciCode":
        from benchmarks.scicode import SciCodeBenchmark
        benchmark = SciCodeBenchmark(name=dataset, file_path="", log_path=str(log_dir))
        benchmark.ensure_ready()
    elif dataset == "Mind2Web":
        from benchmarks.mind2web import Mind2WebBenchmark
        benchmark = Mind2WebBenchmark(name=dataset, file_path="", log_path=str(log_dir))
    else:
        from benchmarks.mmlu import MMLUBenchmark
        benchmark = MMLUBenchmark(name=dataset, file_path="", log_path=str(log_dir))

    base_columns = benchmark.get_result_columns()
    all_columns = base_columns + ["input_tokens", "output_tokens", "total_tokens"]

    max_concurrent = min(MAX_CONCURRENT, getattr(benchmark, "max_concurrent_tasks", MAX_CONCURRENT))
    semaphore = asyncio.Semaphore(max_concurrent)
    mind2web_call_gate = asyncio.Semaphore(getattr(benchmark, "max_llm_calls", 32))

    async def evaluate_one(problem: dict) -> tuple:
        """Evaluate a single query with its own Workflow instance for isolated token tracking."""
        async with semaphore:
            # Fresh instance per query → its TokenUsageTracker captures only this query's LLM calls,
            # even when the workflow internally makes multiple LLM calls (e.g. ScEnsemble).
            if dataset == "Mind2Web":
                from benchmarks.mind2web import Mind2WebWorkflowFactory

                graph = Mind2WebWorkflowFactory(
                    workflow_class=WorkflowClass,
                    name=dataset,
                    llm_config=llm_config,
                    dataset=dataset,
                    max_llm_calls=getattr(benchmark, "max_llm_calls", 32),
                )
                # Share one request gate across every task in this evaluation.
                graph.call_gate = mind2web_call_gate
            else:
                graph = WorkflowClass(name=dataset, llm_config=llm_config, dataset=dataset)
            n_cols = len(base_columns)
            try:
                coro = benchmark.evaluate_problem(problem, graph)
                if QUERY_TIMEOUT > 0 and dataset not in {"SciCode", "Mind2Web"}:
                    result = await asyncio.wait_for(coro, timeout=QUERY_TIMEOUT)
                else:
                    result = await coro
                summary = graph.llm.get_usage_summary()
                in_tok  = summary["total_input_tokens"]
                out_tok = summary["total_output_tokens"]
                return result + (in_tok, out_tok, in_tok + out_tok)
            except asyncio.TimeoutError:
                subject = problem.get("subject", problem.get("category", "unknown"))
                print(f"\n  [timeout] query in '{subject}' exceeded {QUERY_TIMEOUT}s — scored 0")
                # Return a zero-score row with the right number of columns
                failed = (subject,) + ("",) * (n_cols - 3) + (0.0, 0.0)
                return failed + (0, 0, 0)

    if dataset == "SciCode":
        eval_rounds = SCICODE_VALIDATION_ROUNDS
    elif dataset == "Mind2Web":
        eval_rounds = MIND2WEB_VALIDATION_ROUNDS
    else:
        eval_rounds = VALIDATION_ROUNDS
    print(f"\nRunning evaluation on {len(held_out)} queries "
          f"(max_concurrent={max_concurrent}, validation_rounds={eval_rounds}) …")

    accumulated: dict = {}  # subject -> list of per-round scores
    round_averages = []
    round_avg_in: list = []
    round_avg_out: list = []
    scicode_round_metrics: list[dict] = []
    mind2web_round_metrics: list[dict] = []

    for round_i in range(1, eval_rounds + 1):
        if eval_rounds > 1:
            print(f"\n--- Validation round {round_i}/{eval_rounds} ---")
        llm_config.seed = round_i - 1  # seed 0, 1, 2 across rounds

        from tqdm.asyncio import tqdm_asyncio
        tasks = [evaluate_one(p) for p in held_out]
        results_raw = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {dataset} (round {round_i}/{eval_rounds})", total=len(tasks))

        df = pd.DataFrame(results_raw, columns=all_columns)
        if dataset == "SciCode":
            metrics = benchmark.aggregate_dataframe(df)
            avg_score = metrics["fitness"]
            scicode_round_metrics.append(metrics)
        elif dataset == "Mind2Web":
            metrics = benchmark.aggregate_dataframe(df)
            avg_score = metrics["fitness"]
            mind2web_round_metrics.append(metrics)
        else:
            avg_score = df["score"].mean()

        # Save CSV (includes token columns)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"{avg_score:.5f}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Token summary for this round
        total_in  = int(df["input_tokens"].sum())
        total_out = int(df["output_tokens"].sum())
        if dataset == "SciCode":
            token_units = max(int(df["total_subproblems"].sum()), 1)
            avg_in = total_in / token_units
            avg_out = total_out / token_units
        elif dataset == "Mind2Web":
            token_units = max(int(df["num_steps"].sum()), 1)
            avg_in = total_in / token_units
            avg_out = total_out / token_units
        else:
            avg_in = df["input_tokens"].mean()
            avg_out = df["output_tokens"].mean()
        token_label = (
            "subproblem" if dataset == "SciCode"
            else "action" if dataset == "Mind2Web"
            else "query"
        )
        print(f"    Tokens per {token_label}  — avg input: {avg_in:.0f}  avg output: {avg_out:.0f}  avg total: {avg_in + avg_out:.0f}")
        print(f"    Tokens run total  — input: {total_in:,}  output: {total_out:,}  total: {total_in + total_out:,}")

        if dataset == "SciCode":
            per_subject = metrics["field_subproblem_pass_rates"]
        elif dataset == "Mind2Web":
            per_subject = {
                scenario_id: values["macro_step_success_rate"]
                for scenario_id, values in metrics["scenarios"].items()
            }
        else:
            group_col = "category" if dataset == "FullStack" else "subject"
            per_subject = df.groupby(group_col)["score"].mean().to_dict()
        for subj, score in per_subject.items():
            accumulated.setdefault(subj, []).append(score)
        round_averages.append(avg_score)
        round_avg_in.append(avg_in)
        round_avg_out.append(avg_out)

        if eval_rounds > 1:
            print(f"    Round {round_i} average score: {avg_score:.4f}")

    import statistics

    per_subject_avg = {subj: sum(scores) / len(scores) for subj, scores in accumulated.items()}
    per_subject_avg["__average__"] = sum(round_averages) / len(round_averages)
    if dataset == "SciCode":
        per_subject_avg["__main_problem_resolve_rate__"] = sum(
            item["main_problem_resolve_rate"] for item in scicode_round_metrics
        ) / len(scicode_round_metrics)
        per_subject_avg["__global_subproblem_pass_rate__"] = sum(
            item["global_subproblem_pass_rate"] for item in scicode_round_metrics
        ) / len(scicode_round_metrics)
        per_subject_avg["__passed_main_problems__"] = sum(
            item["passed_main_problems"] for item in scicode_round_metrics
        ) / len(scicode_round_metrics)
        per_subject_avg["__total_main_problems__"] = scicode_round_metrics[0][
            "total_main_problems"
        ]
    elif dataset == "Mind2Web":
        metric_keys = (
            "macro_element_acc",
            "macro_action_f1",
            "task_success_rate",
            "macro_candidate_recall",
            "generation_errors",
        )
        for key in metric_keys:
            per_subject_avg[f"__{key}__"] = sum(
                item[key] for item in mind2web_round_metrics
            ) / len(mind2web_round_metrics)
        per_subject_avg["__n_tasks__"] = mind2web_round_metrics[0]["n_tasks"]
        per_subject_avg["__n_steps__"] = mind2web_round_metrics[0]["n_steps"]

    per_subject_std = {subj: statistics.pstdev(scores) for subj, scores in accumulated.items()}
    overall_std = statistics.pstdev(round_averages)

    token_avg = {
        "avg_input_tokens":  sum(round_avg_in)  / len(round_avg_in),
        "avg_output_tokens": sum(round_avg_out) / len(round_avg_out),
        "avg_total_tokens":  sum(round_avg_in)  / len(round_avg_in) + sum(round_avg_out) / len(round_avg_out),
    }

    return per_subject_avg, per_subject_std, overall_std, token_avg


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, dataset: str, best_round: int, token_avg: dict = None, model_name: str = "", per_subject_std: dict = None, overall_std: float = None):
    out_dir = _AFLOW_DIR / f"workspace/{dataset}/workflows"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_") if model_name else ""
    name = f"heldout_eval_{dataset}_{safe_model}_{timestamp}.txt" if safe_model else f"heldout_eval_{dataset}_{timestamp}.txt"
    out_file = out_dir / name

    if dataset == "MATH":
        subjects = MATH_SUBJECTS
    elif dataset == "MMLUPro":
        subjects = MMLU_PRO_SUBJECTS
    elif dataset == "FullStack":
        subjects = FULLSTACK_CATEGORIES
    elif dataset == "SciCode":
        subjects = SCICODE_FIELDS
    elif dataset == "Mind2Web":
        subjects = MIND2WEB_DOMAINS
    else:
        subjects = MMLU_SUBJECTS

    with open(out_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"AFLOW HELD-OUT EVALUATION — {dataset}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Best round:        {best_round}\n")
        if dataset == "SciCode":
            query_label, query_count = "Fixed main problems", 10
            validation_rounds = SCICODE_VALIDATION_ROUNDS
        elif dataset == "Mind2Web":
            query_label, query_count = "Fixed tasks/domain", 100
            validation_rounds = MIND2WEB_VALIDATION_ROUNDS
        else:
            query_label, query_count = "Queries/subject", NUM_EVAL_QUERIES
            validation_rounds = VALIDATION_ROUNDS
        f.write(f"{query_label}:   {query_count}\n")
        f.write(f"Validation rounds: {validation_rounds}\n")
        f.write(f"Sampling seed:     {SEED}\n")
        f.write(f"Date:              {timestamp}\n")
        f.write("-" * 70 + "\n\n")
        for subj in subjects:
            score = results.get(subj, float("nan"))
            std = per_subject_std.get(subj, 0.0) if per_subject_std else 0.0
            f.write(f"  {subj:<35s}  {score:.4f}  ±{std:.4f}\n")
        avg_std = overall_std if overall_std is not None else 0.0
        f.write(f"\n  {'AVERAGE':<35s}  {results['__average__']:.4f}  ±{avg_std:.4f}\n")
        if dataset == "SciCode":
            f.write(
                f"  {'Global subproblem pass rate':<35s}  "
                f"{results['__global_subproblem_pass_rate__']:.4f}\n"
            )
            f.write(
                f"  {'Main problem resolve rate':<35s}  "
                f"{results['__main_problem_resolve_rate__']:.4f}  "
                f"({results['__passed_main_problems__']:.0f}/"
                f"{results['__total_main_problems__']:.0f})\n"
            )
        elif dataset == "Mind2Web":
            f.write(f"  {'Element accuracy':<35s}  {results['__macro_element_acc__']:.4f}\n")
            f.write(f"  {'Action / operation F1':<35s}  {results['__macro_action_f1__']:.4f}\n")
            f.write(f"  {'Task success rate':<35s}  {results['__task_success_rate__']:.4f}\n")
            f.write(f"  {'Candidate recall':<35s}  {results['__macro_candidate_recall__']:.4f}\n")
            f.write(
                f"  {'Generation errors':<35s}  "
                f"{results['__generation_errors__']:.0f}\n"
            )
            f.write(
                f"  {'Evaluated tasks/actions':<35s}  "
                f"{results['__n_tasks__']:.0f}/{results['__n_steps__']:.0f}\n"
            )
        if token_avg:
            f.write("\n" + "-" * 70 + "\n")
            token_unit = (
                "subproblem" if dataset == "SciCode"
                else "action" if dataset == "Mind2Web"
                else "query"
            )
            f.write(f"  {f'Avg input tokens/{token_unit}':<35s}  {token_avg['avg_input_tokens']:.0f}\n")
            f.write(f"  {f'Avg output tokens/{token_unit}':<35s}  {token_avg['avg_output_tokens']:.0f}\n")
            f.write(f"  {f'Avg total tokens/{token_unit}':<35s}  {token_avg['avg_total_tokens']:.0f}\n")
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
    query_count = 10 if DATASET == "SciCode" else 100 if DATASET == "Mind2Web" else NUM_EVAL_QUERIES
    print(f"Queries/subject: {query_count}  |  seed: {SEED}")
    print("=" * 70)

    best_round = find_best_round(DATASET)

    rng = random.Random(SEED)
    print(f"\nBuilding held-out queries (excluding training set) …")
    if DATASET == "MATH":
        held_out = build_math_heldout(rng)
    elif DATASET == "MMLUPro":
        held_out = build_mmlu_pro_heldout(rng)
    elif DATASET == "FullStack":
        held_out = build_fullstack_heldout(rng)
    elif DATASET == "SciCode":
        held_out = build_scicode_heldout()
    elif DATASET == "Mind2Web":
        held_out = build_mind2web_heldout()
    else:
        held_out = build_mmlu_heldout(rng)

    print(f"Total held-out examples: {len(held_out)}")

    results, per_subject_std, overall_std, token_avg = await evaluate(DATASET, best_round, held_out)

    if DATASET == "MATH":
        subjects = MATH_SUBJECTS
    elif DATASET == "MMLUPro":
        subjects = MMLU_PRO_SUBJECTS
    elif DATASET == "FullStack":
        subjects = FULLSTACK_CATEGORIES
    elif DATASET == "SciCode":
        subjects = SCICODE_FIELDS
    elif DATASET == "Mind2Web":
        subjects = MIND2WEB_DOMAINS
    else:
        subjects = MMLU_SUBJECTS
    print("\n" + "=" * 70)
    print(f"RESULTS  —  {DATASET}  (round {best_round})")
    print("=" * 70)
    for subj in subjects:
        avg = results.get(subj, float('nan'))
        std = per_subject_std.get(subj, 0.0)
        print(f"  {subj:<35s}  {avg:.4f}  ±{std:.4f}")
    print(f"\n  {'AVERAGE':<35s}  {results['__average__']:.4f}  ±{overall_std:.4f}")
    if DATASET == "SciCode":
        print(
            f"  {'Global subproblem pass rate':<35s}  "
            f"{results['__global_subproblem_pass_rate__']:.4f}"
        )
        print(
            f"  {'Main problem resolve rate':<35s}  "
            f"{results['__main_problem_resolve_rate__']:.4f}  "
            f"({results['__passed_main_problems__']:.0f}/"
            f"{results['__total_main_problems__']:.0f})"
        )
    elif DATASET == "Mind2Web":
        print(f"  {'Element accuracy':<35s}  {results['__macro_element_acc__']:.4f}")
        print(f"  {'Action / operation F1':<35s}  {results['__macro_action_f1__']:.4f}")
        print(f"  {'Task success rate':<35s}  {results['__task_success_rate__']:.4f}")
        print(f"  {'Candidate recall':<35s}  {results['__macro_candidate_recall__']:.4f}")
        print(f"  {'Generation errors':<35s}  {results['__generation_errors__']:.0f}")
        print(
            f"  {'Evaluated tasks/actions':<35s}  "
            f"{results['__n_tasks__']:.0f}/{results['__n_steps__']:.0f}"
        )
    print("=" * 70)

    exec_llm = get_exec_llm_config()
    save_results(results, DATASET, best_round, token_avg, model_name=exec_llm.model,
                 per_subject_std=per_subject_std, overall_std=overall_std)


if __name__ == "__main__":
    asyncio.run(main())
