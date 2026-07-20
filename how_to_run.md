# How to Run AFlow

All commands run from: `C:\Users\cheikh\Desktop\baseline\AFlow`

---

## Prerequisites (do once, for all benchmarks)

**Install dependencies**
```
pip install -r requirements.txt
```

**Configure models in `config/config2.yaml`**

The model names passed to `--opt_model_name` and `--exec_model_name` must exactly match a key in this file.

| Key | Suggested role |
|---|---|
| `google/gemini-2.5-flash` | opt (search) LLM |
| `openai/gpt-4o-mini-2024-07-18` | exec LLM |
| `meta-llama/llama-4-maverick` | alternative |
| `openai/gpt-oss-20b` | alternative |
| `llama-3.1-8b-instant` | lightweight option |

---

## MMLU-Pro

### Step 1 — Build the validation file
```
python data/build_mmlu_pro_validate.py
```
Output: `data/datasets/mmlupro_validate.jsonl`  
Settings: 4 categories (law, history, philosophy, engineering), 20 examples per category, seed=42, source=HF test split.

### Step 2 — Reset workspace (only if restarting from scratch)
```
python reset_experiment.py --dataset MMLUPro
```

### Step 3 — Run the search
```
python run.py --dataset MMLUPro \
  --opt_model_name "deepseek/deepseek-v4-flash" \
  --exec_model_name "openai/gpt-4.1-nano" \
  --max_rounds 25 --validation_rounds 1
```

### Step 4 — Run held-out evaluation on best workflow
In `eval_best_workflow.py` set `DATASET = "MMLUPro"` (line 37), then:
```
python eval_best_workflow.py
```
Evaluates on up to 50 held-out examples per category (seed=99). Training examples are fingerprinted and excluded — zero overlap guaranteed.

---

## MATH

### Step 1 — Build the validation file
```
python data/build_math_validate.py
```
Output: `data/datasets/math_validate.jsonl`  
Settings: 3 subjects (Number Theory, Precalculus, Counting & Probability), Level 5 only, 20 examples per subject, seed=42.  
Note: downloads MATH.zip from modelscope on first run (~200 MB). Skipped if already cached in `data/math_hf_cache/`.

### Step 2 — Reset workspace (only if restarting from scratch)
```
python reset_experiment.py --dataset MATH
```

### Step 3 — Run the search
```
python run.py --dataset MATH --opt_model_name "deepseek/deepseek-v4-flash" --exec_model_name "deepseek/deepseek-v4-flash-noreason" --max_rounds 25
```

### Step 4 — Run held-out evaluation on best workflow
In `eval_best_workflow.py` set `DATASET = "MATH"` (line 37), then:
```
python eval_best_workflow.py
```

---

## FullStack

### Step 1 — Start SandboxFusion
SandboxFusion must be running before any evaluation call is made.
```
docker run -p 8080:8080 bytedance/sandbox-fusion:latest
```
If running at a different address, set the environment variable:
```
set SANDBOX_FUSION_ENDPOINT=http://<host>:<port>
```

### Step 2 — Build the validation file
`fullstack_validate.jsonl` does not exist yet and must be built before the first run.
```
python data/build_fullstack_validate.py
```
Output: `data/datasets/fullstack_validate.jsonl`  
Settings: 4 categories (Advanced Programming, Scientific Computing, Data Analysis, Desktop and Web Development), all difficulties, 20 examples per category, seed=42, source=HF test split.

### Step 3 — Reset workspace (only if restarting from scratch)
```
python reset_experiment.py --dataset FullStack
```

### Step 4 — Run the search
```
python run.py --dataset FullStack \
  --opt_model_name "google/gemini-2.5-flash" \
  --exec_model_name "openai/gpt-4o-mini-2024-07-18" \
  --max_rounds 20
```

### Step 5 — Run held-out evaluation on best workflow
In `eval_best_workflow.py` set `DATASET = "FullStack"` (line 37), then:
```
python eval_best_workflow.py
```
Evaluates on up to 50 held-out examples per category (seed=99). Training examples are fingerprinted by `id` and excluded — zero overlap guaranteed.
Note: SandboxFusion must be running before calling this step.

---

## SciCode

This integration does not read or copy a ReMAS JSONL. It reconstructs both
splits from the official `SciCode1/SciCode` Hugging Face dataset at revision
`4510f6a6aa27c43fad7b43da2c59602a86e88480`, using frozen problem IDs and
ordering. A generated manifest hashes every complete record and evaluation
fails if the data later drifts.

### Step 1 — Install the official comparison helpers

Some selected official tests import `scicode.compare`. Install the official
package at the same pinned source commit used for reference code:

```
python -m pip install "scicode @ git+https://github.com/scicode-bench/SciCode.git@e3158ea011d4235245a547460d3688d7ccbf9900"
```

### Step 2 — Reconstruct the exact search and held-out records

```
python data/build_scicode_data.py
```

Outputs:

- `data/datasets/scicode_validate.jsonl`: 9 complete main problems, 3 per field.
- `data/datasets/scicode_test.jsonl`: 30 complete main problems, 10 per field.
- `data/datasets/scicode_manifest.json`: pinned source metadata and full-record hashes.

The search set contains 38 scored subproblems and 120 official tests. The
held-out set contains 113 scored subproblems and 358 official tests.

The exact search order is Mathematics `40, 31, 18`; Physics `19, 73, 67`;
Material Science `36, 80, 35`. The exact held-out order is Mathematics
`24, 5, 3, 78, 54, 9, 74, 29, 1, 63`; Physics
`6, 37, 8, 70, 48, 59, 72, 58, 45, 49`; Material Science
`47, 64, 21, 77, 42, 39, 34, 27, 79, 51`. Search and held-out sets have no
overlap.

### Step 3 — Download the official numeric targets

```
python data/download_scicode_data.py
```

This downloads and validates the official approximately 1 GiB
`data/scicode/test_data.h5`. To use an existing copy instead, set
`SCICODE_H5_PATH` to its absolute path.

On this machine, the existing official file already validates all selected
tests, so it can be reused without copying it:

```
set SCICODE_H5_PATH=C:\Users\cheikh\.ReMAS\data\scicode\test_data.h5
```

This reuses only SciCode's official numeric targets; the AFlow splits are still
reconstructed independently and never read a ReMAS JSONL.

### Step 4 — Reset and run search

```
python reset_experiment.py --dataset SciCode
python run.py --dataset SciCode \
  --opt_model_name "deepseek/deepseek-v4-flash" \
  --exec_model_name "deepseek/deepseek-v4-flash-noreason" \
  --max_rounds 30 --validation_rounds 1 --api_concurrency 20
```

`--api_concurrency` is a shared cap on in-flight execution-model requests.
SciCode preserves sequential subproblems within each main-problem trajectory;
different main problems may run concurrently up to this cap.

Each JSONL row is one complete main problem. Its subproblems execute in order,
and code generated for earlier steps is supplied to later steps. Concurrency is
capped at three complete trajectories. Search fitness is the equal-weight mean
of the three field Subproblem Pass Rates, matching the ReMAS setup; the global
subproblem rate and Main Problem Resolve Rate are also reported.

### Step 5 — Run the exact held-out evaluation

Set `DATASET = "SciCode"` near the top of `eval_best_workflow.py`, then run:

```
python eval_best_workflow.py
```

SciCode held-out evaluation uses the frozen 30 records directly, does not
resample them, and runs one evaluation round to match ReMAS.

### Verification

```
python -m unittest discover -s tests -p "test_scicode.py" -v
```

---

## Mind2Web

The ReMAS experiment calls this benchmark Mind2Web (occasionally described as
Web2Mind). This AFlow integration reconstructs the same benchmark from the
official `osunlp/Mind2Web` source and official saved candidate ranks; it does
not read a ReMAS JSONL at runtime. The official dataset revision is pinned to
`17ece8eb89862368edc0cc806acee6fca5163474`, and the 245,190,981-byte
`scores_all_data.pkl` must match SHA-256
`884c97cd9ae0544485d21ea39e0d46422aee0291969a7324e56df3a84466dbd7`.

This is the same public-train proxy used by ReMAS, not the private official
Mind2Web test split. The 1,009 public training tasks are the source pool.

### Step 1 — Reconstruct the exact search and held-out records

The first build downloads about 5.93 GB and can require roughly 15 GB of
temporary disk space. Do not start it on a nearly full drive without checking
free space first.

To reuse the already verified official rank file on this machine, without
copying it, run in Command Prompt:

```
set MIND2WEB_DATA_DIR=C:\Users\cheikh\.ReMAS\data\mind2web
python data/build_mind2web_data.py
```

The existing directory currently has the rank file but not the complete
official train cache, so the builder will still download missing train shards.
It writes only compact, pruned AFlow records:

- `data/datasets/mind2web_validate.jsonl`: 60 tasks, 20 per domain.
- `data/datasets/mind2web_test.jsonl`: 300 disjoint tasks, 100 per domain.
- `data/datasets/mind2web_manifest.json`: pinned settings, ordered IDs, and a
  SHA-256 hash of every complete generated record.

The domain order is Travel, Shopping, Entertainment. Search uses one shared
seed-42 RNG sequentially across those domain pools. Held-out selection removes
all search IDs, then uses one shared seed-99 RNG in the same order. This is
important: resetting the RNG independently for each domain would select
different tasks.

An optional completed full ReMAS search artifact can be used to verify IDs and
order, but never as the source data:

```
python data/build_mind2web_data.py --verify-remas-artifact C:\path\to\eval_data.jsonl
```

The current ReMAS workspace contains only a three-task smoke artifact, not a
completed 60-task default artifact, so it is intentionally not accepted as the
full benchmark.

### Step 2 — Reset and run search

```
python reset_experiment.py --dataset Mind2Web
python run.py --dataset Mind2Web \
  --opt_model_name "deepseek/deepseek-v4-flash" \
  --exec_model_name "deepseek/deepseek-v4-flash-noreason" \
  --max_rounds 20 --validation_rounds 1
```

When these model configs use OpenRouter, AFlow automatically applies the exact
Mind2Web-specific ReMAS policy to both optimizer and executor calls: DeepSeek
only, provider fallback disabled, and reasoning effort set to `none`. Other
benchmarks and non-OpenRouter endpoints are unchanged.

Each task is a complete human action trajectory, but every current action gets
a fresh workflow instance. The prompt is teacher-forced with only the gold
actions before that step; generated actions never alter a later page or prompt.
All actions for a task can run concurrently, with a global limit of 32 LLM
calls. Context uses official top-20 saved ranks and the official DOM
neighborhood pruning. A gold element missed by the ranker is not injected and
that step scores zero without calling the workflow.

The primary fitness is the equal-weight mean of the Travel, Shopping, and
Entertainment task-macro Step Success Rates. A step succeeds only when the
element is acceptable and action F1 is exactly 1. Element accuracy, action F1,
task success, candidate recall, and generation errors are also recorded.

### Step 3 — Run the exact held-out evaluation

Set `DATASET = "Mind2Web"` near the top of `eval_best_workflow.py`, then run:

```
python eval_best_workflow.py
```

Held-out evaluation loads the frozen 300 records directly, validates their full
contents against the manifest, does not resample, and uses one evaluation round
to match ReMAS.

### Verification

```
python -m unittest discover -s tests -p "test_mind2web.py" -v
```

The generated-file/manifest test is skipped until Step 1 has materialized the
full official corpus; all fixture-based protocol and evaluator tests run
without the download.

---

## Additional run.py flags

| Flag | Default | Description |
|---|---|---|
| `--max_rounds` | 20 | Maximum search iterations |
| `--sample` | 4 | Top-k workflows to sample from each round |
| `--check_convergence` | True | Stop early if top-3 score is flat for 5 consecutive rounds |
| `--validation_rounds` | 1 | Evaluation passes per round |
| `--token_budget` | None (unlimited) | Stop when combined search + execution tokens exceed this value |
| `--initial_round` | 1 | Resume from a specific round |

## Notes

- `run.py` calls `download(["datasets"])` automatically but skips silently if `data/datasets/` already exists — it will not re-download anything.
- Round 1 always uses `workspace/{Dataset}/workflows/round_1/graph.py` as the starting workflow.
- Search results accumulate in `workspace/{Dataset}/workflows/results.json`.
- Token usage is saved to `workspace/{Dataset}/token_usage.json` after each run.
