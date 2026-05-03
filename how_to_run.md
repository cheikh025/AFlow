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
python run.py --dataset MATH \
  --opt_model_name "google/gemini-2.5-flash" \
  --exec_model_name "openai/gpt-4o-mini-2024-07-18" \
  --max_rounds 20
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
