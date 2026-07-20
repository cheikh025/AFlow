from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import pandas as pd

from benchmarks.scicode import SciCodeBenchmark, extract_python_script
from benchmarks.scicode_evaluator import SciCodeEvaluator
from benchmarks.scicode_spec import (
    SCICODE_HELDOUT_IDS,
    SCICODE_SEARCH_IDS,
    ordered_pairs,
    validate_manifest_records,
)
from scripts.evaluator import Evaluator


ROOT = Path(__file__).resolve().parents[1]


class _Usage:
    def get_usage_summary(self):
        return {
            "total_cost": 0.25,
            "total_input_tokens": 10,
            "total_output_tokens": 5,
        }


class _Graph:
    def __init__(self):
        self.llm = _Usage()
        self.prompts: list[str] = []

    async def __call__(self, prompt: str):
        self.prompts.append(prompt)
        index = len(self.prompts)
        return f"```python\ndef generated_{index}():\n    return {index}\n```", 0.25


class _PassingEvaluator:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def evaluate_step(self, step, code):
        self.calls.append((str(step["step_number"]), code))
        return {
            "passed": True,
            "status": "pass",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }


class _GateLLM:
    def __init__(self):
        self.call_gate = None


class _GateWorkflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = _GateLLM()


class SciCodeSpecTests(unittest.TestCase):
    def test_operator_surface_includes_answer_generate_and_self_correction(self):
        from run import EXPERIMENT_CONFIGS
        from workspace.SciCode.workflows.template import operator

        expected = ["Custom", "AnswerGenerate", "ScEnsemble", "Review", "Revise"]
        description_path = (
            ROOT / "workspace" / "SciCode" / "workflows" / "template" / "operator.json"
        )
        descriptions = json.loads(description_path.read_text(encoding="utf-8"))

        self.assertEqual(EXPERIMENT_CONFIGS["SciCode"].operators, expected)
        self.assertEqual(operator.__all__, expected)
        self.assertEqual(list(descriptions), expected)
        self.assertTrue(hasattr(operator, "AnswerGenerate"))

    def test_frozen_search_and_heldout_ids_do_not_overlap(self):
        self.assertEqual(
            ordered_pairs("search"),
            [
                ("mathematics", "40"),
                ("mathematics", "31"),
                ("mathematics", "18"),
                ("physics", "19"),
                ("physics", "73"),
                ("physics", "67"),
                ("material_science", "36"),
                ("material_science", "80"),
                ("material_science", "35"),
            ],
        )
        search = {item for values in SCICODE_SEARCH_IDS.values() for item in values}
        heldout = {item for values in SCICODE_HELDOUT_IDS.values() for item in values}
        self.assertFalse(search & heldout)
        self.assertEqual(len(heldout), 30)
        self.assertEqual(
            ordered_pairs("heldout"),
            [
                ("mathematics", "24"),
                ("mathematics", "5"),
                ("mathematics", "3"),
                ("mathematics", "78"),
                ("mathematics", "54"),
                ("mathematics", "9"),
                ("mathematics", "74"),
                ("mathematics", "29"),
                ("mathematics", "1"),
                ("mathematics", "63"),
                ("physics", "6"),
                ("physics", "37"),
                ("physics", "8"),
                ("physics", "70"),
                ("physics", "48"),
                ("physics", "59"),
                ("physics", "72"),
                ("physics", "58"),
                ("physics", "45"),
                ("physics", "49"),
                ("material_science", "47"),
                ("material_science", "64"),
                ("material_science", "21"),
                ("material_science", "77"),
                ("material_science", "42"),
                ("material_science", "39"),
                ("material_science", "34"),
                ("material_science", "27"),
                ("material_science", "79"),
                ("material_science", "51"),
            ],
        )

    def test_generated_files_match_their_full_content_manifest(self):
        data_dir = ROOT / "data" / "datasets"
        manifest = data_dir / "scicode_manifest.json"
        if not manifest.is_file():
            self.skipTest("Run data/build_scicode_data.py to materialize SciCode data.")

        for filename, split, expected_size in (
            ("scicode_validate.jsonl", "search", 9),
            ("scicode_test.jsonl", "heldout", 30),
        ):
            with (data_dir / filename).open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(records), expected_size)
            validate_manifest_records(records, split, manifest)

    def test_field_macro_fitness_matches_remas_scoring(self):
        frame = pd.DataFrame(
            [
                {
                    "field": "mathematics",
                    "passed_subproblems": 1,
                    "total_subproblems": 1,
                    "main_problem_passed": True,
                },
                {
                    "field": "physics",
                    "passed_subproblems": 0,
                    "total_subproblems": 9,
                    "main_problem_passed": False,
                },
                {
                    "field": "material_science",
                    "passed_subproblems": 1,
                    "total_subproblems": 2,
                    "main_problem_passed": False,
                },
            ]
        )
        metrics = SciCodeBenchmark.aggregate_dataframe(frame)
        self.assertAlmostEqual(metrics["fitness"], 0.5)
        self.assertAlmostEqual(metrics["global_subproblem_pass_rate"], 2 / 12)
        self.assertAlmostEqual(metrics["main_problem_resolve_rate"], 1 / 3)

    def test_result_persistence_accepts_aflow_simple_logger(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = SciCodeBenchmark(
                name="SciCode",
                file_path="unused.jsonl",
                log_path=tmp,
                h5py_file="missing.h5",
            )
            details = json.dumps({"steps": [{"step_id": "1.1", "status": "pass"}]})
            rows = [
                (
                    field,
                    str(index),
                    "fixture",
                    "[]",
                    details,
                    1,
                    1,
                    True,
                    1.0,
                    1,
                    0.0,
                )
                for index, field in enumerate(
                    ("mathematics", "physics", "material_science"), 1
                )
            ]

            fitness, _, _ = benchmark.save_results_to_csv(
                rows, benchmark.get_result_columns()
            )

            self.assertEqual(fitness, 1.0)
            self.assertEqual(len(list(Path(tmp).glob("*_metrics.json"))), 1)

    def test_official_code_extraction_removes_imports(self):
        response = (
            "before```python\nimport math\nfrom numpy import array\n"
            "def f():\n    return 3\n```after"
        )
        self.assertEqual(extract_python_script(response), "def f():\n    return 3")


class SciCodeEvaluatorTests(unittest.TestCase):
    def test_hdf5_fixture_executes_official_style_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "targets.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("900.1/test1/answer", data=10)

            evaluator = SciCodeEvaluator(h5_path, timeout=10)
            result = evaluator.evaluate_step(
                {
                    "step_number": "900.1",
                    "test_cases": ["assert square_plus_one(3) == target"],
                },
                "def square_plus_one(value):\n    return value * value + 1",
            )
            self.assertTrue(result["passed"], result["stderr"])
            self.assertEqual(result["status"], "pass")


class SciCodeTrajectoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_api_concurrency_is_capped_at_twenty(self):
        evaluator = Evaluator(eval_path=tempfile.gettempdir())
        workflow = await evaluator._configure_graph(
            "SciCode",
            _GateWorkflow,
            {"dataset": "SciCode", "llm_config": object()},
            max_api_concurrency=20,
        )

        self.assertEqual(SciCodeBenchmark.max_concurrent_tasks, 20)
        self.assertIsNotNone(workflow.llm.call_gate)
        self.assertEqual(workflow.llm.call_gate._value, 20)

    async def test_previous_generated_code_is_propagated_to_later_steps(self):
        benchmark = SciCodeBenchmark(
            name="SciCode",
            file_path="unused.jsonl",
            log_path=tempfile.gettempdir(),
            h5py_file="missing.h5",
        )
        fake_evaluator = _PassingEvaluator()
        benchmark._evaluator = fake_evaluator
        graph = _Graph()
        problem = {
            "field": "mathematics",
            "problem_id": "40",
            "problem_name": "fixture",
            "required_dependencies": "import numpy as np",
            "sub_steps": [
                {
                    "step_number": "40.1",
                    "step_description_prompt": "first description",
                    "step_background": "first background",
                    "function_header": "def generated_1():",
                    "return_line": "return result",
                    "test_cases": [],
                },
                {
                    "step_number": "40.2",
                    "step_description_prompt": "second description",
                    "step_background": "second background",
                    "function_header": "def generated_2():",
                    "return_line": "return result",
                    "test_cases": [],
                },
            ],
        }

        result = await benchmark._run_trajectory(problem, graph)

        self.assertEqual(result[5:7], (2, 2))
        self.assertTrue(result[7])
        self.assertIn("def generated_1():", graph.prompts[1])
        self.assertIn("def generated_1():", fake_evaluator.calls[1][1])
        self.assertTrue(fake_evaluator.calls[1][1].startswith("import numpy as np"))


if __name__ == "__main__":
    unittest.main()
