"""SciCode trajectory benchmark with ReMAS-compatible data and scoring."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pandas as pd

from benchmarks.benchmark import BaseBenchmark
from benchmarks.scicode_evaluator import SciCodeEvaluator
from benchmarks.scicode_spec import (
    SCICODE_FIELD_LABELS,
    SCICODE_REFERENCE_URL,
    SCICODE_SKIPPED_STEPS,
    validate_manifest_records,
)
from scripts.logs import logger
from scripts.utils.common import write_json_file


WITH_BACKGROUND_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""


def extract_python_script(response: str) -> str:
    """Apply the same first-code-block and import stripping used by ReMAS."""

    if "```" in response:
        if "```python" in response:
            code = response.split("```python", 1)[1].split("```", 1)[0]
        else:
            code = response.split("```", 1)[1].split("```", 1)[0]
    else:
        code = response
    return re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)",
        "",
        code,
        flags=re.MULTILINE,
    ).strip()


class SciCodeBenchmark(BaseBenchmark):
    """Evaluate complete SciCode main problems as ordered subproblem trajectories."""

    # Search contains nine complete main problems. Permit all of them to make
    # progress concurrently while the shared LLM gate enforces the API limit.
    max_concurrent_tasks = 20

    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        h5py_file: str | None = None,
        evaluation_timeout: int = 1800,
        generation_timeout: int = 300,
        reference_dir: str | None = None,
    ):
        super().__init__(name, file_path, log_path)
        default_h5 = Path("data") / "scicode" / "test_data.h5"
        self.h5py_file = Path(
            h5py_file or os.environ.get("SCICODE_H5_PATH", str(default_h5))
        ).expanduser().resolve()
        self.reference_dir = Path(
            reference_dir or Path("data") / "scicode" / "reference_steps"
        ).expanduser().resolve()
        self.generation_timeout = generation_timeout
        self._evaluator = SciCodeEvaluator(self.h5py_file, timeout=evaluation_timeout)

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        records = await super().load_data(specific_indices)
        if specific_indices is None:
            filename = Path(self.file_path).name.lower()
            if filename == "scicode_validate.jsonl":
                validate_manifest_records(
                    records,
                    "search",
                    Path(self.file_path).with_name("scicode_manifest.json"),
                )
            elif filename == "scicode_test.jsonl":
                validate_manifest_records(
                    records,
                    "heldout",
                    Path(self.file_path).with_name("scicode_manifest.json"),
                )
        return records

    @staticmethod
    def _step_text(step: dict[str, Any]) -> str:
        return (
            str(step.get("step_description_prompt") or "")
            + "\n"
            + str(step.get("step_background") or "")
        )

    def build_step_prompt(
        self,
        problem: dict[str, Any],
        step_index: int,
        previous_code: list[str],
    ) -> tuple[str, str]:
        steps = problem["sub_steps"]
        if step_index < 0 or step_index >= len(steps):
            raise IndexError(f"Invalid SciCode step index: {step_index}")
        if len(previous_code) != step_index:
            raise ValueError(
                f"Step {step_index + 1} needs {step_index} previous code blocks; "
                f"received {len(previous_code)}."
            )

        previous_sections: list[str] = []
        for index in range(step_index):
            previous_sections.extend(
                [self._step_text(steps[index]), previous_code[index], "------"]
            )
        problem_steps = "\n\n".join(previous_sections[:-1])

        current = steps[step_index]
        function_spec = (
            f"{current.get('function_header', '')}\n\n"
            f"{current.get('return_line', '')}"
        )
        next_step = "\n\n".join([self._step_text(current), function_spec])
        dependencies = str(problem.get("required_dependencies") or "")
        prompt = WITH_BACKGROUND_TEMPLATE.format(
            problem_steps_str=problem_steps,
            next_step_str=next_step,
            dependencies=dependencies,
        )
        previous_code_text = "\n".join(previous_code)
        prefix = f"{dependencies}\n{previous_code_text}\n"
        return prompt, prefix

    def ensure_ready(self) -> None:
        self._evaluator.ensure_ready()
        try:
            helper_spec = importlib.util.find_spec("scicode.compare.cmp")
        except ModuleNotFoundError:
            helper_spec = None
        if self._data_uses_scicode_helpers() and helper_spec is None:
            raise ModuleNotFoundError(
                "The selected official SciCode tests import `scicode.compare`. "
                "Install the pinned official helper package as described in how_to_run.md."
            )

    def _data_uses_scicode_helpers(self) -> bool:
        source = Path(self.file_path) if self.file_path else None
        if source is None or not source.is_file():
            return True
        return "scicode.compare" in source.read_text(encoding="utf-8")

    def _reference_code(self, step_id: str) -> str:
        if step_id not in SCICODE_SKIPPED_STEPS:
            raise ValueError(f"{step_id} is not an official skipped SciCode step.")
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        target = self.reference_dir / f"{step_id}.txt"
        if not target.is_file():
            import requests

            response = requests.get(
                SCICODE_REFERENCE_URL.format(step_id=step_id), timeout=60
            )
            response.raise_for_status()
            target.write_text(response.text, encoding="utf-8")
        return target.read_text(encoding="utf-8")

    @staticmethod
    def score_weight_for_problem(problem: dict[str, Any]) -> int:
        return sum(
            str(step.get("step_number")) not in SCICODE_SKIPPED_STEPS
            for step in problem.get("sub_steps", [])
        )

    @staticmethod
    def _current_cost(graph: Callable) -> float:
        try:
            return float(graph.llm.get_usage_summary()["total_cost"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0.0

    async def _call_graph(self, graph: Callable, prompt: str) -> tuple[str, float]:
        result = await asyncio.wait_for(graph(prompt), timeout=self.generation_timeout)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("SciCode workflow must return (code_text, cumulative_cost).")
        response, cost = result
        return str(response), float(cost)

    async def _run_trajectory(self, problem: dict, graph: Callable) -> tuple:
        previous_code: list[str] = []
        step_results: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        cost = self._current_cost(graph)

        for index, step in enumerate(problem["sub_steps"]):
            step_id = str(step["step_number"])
            if step_id in SCICODE_SKIPPED_STEPS:
                reference_code = self._reference_code(step_id)
                previous_code.append(reference_code)
                step_results.append({"step_id": step_id, "status": "skipped"})
                continue

            prompt, prefix = self.build_step_prompt(problem, index, previous_code)
            try:
                response, cost = await self._call_graph(graph, prompt)
                current_code = extract_python_script(response)
                predictions.append(
                    {
                        "step_id": step_id,
                        "response": response,
                        "extracted_code": current_code,
                    }
                )
            except Exception as exc:
                current_code = ""
                evaluation = {
                    "passed": False,
                    "status": "generation_or_evaluator_error",
                    "returncode": None,
                    "stdout": "",
                    "stderr": str(exc),
                }
                predictions.append(
                    {
                        "step_id": step_id,
                        "response": None,
                        "extracted_code": "",
                        "error": str(exc),
                    }
                )
            else:
                full_code = f"{prefix}\n{current_code}"
                evaluation = await asyncio.to_thread(
                    self._evaluator.evaluate_step, step, full_code
                )

            previous_code.append(current_code)
            step_results.append({"step_id": step_id, **evaluation})

        scored = [result for result in step_results if result["status"] != "skipped"]
        passed = sum(bool(result.get("passed")) for result in scored)
        total = len(scored)
        score = passed / total if total else 0.0
        details = {
            "subproblem_pass_rate": score,
            "passed_subproblems": passed,
            "total_subproblems": total,
            "main_problem_passed": bool(total and passed == total),
            "steps": step_results,
        }
        return (
            str(problem["field"]),
            str(problem["problem_id"]),
            str(problem.get("problem_name", "")),
            json.dumps(predictions, ensure_ascii=False),
            json.dumps(details, ensure_ascii=False),
            passed,
            total,
            details["main_problem_passed"],
            score,
            total,
            cost,
        )

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple:
        try:
            return await self._run_trajectory(problem, graph)
        except Exception as exc:
            total = self.score_weight_for_problem(problem)
            details = {
                "subproblem_pass_rate": 0.0,
                "passed_subproblems": 0,
                "total_subproblems": total,
                "main_problem_passed": False,
                "steps": [],
                "infrastructure_error": str(exc),
            }
            logger.error(
                f"SciCode main problem {problem.get('problem_id')} failed: {exc}"
            )
            return (
                str(problem.get("field", "unknown")),
                str(problem.get("problem_id", "")),
                str(problem.get("problem_name", "")),
                "[]",
                json.dumps(details, ensure_ascii=False),
                0,
                total,
                False,
                0.0,
                total,
                self._current_cost(graph),
            )

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return [
            "field",
            "problem_id",
            "problem_name",
            "prediction",
            "evaluation_details",
            "passed_subproblems",
            "total_subproblems",
            "main_problem_passed",
            "score",
            "score_weight",
            "cost",
        ]

    @staticmethod
    def aggregate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
        field_scores: dict[str, float] = {}
        field_counts: dict[str, dict[str, int]] = {}
        for field in SCICODE_FIELD_LABELS:
            subset = df[df["field"] == field]
            passed = int(subset["passed_subproblems"].sum()) if not subset.empty else 0
            total = int(subset["total_subproblems"].sum()) if not subset.empty else 0
            field_scores[field] = passed / total if total else 0.0
            field_counts[field] = {"passed": passed, "total": total}

        total_passed = int(df["passed_subproblems"].sum()) if not df.empty else 0
        total_subproblems = int(df["total_subproblems"].sum()) if not df.empty else 0
        main_passed = int(df["main_problem_passed"].astype(bool).sum()) if not df.empty else 0
        total_main = len(df)
        return {
            "fitness": sum(field_scores.values()) / len(SCICODE_FIELD_LABELS),
            "field_subproblem_pass_rates": field_scores,
            "field_subproblem_counts": field_counts,
            "global_subproblem_pass_rate": (
                total_passed / total_subproblems if total_subproblems else 0.0
            ),
            "passed_subproblems": total_passed,
            "total_subproblems": total_subproblems,
            "main_problem_resolve_rate": main_passed / total_main if total_main else 0.0,
            "passed_main_problems": main_passed,
            "total_main_problems": total_main,
        }

    def save_results_to_csv(self, results, columns):
        df = pd.DataFrame(results, columns=columns)
        metrics = self.aggregate_dataframe(df)
        fitness = float(metrics["fitness"])
        total_cost = float(df["cost"].max()) if not df.empty else 0.0
        average_cost = total_cost / len(df) if len(df) else 0.0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.log_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{fitness:.5f}_{timestamp}.csv"
        metrics_path = output_dir / f"{fitness:.5f}_{timestamp}_metrics.json"
        df.to_csv(csv_path, index=False)
        write_json_file(metrics_path, metrics, encoding="utf-8", indent=4)

        failures = []
        for row in results:
            record = dict(zip(columns, row))
            if float(record["score"]) >= 1.0:
                continue
            details = json.loads(record["evaluation_details"])
            failed_steps = [
                {
                    "step_id": step.get("step_id"),
                    "status": step.get("status"),
                    "stderr": str(step.get("stderr", ""))[:4000],
                }
                for step in details.get("steps", [])
                if step.get("status") not in {"pass", "skipped"}
            ]
            failures.append(
                {
                    "field": record["field"],
                    "problem_id": record["problem_id"],
                    "problem_name": record["problem_name"],
                    "score": record["score"],
                    "predictions": json.loads(record["prediction"]),
                    "failed_steps": failed_steps,
                }
            )
        if failures:
            log_path = output_dir / "log.json"
            existing = []
            if log_path.is_file():
                try:
                    existing = json.loads(log_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, TypeError):
                    existing = []
            write_json_file(log_path, existing + failures, encoding="utf-8", indent=4)

        logger.info(f"SciCode results saved to {csv_path}")
        for field, score in metrics["field_subproblem_pass_rates"].items():
            counts = metrics["field_subproblem_counts"][field]
            logger.info(
                f"  {SCICODE_FIELD_LABELS[field]}: {score:.5f} "
                f"({counts['passed']}/{counts['total']} subproblems)"
            )
        logger.info(f"SciCode field-macro fitness: {fitness:.5f}")
        logger.info(
            f"SciCode main-problem resolve rate: "
            f"{metrics['main_problem_resolve_rate']:.5f} "
            f"({metrics['passed_main_problems']}/{metrics['total_main_problems']})"
        )
        return fitness, average_cost, total_cost

    def _budget_zero_result(self, problem: dict) -> tuple:
        total = self.score_weight_for_problem(problem)
        details = {
            "subproblem_pass_rate": 0.0,
            "passed_subproblems": 0,
            "total_subproblems": total,
            "main_problem_passed": False,
            "steps": [],
            "budget_skipped": True,
        }
        return (
            str(problem["field"]),
            str(problem["problem_id"]),
            str(problem.get("problem_name", "")),
            "[]",
            json.dumps(details),
            0,
            total,
            False,
            0.0,
            total,
            0.0,
        )

    async def run_evaluation(
        self,
        agent: Callable,
        va_list: List[int],
        max_concurrent_tasks: int = 20,
        budget=None,
    ):
        self.ensure_ready()
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(
            data,
            agent,
            min(max_concurrent_tasks, self.max_concurrent_tasks),
            budget=budget,
        )
        is_partial = any(result is None for result in results)
        complete_results = [
            result if result is not None else self._budget_zero_result(problem)
            for problem, result in zip(data, results)
        ]
        average_score, average_cost, total_cost = self.save_results_to_csv(
            complete_results, self.get_result_columns()
        )

        exec_input_tokens = exec_output_tokens = 0
        try:
            summary = agent.llm.get_usage_summary()
            exec_input_tokens = summary["total_input_tokens"]
            exec_output_tokens = summary["total_output_tokens"]
        except (AttributeError, KeyError, TypeError):
            pass
        scored_units = sum(self.score_weight_for_problem(problem) for problem in data)
        return (
            average_score,
            average_cost,
            total_cost,
            exec_input_tokens,
            exec_output_tokens,
            scored_units,
            is_partial,
        )
