# -*- coding: utf-8 -*-
# @Desc    : FullStackBench benchmark for AFlow (SandboxFusion-based evaluation)

import asyncio
import os
from collections import defaultdict
from typing import Callable, List, Tuple

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class FullStackBenchmark(BaseBenchmark):
    DATASET_NAME = "full_stack_bench_en"

    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        compile_timeout: int = 50,
        run_timeout: int = 50,
    ):
        super().__init__(name, file_path, log_path)
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.sandbox_endpoint = os.environ.get(
            "SANDBOX_FUSION_ENDPOINT", "http://localhost:8080"
        )

    def _call_sandbox(self, prediction: str, raw_example: dict) -> dict:
        """Submit raw model completion to SandboxFusion. Returns pass_rate and accepted."""
        try:
            response = requests.post(
                f"{self.sandbox_endpoint}/submit",
                json={
                    "dataset": self.DATASET_NAME,
                    "id": raw_example["id"],
                    "completion": prediction,
                    "config": {
                        "dataset_type": "AutoEvalDataset",
                        "compile_timeout": self.compile_timeout,
                        "run_timeout": self.run_timeout,
                        "provided_data": raw_example,
                    },
                },
                timeout=self.compile_timeout + self.run_timeout + 10,
            )
            response.raise_for_status()
            result = response.json()

            accepted = result.get("accepted", False)
            tests = result.get("tests", [])
            if tests:
                passed = sum(1 for t in tests if t.get("passed", False))
                pass_rate = passed / len(tests)
            else:
                pass_rate = 1.0 if accepted else 0.0

            return {"pass_rate": pass_rate, "accepted": accepted, "error": result.get("error")}

        except requests.exceptions.RequestException as e:
            logger.error(f"SandboxFusion request failed: {e}")
            return {"pass_rate": 0.0, "accepted": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {"pass_rate": 0.0, "accepted": False, "error": str(e)}

    def calculate_score(self, expected_output, prediction: str) -> Tuple[float, str]:
        # Scoring is handled inline in evaluate_problem via SandboxFusion
        return 0.0, prediction

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, problem: str, programming_language: str):
        return await graph(problem, programming_language)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple:
        category = problem.get("category", "unknown")
        content = problem["content"]
        programming_language = problem["programming_language"]
        raw_example = problem["raw_example"]

        try:
            prediction, cost = await self._generate_output(graph, content, programming_language)

            # SandboxFusion handles code extraction — pass the raw completion
            result = await asyncio.to_thread(self._call_sandbox, prediction, raw_example)
            score = result["pass_rate"]

            if score < 1.0:
                self.log_mismatch(content, "pass all tests", prediction, score)

            return category, content, prediction, score, cost

        except Exception as e:
            logger.info(f"Error evaluating problem: {e}")
            return category, content, str(e), 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["category", "problem", "prediction", "score", "cost"]

    def save_results_to_csv(self, results, columns):
        avg_score, a_cost, t_cost = super().save_results_to_csv(results, columns)

        df = pd.DataFrame(results, columns=columns)
        category_scores = df.groupby("category")["score"].mean()

        logger.info("--- FullStackBench Per-Category Scores ---")
        for cat, score in category_scores.items():
            logger.info(f"  {cat}: {score:.4f} ({score:.2%})")
        logger.info(f"  OVERALL (avg): {avg_score:.4f} ({avg_score:.2%})")
        logger.info("------------------------------------------")

        return avg_score, a_cost, t_cost
