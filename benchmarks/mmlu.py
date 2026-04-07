# -*- coding: utf-8 -*-
# @Desc    : MMLU benchmark for AFlow (4-subject combined, with per-subject logging)

import inspect
import re
from collections import defaultdict
from typing import Callable, List, Optional, Tuple

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MMLUBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    @staticmethod
    def extract_answer_letter(text: str) -> Optional[str]:
        """Extract A/B/C/D from model output using the same logic as MAS optimizer."""
        if not text:
            return None
        text = text.strip()

        # "Answer: B" / "The answer is C"
        match = re.search(r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*\(?([A-Da-d])\)?", text)
        if match:
            return match.group(1).upper()

        # \boxed{B}
        match = re.search(r"\\boxed\{([A-Da-d])\}", text)
        if match:
            return match.group(1).upper()

        # Standalone letter at end of text
        match = re.search(r"\b([A-Da-d])\s*[.)]*\s*$", text)
        if match:
            return match.group(1).upper()

        # First standalone letter in text
        match = re.search(r"\b([A-Da-d])\b", text)
        if match:
            return match.group(1).upper()

        return None

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        extracted = self.extract_answer_letter(prediction) if prediction else None
        score = 1.0 if extracted == expected_output else 0.0
        return score, extracted or ""

    def get_function_code(self, func):
        try:
            return inspect.getsource(func)
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text: str):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple:
        subject = problem["subject"]
        input_text = f"Question: {problem['question']}\n\n{problem['formatted_choices']}"
        expected = problem["answer"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted = self.calculate_score(expected, output)

            if score == 0:
                self.log_mismatch(
                    input_text,
                    expected,
                    output,
                    extracted,
                    extract_answer_code=self.get_function_code(self.extract_answer_letter),
                )

            return subject, input_text, output, expected, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return subject, input_text, str(e), expected, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["subject", "question", "prediction", "expected_output", "score", "cost"]

    def save_results_to_csv(self, results, columns):
        """Save CSV and log per-subject breakdown alongside the combined score."""
        avg_score, a_cost, t_cost = super().save_results_to_csv(results, columns)

        df = pd.DataFrame(results, columns=columns)
        subject_scores = df.groupby("subject")["score"].mean()

        logger.info("--- MMLU Per-Subject Scores ---")
        for subject, score in subject_scores.items():
            logger.info(f"  {subject}: {score:.4f} ({score:.2%})")
        logger.info(f"  OVERALL (avg): {avg_score:.4f} ({avg_score:.2%})")
        logger.info("--------------------------------")

        return avg_score, a_cost, t_cost
