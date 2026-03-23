# -*- coding: utf-8 -*-
# @Desc    : MMLU-Pro benchmark for AFlow (4-category, 10-way MCQ A-J)

import re
from typing import Callable, List, Optional, Tuple

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MMLUProBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    @staticmethod
    def extract_answer_letter(text: str) -> Optional[str]:
        """Extract A-J from model output."""
        if not text:
            return None
        text = text.strip()

        # "Answer: B" / "The answer is C"
        match = re.search(r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*\(?([A-Ja-j])\)?", text)
        if match:
            return match.group(1).upper()

        # \boxed{B}
        match = re.search(r"\\boxed\{([A-Ja-j])\}", text)
        if match:
            return match.group(1).upper()

        # Standalone letter at end of text
        match = re.search(r"\b([A-Ja-j])\s*[.)]*\s*$", text)
        if match:
            return match.group(1).upper()

        # First standalone letter in text
        match = re.search(r"\b([A-Ja-j])\b", text)
        if match:
            return match.group(1).upper()

        return None

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        extracted = self.extract_answer_letter(prediction) if prediction else None
        score = 1.0 if extracted == expected_output else 0.0
        return score, extracted or ""

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text: str):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple:
        category = problem["subject"]  # stored as "subject" for column compatibility
        input_text = f"Question: {problem['question']}\n\n{problem['formatted_choices']}"
        expected = problem["answer"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted = self.calculate_score(expected, output)

            if score == 0:
                self.log_mismatch(input_text, expected, output, extracted)

            return category, input_text, output, expected, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return category, input_text, str(e), expected, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["subject", "question", "prediction", "expected_output", "score", "cost"]

    def save_results_to_csv(self, results, columns):
        """Save CSV and log per-category breakdown alongside the combined score."""
        avg_score, a_cost, t_cost = super().save_results_to_csv(results, columns)

        df = pd.DataFrame(results, columns=columns)
        category_scores = df.groupby("subject")["score"].mean()

        logger.info("--- MMLU-Pro Per-Category Scores ---")
        for category, score in category_scores.items():
            logger.info(f"  {category}: {score:.4f} ({score:.2%})")
        logger.info(f"  OVERALL (avg): {avg_score:.4f} ({avg_score:.2%})")
        logger.info("------------------------------------")

        return avg_score, a_cost, t_cost
