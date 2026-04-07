# -*- coding: utf-8 -*-
# @Desc    : MMLU-Pro benchmark for AFlow (4-category, 10-way MCQ A-J)

import inspect
import json
import random
import re
from typing import Callable, List, Optional, Tuple

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MMLUProBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    _CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    _LETTERS = set(_CHOICES)

    @staticmethod
    def extract_answer_letter(text: str) -> Optional[str]:
        """Extract A-J from model output."""
        if not text:
            return None
        text = text.strip()

        # Step 0: JSON parse — handle {"response": "I", ...} style outputs
        try:
            obj = json.loads(text)
            for key in ("response", "answer", "Answer", "choice"):
                val = str(obj.get(key, "")).strip().upper()
                if len(val) == 1 and val in MMLUProBenchmark._LETTERS:
                    return val
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        # Step 1: "answer is X"
        match = re.search(r"answer is \(?([A-J])\)?", text)
        if match:
            return match.group(1)

        # Step 2: "Answer: X" or "answer: X"
        match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
        if match:
            return match.group(1)

        # Step 3: "Option X" / "option X"
        match = re.search(r'[oO]ption\s+([A-J])\b', text)
        if match:
            return match.group(1)

        # Step 4: last non-empty line is a bare letter
        for line in reversed(text.splitlines()):
            stripped = line.strip().strip('()."\'')
            if len(stripped) == 1 and stripped.upper() in MMLUProBenchmark._LETTERS:
                return stripped.upper()

        # Step 5: \boxed{X}
        match = re.search(r'\\boxed\{([A-J])\}', text)
        if match:
            return match.group(1)

        # Step 6: last standalone A-J in text
        match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
        if match:
            return match.group(0)

        return None

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        extracted = self.extract_answer_letter(prediction) if prediction else None
        if extracted is None:
            extracted = random.choice(self._CHOICES)
        score = 1.0 if extracted == expected_output else 0.0
        return score, extracted

    def get_function_code(self, func):
        try:
            return inspect.getsource(func)
        except OSError:
            return "no code"

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
                self.log_mismatch(
                    input_text,
                    expected,
                    output,
                    extracted,
                    extract_answer_code=self.get_function_code(self.extract_answer_letter),
                )

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
