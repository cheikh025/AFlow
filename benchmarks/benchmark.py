import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from scripts.logs import logger
from scripts.utils.common import write_json_file


class BaseBenchmark(ABC):
    def __init__(self, name: str, file_path: str, log_path: str):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 50, budget=None):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        lock = asyncio.Lock()
        checkpoint = [0]  # mutable container so the nested function can update it

        async def sem_evaluate(problem):
            # If budget already exceeded, skip without running
            if budget is not None and budget.exceeded.is_set():
                return None
            async with semaphore:
                if budget is not None and budget.exceeded.is_set():
                    return None
                result = await self.evaluate_problem(problem, agent)
                # Compute token delta since last checkpoint and consume budget
                if budget is not None:
                    async with lock:
                        try:
                            summary = agent.llm.get_usage_summary()
                            current = summary["total_input_tokens"] + summary["total_output_tokens"]
                        except AttributeError:
                            current = checkpoint[0]
                        delta = current - checkpoint[0]
                        checkpoint[0] = current
                        budget.consume(delta)
                return result

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = 50, budget=None):
        data = await self.load_data(va_list)
        n_total = len(data)
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks, budget=budget)
        columns = self.get_result_columns()

        valid_results = [r for r in results if r is not None]
        n_skipped = n_total - len(valid_results)
        is_partial = n_skipped > 0

        if valid_results:
            average_score, average_cost, total_cost = self.save_results_to_csv(valid_results, columns)
            # Penalize score: skipped problems count as 0
            average_score = average_score * len(valid_results) / n_total
        else:
            average_score, average_cost, total_cost = 0.0, 0.0, 0.0

        if is_partial:
            logger.info(
                f"Budget stop: {len(valid_results)}/{n_total} problems evaluated. "
                f"Adjusted score: {average_score:.5f}"
            )
        else:
            logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")

        # Extract token counts from agent's execution LLM tracker
        exec_input_tokens, exec_output_tokens = 0, 0
        try:
            summary = agent.llm.get_usage_summary()
            exec_input_tokens = summary["total_input_tokens"]
            exec_output_tokens = summary["total_output_tokens"]
        except AttributeError:
            pass
        return average_score, average_cost, total_cost, exec_input_tokens, exec_output_tokens, n_total, is_partial
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost

