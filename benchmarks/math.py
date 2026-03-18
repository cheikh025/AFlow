import inspect
from math import isclose
from typing import Any, Callable, List, Tuple

import pandas as pd
import regex
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


# ── Answer extraction (identical to MAS_pro) ─────────────────────────────────

def _last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx: right_brace_idx + 1]


def _remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left):]
    left = "\\boxed{"
    assert s[:len(left)] == left and s[-1] == "}"
    return s[len(left):-1]


# ── String normalization (MAS_pro strategy) ───────────────────────────────────

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (AssertionError, ValueError):
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _parse_digits(num) -> float | None:
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def _try_parse_numeric(s) -> float | None:
    """Parse a numeric value, also handling LaTeX \\frac{a}{b} patterns."""
    val = _parse_digits(s)
    if val is not None:
        return val
    m = regex.match(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', str(s).strip())
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    return None


class MATHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        boxed = _last_boxed_only_string(text)
        if boxed:
            try:
                return _remove_boxed(boxed)
            except Exception:
                pass
        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = regex.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """
        Two-stage comparison matching MAS_pro strategy:
          1. Symbolic string equivalence after LaTeX normalization (_strip_string)
          2. Numeric comparison with abs_tol=1e-5 (also handles \\frac{a}{b})
        No SymPy — ensures identical scoring to MAS_pro.
        """
        try:
            if _strip_string(str(prediction)) == _strip_string(str(reference)):
                return True
        except Exception:
            if str(prediction) == str(reference):
                return True
        try:
            p = _try_parse_numeric(prediction)
            r = _try_parse_numeric(reference)
            if p is not None and r is not None:
                return isclose(p, r, abs_tol=1e-5)
        except Exception:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, str, int, float]:
        subject = problem.get("subject", problem.get("type", "unknown"))
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return subject, input_text, output, expected_output, uni_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return subject, input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["subject", "question", "prediction", "expected_output", "score", "cost"]

    def save_results_to_csv(self, results, columns):
        """Save CSV and log per-subject breakdown alongside the combined score."""
        avg_score, a_cost, t_cost = super().save_results_to_csv(results, columns)

        df = pd.DataFrame(results, columns=columns)
        subject_scores = df.groupby("subject")["score"].mean()

        logger.info("--- MATH Per-Subject Scores (Level 5) ---")
        for subject, score in subject_scores.items():
            logger.info(f"  {subject}: {score:.4f} ({score:.2%})")
        logger.info(f"  OVERALL (avg): {avg_score:.4f} ({avg_score:.2%})")
        logger.info("------------------------------------------")

        return avg_score, a_cost, t_cost
