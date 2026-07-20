from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmarks.mind2web import (
    Mind2WebBenchmark,
    Mind2WebWorkflowFactory,
    apply_mind2web_llm_routing,
    official_token_f1,
)
from benchmarks.mind2web_dom import Mind2WebDOMError, format_pruned_html
from benchmarks.mind2web_spec import select_protocol_splits, validate_manifest_records
from data.build_mind2web_data import build_step


ROOT = Path(__file__).resolve().parents[1]


class _Usage:
    def __init__(self) -> None:
        self.call_gate = None

    def get_usage_summary(self):
        return {
            "total_input_tokens": 2,
            "total_output_tokens": 1,
            "total_tokens": 3,
            "total_cost": 0.1,
            "call_count": 1,
            "history": [],
        }


class _Workflow:
    instances: list["_Workflow"] = []

    def __init__(self, name, llm_config, dataset) -> None:
        self.llm = _Usage()
        self.__class__.instances.append(self)

    async def __call__(self, prompt):
        return '{"element":"A","operation":"CLICK","value":""}', 0.1


class _PromptGraph:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, prompt: str):
        self.prompts.append(prompt)
        if "step-two-marker" in prompt:
            return '{"element":"B","operation":"TYPE","value":"hello"}', 0.2
        return '{"element":"A","operation":"CLICK","value":""}', 0.1


def _step(index: int, previous: list[str], candidates: str) -> dict:
    return {
        "step": index,
        "previous_actions": previous,
        "page_context": f"page-{index}",
        "candidates": candidates,
        "letter_to_node_id": {"A": "10", "B": "20"},
    }


def _gold(index: int, acceptable: list[str], op: str, value: str = "") -> dict:
    return {
        "step": index,
        "candidate_letters": ["A", "B"],
        "acceptable_letters": acceptable,
        "acceptable_node_ids": ["10" if "A" in acceptable else "20"],
        "op": op,
        "value": value,
    }


class Mind2WebProtocolTests(unittest.TestCase):
    def test_operator_surface_includes_answer_generate_and_self_correction(self):
        from run import EXPERIMENT_CONFIGS
        from workspace.Mind2Web.workflows.template import operator

        expected = ["Custom", "AnswerGenerate", "ScEnsemble", "Review", "Revise"]
        description_path = (
            ROOT / "workspace" / "Mind2Web" / "workflows" / "template" / "operator.json"
        )
        descriptions = json.loads(description_path.read_text(encoding="utf-8"))

        self.assertEqual(EXPERIMENT_CONFIGS["Mind2Web"].operators, expected)
        self.assertEqual(operator.__all__, expected)
        self.assertEqual(list(descriptions), expected)
        self.assertTrue(hasattr(operator, "AnswerGenerate"))

    def test_openrouter_routing_matches_remas_and_is_benchmark_scoped(self):
        openrouter = type(
            "Config",
            (),
            {
                "base_url": "https://openrouter.ai/api/v1",
                "extra_body": {"provider": {"order": ["other"]}, "custom": True},
            },
        )()
        local = type(
            "Config",
            (),
            {"base_url": "http://localhost:8000/v1", "extra_body": {"custom": True}},
        )()

        apply_mind2web_llm_routing(openrouter)
        apply_mind2web_llm_routing(local)

        self.assertEqual(
            openrouter.extra_body["provider"],
            {"only": ["deepseek"], "allow_fallbacks": False},
        )
        self.assertEqual(openrouter.extra_body["reasoning"], {"effort": "none"})
        self.assertTrue(openrouter.extra_body["custom"])
        self.assertEqual(local.extra_body, {"custom": True})

    def test_shared_rng_split_order_is_frozen_and_disjoint(self):
        records = [
            {"id": f"{domain[0]}-{index:03d}", "domain": domain}
            for domain in ("Travel", "Shopping", "Entertainment")
            for index in range(125)
        ]
        search, heldout = select_protocol_splits(records)

        self.assertEqual(
            [item["id"] for item in search[:8]],
            ["T-081", "T-014", "T-003", "T-094", "T-035", "T-031", "T-028", "T-017"],
        )
        shopping = [item for item in search if item["scenario_id"] == "shopping"]
        entertainment = [
            item for item in search if item["scenario_id"] == "entertainment"
        ]
        self.assertEqual(
            [item["id"] for item in shopping[:4]],
            ["S-003", "S-071", "S-025", "S-091"],
        )
        self.assertEqual(
            [item["id"] for item in entertainment[:4]],
            ["E-027", "E-122", "E-097", "E-043"],
        )
        self.assertEqual(len(search), 60)
        self.assertEqual(len(heldout), 300)
        self.assertFalse(
            {item["id"] for item in search} & {item["id"] for item in heldout}
        )

    def test_generated_files_match_full_content_manifest(self):
        data_dir = ROOT / "data" / "datasets"
        manifest = data_dir / "mind2web_manifest.json"
        if not manifest.is_file():
            self.skipTest("Run data/build_mind2web_data.py to materialize Mind2Web data.")

        for filename, split, expected_size in (
            ("mind2web_validate.jsonl", "search", 60),
            ("mind2web_test.jsonl", "heldout", 300),
        ):
            with (data_dir / filename).open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(records), expected_size)
            validate_manifest_records(records, split, manifest)

    def test_builder_does_not_inject_a_gold_candidate_missed_by_ranker(self):
        action = {
            "action_uid": "a1",
            "cleaned_html": (
                '<html backend_node_id="1"><body backend_node_id="2">'
                '<button backend_node_id="10" title="buy"/>'
                '<button backend_node_id="20" title="cancel"/>'
                "</body></html>"
            ),
            "pos_candidates": [{"backend_node_id": "10"}],
            "neg_candidates": [{"backend_node_id": "20"}],
            "operation": {"op": "CLICK", "value": ""},
        }
        ranks = {"task_a1": {"10": 99, "20": 0}}

        step, gold = build_step(action, 0, [], "task", ranks)

        self.assertEqual(set(step["letter_to_node_id"].values()), {"20"})
        self.assertEqual(gold["acceptable_letters"], [])


class Mind2WebDOMTests(unittest.TestCase):
    def test_pruning_keeps_candidate_marker_and_nearby_siblings(self):
        buttons = "".join(
            f'<button backend_node_id="{index}" title="button {index}"/>'
            for index in range(10, 19)
        )
        html = (
            '<html backend_node_id="1"><body backend_node_id="2">'
            f"{buttons}</body></html>"
        )

        page, candidates = format_pruned_html(html, ["14"])

        self.assertIn("14", candidates)
        self.assertIn("button 14", candidates["14"])
        self.assertIn("button 11", page)
        self.assertIn("button 17", page)
        self.assertNotIn("button 10", page)
        self.assertNotIn("button 18", page)

    def test_missing_ranked_node_fails_closed(self):
        with self.assertRaises(Mind2WebDOMError):
            format_pruned_html('<html backend_node_id="1"/>', ["404"])


class Mind2WebScoringTests(unittest.TestCase):
    def test_parser_and_official_action_f1(self):
        parsed = Mind2WebBenchmark.parse_action(
            "```json\n{element: 'b', operation: 'type', value: 'red shoes'}\n```",
            {"A", "B"},
        )
        self.assertEqual(
            parsed,
            {"element": "B", "operation": "TYPE", "value": "red shoes"},
        )
        self.assertEqual(official_token_f1("TYPE shoes red", "TYPE red shoes"), 1.0)
        self.assertLess(official_token_f1("TYPE red shoes now", "TYPE red shoes"), 1.0)

    def test_step_requires_both_element_and_exact_action_f1(self):
        gold = _gold(0, ["A"], "TYPE", "red shoes")
        right = Mind2WebBenchmark.evaluate_action(
            {"element": "A", "operation": "TYPE", "value": "shoes red"}, gold
        )
        wrong_element = Mind2WebBenchmark.evaluate_action(
            {"element": "B", "operation": "TYPE", "value": "red shoes"}, gold
        )
        extra_value = Mind2WebBenchmark.evaluate_action(
            {"element": "A", "operation": "TYPE", "value": "red shoes now"}, gold
        )
        self.assertTrue(right["step_success"])
        self.assertFalse(wrong_element["step_success"])
        self.assertFalse(extra_value["step_success"])

    def test_domain_macro_uses_task_macro_step_success(self):
        frame = pd.DataFrame(
            [
                {"scenario_id": "travel", "element_acc": 1.0, "action_f1": 1.0, "step_success_rate": 1.0, "task_success": 1.0, "candidate_recall": 1.0, "num_steps": 1, "generation_errors": 0},
                {"scenario_id": "shopping", "element_acc": 0.0, "action_f1": 0.0, "step_success_rate": 0.0, "task_success": 0.0, "candidate_recall": 1.0, "num_steps": 100, "generation_errors": 0},
                {"scenario_id": "entertainment", "element_acc": 0.5, "action_f1": 0.5, "step_success_rate": 0.5, "task_success": 0.0, "candidate_recall": 1.0, "num_steps": 2, "generation_errors": 0},
            ]
        )

        metrics = Mind2WebBenchmark.aggregate_dataframe(frame)

        self.assertEqual(metrics["fitness"], 0.5)
        self.assertNotAlmostEqual(metrics["fitness"], (1 + 1) / 103)


class Mind2WebTrajectoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_actions_are_independent_and_teacher_forced(self):
        benchmark = Mind2WebBenchmark(
            name="Mind2Web", file_path="unused.jsonl", log_path=tempfile.gettempdir()
        )
        graph = _PromptGraph()
        problem = {
            "scenario_id": "travel",
            "domain": "Travel",
            "id": "fixture",
            "task": "perform fixture",
            "website": "example",
            "num_steps": 2,
            "steps": [
                _step(0, [], "A) first-marker\nB) unused"),
                _step(1, ["[button] first -> CLICK"], "A) unused\nB) step-two-marker"),
            ],
            "gold_actions": [
                _gold(0, ["A"], "CLICK"),
                _gold(1, ["B"], "TYPE", "hello"),
            ],
        }

        result = await benchmark.evaluate_problem(problem, graph)

        self.assertEqual(result[10], 1.0)
        self.assertEqual(len(graph.prompts), 2)
        first_prompt = next(item for item in graph.prompts if "first-marker" in item)
        second_prompt = next(item for item in graph.prompts if "step-two-marker" in item)
        self.assertIn("GOLD PREVIOUS ACTIONS", first_prompt)
        self.assertIn("None", first_prompt)
        self.assertNotIn("[button] first -> CLICK", first_prompt)
        self.assertIn("[button] first -> CLICK", second_prompt)

    async def test_candidate_miss_skips_generation(self):
        benchmark = Mind2WebBenchmark(
            name="Mind2Web", file_path="unused.jsonl", log_path=tempfile.gettempdir()
        )
        graph = _PromptGraph()
        problem = {
            "scenario_id": "shopping",
            "domain": "Shopping",
            "id": "miss",
            "task": "fixture",
            "website": "example",
            "num_steps": 1,
            "steps": [_step(0, [], "A) wrong")],
            "gold_actions": [_gold(0, [], "CLICK")],
        }

        result = await benchmark.evaluate_problem(problem, graph)

        self.assertEqual(result[10], 0.0)
        self.assertEqual(result[12], 0.0)
        self.assertEqual(graph.prompts, [])

    async def test_factory_creates_one_workflow_per_action_with_shared_gate(self):
        _Workflow.instances.clear()
        factory = Mind2WebWorkflowFactory(
            workflow_class=_Workflow,
            name="Mind2Web",
            llm_config=object(),
            dataset="Mind2Web",
            max_llm_calls=2,
        )

        await asyncio.gather(factory("one"), factory("two"))

        self.assertEqual(len(_Workflow.instances), 2)
        self.assertIsNot(_Workflow.instances[0], _Workflow.instances[1])
        self.assertIs(_Workflow.instances[0].llm.call_gate, factory.call_gate)
        self.assertIs(_Workflow.instances[1].llm.call_gate, factory.call_gate)
        self.assertEqual(factory.get_usage_summary()["total_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
