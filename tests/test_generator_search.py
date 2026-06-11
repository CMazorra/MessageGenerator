import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from evaluator import DynamicEvaluator
from generator import DynamicGenerator


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, model, prompt, options):
        self.prompts.append(prompt)
        if not self.responses:
            return {"response": "[]"}
        return {"response": self.responses.pop(0)}


class FakeSemanticScorer:
    def score(self, text, tone):
        if "sorry" in text.lower() or "care" in text.lower():
            return 1.0
        return 0.0


class FakeToneEvaluator:
    def __init__(self):
        self.hard_evaluator = DynamicEvaluator()

    def evaluate(self, generated_text, constraints, judge_model="llama3"):
        result = self.hard_evaluator.evaluate(
            generated_text,
            {key: value for key, value in constraints.items() if key != "tone"},
            judge_model=judge_model
        )
        if "tone" in constraints:
            tone_passed = "care" in generated_text.lower() or "sorry" in generated_text.lower()
            details = dict(result["details"])
            details["tone"] = tone_passed
            total = len(constraints)
            passed = sum(1 for value in details.values() if value)
            return {"global_score": passed / total, "details": details}
        return result


def build_generator(responses, semantic_scorer=None):
    generator = DynamicGenerator(evaluator=DynamicEvaluator(), semantic_scorer=semantic_scorer)
    generator.client = FakeClient(responses)
    return generator


def build_generator_with_evaluator(responses, evaluator, semantic_scorer=None):
    generator = DynamicGenerator(evaluator=evaluator, semantic_scorer=semantic_scorer)
    generator.client = FakeClient(responses)
    return generator


def test_parse_candidate_steps_from_json_array():
    generator = build_generator([])

    candidates = generator._parse_candidate_steps(
        '["First option.", "Second option.", "First option."]',
        branching_factor=4
    )

    assert candidates == ["First option.", "Second option."]


def test_search_selects_branch_with_required_words_and_no_forbidden_words():
    responses = [
        json.dumps([
            "We are sorry for the shipping delay and will issue a refund.",
            "The mistake was our fault and your package is late.",
            "Your package is delayed."
        ])
    ]
    generator = build_generator(responses, semantic_scorer=FakeSemanticScorer())
    constraints = {
        "max_words": 20,
        "mandatory_words": ["sorry", "shipping", "refund"],
        "forbidden_words": ["mistake", "fault"]
    }

    text, metrics = generator.generate_with_search(
        "Write an apology for a delayed package.",
        constraints,
        "fake-model",
        max_depth=1,
        evaluate_soft=False
    )

    assert text == "We are sorry for the shipping delay and will issue a refund."
    assert metrics["algorithm"] == "heuristic_graph_search"
    assert metrics["nodes_expanded"] == 1
    assert metrics["candidates_evaluated"] == 3
    assert metrics["best_hard_score"] == 1.0


def test_search_expands_exact_lines_as_partial_states():
    responses = [
        json.dumps(["AI cloud tools", "crypto cloud tools"]),
        json.dumps(["workflow automation", "blockchain analytics"])
    ]
    generator = build_generator(responses, semantic_scorer=FakeSemanticScorer())
    constraints = {
        "exact_lines": 2,
        "mandatory_words": ["AI", "cloud"],
        "forbidden_words": ["crypto", "blockchain"]
    }

    text, metrics = generator.generate_with_search(
        "List startup ideas.",
        constraints,
        "fake-model",
        branching_factor=2,
        max_depth=2,
        evaluate_soft=False
    )

    assert text == "AI cloud tools\nworkflow automation"
    assert metrics["search_depth"] == 2
    assert metrics["best_hard_score"] == 1.0


def test_length_lookahead_penalizes_nearly_exhausted_word_budget():
    generator = build_generator([])
    constraints = {
        "max_words": 10,
        "mandatory_words": ["refund"]
    }

    roomy_score = generator._score_length_lookahead(
        "We are sorry today",
        constraints,
        depth=1,
        max_depth=3
    )
    tight_score = generator._score_length_lookahead(
        "We are sorry your package is delayed after shipping",
        constraints,
        depth=1,
        max_depth=3
    )

    assert tight_score < roomy_score


def test_semantic_guidance_prefers_tone_aligned_branch():
    responses = [
        json.dumps([
            "Your package is delayed.",
            "We are sorry and care about making this right."
        ])
    ]
    generator = build_generator(responses, semantic_scorer=FakeSemanticScorer())
    constraints = {
        "max_words": 20,
        "tone": "empathetic"
    }

    text, metrics = generator.generate_with_search(
        "Write an apology for a delayed package.",
        constraints,
        "fake-model",
        max_depth=1,
        evaluate_soft=False,
        semantic_guidance=True,
        semantic_weight=0.5,
        pruning_threshold=-1.0
    )

    assert text == "We are sorry and care about making this right."
    assert metrics["semantic_guidance"] == True


def test_forbidden_word_branch_is_pruned():
    responses = [
        json.dumps([
            "This contains crypto.",
            "This mentions AI and cloud."
        ])
    ]
    generator = build_generator(responses)
    constraints = {
        "mandatory_words": ["AI", "cloud"],
        "forbidden_words": ["crypto"]
    }

    text, metrics = generator.generate_with_search(
        "List a startup idea.",
        constraints,
        "fake-model",
        max_depth=1,
        evaluate_soft=False
    )

    assert text == "This mentions AI and cloud."
    assert metrics["candidates_pruned"] == 1


def test_mcts_selects_branch_with_best_rollout_reward():
    responses = [
        json.dumps([
            "We are sorry for the shipping delay and will issue a refund.",
            "The mistake was our fault."
        ])
    ]
    generator = build_generator(responses)
    constraints = {
        "max_words": 20,
        "mandatory_words": ["sorry", "shipping", "refund"],
        "forbidden_words": ["mistake", "fault"]
    }

    text, metrics = generator.generate_with_mcts(
        "Write an apology for a delayed package.",
        constraints,
        "fake-model",
        iterations=2,
        branching_factor=2,
        max_depth=1,
        evaluate_soft=False,
        semantic_guidance=False,
        rollout_policy="cheap"
    )

    assert text == "We are sorry for the shipping delay and will issue a refund."
    assert metrics["algorithm"] == "mcts"
    assert metrics["rollouts"] == 2
    assert metrics["root_children"] == 2
    assert metrics["best_hard_score"] == 1.0


def test_mcts_rollout_synthesizes_valid_json():
    generator = build_generator([])
    constraints = {
        "format": "json",
        "required_json_keys": ["product_name", "description"],
        "mandatory_words": ["ergonomic", "battery"]
    }

    rollout = generator._quick_rollout_complete(
        "{not-valid",
        "Create a product description.",
        constraints,
        max_depth=1
    )
    parsed = json.loads(rollout)

    assert set(parsed.keys()) == {"product_name", "description"}
    assert "ergonomic" in rollout
    assert "battery" in rollout


def test_mcts_repairs_tone_once_when_final_judge_fails():
    responses = [
        json.dumps(["Your package is delayed."]),
        "We are sorry and care about making this right."
    ]
    generator = build_generator_with_evaluator(responses, FakeToneEvaluator())
    constraints = {
        "max_words": 20,
        "tone": "empathetic"
    }

    text, metrics = generator.generate_with_mcts(
        "Write an apology for a delayed package.",
        constraints,
        "fake-model",
        iterations=1,
        branching_factor=1,
        max_depth=1,
        evaluate_soft=True,
        semantic_guidance=False,
        tone_repair=True,
        rollout_policy="cheap"
    )

    assert text == "We are sorry and care about making this right."
    assert metrics["tone_repair_used"] == 1
    assert metrics["final_details"]["tone"] == True


def test_mcts_can_use_llm_rollout_for_fluent_completion():
    responses = [
        json.dumps(["Your package is delayed"]),
        "We are sorry your package is delayed, and we will arrange a refund for the shipping issue."
    ]
    generator = build_generator(responses, semantic_scorer=FakeSemanticScorer())
    constraints = {
        "max_words": 20,
        "mandatory_words": ["sorry", "shipping", "refund"],
        "tone": "empathetic"
    }

    text, metrics = generator.generate_with_mcts(
        "Write an apology for a delayed package.",
        constraints,
        "fake-model",
        iterations=1,
        branching_factor=1,
        max_depth=1,
        evaluate_soft=False,
        semantic_guidance=True,
        rollout_policy="llm"
    )

    assert text == "We are sorry your package is delayed, and we will arrange a refund for the shipping issue."
    assert metrics["rollout_policy"] == "llm"
