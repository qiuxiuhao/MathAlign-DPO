from __future__ import annotations

import unittest

from mathalign_dpo.data.build_preferences import build_dpo_examples, build_manual_review_examples
from mathalign_dpo.data.build_sft import build_sft_example
from mathalign_dpo.data.mutate_steps import mutate_number, mutate_operator, mutate_step


CONFIG = {
    "project": {"seed": 42},
    "preprocessing": {
        "system_prompt": "You are careful.",
        "user_instruction": "Solve step by step.",
        "require_final_answer_for_dpo": True,
    },
    "negative_sampling": {
        "strategy": "mixed",
        "number_offset_choices": [-2, -1, 1, 2],
    },
}


def _step_example() -> dict[str, object]:
    return {
        "id": "numina_train_00000010",
        "source_id": "00000010",
        "problem": "Compute 2+3.",
        "solution": "1. We compute 2+3=5.\n2. Therefore \\boxed{5}.",
        "steps": ["1. We compute 2+3=5.", "2. Therefore the answer is \\boxed{5}."],
        "final_answer": "5",
        "parse_status": "success",
        "metadata": {"answer_confidence": "high"},
    }


def _many_step_example(source_id: str, normalized_id: str, confidence: str = "high") -> dict[str, object]:
    return {
        "id": normalized_id,
        "source_id": source_id,
        "problem": f"Compute values for {source_id}.",
        "solution": "1. We compute 1+1=2.\n2. Then 2+2=4.\n3. Finally \\boxed{4}.",
        "steps": ["1. We compute 1+1=2.", "2. Then 2+2=4.", "3. Finally \\boxed{4}."],
        "final_answer": "4" if confidence in {"high", "medium"} else None,
        "parse_status": "success" if confidence in {"high", "medium"} else "partial",
        "metadata": {"answer_confidence": confidence},
    }


class Stage2BuilderTests(unittest.TestCase):
    def test_sft_messages_preserve_full_solution_and_null_token_count(self) -> None:
        sft = build_sft_example(_step_example(), CONFIG)

        self.assertEqual([message["role"] for message in sft["messages"]], ["system", "user", "assistant"])
        self.assertEqual(sft["messages"][2]["content"], _step_example()["solution"])
        self.assertIsNone(sft["token_count"])

    def test_number_mutation_is_deterministic_and_skips_step_label(self) -> None:
        first = mutate_number("1. Add 2+3=5.", "00000010", 0, 42, [-1, 1])
        second = mutate_number("1. Add 2+3=5.", "00000010", 0, 42, [-1, 1])

        self.assertTrue(first.success)
        self.assertEqual(first, second)
        self.assertNotEqual(first.changed_span, (0, 1))
        self.assertNotEqual(first.text.strip(), "1. Add 2+3=5.")

    def test_operator_mutation_avoids_unary_minus(self) -> None:
        result = mutate_operator("Use -3+5=2.", "00000010", 0, 42)

        self.assertTrue(result.success)
        self.assertIn("-3", result.text)
        self.assertNotEqual(result.text, "Use -3+5=2.")

    def test_mixed_falls_back_when_first_strategy_cannot_mutate(self) -> None:
        result = mutate_step("No numbers but x+y=z.", "abc", 0, "mixed", 42, [1])

        self.assertTrue(result.success)
        self.assertEqual(result.strategy, "operator_mutation")

    def test_dpo_prompt_contains_only_previous_steps(self) -> None:
        pairs, failures = build_dpo_examples([_step_example()], CONFIG, maximum=10)

        self.assertEqual(failures, {})
        self.assertGreaterEqual(len(pairs), 2)
        second = next(row for row in pairs if row["step_index"] == 1)
        prompt_text = "\n".join(message["content"] for message in second["prompt"])
        self.assertIn("1. We compute 2+3=5.", prompt_text)
        self.assertNotIn("2. Therefore the answer is", prompt_text)
        self.assertNotEqual(second["chosen"][0]["content"], second["rejected"][0]["content"])
        self.assertIsNone(second["token_count"])
        self.assertNotIn("_mixed", second["id"])
        self.assertIn(second["metadata"]["mutation"]["strategy"], second["id"])

    def test_dpo_sampling_is_deterministic_and_limits_per_source(self) -> None:
        examples = [
            _many_step_example("src_a", "numina_train_a"),
            _many_step_example("src_b", "numina_train_b"),
            _many_step_example("src_c", "numina_train_c"),
        ]

        first, _ = build_dpo_examples(examples, CONFIG, maximum=6, max_pairs_per_source=1)
        second, _ = build_dpo_examples(list(reversed(examples)), CONFIG, maximum=6, max_pairs_per_source=1)

        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        source_counts = {}
        for row in first:
            source_counts[row["source_id"]] = source_counts.get(row["source_id"], 0) + 1
        self.assertLessEqual(max(source_counts.values()), 1)

    def test_low_confidence_does_not_enter_dpo(self) -> None:
        low = _many_step_example("src_low", "numina_train_low", confidence="low")
        low["parse_status"] = "success"
        low["final_answer"] = "4"
        examples = [low]

        pairs, failures = build_dpo_examples(examples, CONFIG, maximum=3)

        self.assertEqual(pairs, [])
        self.assertEqual(failures["skipped_answer_confidence_low"], 1)

    def test_manual_review_sampling_is_deterministic(self) -> None:
        pairs, _ = build_dpo_examples([_step_example()], CONFIG, maximum=10)

        first = build_manual_review_examples(pairs, sample_count=1, seed=42)
        second = build_manual_review_examples(pairs, sample_count=1, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first[0]["dpo_id"], second[0]["dpo_id"])
