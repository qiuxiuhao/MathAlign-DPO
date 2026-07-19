from __future__ import annotations

import unittest

from mathalign_dpo.data.parse_steps import extract_final_answer, parse_normalized_example, parse_solution


def _example(solution: str) -> dict[str, object]:
    return {
        "id": "numina_train_00000012",
        "source_id": "00000012",
        "problem": "Compute 2+2.",
        "solution": solution,
    }


class ParseStepsTests(unittest.TestCase):
    def test_numbered_solution_success_preserves_source_id(self) -> None:
        parsed = parse_normalized_example(
            _example("1. Compute 2+2=4.\n2. Therefore the final answer is \\boxed{4}."),
            minimum_steps=2,
        )

        self.assertEqual(parsed["id"], "numina_train_00000012")
        self.assertEqual(parsed["source_id"], "00000012")
        self.assertEqual(parsed["parse_status"], "success")
        self.assertEqual(parsed["steps"][0], "1. Compute 2+2=4.")
        self.assertEqual(parsed["final_answer"], "4")
        self.assertIsNone(parsed["metadata"]["parse_failure_reason"])

    def test_partial_when_steps_exist_without_final_answer(self) -> None:
        parsed = parse_solution("First compute the area.\n\nThen simplify the expression.", minimum_steps=2)

        self.assertEqual(parsed.parse_status, "partial")
        self.assertEqual(parsed.steps, ["First compute the area.", "Then simplify the expression."])
        self.assertIsNone(parsed.final_answer)

    def test_failed_when_no_reliable_steps(self) -> None:
        parsed = parse_solution("42", minimum_steps=2)

        self.assertEqual(parsed.parse_status, "failed")
        self.assertEqual(parsed.steps, [])
        self.assertEqual(parsed.failure_reason, "insufficient_steps")

    def test_answer_priority_and_last_boxed(self) -> None:
        boxed = extract_final_answer("First \\boxed{1}. Finally \\fbox{2}.")
        self.assertEqual(boxed.answer, "2")
        self.assertEqual(boxed.confidence, "high")
        self.assertEqual(extract_final_answer("Work\n#### 7").method, "hash_answer")
        self.assertEqual(extract_final_answer("Therefore the correct answer is (C).").confidence, "high")
        self.assertEqual(extract_final_answer("Thus option B is correct.").confidence, "medium")
        self.assertEqual(extract_final_answer("Work is done.\n3/5").method, "last_line_answer")
        self.assertEqual(extract_final_answer("The last line has 3/5.").confidence, "low")

    def test_low_confidence_answer_becomes_partial(self) -> None:
        parsed = parse_solution("First compute a value.\n\nThen the text mentions 3/5.", minimum_steps=2)

        self.assertEqual(parsed.parse_status, "partial")
        self.assertIsNone(parsed.final_answer)
        self.assertEqual(parsed.answer_candidate, "3/5")
        self.assertEqual(parsed.answer_confidence, "low")
