from __future__ import annotations

import unittest

from mathalign_dpo.evaluation.answer_normalization import extract_and_normalize_answer, exact_match, normalize_answer


class AnswerNormalizationTests(unittest.TestCase):
    def test_extracts_boxed_answer(self) -> None:
        answer = extract_and_normalize_answer("Work first. Therefore \\boxed{007}.")

        self.assertTrue(answer.extracted)
        self.assertEqual(answer.normalized_answer, "7")

    def test_normalizes_choices(self) -> None:
        self.assertEqual(normalize_answer("(c)"), "C")
        self.assertTrue(exact_match("option B", "B"))

    def test_reduces_integer_fractions(self) -> None:
        self.assertEqual(normalize_answer("\\frac{6}{8}"), "3/4")
        self.assertEqual(normalize_answer("10/5"), "2")

    def test_normalizes_decimal_surface(self) -> None:
        self.assertEqual(normalize_answer("+0012.3400"), "12.34")

    def test_symbolic_fraction_only_surface_normalizes(self) -> None:
        self.assertEqual(normalize_answer("\\frac{7\\sqrt{2}}{10}"), "\\frac{7\\sqrt{2}}{10}")

    def test_unextractable_answer_scores_incorrect(self) -> None:
        answer = extract_and_normalize_answer("I cannot solve it.")

        self.assertFalse(answer.extracted)
        self.assertIsNone(answer.normalized_answer)
