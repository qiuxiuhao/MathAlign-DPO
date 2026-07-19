from __future__ import annotations

import unittest

from mathalign_dpo.data.load_numina import audit_rows, build_source_id, normalize_rows, normalize_text


PREPROCESSING = {
    "normalize_line_endings": True,
    "strip_outer_whitespace": True,
    "max_consecutive_blank_lines": 1,
}


class NormalizeNuminaTests(unittest.TestCase):
    def test_native_id_is_preferred_when_unique(self) -> None:
        rows = [
            {"id": "a-1", "problem": "What is 1+1?", "solution": "2", "source": "fixture"},
            {"id": "a-2", "problem": "What is 2+2?", "solution": "4", "source": "fixture"},
        ]

        result = normalize_rows(rows, "AI-MO/NuminaMath-CoT", "rev", "train", PREPROCESSING)

        self.assertEqual(result.audit.id_strategy, "native_field")
        self.assertEqual(result.audit.id_field, "id")
        self.assertEqual(result.examples[0]["source_id"], "a-1")
        self.assertEqual(result.examples[0]["id"], "numina_train_a-1")

    def test_row_index_fallback_when_no_native_id(self) -> None:
        rows = [
            {"problem": "What is 1+1?", "solution": "2", "source": "fixture"},
            {"problem": "What is 2+2?", "solution": "4", "source": "fixture"},
        ]

        result = normalize_rows(rows, "AI-MO/NuminaMath-CoT", "rev", "train", PREPROCESSING)

        self.assertEqual(result.audit.id_strategy, "row_index_fallback")
        self.assertIsNone(result.audit.id_field)
        self.assertEqual(result.examples[1]["source_id"], "00000001")
        self.assertEqual(result.examples[1]["id"], "numina_train_00000001")

    def test_invalid_rows_are_rejected_with_reasons(self) -> None:
        rows = [
            {"problem": "", "solution": "2"},
            {"problem": "same", "solution": "same"},
            {"problem": 1, "solution": "2"},
            {"problem": "ok", "solution": None},
            {"problem": "ok", "solution": "fine"},
        ]

        result = normalize_rows(rows, "AI-MO/NuminaMath-CoT", "rev", "train", PREPROCESSING)

        self.assertEqual(len(result.examples), 1)
        self.assertEqual(
            result.rejected,
            {
                "empty_problem": 1,
                "problem_equals_solution": 1,
                "problem_not_string": 1,
                "solution_not_string": 1,
            },
        )

    def test_text_cleanup_preserves_math_but_limits_blank_lines(self) -> None:
        text = "  Let x=1.\r\n\r\n\r\nThen \\boxed{1}.  "

        self.assertEqual(normalize_text(text, PREPROCESSING), "Let x=1.\n\nThen \\boxed{1}.")

    def test_audit_reports_field_types_and_empty_counts(self) -> None:
        rows = [
            {"problem": "p", "solution": "s", "source": ""},
            {"problem": "p2", "solution": "s2", "source": "olympiad"},
        ]

        audit = audit_rows(rows)

        self.assertEqual(audit.fields, ["problem", "solution", "source"])
        self.assertEqual(audit.field_types["problem"], ["str"])
        self.assertEqual(audit.empty_counts["source"], 1)

    def test_source_id_sanitizes_native_values(self) -> None:
        self.assertEqual(build_source_id({"id": " A/B C "}, 3, "id"), "A_B_C")
