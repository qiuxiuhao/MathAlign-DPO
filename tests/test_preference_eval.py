from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mathalign_dpo.evaluation.preference_eval import load_preference_validation_rows, preference_summary


class PreferenceEvalTests(unittest.TestCase):
    def test_preference_summary_counts_strict_chosen_wins(self) -> None:
        rows = [
            {"model_stage": "dpo", "logp_margin": 0.5, "preferred_chosen": True},
            {"model_stage": "dpo", "logp_margin": 0.0, "preferred_chosen": False},
            {"model_stage": "base", "logp_margin": -1.0, "preferred_chosen": False},
        ]

        summary = preference_summary(rows, "dpo")

        self.assertEqual(summary["num_examples"], 2)
        self.assertEqual(summary["preference_accuracy"], 0.5)
        self.assertEqual(summary["preference_margin_mean"], 0.25)

    def test_empty_preference_summary_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "No preference rows"):
            preference_summary([], "dpo")

    def test_loads_selected_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dpo_validation.jsonl"
            rows = [_row("a"), _row("b")]
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")
            config = {"data": {"dpo_validation_file": str(path)}}
            metadata = {"token_statistics": {"validation": {"selected_ids": ["b"]}}}

            selected = load_preference_validation_rows(config, metadata, 1)

            self.assertEqual(selected[0]["id"], "b")


def _row(row_id: str) -> dict[str, object]:
    return {"id": row_id, "source_id": row_id, "prompt": [], "chosen": [], "rejected": []}
