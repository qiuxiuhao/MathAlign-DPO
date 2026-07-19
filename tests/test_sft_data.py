from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mathalign_dpo.data.write_outputs import sha256_file
from mathalign_dpo.training.sft_data import (
    load_sft_candidate_pools,
    load_sft_data,
    select_tokenized_sft_data,
    tokenize_and_filter_sft_data,
    validate_stage2_sft_row,
    validate_tokenizer_chat_template,
)


class FakeTokenizer:
    chat_template = "{{ messages }}"
    pad_token = None
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        return [0] * sum(len(message["content"].split()) for message in messages)


class BoundaryTokenizer:
    chat_template = "{{ messages }}"
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        user = messages[1]["content"]
        count = int(user.removeprefix("tokens:"))
        return [0] * count


class SFTDataTests(unittest.TestCase):
    def test_loads_mini_rows_from_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_fixture(root)

            bundle = load_sft_data(config)

            self.assertEqual([row["id"] for row in bundle.train_rows], ["train_a"])
            self.assertEqual([row["id"] for row in bundle.validation_rows], ["val_a"])
            self.assertEqual(bundle.selected_counts, {"train": 1, "validation": 1})

    def test_manifest_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_fixture(root)
            _write_jsonl(root / "sft_train.jsonl", [_sft_row("train_a"), _sft_row("changed")])

            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_sft_data(config)

    def test_missing_mini_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_fixture(root, mini_train_ids=["missing"])

            with self.assertRaisesRegex(ValueError, "missing train SFT IDs"):
                load_sft_data(config)

    def test_stage2_sft_row_requires_null_token_count(self) -> None:
        row = _sft_row("bad")
        row["token_count"] = 10

        with self.assertRaisesRegex(ValueError, "token_count must be null"):
            validate_stage2_sft_row(row)

    def test_tokenizer_template_and_padding_are_validated(self) -> None:
        tokenizer = FakeTokenizer()

        metadata = validate_tokenizer_chat_template(tokenizer)

        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertTrue(metadata["pad_token_set_from_eos"])

    def test_token_filtering_is_stable_and_does_not_truncate(self) -> None:
        rows = [
            _sft_row("short", user="one two", assistant="three"),
            _sft_row("long", user="one two three", assistant="four five six"),
        ]

        tokenized = tokenize_and_filter_sft_data(rows, rows[:1], FakeTokenizer(), max_length=4)

        self.assertEqual([row["id"] for row in tokenized.train_rows], ["short"])
        self.assertEqual(tokenized.train_rows[0]["prompt"][1]["content"], "one two")
        self.assertEqual(tokenized.token_statistics["train"]["filtered_ids"], ["long"])

    def test_boundary_lengths_keep_511_and_512_filter_513(self) -> None:
        rows = [
            _sft_row("len511", user="tokens:511"),
            _sft_row("len512", user="tokens:512"),
            _sft_row("len513", user="tokens:513"),
        ]

        tokenized = tokenize_and_filter_sft_data(rows, rows[:2], BoundaryTokenizer(), max_length=512)

        self.assertEqual([row["id"] for row in tokenized.train_rows], ["len511", "len512"])
        self.assertEqual(tokenized.token_statistics["train"]["filtered_ids"], ["len513"])

    def test_selects_exact_target_after_expanding_candidate_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_fixture(
                root,
                mini_train_ids=["short_a"],
                formal_train_ids=["short_a", "short_b", "short_c"],
                train_rows=[
                    _sft_row("short_a", user="one"),
                    _sft_row("short_b", user="two"),
                    _sft_row("short_c", user="three"),
                ],
            )
            pools = load_sft_candidate_pools(config)

            tokenized = select_tokenized_sft_data(
                pools,
                FakeTokenizer(),
                max_length=4,
                seed=42,
                target_train_count=2,
                target_validation_count=1,
            )

            self.assertEqual(len(tokenized.train_rows), 2)
            self.assertEqual(tokenized.token_statistics["train"]["selected_pool"], "expanded")
            self.assertTrue(tokenized.token_statistics["train"]["selection_hash"])


def _write_stage2_fixture(
    root: Path,
    mini_train_ids: list[str] | None = None,
    formal_train_ids: list[str] | None = None,
    train_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    stage1_manifest = root / "split_manifest.json"
    stage1_manifest.write_text(json.dumps({"completed": True}), encoding="utf-8")
    train_path = root / "sft_train.jsonl"
    validation_path = root / "sft_validation.jsonl"
    actual_train_rows = train_rows or [_sft_row("train_a"), _sft_row("train_b")]
    _write_jsonl(train_path, actual_train_rows)
    _write_jsonl(validation_path, [_sft_row("val_a")])
    manifest = {
        "schema_version": "1.0",
        "stage": 2,
        "completed": True,
        "token_length_status": "not_checked_no_tokenizer",
        "stage1_manifest_file": {"path": str(stage1_manifest), "sha256": sha256_file(stage1_manifest)},
        "files": {
            "sft_train": {"path": str(train_path), "rows": len(actual_train_rows), "sha256": sha256_file(train_path)},
            "sft_validation": {"path": str(validation_path), "rows": 1, "sha256": sha256_file(validation_path)},
        },
        "views": {
            "mini": {
                "sft": {
                    "train": mini_train_ids or ["train_a"],
                    "validation": ["val_a"],
                }
            },
            "formal": {
                "sft": {
                    "train": formal_train_ids or ["train_a", "train_b"],
                    "validation": ["val_a"],
                }
            }
        },
    }
    manifest_path = root / "stage2_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"project": {"run_mode": "mini"}, "data": {"stage2_manifest_file": str(manifest_path)}}


def _sft_row(row_id: str, user: str = "problem", assistant: str = "solution") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "id": row_id,
        "source_id": row_id.removesuffix("_sft"),
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "token_count": None,
        "metadata": {"token_length_status": "not_checked_no_tokenizer"},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
