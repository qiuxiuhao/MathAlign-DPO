from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mathalign_dpo.data.write_outputs import sha256_file
from mathalign_dpo.training.dpo_data import (
    count_dpo_tokens,
    load_dpo_candidate_pools,
    select_tokenized_dpo_data,
    validate_stage2_dpo_row,
)


class FakeDPOTokenizer:
    chat_template = "{{ messages }}"
    pad_token = None
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, return_dict=False):
        assert tokenize is True
        count = sum(_message_tokens(message["content"]) for message in messages)
        if add_generation_prompt:
            count += 1
        ids = list(range(count))
        if return_dict:
            return {"input_ids": ids}
        return ids


class DPODataTests(unittest.TestCase):
    def test_loads_only_run_mode_pool_from_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_dpo_fixture(root)

            pools = load_dpo_candidate_pools(config)

            self.assertEqual([row["id"] for row in pools.train_rows], ["train_a"])
            self.assertEqual(pools.candidate_counts["train"], {"run_mode": 1})

    def test_manifest_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_dpo_fixture(root)
            _write_jsonl(root / "dpo_train.jsonl", [_dpo_row("train_a"), _dpo_row("changed")])

            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_dpo_candidate_pools(config)

    def test_missing_mini_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_dpo_fixture(root, mini_train_ids=["missing"])

            with self.assertRaisesRegex(ValueError, "missing train DPO IDs"):
                load_dpo_candidate_pools(config)

    def test_stage2_dpo_row_requires_null_token_count(self) -> None:
        row = _dpo_row("bad")
        row["token_count"] = {"prompt": 1}

        with self.assertRaisesRegex(ValueError, "token_count must be null"):
            validate_stage2_dpo_row(row)

    def test_dpo_schema_rejects_bad_roles_and_identical_pair(self) -> None:
        bad_role = _dpo_row("bad_role")
        bad_role["chosen"][0]["role"] = "user"
        with self.assertRaisesRegex(ValueError, "chosen must be exactly one assistant"):
            validate_stage2_dpo_row(bad_role)

        identical = _dpo_row("identical")
        identical["rejected"][0]["content"] = identical["chosen"][0]["content"]
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_stage2_dpo_row(identical)

    def test_dpo_schema_rejects_rejected_prompt_leak(self) -> None:
        row = _dpo_row("leak", history="bad step")
        row["rejected"][0]["content"] = "bad step"

        with self.assertRaisesRegex(ValueError, "leaks rejected"):
            validate_stage2_dpo_row(row)

    def test_count_dpo_tokens_uses_generation_prompt_boundary(self) -> None:
        row = _dpo_row("tokens", user="u u", chosen="c c c", rejected="r")

        counts = count_dpo_tokens(FakeDPOTokenizer(), row)

        self.assertEqual(counts["prompt"], 4)
        self.assertEqual(counts["chosen_total"], 6)
        self.assertEqual(counts["rejected_total"], 4)
        self.assertEqual(counts["chosen_completion"], 2)

    def test_rejects_source_ids_outside_mini_source_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_dpo_fixture(
                root,
                mini_train_ids=["train_a"],
                mini_train_source_ids=["other_source"],
            )

            with self.assertRaisesRegex(ValueError, "outside the mini source view"):
                load_dpo_candidate_pools(config)

    def test_fails_instead_of_expanding_candidate_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_dpo_fixture(
                root,
                mini_train_ids=["short_a"],
                formal_train_ids=["short_a", "short_b", "short_c"],
                train_rows=[
                    _dpo_row("short_a", user="u", chosen="c c", rejected="r r"),
                    _dpo_row("short_b", user="u", chosen="c c", rejected="r r"),
                    _dpo_row("short_c", user="u", chosen="c c", rejected="r r"),
                ],
            )
            pools = load_dpo_candidate_pools(config)

            with self.assertRaisesRegex(ValueError, "must not borrow rows from the formal pool"):
                select_tokenized_dpo_data(
                    pools,
                    FakeDPOTokenizer(),
                    max_length=8,
                    max_prompt_length=4,
                    seed=42,
                    target_train_count=2,
                    target_validation_count=1,
                )

    def test_token_filtering_checks_prompt_and_total_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_stage2_dpo_fixture(
                root,
                mini_train_ids=["prompt_long", "total_long", "ok"],
                formal_train_ids=["prompt_long", "total_long", "ok"],
                train_rows=[
                    _dpo_row("prompt_long", user="u " * 5, chosen="c", rejected="r"),
                    _dpo_row("total_long", user="u", chosen="c " * 10, rejected="r"),
                    _dpo_row("ok", user="u", chosen="c c", rejected="r r"),
                ],
            )
            pools = load_dpo_candidate_pools(config)

            tokenized = select_tokenized_dpo_data(
                pools,
                FakeDPOTokenizer(),
                max_length=7,
                max_prompt_length=4,
                seed=42,
                target_train_count=1,
                target_validation_count=1,
            )

            reasons = {row["id"]: row["reason"] for row in tokenized.token_statistics["train"]["filtered"]}
            self.assertEqual(reasons["prompt_long"], "prompt_too_long")
            self.assertEqual(reasons["total_long"], "chosen_too_long")


def _write_stage2_dpo_fixture(
    root: Path,
    mini_train_ids: list[str] | None = None,
    mini_train_source_ids: list[str] | None = None,
    formal_train_ids: list[str] | None = None,
    train_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    stage1_manifest = root / "split_manifest.json"
    stage1_manifest.write_text(json.dumps({"completed": True}), encoding="utf-8")
    train_path = root / "dpo_train.jsonl"
    validation_path = root / "dpo_validation.jsonl"
    actual_train_rows = train_rows or [_dpo_row("train_a"), _dpo_row("train_b")]
    _write_jsonl(train_path, actual_train_rows)
    _write_jsonl(validation_path, [_dpo_row("val_a")])
    manifest = {
        "schema_version": "1.0",
        "stage": 2,
        "completed": True,
        "token_length_status": "not_checked_no_tokenizer",
        "stage1_manifest_file": {"path": str(stage1_manifest), "sha256": sha256_file(stage1_manifest)},
        "files": {
            "dpo_train": {"path": str(train_path), "rows": len(actual_train_rows), "sha256": sha256_file(train_path)},
            "dpo_validation": {"path": str(validation_path), "rows": 1, "sha256": sha256_file(validation_path)},
        },
        "views": {
            "mini": {
                "dpo": {
                    "train": mini_train_ids or ["train_a"],
                    "validation": ["val_a"],
                },
                "dpo_source_ids": {
                    "train": mini_train_source_ids or mini_train_ids or ["train_a"],
                    "validation": ["val_a"],
                },
            },
            "formal": {
                "dpo": {
                    "train": formal_train_ids or ["train_a", "train_b"],
                    "validation": ["val_a"],
                }
            },
        },
    }
    manifest_path = root / "stage2_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"project": {"run_mode": "mini"}, "data": {"stage2_manifest_file": str(manifest_path)}}


def _dpo_row(row_id: str, user: str = "problem", history: str | None = None, chosen: str = "correct step", rejected: str = "wrong step") -> dict[str, object]:
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": user},
    ]
    if history is not None:
        prompt.append({"role": "assistant", "content": history})
    return {
        "schema_version": "1.0",
        "id": row_id,
        "source_id": row_id,
        "step_index": 0,
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "token_count": None,
        "metadata": {"token_length_status": "not_checked_no_tokenizer"},
    }


def _message_tokens(text: str) -> int:
    return len(text.split())


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
