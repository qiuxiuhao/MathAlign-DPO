from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mathalign_dpo.data.write_outputs import sha256_file
from mathalign_dpo.evaluation.eval_data import (
    load_stage5_eval_dataset,
    validate_dpo_eval_source,
    validate_no_training_leakage,
    validate_sft_eval_source,
)


class EvaluationDataTests(unittest.TestCase):
    def test_loads_mini_evaluation_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fixture(root)

            rows, lineage = load_stage5_eval_dataset(config, sample_count=2)

            self.assertEqual([row["id"] for row in rows], ["eval_a", "eval_b"])
            self.assertEqual(lineage["selected_count"], 2)
            self.assertEqual(rows[0]["prompt_messages"][0]["role"], "system")

    def test_rejects_training_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fixture(root, eval_rows=[_eval_row("eval_a", "001")])
            eval_rows, _ = load_stage5_eval_dataset(config, sample_count=1)
            sft = {"token_statistics": {"train": {"selected_ids": ["numina_train_001_sft"]}, "validation": {"selected_ids": []}}}
            dpo = {"token_statistics": {"train": {"selected_ids": []}, "validation": {"selected_ids": ["numina_train_003_step_001_number_mutation"]}}}

            with self.assertRaisesRegex(ValueError, "overlaps"):
                validate_no_training_leakage(eval_rows, sft, dpo, config["data"]["stage2_manifest_file"])

    def test_validates_sft_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fixture(root)
            sft_dir = _write_run(root / "sft", config, stage=3, training_stage="sft", counts={"train": 256, "validation": 32})

            source = validate_sft_eval_source(sft_dir, config)

            self.assertEqual(source["run_id"], "sft_run")

    def test_rejects_superseded_dpo_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fixture(root)
            sft_source = {"run_id": "sft_run"}
            dpo_dir = _write_run(root / "dpo", config, stage=4, training_stage="dpo", counts={"train": 256, "validation": 32})

            with self.assertRaisesRegex(ValueError, "Mini-only sample policy"):
                validate_dpo_eval_source(dpo_dir, sft_source, config)

    def test_accepts_mini_only_dpo_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fixture(root)
            dpo_dir = _write_run(
                root / "dpo",
                config,
                stage=4,
                training_stage="dpo",
                counts={"train": config["dpo"]["train_samples"], "validation": config["dpo"]["validation_samples"]},
                selected_pool="run_mode",
                stability=True,
            )

            source = validate_dpo_eval_source(dpo_dir, {"run_id": "sft_run"}, config)

            self.assertEqual(source["run_id"], "dpo_run")


def _write_fixture(root: Path, eval_rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    step_eval = root / "step_eval.jsonl"
    sft_train = root / "sft_train.jsonl"
    sft_validation = root / "sft_validation.jsonl"
    dpo_train = root / "dpo_train.jsonl"
    dpo_validation = root / "dpo_validation.jsonl"
    actual_eval = eval_rows or [_eval_row("eval_a", "101"), _eval_row("eval_b", "102"), _eval_row("eval_c", "103")]
    _write_jsonl(step_eval, actual_eval)
    _write_jsonl(sft_train, [_sft_row("numina_train_001_sft", "001")])
    _write_jsonl(sft_validation, [])
    _write_jsonl(dpo_train, [_dpo_row("numina_train_002_step_001_number_mutation", "002")])
    _write_jsonl(dpo_validation, [_dpo_row("numina_train_003_step_001_number_mutation", "003")])
    manifest = {
        "stage": 2,
        "completed": True,
        "files": {
            "step_evaluation": {"path": str(step_eval), "rows": len(actual_eval), "sha256": sha256_file(step_eval)},
            "sft_train": {"path": str(sft_train), "rows": 1, "sha256": sha256_file(sft_train)},
            "sft_validation": {"path": str(sft_validation), "rows": 0, "sha256": sha256_file(sft_validation)},
            "dpo_train": {"path": str(dpo_train), "rows": 1, "sha256": sha256_file(dpo_train)},
            "dpo_validation": {"path": str(dpo_validation), "rows": 1, "sha256": sha256_file(dpo_validation)},
        },
        "views": {"mini": {"step": {"evaluation": [row["id"] for row in actual_eval]}}},
    }
    manifest_path = root / "stage2_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "project": {"run_mode": "mini", "seed": 42},
        "model": {"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "rev", "torch_dtype": "float16"},
        "runtime": {"backend": "mps"},
        "data": {
            "stage2_manifest_file": str(manifest_path),
            "dpo_validation_file": str(dpo_validation),
        },
        "preprocessing": {"system_prompt": "system", "user_instruction": "solve"},
        "dpo": {"train_samples": 179, "validation_samples": 21},
    }


def _write_run(root: Path, config: dict[str, object], stage: int, training_stage: str, counts: dict[str, int], selected_pool="expanded", stability=False) -> Path:
    adapter = root / "final_adapter"
    tokenizer = root / "tokenizer"
    adapter.mkdir(parents=True)
    tokenizer.mkdir()
    (adapter / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    metadata = {
        "status": "completed",
        "stage": stage,
        "training_stage": training_stage,
        "run_mode": "mini",
        "smoke_test": False,
        "run_id": f"{training_stage}_run",
        "model": config["model"],
        "runtime": config["runtime"],
        "dataset_counts": {"final_actual": counts},
        "token_statistics": {
            "train": {"selected_pool": selected_pool, "selected_ids": []},
            "validation": {"selected_pool": selected_pool, "selected_ids": ["numina_train_003_step_001_number_mutation"]},
        },
        "sft_source": {"run_id": "sft_run"},
        "numerical_stability": {"passed": stability},
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _eval_row(row_id: str, source_id: str) -> dict[str, object]:
    return {"id": row_id, "source_id": source_id, "problem": "2+2?", "final_answer": "4", "parse_status": "success", "metadata": {}}


def _sft_row(row_id: str, source_id: str) -> dict[str, object]:
    return {"id": row_id, "source_id": source_id, "metadata": {"normalized_id": f"numina_train_{source_id}"}}


def _dpo_row(row_id: str, source_id: str) -> dict[str, object]:
    return {"id": row_id, "source_id": source_id, "prompt": [], "chosen": [], "rejected": [], "metadata": {"normalized_id": f"numina_train_{source_id}"}}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
