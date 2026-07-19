from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.training.train_dpo import (
    _apply_runtime_overrides,
    assert_dpo_trainer_input_lengths,
    cli_payload,
    prepare_dpo_staged_output_dir,
    resolve_dpo_output_dir,
    validate_sft_run_dir,
)
from scripts import train_dpo


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "configs/qwen25_0_5b_m5_24gb_mini.yaml"
FORMAL = ROOT / "configs/qwen25_3b_4090.yaml"


class TrainDPOCLITests(unittest.TestCase):
    def test_cli_parser_requires_sft_run_dir(self) -> None:
        with self.assertRaises(SystemExit):
            train_dpo.build_parser().parse_args(["--config", str(MINI)])

    def test_cli_parser_accepts_stage4_options(self) -> None:
        args = train_dpo.build_parser().parse_args(
            [
                "--config",
                str(MINI),
                "--sft-run-dir",
                "sft",
                "--smoke-test",
                "--output-dir",
                "out",
                "--train-samples",
                "2",
                "--validation-samples",
                "1",
                "--max-steps",
                "1",
                "--overwrite",
            ]
        )

        self.assertEqual(args.sft_run_dir, "sft")
        self.assertTrue(args.smoke_test)
        self.assertEqual(args.train_samples, 2)

    def test_main_calls_training_function(self) -> None:
        with mock.patch.object(
            train_dpo,
            "train_dpo_from_config",
            return_value={"status": "completed", "run_id": "r", "output_dir": "o", "elapsed_seconds": 0},
        ):
            train_dpo.main(["--config", str(MINI), "--sft-run-dir", "sft", "--smoke-test"])

    def test_resolves_default_run_dir_under_dpo_output_dir(self) -> None:
        config = load_single_config(MINI)

        run_dir = resolve_dpo_output_dir(config, "run123", None)

        self.assertEqual(run_dir, Path("outputs/checkpoints/mini/dpo") / "run123")

    def test_output_dir_collision_requires_overwrite(self) -> None:
        config = load_single_config(MINI)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "existing.txt").write_text("x", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_dpo_staged_output_dir(config, "new", run_dir, overwrite=False)

    def test_smoke_overrides_use_dpo_samples_and_dpo_steps(self) -> None:
        config = load_single_config(MINI)

        effective, overrides = _apply_runtime_overrides(config, True, None, None, None)

        self.assertEqual(effective["dpo"]["max_steps"], config["smoke_test"]["max_steps"])
        self.assertEqual(effective["data"]["train_samples"], config["smoke_test"]["dpo_samples"])
        self.assertEqual(effective["data"]["validation_samples"], config["smoke_test"]["validation_samples"])
        self.assertEqual(overrides["applied"]["dpo.max_steps"], config["smoke_test"]["max_steps"])

    def test_formal_config_still_parses_for_compatibility(self) -> None:
        config = load_single_config(FORMAL)

        self.assertEqual(config["runtime"]["backend"], "cuda")
        self.assertTrue(config["quantization"]["load_in_4bit"])

    def test_validate_sft_run_dir_accepts_completed_256_row_run(self) -> None:
        config = load_single_config(MINI)
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_sft_run(Path(tmp), config, smoke_test=False, train_count=256)

            metadata = validate_sft_run_dir(root, config, smoke_test=False)

            self.assertEqual(metadata["final_train_count"], 256)
            self.assertEqual(metadata["adapter_dir"], str(root / "final_adapter"))

    def test_validate_sft_run_dir_rejects_smoke_for_normal_dpo(self) -> None:
        config = load_single_config(MINI)
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_sft_run(Path(tmp), config, smoke_test=True, train_count=64)

            with self.assertRaisesRegex(ValueError, "smoke SFT"):
                validate_sft_run_dir(root, config, smoke_test=False)

    def test_validate_sft_run_dir_rejects_short_normal_sft(self) -> None:
        config = load_single_config(MINI)
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_sft_run(Path(tmp), config, smoke_test=False, train_count=135)

            with self.assertRaisesRegex(ValueError, "256-row"):
                validate_sft_run_dir(root, config, smoke_test=False)

    def test_trainer_length_assertion_fails_on_long_rows(self) -> None:
        trainer = _FakeTrainer(
            [
                {"prompt_ids": [1, 2], "chosen_ids": [3], "rejected_ids": [4]},
                {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6]},
            ]
        )

        with self.assertRaisesRegex(ValueError, "longer than max_length"):
            assert_dpo_trainer_input_lengths(trainer, max_length=4)

    def test_cli_payload_contains_sft_source(self) -> None:
        payload = json.loads(
            cli_payload(
                {
                    "status": "completed",
                    "run_id": "run",
                    "output_dir": "out",
                    "elapsed_seconds": 1,
                    "sft_source": {"run_id": "sft"},
                }
            )
        )

        self.assertEqual(payload["sft_source"]["run_id"], "sft")


class _FakeTrainer:
    def __init__(self, rows):
        self.train_dataset = rows
        self.eval_dataset = rows[:1]


def _write_sft_run(root: Path, config: dict[str, object], smoke_test: bool, train_count: int) -> Path:
    run_dir = root / "sft_run"
    adapter_dir = run_dir / "final_adapter"
    tokenizer_dir = run_dir / "tokenizer"
    adapter_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    metadata = {
        "status": "completed",
        "stage": 3,
        "training_stage": "sft",
        "run_mode": "mini",
        "smoke_test": smoke_test,
        "run_id": "sft",
        "git_commit": "abc",
        "seed": config["project"]["seed"],
        "project": config["project"],
        "model": config["model"],
        "runtime": config["runtime"],
        "model_loader": {"lora": {"target_modules": config["lora"]["target_modules"]}},
        "dataset_counts": {"final_actual": {"train": train_count, "validation": 32}},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir
