from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.training.train_sft import (
    RunDirectories,
    _write_trainer_artifacts,
    _apply_runtime_overrides,
    prepare_output_dir,
    prepare_staged_output_dir,
    publish_staged_output,
    resolve_output_dir,
)
from scripts import train_sft


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "configs/qwen25_0_5b_m5_24gb_mini.yaml"
FORMAL = ROOT / "configs/qwen25_3b_4090.yaml"


class TrainSFTCLITests(unittest.TestCase):
    def test_cli_parser_accepts_stage3_options(self) -> None:
        args = train_sft.build_parser().parse_args(
            [
                "--config",
                str(MINI),
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

        self.assertTrue(args.smoke_test)
        self.assertEqual(args.train_samples, 2)
        self.assertEqual(args.validation_samples, 1)
        self.assertEqual(args.max_steps, 1)

    def test_main_calls_training_function(self) -> None:
        with mock.patch.object(train_sft, "train_sft_from_config", return_value={"status": "completed", "run_id": "r", "output_dir": "o", "elapsed_seconds": 0}):
            train_sft.main(["--config", str(MINI), "--smoke-test"])

    def test_output_dir_collision_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "existing.txt").write_text("x", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_output_dir(run_dir, overwrite=False)

    def test_resolves_default_run_dir_under_config_output_dir(self) -> None:
        config = load_single_config(MINI)

        run_dir = resolve_output_dir(config, "run123", None)

        self.assertEqual(run_dir, Path("outputs/checkpoints/mini/sft") / "run123")

    def test_formal_config_still_parses_for_compatibility(self) -> None:
        config = load_single_config(FORMAL)

        self.assertEqual(config["runtime"]["backend"], "cuda")
        self.assertTrue(config["quantization"]["load_in_4bit"])

    def test_writes_trainer_state_with_save_to_json_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            trainer = _FakeTrainer()

            _write_trainer_artifacts(trainer, {"train_loss": 0.5}, {"eval_loss": 0.4}, run_dir)

            self.assertEqual(json.loads((run_dir / "trainer_state.json").read_text(encoding="utf-8")), {"global_step": 10})
            self.assertEqual(json.loads((run_dir / "train_metrics.json").read_text(encoding="utf-8"))["train_loss"], 0.5)
            self.assertEqual(json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))["eval_loss"], 0.4)
            self.assertTrue((run_dir / "loss_history.jsonl").exists())

    def test_smoke_overrides_are_applied_before_metadata(self) -> None:
        config = load_single_config(MINI)

        effective, overrides = _apply_runtime_overrides(config, True, None, None, None)

        self.assertEqual(effective["sft"]["max_steps"], config["smoke_test"]["max_steps"])
        self.assertEqual(effective["data"]["train_samples"], config["smoke_test"]["train_samples"])
        self.assertEqual(effective["data"]["validation_samples"], config["smoke_test"]["validation_samples"])
        self.assertEqual(overrides["applied"]["sft.max_steps"], config["smoke_test"]["max_steps"])

    def test_staged_publish_preserves_old_output_until_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_dir = root / "run"
            final_dir.mkdir()
            old_file = final_dir / "old.txt"
            old_file.write_text("old", encoding="utf-8")
            dirs = prepare_staged_output_dir({"sft": {"output_dir": str(root)}}, "new", final_dir, overwrite=True)

            self.assertEqual(old_file.read_text(encoding="utf-8"), "old")
            (dirs.staging_dir / "new.txt").write_text("new", encoding="utf-8")
            publish_staged_output(dirs, overwrite=True)

            self.assertFalse(old_file.exists())
            self.assertEqual((final_dir / "new.txt").read_text(encoding="utf-8"), "new")

    def test_failed_staging_does_not_modify_old_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_dir = root / "run"
            final_dir.mkdir()
            old_file = final_dir / "old.txt"
            old_file.write_text("old", encoding="utf-8")
            staging_dir = root / ".run.failed.staging"
            staging_dir.mkdir()
            (staging_dir / "run_metadata.json").write_text("failed", encoding="utf-8")

            self.assertEqual(old_file.read_text(encoding="utf-8"), "old")
            self.assertTrue(staging_dir.exists())


class _FakeState:
    log_history = [{"loss": 0.5}, {"eval_loss": 0.4}, {"learning_rate": 0.0}]

    def save_to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps({"global_step": 10}), encoding="utf-8")


class _FakeTrainer:
    state = _FakeState()
