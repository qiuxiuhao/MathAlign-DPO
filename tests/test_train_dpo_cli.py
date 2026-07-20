from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.training.train_dpo import (
    _apply_runtime_overrides,
    _assert_numerical_stability,
    _compute_trainable_grad_norm,
    _copy_default_adapter_to_ref,
    _dpo_config,
    _dpo_callbacks,
    _precompute_mps_reference_logps,
    _validate_reference_adapters,
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
        self.assertEqual(effective["dpo"]["train_samples"], config["smoke_test"]["dpo_samples"])
        self.assertEqual(effective["dpo"]["validation_samples"], config["smoke_test"]["validation_samples"])
        self.assertEqual(effective["data"]["train_samples"], config["data"]["train_samples"])
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
            assert_dpo_trainer_input_lengths(trainer, max_length=4, max_prompt_length=4)

    def test_trainer_length_assertion_fails_on_missing_token_fields(self) -> None:
        trainer = _FakeTrainer([{"prompt_ids": [1], "chosen_ids": [2]}])

        with self.assertRaisesRegex(ValueError, "missing prompt_ids/chosen_ids/rejected_ids"):
            assert_dpo_trainer_input_lengths(trainer, max_length=4, max_prompt_length=4)

    def test_numerical_stability_rejects_nonfinite_core_metrics(self) -> None:
        trainer = _FakeTrainer([{"prompt_ids": [1], "chosen_ids": [2], "rejected_ids": [3]}])
        trainer.state.log_history = [{"loss": float("nan")}]

        with self.assertRaisesRegex(ValueError, "Non-finite DPO metrics"):
            _assert_numerical_stability(trainer, {"train_loss": 1.0}, {"eval_loss": 1.0}, [])

    def test_numerical_stability_ignores_nonfinite_diagnostic_entropy(self) -> None:
        trainer = _FakeTrainer([{"prompt_ids": [1], "chosen_ids": [2], "rejected_ids": [3]}])
        trainer.state.log_history = [
            {
                "entropy": float("nan"),
                "loss": 0.7,
                "grad_norm": 1.0,
                "logps/chosen": -2.0,
                "logps/rejected": -3.0,
                "rewards/margins": 0.1,
            }
        ]

        report = _assert_numerical_stability(trainer, {"train_loss": 0.7}, {"eval_loss": 0.7}, [])

        self.assertTrue(report["passed"])

    def test_numerical_stability_rejects_all_zero_dpo_signal(self) -> None:
        trainer = _FakeTrainer([{"prompt_ids": [1], "chosen_ids": [2], "rejected_ids": [3]}])
        trainer.state.log_history = [{"loss": 0.0, "logps/chosen": 0.0, "logps/rejected": 0.0}]

        with self.assertRaisesRegex(ValueError, "all zero"):
            _assert_numerical_stability(trainer, {"train_loss": 0.0}, {"eval_loss": 0.0}, [])

    def test_dpo_config_does_not_filter_nan_inf_logs(self) -> None:
        trl = __import__("trl")
        config = load_single_config(MINI)

        args = _dpo_config(trl, config, Path("out"))

        self.assertFalse(args.logging_nan_inf_filter)

    def test_mps_dpo_uses_callback_grad_norm_instead_of_trainer_clip(self) -> None:
        trl = __import__("trl")
        config = load_single_config(MINI)

        args = _dpo_config(trl, config, Path("out"))
        callbacks = _dpo_callbacks(config)

        self.assertEqual(args.max_grad_norm, 0.0)
        self.assertEqual(len(callbacks), 1)

    def test_trainable_grad_norm_is_computed_on_cpu_copy(self) -> None:
        torch = __import__("torch")
        model = _FakeGradModel(torch.tensor([3.0, 4.0]))

        norm = _compute_trainable_grad_norm(model)

        self.assertEqual(norm, 5.0)

    def test_mps_reference_logps_are_precomputed_after_ref_validation(self) -> None:
        config = load_single_config(MINI)
        trainer = _FakePrecomputeTrainer()

        report = _precompute_mps_reference_logps(config, trainer)

        self.assertTrue(report["enabled"])
        self.assertTrue(trainer.precompute_ref_logps)
        self.assertEqual(report["train"]["rows"], 1)

    def test_reference_adapter_validation_checks_default_and_ref(self) -> None:
        model = _FakePeftModel()

        report = _validate_reference_adapters(model)

        self.assertTrue(report["initial_weights_allclose"])

    def test_reference_adapter_copy_syncs_ref_weights(self) -> None:
        model = _FakePeftModel(default_value=7, ref_value=3)

        _copy_default_adapter_to_ref(model)
        report = _validate_reference_adapters(model)

        self.assertTrue(report["initial_weights_allclose"])

    def test_reference_adapter_validation_rejects_large_weight_difference(self) -> None:
        model = _FakePeftModel(default_value=7, ref_value=3)

        with self.assertRaisesRegex(ValueError, "weights differ"):
            _validate_reference_adapters(model)

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
        self.state = type("State", (), {"log_history": []})()
        self._metrics = {}


class _FakeArgs:
    precompute_ref_batch_size = None
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1


class _FakePrecomputeTrainer:
    def __init__(self) -> None:
        self.args = _FakeArgs()
        self.train_dataset = [{"id": "train"}]
        self.eval_dataset = [{"id": "eval"}]
        self.precompute_ref_logps = False

    def _precompute_ref_logps(self, dataset, name, batch_size):
        return [
            {
                **row,
                "ref_chosen_logps": -1.0 if name == "train" else -2.0,
                "ref_rejected_logps": -1.5 if name == "train" else -2.5,
            }
            for row in dataset
        ]


class _FakeParam:
    def __init__(self, value, requires_grad):
        self.value = value
        self.requires_grad = requires_grad

    def detach(self):
        return self

    def copy_(self, other):
        self.value = other.value
        return self

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def __eq__(self, other):
        return self.value == other.value


class _FakePeftModel:
    peft_config = {"default": object(), "ref": object()}

    def __init__(self, default_value: int = 1, ref_value: int = 1) -> None:
        self.params = [
            ("base.lora_A.default.weight", _FakeParam(default_value, True)),
            ("base.lora_A.ref.weight", _FakeParam(ref_value, False)),
        ]

    def named_parameters(self):
        return iter(self.params)


class _FakeGradModel:
    def __init__(self, grad):
        self.param = type("Param", (), {"requires_grad": True, "grad": grad})()

    def named_parameters(self):
        return iter([("adapter.weight", self.param)])


def _write_sft_run(root: Path, config: dict[str, object], smoke_test: bool, train_count: int) -> Path:
    run_dir = root / "sft_run"
    adapter_dir = run_dir / "final_adapter"
    tokenizer_dir = run_dir / "tokenizer"
    adapter_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    adapter_config = {
        "r": config["lora"]["rank"],
        "lora_alpha": config["lora"]["alpha"],
        "lora_dropout": config["lora"]["dropout"],
        "bias": config["lora"]["bias"],
        "target_modules": list(config["lora"]["target_modules"]),
        "base_model_name_or_path": config["model"]["name_or_path"],
        "revision": config["model"]["revision"],
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
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
        "tokenizer": {
            "vocab_size": 10,
            "pad_token_after": "<eos>",
            "pad_token_id": 0,
            "eos_token": "<eos>",
            "eos_token_id": 0,
            "chat_template_sha256": "abc",
        },
        "effective_config": config,
        "data_lineage": {
            "stage2_manifest_file": config["data"]["stage2_manifest_file"],
            "stage2_manifest_sha256": "hash",
        },
        "dataset_counts": {"final_actual": {"train": train_count, "validation": 32}},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir
