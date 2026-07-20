from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.evaluation.evaluate_math import _apply_runtime_overrides, _generation_config, cli_payload, evaluate_math_from_config
from scripts import evaluate_math


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "configs/qwen25_0_5b_m5_24gb_mini.yaml"


class EvaluateMathCLITests(unittest.TestCase):
    def test_parser_requires_sources(self) -> None:
        with self.assertRaises(SystemExit):
            evaluate_math.build_parser().parse_args(["--config", str(MINI)])

    def test_parser_accepts_stage5_options(self) -> None:
        args = evaluate_math.build_parser().parse_args(
            [
                "--config",
                str(MINI),
                "--sft-run-dir",
                "sft",
                "--dpo-run-dir",
                "dpo",
                "--smoke-test",
                "--samples",
                "2",
                "--overwrite",
            ]
        )

        self.assertTrue(args.smoke_test)
        self.assertEqual(args.samples, 2)

    def test_smoke_overrides_evaluation_samples(self) -> None:
        config = load_single_config(MINI)

        effective, overrides = _apply_runtime_overrides(config, smoke_test=True, samples=None)

        self.assertEqual(effective["evaluation"]["samples"], config["smoke_test"]["evaluation_samples"])
        self.assertEqual(overrides["applied"]["evaluation.samples"], config["smoke_test"]["evaluation_samples"])

    def test_generation_config_is_deterministic(self) -> None:
        config = load_single_config(MINI)
        tokenizer = type("Tok", (), {"pad_token_id": 1, "eos_token_id": 2})()

        generation = _generation_config(config, tokenizer)

        self.assertFalse(generation["do_sample"])
        self.assertEqual(generation["num_beams"], 1)
        self.assertNotIn("temperature", generation)
        self.assertNotIn("top_p", generation)

    def test_cli_payload_contains_metrics(self) -> None:
        payload = json.loads(
            cli_payload(
                {
                    "status": "completed",
                    "run_id": "run",
                    "output_dir": "out",
                    "elapsed_seconds": 1,
                    "metrics": {"models": {}},
                }
            )
        )

        self.assertEqual(payload["metrics"], {"models": {}})

    def test_rejects_superseded_dpo_before_mps_model_loading(self) -> None:
        config = load_single_config(MINI)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft = _write_source_run(root / "sft", config, stage=3, training_stage="sft", counts={"train": 256, "validation": 32})
            dpo = _write_source_run(root / "dpo", config, stage=4, training_stage="dpo", counts={"train": 256, "validation": 32})

            with self.assertRaisesRegex(ValueError, "Mini-only sample policy"):
                evaluate_math_from_config(MINI, sft, dpo, output_dir=root / "eval")

            self.assertTrue(any(root.glob(".eval.*.staging/run_metadata.json")))


def _write_source_run(root: Path, config: dict[str, object], stage: int, training_stage: str, counts: dict[str, int]) -> Path:
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
        "token_statistics": {"train": {"selected_ids": []}, "validation": {"selected_ids": []}},
    }
    if training_stage == "dpo":
        metadata["token_statistics"]["train"]["selected_pool"] = "expanded"
        metadata["token_statistics"]["validation"]["selected_pool"] = "expanded"
        metadata["sft_source"] = {"run_id": "sft_run"}
    (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root
