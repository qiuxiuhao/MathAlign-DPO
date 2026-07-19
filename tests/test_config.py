from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mathalign_dpo.config.load_config import load_stage1_configs, sample_counts, split_ratios


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "configs/qwen25_0_5b_m5_24gb_mini.yaml"
FORMAL = ROOT / "configs/qwen25_3b_4090.yaml"


class ConfigTests(unittest.TestCase):
    def test_stage1_configs_parse_and_share_data_settings(self) -> None:
        configs = load_stage1_configs(MINI, FORMAL)

        self.assertEqual(configs.mini["project"]["run_mode"], "mini")
        self.assertEqual(configs.formal["project"]["run_mode"], "formal")
        self.assertNotIn("stage", configs.mini["project"])
        self.assertNotIn("stage", configs.formal["project"])
        self.assertEqual(configs.dataset_revision, configs.mini["data"]["dataset_revision"])
        self.assertTrue(configs.dataset_revision)
        self.assertLess(abs(sum(split_ratios(configs.formal).values()) - 1.0), 1e-9)

        mini_counts = sample_counts(configs.mini)
        formal_counts = sample_counts(configs.formal)
        for split in ("train", "validation", "evaluation"):
            self.assertLessEqual(mini_counts[split], formal_counts[split])

    def test_mps_and_cuda_quantization_rules(self) -> None:
        configs = load_stage1_configs(MINI, FORMAL)

        self.assertEqual(configs.mini["runtime"]["backend"], "mps")
        self.assertIs(configs.mini["quantization"]["enabled"], False)
        self.assertIs(configs.mini["quantization"]["load_in_4bit"], False)
        self.assertEqual(configs.mini["sft"]["optimizer"], "adamw_torch")

        self.assertEqual(configs.formal["runtime"]["backend"], "cuda")
        self.assertIs(configs.formal["quantization"]["enabled"], True)
        self.assertIs(configs.formal["quantization"]["load_in_4bit"], True)
        self.assertEqual(configs.formal["quantization"]["quant_type"], "nf4")

    def test_rejects_project_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini_path = tmp_path / "mini.yaml"
            formal_path = tmp_path / "formal.yaml"
            mini_text = MINI.read_text(encoding="utf-8").replace("  seed: 42\n", "  seed: 42\n  stage: 1\n", 1)
            mini_path.write_text(mini_text, encoding="utf-8")
            formal_path.write_text(FORMAL.read_text(encoding="utf-8"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "project.stage"):
                load_stage1_configs(mini_path, formal_path)
