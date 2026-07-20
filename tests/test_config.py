from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mathalign_dpo.config.load_config import load_project_configs, load_single_config, sample_counts, split_ratios


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "configs/qwen25_0_5b_m5_24gb_mini.yaml"
FORMAL = ROOT / "configs/qwen25_3b_4090.yaml"


class ConfigTests(unittest.TestCase):
    def test_stage1_configs_parse_and_share_data_settings(self) -> None:
        configs = load_project_configs(MINI, FORMAL)

        self.assertEqual(configs.mini["project"]["run_mode"], "mini")
        self.assertEqual(configs.formal["project"]["run_mode"], "formal")
        self.assertNotIn("stage", configs.mini["project"])
        self.assertNotIn("stage", configs.formal["project"])
        self.assertEqual(configs.dataset_revision, configs.mini["data"]["dataset_revision"])
        self.assertNotEqual(configs.formal["data"]["split_manifest_file"], configs.formal["data"]["stage2_manifest_file"])
        self.assertNotEqual(configs.formal["data"]["statistics_file"], configs.formal["data"]["stage2_statistics_file"])
        self.assertTrue(configs.dataset_revision)
        self.assertLess(abs(sum(split_ratios(configs.formal).values()) - 1.0), 1e-9)

        mini_counts = sample_counts(configs.mini)
        formal_counts = sample_counts(configs.formal)
        for split in ("train", "validation", "evaluation"):
            self.assertLessEqual(mini_counts[split], formal_counts[split])

    def test_mps_and_cuda_quantization_rules(self) -> None:
        configs = load_project_configs(MINI, FORMAL)

        self.assertEqual(configs.mini["runtime"]["backend"], "mps")
        self.assertIs(configs.mini["quantization"]["enabled"], False)
        self.assertIs(configs.mini["quantization"]["load_in_4bit"], False)
        self.assertEqual(configs.mini["sft"]["optimizer"], "adamw_torch")
        self.assertEqual(configs.mini["dpo"]["optimizer"], "adamw_torch")
        self.assertGreater(configs.mini["dpo"]["beta"], 0)
        self.assertEqual(configs.mini["dpo"]["loss_type"], "sigmoid")
        self.assertLess(configs.mini["dpo"]["max_prompt_length"], configs.mini["dpo"]["max_length"])
        self.assertEqual(configs.mini["dpo"]["adapter_reload_samples"], 1)
        self.assertEqual(configs.mini["dpo"]["adapter_reload_max_new_tokens"], 32)
        self.assertEqual(configs.mini["dpo"]["train_samples"], 179)
        self.assertEqual(configs.mini["dpo"]["validation_samples"], 21)
        self.assertEqual(configs.mini["model"]["torch_dtype"], "float16")
        self.assertRegex(configs.mini["model"]["revision"], r"^[0-9a-f]{40}$")
        self.assertIs(configs.mini["runtime"]["allow_cpu_fallback"], False)
        self.assertEqual(configs.mini["sft"]["adapter_reload_samples"], 3)
        self.assertFalse(configs.mini["evaluation"]["do_sample"])
        self.assertEqual(configs.mini["evaluation"]["num_beams"], 1)
        self.assertEqual(configs.mini["evaluation"]["samples"], 32)

        self.assertEqual(configs.formal["runtime"]["backend"], "cuda")
        self.assertIs(configs.formal["quantization"]["enabled"], True)
        self.assertIs(configs.formal["quantization"]["load_in_4bit"], True)
        self.assertEqual(configs.formal["quantization"]["quant_type"], "nf4")
        self.assertEqual(configs.formal["quantization"]["compute_dtype"], "bfloat16")
        self.assertEqual(configs.formal["dpo"]["optimizer"], "paged_adamw_8bit")
        self.assertLess(configs.formal["dpo"]["max_prompt_length"], configs.formal["dpo"]["max_length"])
        self.assertRegex(configs.formal["model"]["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(configs.formal["sft"]["adapter_reload_samples"], 3)

    def test_single_config_loader_parses_approved_configs(self) -> None:
        mini = load_single_config(MINI)
        formal = load_single_config(FORMAL)

        self.assertEqual(mini["project"]["run_mode"], "mini")
        self.assertEqual(formal["project"]["run_mode"], "formal")

    def test_rejects_project_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini_path = tmp_path / "mini.yaml"
            formal_path = tmp_path / "formal.yaml"
            mini_text = MINI.read_text(encoding="utf-8").replace("  seed: 42\n", "  seed: 42\n  stage: 1\n", 1)
            mini_path.write_text(mini_text, encoding="utf-8")
            formal_path.write_text(FORMAL.read_text(encoding="utf-8"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "project.stage"):
                load_project_configs(mini_path, formal_path)

    def test_rejects_invalid_dpo_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini_path = tmp_path / "mini.yaml"
            text = MINI.read_text(encoding="utf-8").replace("  max_prompt_length: 384\n", "  max_prompt_length: 512\n", 1)
            mini_path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dpo.max_prompt_length"):
                load_single_config(mini_path)

    def test_rejects_unsupported_dpo_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini_path = tmp_path / "mini.yaml"
            text = MINI.read_text(encoding="utf-8").replace("  loss_type: sigmoid\n", "  loss_type: hinge\n", 1)
            mini_path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dpo.loss_type"):
                load_single_config(mini_path)

    def test_rejects_nondeterministic_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini_path = tmp_path / "mini.yaml"
            text = MINI.read_text(encoding="utf-8").replace("  do_sample: false\n", "  do_sample: true\n", 1)
            mini_path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "evaluation.do_sample"):
                load_single_config(mini_path)
