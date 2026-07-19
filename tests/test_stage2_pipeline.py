from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from mathalign_dpo.data.stage2_pipeline import build_stage2_data, stage2_output_paths, validate_mini_stage2_views
from mathalign_dpo.data.write_outputs import publish_stage1_outputs, sha256_file


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "configs/qwen25_0_5b_m5_24gb_mini.yaml"
FORMAL = ROOT / "configs/qwen25_3b_4090.yaml"


class Stage2PipelineTests(unittest.TestCase):
    def test_stage2_uses_separate_manifest_and_preserves_stage1_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mini_path, formal_path, stage1_manifest_path = _write_configs_and_stage1_outputs(root)
            before = sha256_file(stage1_manifest_path)

            result = build_stage2_data(
                mini_config=mini_path,
                formal_config=formal_path,
                output_dir=root / "stage2",
                overwrite=False,
            )

            after = sha256_file(stage1_manifest_path)
            self.assertEqual(before, after)
            stage2_manifest_path = Path(result["paths"]["manifest"])
            self.assertEqual(stage2_manifest_path.name, "stage2_manifest.json")
            manifest = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage1_manifest_file"]["path"], str(stage1_manifest_path))
            self.assertEqual(manifest["stage1_manifest_file"]["sha256"], before)
            validate_mini_stage2_views(
                manifest["views"],
                json.loads(stage1_manifest_path.read_text(encoding="utf-8")),
            )

    def test_stage2_output_paths_do_not_use_stage1_manifest_names(self) -> None:
        config = yaml.safe_load(FORMAL.read_text(encoding="utf-8"))
        paths = stage2_output_paths(config)

        self.assertEqual(paths["manifest"], Path("data/processed/stage2_manifest.json"))
        self.assertEqual(paths["statistics"], Path("data/processed/stage2_statistics.json"))


def _write_configs_and_stage1_outputs(root: Path) -> tuple[Path, Path, Path]:
    mini = yaml.safe_load(MINI.read_text(encoding="utf-8"))
    formal = yaml.safe_load(FORMAL.read_text(encoding="utf-8"))
    stage1_paths = {
        "train": root / "normalized_train.jsonl",
        "validation": root / "normalized_validation.jsonl",
        "evaluation": root / "normalized_eval.jsonl",
        "statistics": root / "data_statistics.json",
        "manifest": root / "split_manifest.json",
    }
    stage2_paths = {
        "step_train_file": root / "stage2" / "step_train.jsonl",
        "step_validation_file": root / "stage2" / "step_validation.jsonl",
        "step_eval_file": root / "stage2" / "step_eval.jsonl",
        "sft_train_file": root / "stage2" / "sft_train.jsonl",
        "sft_validation_file": root / "stage2" / "sft_validation.jsonl",
        "dpo_train_file": root / "stage2" / "dpo_train.jsonl",
        "dpo_validation_file": root / "stage2" / "dpo_validation.jsonl",
        "manual_review_file": root / "stage2" / "manual_review_preferences.jsonl",
        "stage2_statistics_file": root / "stage2" / "stage2_statistics.json",
        "stage2_manifest_file": root / "stage2" / "stage2_manifest.json",
    }
    for config, mode in ((mini, "mini"), (formal, "formal")):
        data = config["data"]
        data["processed_dir"] = str(root)
        data["normalized_train_file"] = str(stage1_paths["train"])
        data["normalized_validation_file"] = str(stage1_paths["validation"])
        data["normalized_eval_file"] = str(stage1_paths["evaluation"])
        data["statistics_file"] = str(stage1_paths["statistics"])
        data["split_manifest_file"] = str(stage1_paths["manifest"])
        for key, path in stage2_paths.items():
            data[key] = str(path)
        data["train_samples"] = 2 if mode == "mini" else 4
        data["validation_samples"] = 1 if mode == "mini" else 2
        data["evaluation_samples"] = 1
        config["negative_sampling"]["minimum_dpo_examples"] = 1
        config["negative_sampling"]["maximum_dpo_examples"] = 2 if mode == "mini" else 4
        config["negative_sampling"]["save_manual_review_samples"] = 1

    rows = [_row(index) for index in range(1, 8)]
    publish_stage1_outputs(
        canonical={
            "train": rows[:4],
            "validation": rows[4:6],
            "evaluation": rows[6:],
        },
        statistics={"schema_version": "1.0", "stage": 1},
        manifest={
            "schema_version": "1.0",
            "stage": 1,
            "completed": False,
            "dataset_name": formal["data"]["dataset_name"],
            "dataset_revision": formal["data"]["dataset_revision"],
            "source_split": formal["data"]["source_split"],
            "seed": formal["project"]["seed"],
            "views": {
                "formal": {
                    "train": [row["id"] for row in rows[:4]],
                    "validation": [row["id"] for row in rows[4:6]],
                    "evaluation": [rows[6]["id"]],
                },
                "mini": {
                    "train": [row["id"] for row in rows[:2]],
                    "validation": [rows[4]["id"]],
                    "evaluation": [rows[6]["id"]],
                },
            },
        },
        output_paths=stage1_paths,
        overwrite=False,
        run_id="stage1_fixture",
    )

    mini_path = root / "mini.yaml"
    formal_path = root / "formal.yaml"
    mini_path.write_text(yaml.safe_dump(mini, sort_keys=False), encoding="utf-8")
    formal_path.write_text(yaml.safe_dump(formal, sort_keys=False), encoding="utf-8")
    return mini_path, formal_path, stage1_paths["manifest"]


def _row(index: int) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "id": f"numina_train_{index:08d}",
        "source": "AI-MO/NuminaMath-CoT",
        "source_split": "train",
        "source_id": f"{index:08d}",
        "problem": f"Compute {index}+{index}.",
        "solution": f"1. Add {index}+{index}={index * 2}.\n2. Therefore the final answer is \\boxed{{{index * 2}}}.",
        "metadata": {"source_subset": None, "original_fields": ["problem", "solution"]},
    }
