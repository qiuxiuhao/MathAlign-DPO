from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mathalign_dpo.data.select_views import load_normalized_views, load_stage1_manifest
from mathalign_dpo.data.write_outputs import publish_stage1_outputs


def _config(root: Path, mode: str) -> dict[str, object]:
    return {
        "project": {"seed": 42, "run_mode": mode},
        "data": {
            "dataset_name": "AI-MO/NuminaMath-CoT",
            "dataset_revision": "abc123",
            "source_split": "train",
            "split_manifest_file": str(root / "split_manifest.json"),
        },
    }


def _row(index: int) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "id": f"numina_train_{index:08d}",
        "source": "AI-MO/NuminaMath-CoT",
        "source_split": "train",
        "source_id": f"{index:08d}",
        "problem": f"Problem {index}",
        "solution": f"1. Work {index}.\n2. Answer is \\boxed{{{index}}}.",
        "metadata": {"original_fields": ["problem", "solution"]},
    }


class SelectViewsTests(unittest.TestCase):
    def test_loads_normalized_examples_in_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "train": root / "normalized_train.jsonl",
                "validation": root / "normalized_validation.jsonl",
                "evaluation": root / "normalized_eval.jsonl",
                "statistics": root / "data_statistics.json",
                "manifest": root / "split_manifest.json",
            }
            formal_ids = {
                "train": ["numina_train_00000002", "numina_train_00000001"],
                "validation": ["numina_train_00000003"],
                "evaluation": ["numina_train_00000004"],
            }
            mini_ids = {
                "train": ["numina_train_00000002"],
                "validation": ["numina_train_00000003"],
                "evaluation": ["numina_train_00000004"],
            }
            publish_stage1_outputs(
                canonical={
                    "train": [_row(2), _row(1)],
                    "validation": [_row(3)],
                    "evaluation": [_row(4)],
                },
                statistics={"schema_version": "1.0", "stage": 1},
                manifest={
                    "schema_version": "1.0",
                    "stage": 1,
                    "completed": False,
                    "dataset_name": "AI-MO/NuminaMath-CoT",
                    "dataset_revision": "abc123",
                    "source_split": "train",
                    "seed": 42,
                    "views": {"formal": formal_ids, "mini": mini_ids},
                },
                output_paths=paths,
                overwrite=False,
                run_id="select",
            )

            manifest = load_stage1_manifest(paths["manifest"], _config(root, "mini"), _config(root, "formal"))
            views = load_normalized_views(manifest)

            self.assertEqual([row["id"] for row in views["train"]], formal_ids["train"])

    def test_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "train": root / "normalized_train.jsonl",
                "validation": root / "normalized_validation.jsonl",
                "evaluation": root / "normalized_eval.jsonl",
                "statistics": root / "data_statistics.json",
                "manifest": root / "split_manifest.json",
            }
            publish_stage1_outputs(
                canonical={"train": [_row(1)], "validation": [_row(2)], "evaluation": [_row(3)]},
                statistics={"schema_version": "1.0", "stage": 1},
                manifest={
                    "schema_version": "1.0",
                    "stage": 1,
                    "completed": False,
                    "dataset_name": "AI-MO/NuminaMath-CoT",
                    "dataset_revision": "abc123",
                    "source_split": "train",
                    "seed": 42,
                    "views": {
                        "formal": {
                            "train": ["numina_train_00000001"],
                            "validation": ["numina_train_00000002"],
                            "evaluation": ["numina_train_00000003"],
                        },
                        "mini": {
                            "train": ["numina_train_00000001"],
                            "validation": ["numina_train_00000002"],
                            "evaluation": ["numina_train_00000003"],
                        },
                    },
                },
                output_paths=paths,
                overwrite=False,
                run_id="select",
            )
            paths["train"].write_text(json.dumps(_row(9), sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sha256"):
                load_stage1_manifest(paths["manifest"], _config(root, "mini"), _config(root, "formal"))
