from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from mathalign_dpo.data.write_outputs import publish_stage1_outputs, sha256_file


def _row(index: int) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "id": f"numina_train_{index:08d}",
        "source": "AI-MO/NuminaMath-CoT",
        "source_split": "train",
        "source_id": f"{index:08d}",
        "problem": f"Problem {index}",
        "solution": f"Solution {index}",
        "metadata": {"source_subset": None, "original_fields": ["problem", "solution"]},
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "train": root / "normalized_train.jsonl",
        "validation": root / "normalized_validation.jsonl",
        "evaluation": root / "normalized_eval.jsonl",
        "statistics": root / "data_statistics.json",
        "manifest": root / "split_manifest.json",
    }


class WriteOutputsTests(unittest.TestCase):
    def test_publish_writes_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = _paths(tmp_path)

            published = publish_stage1_outputs(
                canonical={"train": [_row(1)], "validation": [_row(2)], "evaluation": [_row(3)]},
                statistics={"schema_version": "1.0", "stage": 1},
                manifest={"schema_version": "1.0", "stage": 1, "completed": False},
                output_paths=paths,
                overwrite=False,
                run_id="test",
            )

            self.assertTrue(paths["manifest"].exists())
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertIs(manifest["completed"], True)
            self.assertEqual(manifest["files"]["train"]["rows"], 1)
            self.assertEqual(manifest["files"]["train"]["sha256"], sha256_file(paths["train"]))
            self.assertFalse(published.staging_dir.exists())

    def test_existing_outputs_fail_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = _paths(tmp_path)
            paths["train"].parent.mkdir(parents=True, exist_ok=True)
            paths["train"].write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                publish_stage1_outputs(
                    canonical={"train": [_row(1)], "validation": [_row(2)], "evaluation": [_row(3)]},
                    statistics={"schema_version": "1.0", "stage": 1},
                    manifest={"schema_version": "1.0", "stage": 1, "completed": False},
                    output_paths=paths,
                    overwrite=False,
                    run_id="test",
                )

    def test_failed_staging_does_not_publish_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = _paths(tmp_path)

            with self.assertRaises(ValueError):
                publish_stage1_outputs(
                    canonical={
                        "train": [{"id": "bad"}],
                        "validation": [_row(2)],
                        "evaluation": [_row(3)],
                    },
                    statistics={"schema_version": "1.0", "stage": 1},
                    manifest={"schema_version": "1.0", "stage": 1, "completed": False},
                    output_paths=paths,
                    overwrite=False,
                    run_id="test",
                )

            self.assertFalse(paths["manifest"].exists())
            self.assertFalse((tmp_path / ".stage_test").exists())
