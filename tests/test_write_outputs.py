from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from mathalign_dpo.data import write_outputs
from mathalign_dpo.data.write_outputs import JsonOutput, publish_json_outputs, publish_stage1_outputs, sha256_file, validate_completed_manifest


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
            validate_completed_manifest(paths["manifest"])

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

    def test_overwrite_staging_write_failure_preserves_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            self._publish_old_outputs(paths)
            before = _snapshot(paths)

            def failing_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
                if path.name == "normalized_validation.jsonl":
                    raise OSError("injected staging write failure")
                write_outputs._write_jsonl_original(path, rows)

            write_outputs._write_jsonl_original = write_outputs._write_jsonl
            try:
                with mock.patch.object(write_outputs, "_write_jsonl", side_effect=failing_write_jsonl):
                    with self.assertRaisesRegex(OSError, "injected"):
                        publish_stage1_outputs(
                            canonical={"train": [_row(10)], "validation": [_row(20)], "evaluation": [_row(30)]},
                            statistics={"schema_version": "1.0", "stage": 1, "version": "new"},
                            manifest={"schema_version": "1.0", "stage": 1, "completed": False, "version": "new"},
                            output_paths=paths,
                            overwrite=True,
                            run_id="new",
                        )
            finally:
                delattr(write_outputs, "_write_jsonl_original")

            self.assertEqual(before, _snapshot(paths))
            self.assertFalse((paths["manifest"].parent / ".stage_new").exists())

    def test_overwrite_staging_validation_failure_preserves_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            self._publish_old_outputs(paths)
            before = _snapshot(paths)

            with mock.patch.object(write_outputs, "_count_jsonl_rows", return_value=999):
                with self.assertRaisesRegex(ValueError, "row count mismatch"):
                    publish_stage1_outputs(
                        canonical={"train": [_row(10)], "validation": [_row(20)], "evaluation": [_row(30)]},
                        statistics={"schema_version": "1.0", "stage": 1, "version": "new"},
                        manifest={"schema_version": "1.0", "stage": 1, "completed": False, "version": "new"},
                        output_paths=paths,
                        overwrite=True,
                        run_id="new",
                    )

            self.assertEqual(before, _snapshot(paths))
            self.assertFalse((paths["manifest"].parent / ".stage_new").exists())

    def test_overwrite_publish_failure_rolls_back_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            self._publish_old_outputs(paths)
            before = _snapshot(paths)

            def failing_replace(source: Path, target: Path) -> None:
                source_path = Path(source)
                if source_path.name == "normalized_validation.jsonl" and source_path.parent.name != "backup":
                    raise OSError("injected publish failure")
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                source_path.replace(target)

            with self.assertRaisesRegex(OSError, "injected"):
                publish_stage1_outputs(
                    canonical={"train": [_row(10)], "validation": [_row(20)], "evaluation": [_row(30)]},
                    statistics={"schema_version": "1.0", "stage": 1, "version": "new"},
                    manifest={"schema_version": "1.0", "stage": 1, "completed": False, "version": "new"},
                    output_paths=paths,
                    overwrite=True,
                    run_id="new",
                    replace_file=failing_replace,
                )

            self.assertEqual(before, _snapshot(paths))
            validate_completed_manifest(paths["manifest"])

    def test_generic_stage2_write_failure_preserves_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _stage2_paths(root)
            self._publish_old_stage2_outputs(paths)
            before = _snapshot(paths)

            def failing_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
                if path.name == "sft_train.jsonl":
                    raise OSError("injected stage2 write failure")
                write_outputs._write_jsonl_original(path, rows)

            write_outputs._write_jsonl_original = write_outputs._write_jsonl
            try:
                with mock.patch.object(write_outputs, "_write_jsonl", side_effect=failing_write_jsonl):
                    with self.assertRaisesRegex(OSError, "injected"):
                        publish_json_outputs(
                            outputs=_stage2_outputs(paths, version="new"),
                            manifest_name="manifest",
                            overwrite=True,
                            run_id="new_stage2",
                            manifest_builder=_stage2_manifest_builder,
                        )
            finally:
                delattr(write_outputs, "_write_jsonl_original")

            self.assertEqual(before, _snapshot(paths))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage2"]["version"], "old")
            self.assertFalse((root / ".stage_new_stage2").exists())

    def test_generic_stage2_validation_failure_preserves_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _stage2_paths(root)
            self._publish_old_stage2_outputs(paths)
            before = _snapshot(paths)

            outputs = _stage2_outputs(paths, version="new")
            outputs["dpo_train"] = JsonOutput(paths["dpo_train"], "jsonl", [{"id": "new"}], rows=2)
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                publish_json_outputs(
                    outputs=outputs,
                    manifest_name="manifest",
                    overwrite=True,
                    run_id="new_stage2",
                    manifest_builder=_stage2_manifest_builder,
                )

            self.assertEqual(before, _snapshot(paths))
            self.assertFalse((root / ".stage_new_stage2").exists())

    def test_generic_stage2_publish_failure_rolls_back_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _stage2_paths(root)
            self._publish_old_stage2_outputs(paths)
            before = _snapshot(paths)

            def failing_replace(source: Path, target: Path) -> None:
                source_path = Path(source)
                if source_path.name == "sft_train.jsonl" and source_path.parent.name != "backup":
                    raise OSError("injected stage2 publish failure")
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                source_path.replace(target)

            with self.assertRaisesRegex(OSError, "injected"):
                publish_json_outputs(
                    outputs=_stage2_outputs(paths, version="new"),
                    manifest_name="manifest",
                    overwrite=True,
                    run_id="new_stage2",
                    replace_file=failing_replace,
                    manifest_builder=_stage2_manifest_builder,
                )

            self.assertEqual(before, _snapshot(paths))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["stage2"]["completed"])
            self.assertEqual(manifest["stage2"]["version"], "old")

    def _publish_old_outputs(self, paths: dict[str, Path]) -> None:
        publish_stage1_outputs(
            canonical={"train": [_row(1)], "validation": [_row(2)], "evaluation": [_row(3)]},
            statistics={"schema_version": "1.0", "stage": 1, "version": "old"},
            manifest={"schema_version": "1.0", "stage": 1, "completed": False, "version": "old"},
            output_paths=paths,
            overwrite=False,
            run_id="old",
        )

    def _publish_old_stage2_outputs(self, paths: dict[str, Path]) -> None:
        publish_json_outputs(
            outputs=_stage2_outputs(paths, version="old"),
            manifest_name="manifest",
            overwrite=False,
            run_id="old_stage2",
            manifest_builder=_stage2_manifest_builder,
        )


def _snapshot(paths: dict[str, Path]) -> dict[str, tuple[str, str]]:
    return {name: (path.read_text(encoding="utf-8"), sha256_file(path)) for name, path in paths.items()}


def _stage2_paths(root: Path) -> dict[str, Path]:
    return {
        "step_train": root / "step_train.jsonl",
        "step_validation": root / "step_validation.jsonl",
        "step_evaluation": root / "step_eval.jsonl",
        "sft_train": root / "sft_train.jsonl",
        "sft_validation": root / "sft_validation.jsonl",
        "dpo_train": root / "dpo_train.jsonl",
        "dpo_validation": root / "dpo_validation.jsonl",
        "manual_review": root / "manual_review_preferences.jsonl",
        "statistics": root / "data_statistics.json",
        "manifest": root / "split_manifest.json",
    }


def _stage2_outputs(paths: dict[str, Path], version: str) -> dict[str, JsonOutput]:
    row = {"id": version, "value": version}
    return {
        "step_train": JsonOutput(paths["step_train"], "jsonl", [row], rows=1),
        "step_validation": JsonOutput(paths["step_validation"], "jsonl", [row], rows=1),
        "step_evaluation": JsonOutput(paths["step_evaluation"], "jsonl", [row], rows=1),
        "sft_train": JsonOutput(paths["sft_train"], "jsonl", [row], rows=1),
        "sft_validation": JsonOutput(paths["sft_validation"], "jsonl", [row], rows=1),
        "dpo_train": JsonOutput(paths["dpo_train"], "jsonl", [row], rows=1),
        "dpo_validation": JsonOutput(paths["dpo_validation"], "jsonl", [row], rows=1),
        "manual_review": JsonOutput(paths["manual_review"], "jsonl", [row], rows=1),
        "statistics": JsonOutput(paths["statistics"], "json", {"version": version}),
        "manifest": JsonOutput(paths["manifest"], "json", {"stage2": {"completed": False, "version": version}}),
    }


def _stage2_manifest_builder(
    manifest: dict[str, object],
    paths: dict[str, Path],
    hashes: dict[str, str],
    counts: dict[str, int],
) -> dict[str, object]:
    payload = dict(manifest)
    stage2 = dict(payload["stage2"])
    stage2["completed"] = True
    stage2["files"] = {
        name: {"path": str(paths[name]), "rows": counts[name], "sha256": hashes[name]}
        for name in counts
    }
    payload["stage2"] = stage2
    return payload
