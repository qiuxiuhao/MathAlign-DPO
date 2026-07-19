from __future__ import annotations

import tempfile
import unittest
import math
from pathlib import Path

from mathalign_dpo.training.runtime_metadata import (
    RunClock,
    build_run_id,
    collect_base_metadata,
    finalize_metadata,
    json_safe,
    peak_process_memory_mb,
    write_json,
    write_jsonl,
)


class RuntimeMetadataTests(unittest.TestCase):
    def test_run_id_contains_stage_and_smoke_state(self) -> None:
        self.assertIn("stage3_sft_smoke", build_run_id("sft", True))
        self.assertIn("stage3_sft_mini", build_run_id("sft", False))
        self.assertIn("stage4_dpo_smoke", build_run_id("dpo", True, stage_number=4))
        self.assertNotEqual(build_run_id("sft", False), build_run_id("sft", False))

    def test_finalize_metadata_records_elapsed_and_peak_memory(self) -> None:
        clock = RunClock.start()
        metadata = {"run_id": "run", "status": "running"}

        finalized = finalize_metadata(metadata, clock, "completed", {"metrics": {"loss": 1.0}})

        self.assertEqual(finalized["status"], "completed")
        self.assertIn("elapsed_seconds", finalized)
        self.assertIn("peak_process_memory_mb", finalized)
        self.assertGreaterEqual(peak_process_memory_mb(), 0)

    def test_collect_base_metadata_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = collect_base_metadata(
                _config(),
                "config.yaml",
                tmp,
                "run",
                stage_number=3,
                training_stage="sft",
                run_mode="mini",
                smoke_test=True,
                runtime_overrides={"applied": {"sft.max_steps": 10}},
            )

            self.assertEqual(metadata["stage"], 3)
            self.assertEqual(metadata["training_stage"], "sft")
            self.assertEqual(metadata["run_mode"], "mini")
            self.assertEqual(metadata["effective_config"]["sft"]["max_steps"], 1)
            self.assertEqual(metadata["dpo"]["max_steps"], 2)
            self.assertEqual(metadata["runtime_overrides"]["applied"]["sft.max_steps"], 10)
            self.assertEqual(metadata["device"]["backend"], "mps")
            write_json(Path(tmp) / "metadata.json", metadata)
            self.assertTrue((Path(tmp) / "metadata.json").exists())

    def test_json_writers_replace_nonfinite_floats_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "metrics.json", {"loss": math.nan, "ok": 1.0, "nested": {"inf": math.inf}})
            write_jsonl(root / "history.jsonl", [{"loss": math.nan}])

            self.assertEqual((root / "metrics.json").read_text(encoding="utf-8").count("null"), 2)
            self.assertEqual((root / "history.jsonl").read_text(encoding="utf-8").strip(), '{"loss": null}')
            self.assertEqual(json_safe({"x": math.nan})["x"], None)


def _config() -> dict[str, object]:
    return {
        "project": {"run_mode": "mini", "seed": 42},
        "model": {"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"},
        "runtime": {"backend": "mps", "device": "mps", "allow_cpu_fallback": False},
        "sft": {"max_steps": 1},
        "dpo": {"max_steps": 2},
    }
