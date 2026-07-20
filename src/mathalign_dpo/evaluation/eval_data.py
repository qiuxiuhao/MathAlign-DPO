"""Stage 5 evaluation data and run metadata validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mathalign_dpo.data.write_outputs import sha256_file


def load_stage5_eval_dataset(config: Mapping[str, Any], sample_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load fixed Mini evaluation examples from Stage 2 step evaluation data."""

    manifest_path = Path(str(config["data"]["stage2_manifest_file"]))
    manifest = _load_stage2_manifest(manifest_path)
    run_mode = str(config["project"]["run_mode"])
    view_ids = manifest["views"][run_mode]["step"]["evaluation"]
    step_info = manifest["files"]["step_evaluation"]
    step_path = Path(str(step_info["path"]))
    rows = _read_jsonl(step_path)
    if len(rows) != int(step_info["rows"]):
        raise ValueError(f"Stage 2 step_evaluation row count mismatch: expected {step_info['rows']}, got {len(rows)}")
    if sha256_file(step_path) != step_info["sha256"]:
        raise ValueError(f"Stage 2 step_evaluation sha256 mismatch: {step_path}")
    by_id = {str(row["id"]): row for row in rows}
    missing = [row_id for row_id in view_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Stage 2 evaluation view references missing rows: {missing[:3]}")
    selected = [_eval_row(by_id[row_id], config) for row_id in view_ids[:sample_count]]
    if len(selected) != sample_count:
        raise ValueError(f"Not enough evaluation rows: needed {sample_count}, got {len(selected)}")
    return selected, {
        "stage2_manifest_file": str(manifest_path),
        "stage2_manifest_sha256": sha256_file(manifest_path),
        "step_eval_file": str(step_path),
        "step_eval_sha256": step_info["sha256"],
        "view_count": len(view_ids),
        "selected_count": len(selected),
        "selected_ids": [row["id"] for row in selected],
    }


def validate_no_training_leakage(
    eval_rows: Sequence[Mapping[str, Any]],
    sft_metadata: Mapping[str, Any],
    dpo_metadata: Mapping[str, Any],
    stage2_manifest_path: str | Path,
) -> dict[str, Any]:
    """Ensure evaluation examples are disjoint from selected SFT/DPO training examples."""

    train_aliases = _selected_aliases_from_run(sft_metadata, stage2_manifest_path, "sft")
    train_aliases.update(_selected_aliases_from_run(dpo_metadata, stage2_manifest_path, "dpo"))
    eval_aliases = {alias for row in eval_rows for alias in _row_aliases(row)}
    overlap = sorted(eval_aliases & train_aliases)
    if overlap:
        raise ValueError(f"Evaluation data overlaps selected SFT/DPO training sources: {overlap[:3]}")
    return {"passed": True, "checked_eval_rows": len(eval_rows), "training_alias_count": len(train_aliases)}


def validate_sft_eval_source(sft_run_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a completed Stage 3 Mini SFT run for Stage 5 evaluation."""

    root = Path(sft_run_dir)
    metadata = _read_json(root / "run_metadata.json")
    if metadata.get("status") != "completed" or int(metadata.get("stage", -1)) != 3 or metadata.get("training_stage") != "sft":
        raise ValueError(f"Stage 5 requires a completed Stage 3 SFT run: {root}")
    if metadata.get("run_mode") != "mini" or metadata.get("smoke_test") is True:
        raise ValueError("Stage 5 requires a non-smoke Mini SFT run")
    final_train = _final_train_count(metadata)
    if final_train != 256:
        raise ValueError(f"Stage 5 requires a 256-row SFT run, got {final_train}")
    _assert_model_metadata_matches(metadata, config, "SFT")
    adapter_dir = root / "final_adapter"
    tokenizer_dir = root / "tokenizer"
    _require_files([adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_config.json", tokenizer_dir / "tokenizer.json"])
    return {
        "run_dir": str(root),
        "run_id": metadata.get("run_id"),
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "metadata": metadata,
    }


def validate_dpo_eval_source(dpo_run_dir: str | Path, sft_source: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a completed Stage 4 Mini-only DPO run for Stage 5 evaluation."""

    root = Path(dpo_run_dir)
    metadata = _read_json(root / "run_metadata.json")
    if metadata.get("status") != "completed" or int(metadata.get("stage", -1)) != 4 or metadata.get("training_stage") != "dpo":
        raise ValueError(f"Stage 5 requires a completed Stage 4 DPO run: {root}")
    if metadata.get("run_mode") != "mini" or metadata.get("smoke_test") is True:
        raise ValueError("Stage 5 requires a non-smoke Mini DPO run")
    counts = metadata.get("dataset_counts", {}).get("final_actual", {})
    expected = {"train": int(config["dpo"]["train_samples"]), "validation": int(config["dpo"]["validation_samples"])}
    if counts != expected:
        raise ValueError(f"Stage 5 rejects DPO runs outside the Mini-only sample policy: expected {expected}, got {counts}")
    token_stats = metadata.get("token_statistics", {})
    if token_stats.get("train", {}).get("selected_pool") != "run_mode" or token_stats.get("validation", {}).get("selected_pool") != "run_mode":
        raise ValueError("Stage 5 requires a Mini-only DPO run with selected_pool=run_mode")
    stability = metadata.get("numerical_stability")
    if not isinstance(stability, Mapping) or stability.get("passed") is not True:
        raise ValueError("Stage 5 requires a DPO run that passed numerical stability gates")
    sft_run_id = metadata.get("sft_source", {}).get("run_id")
    if sft_run_id and sft_run_id != sft_source.get("run_id"):
        raise ValueError(f"DPO run was initialized from a different SFT run: {sft_run_id}")
    _assert_model_metadata_matches(metadata, config, "DPO")
    adapter_dir = root / "final_adapter"
    tokenizer_dir = root / "tokenizer"
    _require_files([adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_config.json", tokenizer_dir / "tokenizer.json"])
    return {
        "run_dir": str(root),
        "run_id": metadata.get("run_id"),
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "metadata": metadata,
    }


def _eval_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("parse_status") != "success" or not row.get("final_answer"):
        raise ValueError(f"Evaluation row must have parse_status=success and final_answer: {row.get('id')}")
    messages = [
        {"role": "system", "content": str(config["preprocessing"]["system_prompt"])},
        {
            "role": "user",
            "content": f"{config['preprocessing']['user_instruction']}\n\nProblem:\n{row['problem']}",
        },
    ]
    return {
        "schema_version": "1.0",
        "id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "problem": str(row["problem"]),
        "reference_answer": str(row["final_answer"]),
        "prompt_messages": messages,
        "metadata": dict(row.get("metadata", {})),
    }


def _load_stage2_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("completed") is not True or int(manifest.get("stage", -1)) != 2:
        raise ValueError(f"Expected completed Stage 2 manifest: {manifest_path}")
    if "step_evaluation" not in manifest.get("files", {}):
        raise ValueError("Stage 2 manifest is missing files.step_evaluation")
    return manifest


def _selected_aliases_from_run(metadata: Mapping[str, Any], stage2_manifest_path: str | Path, kind: str) -> set[str]:
    selected_ids = _selected_ids(metadata)
    if not selected_ids:
        raise ValueError(f"{kind.upper()} metadata is missing selected IDs; cannot prove evaluation leakage safety")
    manifest = _load_stage2_manifest(Path(stage2_manifest_path))
    aliases = set(selected_ids)
    file_keys = ("sft_train", "sft_validation") if kind == "sft" else ("dpo_train", "dpo_validation")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for key in file_keys:
        for row in _read_jsonl(Path(str(manifest["files"][key]["path"]))):
            rows_by_id[str(row["id"])] = row
    for row_id in selected_ids:
        row = rows_by_id.get(row_id)
        if row is None:
            aliases.update(_aliases_from_id(row_id))
            continue
        aliases.update(_row_aliases(row))
    if len(aliases) == len(selected_ids):
        raise ValueError(f"{kind.upper()} selected IDs could not be mapped to sources; cannot prove evaluation leakage safety")
    return aliases


def _selected_ids(metadata: Mapping[str, Any]) -> list[str]:
    token_stats = metadata.get("token_statistics", {})
    ids: list[str] = []
    for split in ("train", "validation"):
        ids.extend(str(row_id) for row_id in token_stats.get(split, {}).get("selected_ids", []))
    return ids


def _row_aliases(row: Mapping[str, Any]) -> set[str]:
    aliases = {str(row.get("id", "")), str(row.get("source_id", ""))}
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("normalized_id"):
        aliases.add(str(metadata["normalized_id"]))
    aliases.update(_aliases_from_id(str(row.get("id", ""))))
    source_id = str(row.get("source_id", ""))
    if source_id and not source_id.startswith("numina_"):
        aliases.add(f"numina_train_{source_id}")
    return {alias for alias in aliases if alias}


def _aliases_from_id(row_id: str) -> set[str]:
    aliases = {row_id}
    for suffix in ("_sft",):
        if row_id.endswith(suffix):
            aliases.add(row_id[: -len(suffix)])
    if "_step_" in row_id:
        aliases.add(row_id.split("_step_", 1)[0])
    return aliases


def _final_train_count(metadata: Mapping[str, Any]) -> int:
    counts = metadata.get("dataset_counts", {})
    return int(counts.get("final_actual", {}).get("train", 0))


def _assert_model_metadata_matches(metadata: Mapping[str, Any], config: Mapping[str, Any], label: str) -> None:
    for key in ("name_or_path", "revision", "torch_dtype"):
        if metadata.get("model", {}).get(key) != config["model"][key]:
            raise ValueError(f"{label} metadata mismatch for model.{key}")
    if metadata.get("runtime", {}).get("backend") != config["runtime"]["backend"]:
        raise ValueError(f"{label} metadata mismatch for runtime.backend")


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required evaluation source artifacts: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows
