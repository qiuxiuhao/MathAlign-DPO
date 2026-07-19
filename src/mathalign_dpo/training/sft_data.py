"""Load, validate, and tokenize Stage 2 SFT data for training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from mathalign_dpo.data.write_outputs import sha256_file


SFT_SPLITS = ("train", "validation")
SFT_FILE_KEYS = {"train": "sft_train", "validation": "sft_validation"}


@dataclass(frozen=True)
class SFTDataBundle:
    """Selected Stage 2 SFT rows and their manifest."""

    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    selected_counts: dict[str, int]


@dataclass(frozen=True)
class TokenizedSFTData:
    """Prompt/completion rows after tokenizer length filtering."""

    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    token_statistics: dict[str, Any]


def load_sft_data(
    config: Mapping[str, Any],
    train_limit: int | None = None,
    validation_limit: int | None = None,
) -> SFTDataBundle:
    """Load SFT rows selected by the Stage 2 manifest view for the run mode."""

    manifest_path = Path(str(config["data"]["stage2_manifest_file"]))
    manifest = load_and_validate_stage2_manifest(manifest_path)
    run_mode = str(config["project"]["run_mode"])
    views = manifest.get("views", {})
    if run_mode not in views:
        raise ValueError(f"Stage 2 manifest does not contain {run_mode!r} views: {manifest_path}")

    selected: dict[str, list[dict[str, Any]]] = {}
    selected_counts: dict[str, int] = {}
    limits = {"train": train_limit, "validation": validation_limit}
    for split in SFT_SPLITS:
        all_rows = _load_and_validate_manifest_file(manifest, split)
        ids = list(views[run_mode]["sft"][split])
        if limits[split] is not None:
            ids = ids[: int(limits[split])]
        selected[split] = _select_manifest_rows(all_rows, ids, split)
        selected_counts[split] = len(selected[split])

    return SFTDataBundle(
        train_rows=selected["train"],
        validation_rows=selected["validation"],
        manifest=manifest,
        selected_counts=selected_counts,
    )


def load_and_validate_stage2_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and validate the Stage 2 manifest envelope."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 2 manifest file does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("completed") is not True:
        raise ValueError(f"Stage 2 manifest is not completed: {manifest_path}")
    if int(manifest.get("stage", -1)) != 2:
        raise ValueError(f"Expected Stage 2 manifest, got stage={manifest.get('stage')!r}: {manifest_path}")
    if manifest.get("token_length_status") != "not_checked_no_tokenizer":
        raise ValueError(f"Unexpected Stage 2 token length status: {manifest.get('token_length_status')!r}")

    stage1_info = manifest.get("stage1_manifest_file")
    if not isinstance(stage1_info, dict):
        raise ValueError("Stage 2 manifest is missing stage1_manifest_file")
    stage1_path = Path(str(stage1_info.get("path", "")))
    if not stage1_path.exists():
        raise FileNotFoundError(f"Referenced Stage 1 manifest does not exist: {stage1_path}")
    if sha256_file(stage1_path) != stage1_info.get("sha256"):
        raise ValueError(f"Stage 1 manifest sha256 mismatch: {stage1_path}")

    for split in SFT_SPLITS:
        _validate_manifest_file_entry(manifest, SFT_FILE_KEYS[split])
    return manifest


def tokenize_and_filter_sft_data(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> TokenizedSFTData:
    """Convert SFT messages to prompt/completion rows and filter by token length."""

    train_converted, train_stats = _convert_and_filter_split(train_rows, tokenizer, max_length)
    validation_converted, validation_stats = _convert_and_filter_split(validation_rows, tokenizer, max_length)
    if not train_converted:
        raise ValueError("No SFT train rows remain after token length filtering")
    if not validation_converted:
        raise ValueError("No SFT validation rows remain after token length filtering")
    return TokenizedSFTData(
        train_rows=train_converted,
        validation_rows=validation_converted,
        token_statistics={
            "max_length": int(max_length),
            "train": train_stats,
            "validation": validation_stats,
        },
    )


def validate_tokenizer_chat_template(tokenizer: Any) -> dict[str, Any]:
    """Require a real tokenizer chat template and return padding metadata."""

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer must provide a chat_template; Stage 3 does not define a custom template")
    pad_token_before = getattr(tokenizer, "pad_token", None)
    pad_token_set_from_eos = False
    if pad_token_before is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = eos_token
        pad_token_set_from_eos = True
    return {
        "chat_template_present": True,
        "pad_token_before": pad_token_before,
        "pad_token_after": getattr(tokenizer, "pad_token", None),
        "pad_token_set_from_eos": pad_token_set_from_eos,
    }


def count_chat_tokens(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> int:
    """Count tokens in a rendered chat using the configured tokenizer template."""

    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(rendered, "shape"):
        return int(rendered.shape[-1])
    return len(rendered)


def sft_row_to_prompt_completion(row: Mapping[str, Any], token_count: int) -> dict[str, Any]:
    """Convert one Stage 2 SFT row to TRL prompt/completion format."""

    messages = list(row["messages"])
    return {
        "id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "prompt": messages[:2],
        "completion": [messages[2]],
        "token_count": int(token_count),
        "metadata": dict(row.get("metadata", {})),
    }


def _load_and_validate_manifest_file(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    key = SFT_FILE_KEYS[split]
    info = manifest["files"][key]
    path = Path(str(info["path"]))
    rows = _read_jsonl(path)
    if len(rows) != int(info["rows"]):
        raise ValueError(f"Manifest row count mismatch for {key}: expected {info['rows']}, got {len(rows)}")
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"Manifest sha256 mismatch for {key}: {path}")
    seen: set[str] = set()
    for row in rows:
        validate_stage2_sft_row(row)
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"Duplicate SFT row id in {path}: {row_id}")
        seen.add(row_id)
    return rows


def _validate_manifest_file_entry(manifest: Mapping[str, Any], key: str) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or key not in files:
        raise ValueError(f"Stage 2 manifest is missing files.{key}")
    info = files[key]
    path = Path(str(info.get("path", "")))
    if not path.exists():
        raise FileNotFoundError(f"Manifest file is missing for {key}: {path}")
    if "rows" not in info or "sha256" not in info:
        raise ValueError(f"Manifest file entry for {key} must include rows and sha256")


def validate_stage2_sft_row(row: Mapping[str, Any]) -> None:
    """Validate the Stage 2 SFT schema expected by training."""

    required = {"schema_version", "id", "source_id", "messages", "token_count", "metadata"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"SFT row missing fields: {missing}")
    if row["schema_version"] != "1.0":
        raise ValueError(f"Unsupported SFT schema version for {row.get('id')}: {row['schema_version']!r}")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise ValueError("SFT row id must be a non-empty string")
    messages = row["messages"]
    if not isinstance(messages, list) or [msg.get("role") for msg in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"SFT messages must be exactly system/user/assistant: {row['id']}")
    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"SFT message content must be non-empty: {row['id']}")
    if row["token_count"] is not None:
        raise ValueError(f"Stage 2 SFT token_count must be null before Stage 3 filtering: {row['id']}")


def _select_manifest_rows(rows: Sequence[dict[str, Any]], ids: Sequence[str], split: str) -> list[dict[str, Any]]:
    rows_by_id = {str(row["id"]): row for row in rows}
    missing = [row_id for row_id in ids if row_id not in rows_by_id]
    if missing:
        raise ValueError(f"Stage 2 manifest references missing {split} SFT IDs: {missing[:3]}")
    return [rows_by_id[row_id] for row_id in ids]


def _convert_and_filter_split(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    filtered_ids: list[str] = []
    token_counts: list[int] = []
    for row in rows:
        messages = list(row["messages"])
        token_count = count_chat_tokens(tokenizer, messages)
        token_counts.append(token_count)
        if token_count <= max_length:
            converted.append(sft_row_to_prompt_completion(row, token_count))
        else:
            filtered_ids.append(str(row["id"]))
    return converted, _token_stats(len(rows), converted, filtered_ids, token_counts)


def _token_stats(
    input_count: int,
    kept_rows: Sequence[Mapping[str, Any]],
    filtered_ids: Sequence[str],
    token_counts: Sequence[int],
) -> dict[str, Any]:
    sorted_counts = sorted(int(count) for count in token_counts)
    return {
        "input_count": int(input_count),
        "kept_count": len(kept_rows),
        "filtered_count": len(filtered_ids),
        "filtered_ids": list(filtered_ids),
        "kept_ids": [str(row["id"]) for row in kept_rows],
        "min": min(sorted_counts) if sorted_counts else None,
        "max": max(sorted_counts) if sorted_counts else None,
        "mean": float(mean(sorted_counts)) if sorted_counts else None,
        "p95": _percentile(sorted_counts, 0.95),
    }


def _percentile(sorted_values: Sequence[int], percentile: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * percentile)))
    return int(sorted_values[index])


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
