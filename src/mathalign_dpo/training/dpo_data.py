"""Load, validate, and tokenize Stage 2 DPO data for training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from mathalign_dpo.data.write_outputs import sha256_file


DPO_SPLITS = ("train", "validation")
DPO_FILE_KEYS = {"train": "dpo_train", "validation": "dpo_validation"}


@dataclass(frozen=True)
class DPOCandidatePools:
    """DPO candidate pools from the configured Stage 2 run-mode view."""

    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    candidate_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class TokenizedDPOData:
    """Prompt/chosen/rejected rows after tokenizer length filtering."""

    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    token_statistics: dict[str, Any]


def load_dpo_candidate_pools(config: Mapping[str, Any]) -> DPOCandidatePools:
    """Load DPO candidates from the configured Stage 2 run-mode view only."""

    manifest_path = Path(str(config["data"]["stage2_manifest_file"]))
    manifest = load_and_validate_stage2_dpo_manifest(manifest_path)
    views = manifest.get("views", {})
    run_mode = str(config["project"]["run_mode"])
    if run_mode not in views:
        raise ValueError(f"Stage 2 manifest does not contain {run_mode!r} views: {manifest_path}")

    pools: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in DPO_SPLITS:
        all_rows = _load_and_validate_manifest_file(manifest, split)
        ids = list(views[run_mode]["dpo"][split])
        pools[split] = _select_manifest_rows(all_rows, ids, split)
        _assert_rows_use_run_mode_source_ids(pools[split], views[run_mode], split, run_mode)
        counts[split] = {
            "run_mode": len(pools[split]),
        }
    return DPOCandidatePools(
        train_rows=pools["train"],
        validation_rows=pools["validation"],
        manifest=manifest,
        candidate_counts=counts,
    )


def load_and_validate_stage2_dpo_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and validate the Stage 2 DPO manifest envelope."""

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

    for split in DPO_SPLITS:
        _validate_manifest_file_entry(manifest, DPO_FILE_KEYS[split])
    return manifest


def select_tokenized_dpo_data(
    candidate_pools: DPOCandidatePools,
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
    seed: int,
    target_train_count: int,
    target_validation_count: int,
) -> TokenizedDPOData:
    """Filter candidate pools by true token length and select exact target counts."""

    train_rows, train_stats = _filter_rank_and_select_split(
        rows=candidate_pools.train_rows,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        seed=seed,
        split="train",
        target_count=target_train_count,
    )
    validation_rows, validation_stats = _filter_rank_and_select_split(
        rows=candidate_pools.validation_rows,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        seed=seed,
        split="validation",
        target_count=target_validation_count,
    )
    return TokenizedDPOData(
        train_rows=train_rows,
        validation_rows=validation_rows,
        token_statistics={
            "max_length": int(max_length),
            "max_prompt_length": int(max_prompt_length),
            "train": train_stats,
            "validation": validation_stats,
        },
    )


def assert_dpo_rows_within_limits(rows: Sequence[Mapping[str, Any]], max_length: int, max_prompt_length: int, label: str) -> None:
    """Fail if selected DPO rows exceed configured prompt or total lengths."""

    too_long: list[str] = []
    for row in rows:
        token_count = row["token_count"]
        if int(token_count["prompt"]) > max_prompt_length:
            too_long.append(str(row["id"]))
        if int(token_count["chosen_total"]) > max_length or int(token_count["rejected_total"]) > max_length:
            too_long.append(str(row["id"]))
    if too_long:
        raise ValueError(f"{label} DPO rows exceed configured token limits: {too_long[:3]}")


def validate_stage2_dpo_row(row: Mapping[str, Any]) -> None:
    """Validate the Stage 2 DPO schema expected by training."""

    required = {"schema_version", "id", "source_id", "prompt", "chosen", "rejected", "metadata", "token_count"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"DPO row missing fields: {missing}")
    if row["schema_version"] != "1.0":
        raise ValueError(f"Unsupported DPO schema version for {row.get('id')}: {row['schema_version']!r}")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise ValueError("DPO row id must be a non-empty string")
    if not isinstance(row["source_id"], str) or not row["source_id"].strip():
        raise ValueError(f"DPO source_id must be non-empty: {row['id']}")
    prompt = row["prompt"]
    if not isinstance(prompt, list) or len(prompt) < 2:
        raise ValueError(f"DPO prompt must contain at least system and user: {row['id']}")
    if [message.get("role") for message in prompt[:2]] != ["system", "user"]:
        raise ValueError(f"DPO prompt must begin with system/user: {row['id']}")
    for message in prompt:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"DPO prompt role is invalid: {row['id']}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"DPO prompt content must be non-empty: {row['id']}")
    chosen = _single_assistant_text(row["chosen"], "chosen", str(row["id"]))
    rejected = _single_assistant_text(row["rejected"], "rejected", str(row["id"]))
    if chosen.strip() == rejected.strip():
        raise ValueError(f"DPO chosen and rejected are identical: {row['id']}")
    prompt_text = "\n".join(str(message.get("content", "")) for message in prompt)
    if rejected in prompt_text:
        raise ValueError(f"DPO prompt leaks rejected step text: {row['id']}")
    if row["token_count"] is not None:
        raise ValueError(f"Stage 2 DPO token_count must be null before Stage 4 filtering: {row['id']}")


def count_dpo_tokens(tokenizer: Any, row: Mapping[str, Any]) -> dict[str, int]:
    """Count DPO prompt and completion tokens using TRL-compatible chat rendering."""

    prompt = list(row["prompt"])
    chosen = list(row["chosen"])
    rejected = list(row["rejected"])
    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    chosen_processed = tokenizer.apply_chat_template(
        prompt + chosen,
        tokenize=True,
        return_dict=True,
    )
    rejected_processed = tokenizer.apply_chat_template(
        prompt + rejected,
        tokenize=True,
        return_dict=True,
    )
    prompt_ids = _flatten_ids(prompt_ids)
    chosen_ids = _flatten_ids(chosen_processed["input_ids"])
    rejected_ids = _flatten_ids(rejected_processed["input_ids"])
    prompt_len = len(prompt_ids)
    return {
        "prompt": prompt_len,
        "chosen_total": len(chosen_ids),
        "rejected_total": len(rejected_ids),
        "chosen_completion": len(chosen_ids) - prompt_len,
        "rejected_completion": len(rejected_ids) - prompt_len,
    }


def selection_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash selected row IDs in final training order."""

    payload = "\n".join(str(row["id"]) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_and_validate_manifest_file(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    key = DPO_FILE_KEYS[split]
    info = manifest["files"][key]
    path = Path(str(info["path"]))
    rows = _read_jsonl(path)
    if len(rows) != int(info["rows"]):
        raise ValueError(f"Manifest row count mismatch for {key}: expected {info['rows']}, got {len(rows)}")
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"Manifest sha256 mismatch for {key}: {path}")
    seen: set[str] = set()
    for row in rows:
        validate_stage2_dpo_row(row)
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"Duplicate DPO row id in {path}: {row_id}")
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


def _select_manifest_rows(rows: Sequence[dict[str, Any]], ids: Sequence[str], split: str) -> list[dict[str, Any]]:
    rows_by_id = {str(row["id"]): row for row in rows}
    missing = [row_id for row_id in ids if row_id not in rows_by_id]
    if missing:
        raise ValueError(f"Stage 2 manifest references missing {split} DPO IDs: {missing[:3]}")
    return [rows_by_id[row_id] for row_id in ids]


def _assert_rows_use_run_mode_source_ids(rows: Sequence[Mapping[str, Any]], view: Mapping[str, Any], split: str, run_mode: str) -> None:
    source_view = view.get("dpo_source_ids", {})
    source_ids = set(source_view.get(split, [])) if isinstance(source_view, Mapping) else set()
    if not source_ids:
        raise ValueError(f"Stage 2 manifest is missing {run_mode} dpo_source_ids.{split}")
    outside = sorted({_row_source_view_id(row) for row in rows if _row_source_view_id(row) not in source_ids})
    if outside:
        raise ValueError(f"{run_mode} {split} DPO rows include source IDs outside the {run_mode} source view: {outside[:3]}")


def _row_source_view_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("normalized_id"):
        return str(metadata["normalized_id"])
    return str(row["source_id"])


def _filter_rank_and_select_split(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
    seed: int,
    split: str,
    target_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_rows, pool_stats = _convert_and_filter_split(rows, tokenizer, max_length, max_prompt_length)
    if len(eligible_rows) < target_count:
        raise ValueError(
            f"Not enough Mini {split} DPO rows in Stage 2 {split} view after token filtering at "
            f"max_length={max_length}, max_prompt_length={max_prompt_length}: "
            f"needed {target_count}, got {len(eligible_rows)}. "
            "Stage 4 Mini DPO must not borrow rows from the formal pool."
        )
    ranked = sorted(eligible_rows, key=lambda row: (_stable_rank(seed, split, str(row["id"])), str(row["id"])))
    selected = ranked[:target_count]
    assert_dpo_rows_within_limits(selected, max_length, max_prompt_length, f"selected {split}")
    stats = dict(pool_stats)
    stats.update(
        {
            "target_count": int(target_count),
            "final_count": len(selected),
            "selected_pool": "run_mode",
            "candidate_count": len(rows),
            "length_filtered_count": int(stats["filtered_count"]),
            "selection_hash": selection_hash(selected),
            "selected_ids": [str(row["id"]) for row in selected],
        }
    )
    return selected, stats


def _convert_and_filter_split(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    prompt_counts: list[int] = []
    total_counts: list[int] = []
    completion_counts: list[int] = []
    for row in rows:
        token_count = count_dpo_tokens(tokenizer, row)
        prompt_counts.append(int(token_count["prompt"]))
        total_counts.append(max(int(token_count["chosen_total"]), int(token_count["rejected_total"])))
        completion_counts.append(max(int(token_count["chosen_completion"]), int(token_count["rejected_completion"])))
        reason = _filter_reason(token_count, max_length, max_prompt_length)
        if reason is None:
            converted.append(dpo_row_to_trainer_row(row, token_count))
        else:
            filtered.append({"id": str(row["id"]), "reason": reason, "token_count": token_count})
    return converted, _token_stats(len(rows), converted, filtered, prompt_counts, total_counts, completion_counts)


def dpo_row_to_trainer_row(row: Mapping[str, Any], token_count: Mapping[str, int]) -> dict[str, Any]:
    """Convert one Stage 2 DPO row to the TRL conversational preference format."""

    return {
        "id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "prompt": list(row["prompt"]),
        "chosen": list(row["chosen"]),
        "rejected": list(row["rejected"]),
        "token_count": dict(token_count),
        "metadata": dict(row.get("metadata", {})),
    }


def _filter_reason(token_count: Mapping[str, int], max_length: int, max_prompt_length: int) -> str | None:
    if int(token_count["prompt"]) > max_prompt_length:
        return "prompt_too_long"
    if int(token_count["chosen_total"]) > max_length:
        return "chosen_too_long"
    if int(token_count["rejected_total"]) > max_length:
        return "rejected_too_long"
    if int(token_count["chosen_completion"]) <= 0:
        return "chosen_completion_empty"
    if int(token_count["rejected_completion"]) <= 0:
        return "rejected_completion_empty"
    return None


def _token_stats(
    input_count: int,
    kept_rows: Sequence[Mapping[str, Any]],
    filtered: Sequence[Mapping[str, Any]],
    prompt_counts: Sequence[int],
    total_counts: Sequence[int],
    completion_counts: Sequence[int],
) -> dict[str, Any]:
    return {
        "input_count": int(input_count),
        "kept_count": len(kept_rows),
        "filtered_count": len(filtered),
        "filtered": list(filtered),
        "filtered_ids": [str(row["id"]) for row in filtered],
        "kept_ids": [str(row["id"]) for row in kept_rows],
        "prompt": _series_stats(prompt_counts),
        "total": _series_stats(total_counts),
        "completion": _series_stats(completion_counts),
    }


def _series_stats(values: Sequence[int]) -> dict[str, Any]:
    sorted_values = sorted(int(value) for value in values)
    return {
        "min": min(sorted_values) if sorted_values else None,
        "max": max(sorted_values) if sorted_values else None,
        "mean": float(mean(sorted_values)) if sorted_values else None,
        "p95": _percentile(sorted_values, 0.95),
    }


def _percentile(sorted_values: Sequence[int], percentile: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * percentile)))
    return int(sorted_values[index])


def _stable_rank(seed: int, split: str, row_id: str) -> str:
    payload = f"{seed}|dpo|{split}|{row_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _single_assistant_text(messages: Any, field: str, example_id: str) -> str:
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "assistant":
        raise ValueError(f"DPO {field} must be exactly one assistant message: {example_id}")
    content = messages[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"DPO {field} content must be non-empty: {example_id}")
    return content


def _flatten_ids(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        return [int(value) for value in ids[0]]
    return [int(value) for value in ids]


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
