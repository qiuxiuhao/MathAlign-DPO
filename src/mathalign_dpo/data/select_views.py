"""Load Stage 1 normalized outputs in manifest order."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.data.write_outputs import validate_completed_manifest


NORMALIZED_SPLIT_KEYS = {
    "train": "train",
    "validation": "validation",
    "evaluation": "evaluation",
}


def load_stage1_manifest(manifest_path: str | Path, mini_config: Mapping[str, Any], formal_config: Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate the completed Stage 1 split manifest."""

    path = Path(manifest_path)
    validate_completed_manifest(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset_name") != formal_config["data"]["dataset_name"]:
        raise ValueError("Stage 1 manifest dataset_name does not match config")
    if manifest.get("dataset_revision") != formal_config["data"]["dataset_revision"]:
        raise ValueError("Stage 1 manifest dataset_revision does not match config")
    if manifest.get("source_split") != formal_config["data"]["source_split"]:
        raise ValueError("Stage 1 manifest source_split does not match config")
    if manifest.get("seed") != formal_config["project"]["seed"]:
        raise ValueError("Stage 1 manifest seed does not match config")
    if mini_config["data"]["dataset_revision"] != formal_config["data"]["dataset_revision"]:
        raise ValueError("Mini and formal configs must share dataset_revision")
    return manifest


def load_normalized_views(manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Load normalized examples in formal manifest order for each split."""

    loaded: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "evaluation"):
        file_info = manifest["files"][NORMALIZED_SPLIT_KEYS[split]]
        rows = _load_jsonl_by_id(Path(str(file_info["path"])))
        ordered_ids = list(manifest["views"]["formal"][split])
        missing = [example_id for example_id in ordered_ids if example_id not in rows]
        if missing:
            raise ValueError(f"Normalized file missing manifest IDs for {split}: {missing[:3]}")
        loaded[split] = [rows[example_id] for example_id in ordered_ids]
    return loaded


def mini_ids_for_stage1(manifest: Mapping[str, Any], split: str) -> list[str]:
    """Return Stage 1 Mini IDs for a split."""

    return list(manifest["views"]["mini"][split])


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            item = json.loads(line)
            example_id = item.get("id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"JSONL row missing id in {path}: line {line_number}")
            if example_id in rows:
                raise ValueError(f"Duplicate id in {path}: {example_id}")
            rows[example_id] = item
    return rows
