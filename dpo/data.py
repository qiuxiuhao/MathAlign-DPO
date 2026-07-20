"""Dataset loading for the standalone DPO stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, load_from_disk


DPO_COLUMNS = {
    "schema_version",
    "id",
    "source_id",
    "step_index",
    "prompt",
    "chosen",
    "rejected",
    "token_count",
    "split",
    "metadata",
}


@dataclass(frozen=True)
class DPODatasets:
    """Stage 1 DPO datasets ready for TRL."""

    train: Dataset
    validation: Dataset
    path: Path


def dpo_dataset_path(config: Mapping[str, Any]) -> Path:
    """Return the local DPO DatasetDict path for the configured run mode."""

    configured = config.get("dpo", {}).get("dataset_dir")
    if configured:
        return Path(str(configured))
    mode = str(config["project"]["run_mode"])
    return Path(str(config["data"][f"{mode}_dir"])) / "dpo"


def load_dpo_datasets(
    config: Mapping[str, Any],
    train_limit: int | None = None,
    validation_limit: int | None = None,
) -> DPODatasets:
    """Load Stage 1 DPO train/validation datasets without changing row order."""

    path = dpo_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(f"DPO Dataset path does not exist: {path}")
    loaded = load_from_disk(str(path))
    if "train" not in loaded or "validation" not in loaded:
        raise ValueError(f"DPO Dataset must contain train and validation splits: {path}")
    train = _limit_dataset(loaded["train"], train_limit)
    validation = _limit_dataset(loaded["validation"], validation_limit)
    _validate_split(train, config, "dpo/train", path)
    _validate_split(validation, config, "dpo/validation", path)
    return DPODatasets(train=train, validation=validation, path=path)


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    count = min(len(dataset), int(limit))
    return dataset.select(range(count))


def _validate_split(dataset: Dataset, config: Mapping[str, Any], label: str, path: Path) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{label} split is empty: {path}")
    missing = sorted(DPO_COLUMNS - set(dataset.column_names))
    if missing:
        raise ValueError(f"{label} split missing required columns {missing}: {path}")
    max_length = int(config["dpo"]["max_length"])
    max_prompt_length = int(config["dpo"]["max_prompt_length"])
    for row in dataset:
        _validate_row(row, label, max_length=max_length, max_prompt_length=max_prompt_length)


def _validate_row(row: Mapping[str, Any], label: str, max_length: int, max_prompt_length: int) -> None:
    row_id = str(row.get("id", ""))
    if not row_id:
        raise ValueError(f"{label} row id must be non-empty")
    if row.get("chosen") == row.get("rejected"):
        raise ValueError(f"{label} chosen and rejected are identical: {row_id}")
    if not isinstance(row.get("prompt"), list) or not row["prompt"]:
        raise ValueError(f"{label} prompt must be a non-empty message list: {row_id}")
    if not isinstance(row.get("chosen"), list) or not row["chosen"]:
        raise ValueError(f"{label} chosen must be a non-empty message list: {row_id}")
    if not isinstance(row.get("rejected"), list) or not row["rejected"]:
        raise ValueError(f"{label} rejected must be a non-empty message list: {row_id}")
    token_count = row.get("token_count")
    if not isinstance(token_count, Mapping):
        raise ValueError(f"{label} token_count must be present: {row_id}")
    prompt_tokens = int(token_count.get("prompt", -1))
    chosen_total = int(token_count.get("chosen_total", -1))
    rejected_total = int(token_count.get("rejected_total", -1))
    if prompt_tokens < 0 or chosen_total < 0 or rejected_total < 0:
        raise ValueError(f"{label} token_count has invalid values: {row_id}")
    if prompt_tokens > max_prompt_length:
        raise ValueError(f"{label} prompt exceeds configured max_prompt_length: {row_id}")
    if chosen_total > max_length or rejected_total > max_length:
        raise ValueError(f"{label} row exceeds configured max_length: {row_id}")
