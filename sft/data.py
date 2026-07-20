"""Dataset loading for the standalone SFT stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, load_from_disk


SFT_COLUMNS = {"schema_version", "id", "source_id", "prompt", "completion", "messages", "token_count", "split", "metadata"}
EVALUATION_COLUMNS = {
    "schema_version",
    "id",
    "source_id",
    "problem",
    "reference_answer",
    "prompt_messages",
    "prompt_token_count",
    "split",
    "metadata",
}


@dataclass(frozen=True)
class SFTDatasets:
    """Stage 1 SFT datasets ready for TRL."""

    train: Dataset
    validation: Dataset
    path: Path


def sft_dataset_path(config: Mapping[str, Any]) -> Path:
    """Return the local SFT DatasetDict path for the configured run mode."""

    mode = str(config["project"]["run_mode"])
    return Path(str(config["data"][f"{mode}_dir"])) / "sft"


def evaluation_dataset_path(config: Mapping[str, Any]) -> Path:
    """Return the local evaluation Dataset path for the configured run mode."""

    mode = str(config["project"]["run_mode"])
    return Path(str(config["data"][f"{mode}_dir"])) / "evaluation"


def load_sft_datasets(
    config: Mapping[str, Any],
    train_limit: int | None = None,
    validation_limit: int | None = None,
) -> SFTDatasets:
    """Load Stage 1 SFT train/validation datasets without changing row order."""

    path = sft_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(f"SFT Dataset path does not exist: {path}")
    loaded = load_from_disk(str(path))
    if "train" not in loaded or "validation" not in loaded:
        raise ValueError(f"SFT Dataset must contain train and validation splits: {path}")
    train = _limit_dataset(loaded["train"], train_limit)
    validation = _limit_dataset(loaded["validation"], validation_limit)
    _validate_split(train, SFT_COLUMNS, "sft/train", path)
    _validate_split(validation, SFT_COLUMNS, "sft/validation", path)
    return SFTDatasets(train=train, validation=validation, path=path)


def load_evaluation_dataset(config: Mapping[str, Any], limit: int | None = None) -> Dataset:
    """Load the Stage 1 evaluation dataset without changing row order."""

    path = evaluation_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation Dataset path does not exist: {path}")
    loaded = load_from_disk(str(path))
    dataset = _limit_dataset(loaded, limit)
    _validate_split(dataset, EVALUATION_COLUMNS, "evaluation", path)
    return dataset


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    count = min(len(dataset), int(limit))
    return dataset.select(range(count))


def _validate_split(dataset: Dataset, required_columns: set[str], label: str, path: Path) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{label} split is empty: {path}")
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        raise ValueError(f"{label} split missing required columns {missing}: {path}")
