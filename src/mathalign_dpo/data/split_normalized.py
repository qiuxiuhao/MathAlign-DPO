"""Deterministic source-level splits for normalized examples."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from mathalign_dpo.config.load_config import sample_counts


SPLIT_NAMES = ("train", "validation", "evaluation")


@dataclass(frozen=True)
class SplitResult:
    """Canonical formal splits plus Mini/formal ID views."""

    canonical: dict[str, list[dict[str, Any]]]
    formal_ids: dict[str, list[str]]
    mini_ids: dict[str, list[str]]


def split_examples(
    examples: list[dict[str, Any]],
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    seed: int,
    ratios: Mapping[str, float],
    mini_config: Mapping[str, Any],
    formal_config: Mapping[str, Any],
) -> SplitResult:
    """Assign examples to stable splits and produce run-mode views."""

    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    for example in examples:
        split = assign_split(
            source_id=str(example["source_id"]),
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            source_split=source_split,
            seed=seed,
            ratios=ratios,
        )
        assigned[split].append(example)

    for split, split_examples_list in assigned.items():
        split_examples_list.sort(
            key=lambda item: stable_rank(
                source_id=str(item["source_id"]),
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
                source_split=source_split,
                split=split,
                seed=seed,
            )
        )

    mini_counts = sample_counts(dict(mini_config))
    formal_counts = sample_counts(dict(formal_config))
    canonical: dict[str, list[dict[str, Any]]] = {}
    formal_ids: dict[str, list[str]] = {}
    mini_ids: dict[str, list[str]] = {}

    for split in SPLIT_NAMES:
        if len(assigned[split]) < formal_counts[split]:
            raise ValueError(
                f"Not enough normalized examples for {split}: "
                f"need {formal_counts[split]}, got {len(assigned[split])}"
            )
        canonical[split] = assigned[split][: formal_counts[split]]
        formal_ids[split] = [str(example["id"]) for example in canonical[split]]
        mini_ids[split] = formal_ids[split][: mini_counts[split]]

    return SplitResult(canonical=canonical, formal_ids=formal_ids, mini_ids=mini_ids)


def assign_split(
    source_id: str,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    seed: int,
    ratios: Mapping[str, float],
) -> str:
    """Return a deterministic split name for a source ID."""

    bucket = _bucket(
        "split",
        dataset_name,
        dataset_revision,
        source_split,
        source_id,
        str(seed),
    )
    train_cutoff = float(ratios["train"])
    validation_cutoff = train_cutoff + float(ratios["validation"])
    if bucket < train_cutoff:
        return "train"
    if bucket < validation_cutoff:
        return "validation"
    return "evaluation"


def stable_rank(
    source_id: str,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    split: str,
    seed: int,
) -> str:
    """Return a stable hexadecimal rank for sorting within a split."""

    return hashlib.sha256(
        "|".join(["rank", dataset_name, dataset_revision, source_split, split, source_id, str(seed)]).encode("utf-8")
    ).hexdigest()


def _bucket(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(16**16)
