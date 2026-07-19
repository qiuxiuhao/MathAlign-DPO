"""Stage 2 orchestration for step, SFT, DPO, and review data."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.config.load_config import load_project_configs, sample_counts
from mathalign_dpo.data.build_preferences import (
    build_dpo_examples,
    build_manual_review_examples,
    dpo_strategy_counts,
    validate_dpo_example,
)
from mathalign_dpo.data.build_sft import TOKEN_LENGTH_STATUS, build_sft_examples, sft_status_counts, validate_sft_example
from mathalign_dpo.data.parse_steps import (
    answer_confidence_counts,
    answer_method_counts,
    parse_failure_reason_counts,
    parse_normalized_example,
    parse_status_counts,
    validate_step_example,
)
from mathalign_dpo.data.select_views import load_normalized_views, load_stage1_manifest
from mathalign_dpo.data.write_outputs import JsonOutput, PublishedOutputs, publish_json_outputs, sha256_file


SPLITS = ("train", "validation", "evaluation")
MAX_DPO_PAIRS_PER_SOURCE = 2
DPO_CONFIDENCES = {"high", "medium"}


def build_stage2_data(
    mini_config: str | Path,
    formal_config: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 2 and publish outputs transactionally."""

    configs = load_project_configs(mini_config, formal_config)
    mini = _config_with_stage2_smoke_caps(configs.mini) if smoke_test else configs.mini
    formal = _config_with_stage2_smoke_caps(configs.formal) if smoke_test else configs.formal
    _validate_stage2_configs(mini, formal)

    stage1_manifest_path = Path(str(formal["data"]["split_manifest_file"]))
    stage1_manifest_sha256 = sha256_file(stage1_manifest_path)
    stage1_manifest = load_stage1_manifest(stage1_manifest_path, mini, formal)
    normalized = _cap_normalized_for_run(load_normalized_views(stage1_manifest), formal, smoke_test=smoke_test)

    step_examples = _parse_steps(normalized, int(formal["preprocessing"]["minimum_steps"]))
    mini_step_examples = _select_mini_step_examples(step_examples, stage1_manifest, mini)

    sft_train = build_sft_examples(step_examples["train"], formal, int(formal["data"]["train_samples"]))
    sft_validation = build_sft_examples(step_examples["validation"], formal, int(formal["data"]["validation_samples"]))
    mini_sft_train = build_sft_examples(mini_step_examples["train"], formal, int(mini["data"]["train_samples"]))
    mini_sft_validation = build_sft_examples(mini_step_examples["validation"], formal, int(mini["data"]["validation_samples"]))

    dpo_train, mini_dpo_train, dpo_train_failures = _build_formal_and_mini_dpo(
        formal_steps=step_examples["train"],
        mini_steps=mini_step_examples["train"],
        formal_config=formal,
        mini_config=mini,
        formal_maximum=int(formal["negative_sampling"]["maximum_dpo_examples"]),
        mini_maximum=int(mini["negative_sampling"]["maximum_dpo_examples"]),
    )
    dpo_validation, mini_dpo_validation, dpo_validation_failures = _build_formal_and_mini_dpo(
        formal_steps=step_examples["validation"],
        mini_steps=mini_step_examples["validation"],
        formal_config=formal,
        mini_config=mini,
        formal_maximum=int(formal["data"]["validation_samples"]),
        mini_maximum=int(mini["data"]["validation_samples"]),
    )

    _validate_minimum_counts(mini, formal, mini_sft_train, mini_sft_validation, dpo_train, mini_dpo_train, mini_dpo_validation)
    _validate_examples([*sft_train, *sft_validation], [*dpo_train, *dpo_validation])

    manual_review = build_manual_review_examples(
        [*dpo_train, *dpo_validation],
        sample_count=min(int(formal["negative_sampling"]["save_manual_review_samples"]), len(dpo_train) + len(dpo_validation)),
        seed=int(formal["project"]["seed"]),
    )

    output_paths = stage2_output_paths(formal, output_dir=output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_stage2")
    stage2_views = _build_stage2_views(
        stage1_manifest=stage1_manifest,
        mini=mini,
        step_examples=step_examples,
        mini_step_examples=mini_step_examples,
        sft_train=sft_train,
        sft_validation=sft_validation,
        mini_sft_train=mini_sft_train,
        mini_sft_validation=mini_sft_validation,
        dpo_train=dpo_train,
        dpo_validation=dpo_validation,
        mini_dpo_train=mini_dpo_train,
        mini_dpo_validation=mini_dpo_validation,
        manual_review=manual_review,
    )
    validate_mini_stage2_views(stage2_views, stage1_manifest)

    statistics = _build_statistics(
        stage1_manifest_path=stage1_manifest_path,
        stage1_manifest_sha256=stage1_manifest_sha256,
        stage1_manifest=stage1_manifest,
        step_examples=step_examples,
        mini_step_examples=mini_step_examples,
        sft_train=sft_train,
        sft_validation=sft_validation,
        mini_sft_train=mini_sft_train,
        mini_sft_validation=mini_sft_validation,
        dpo_train=dpo_train,
        dpo_validation=dpo_validation,
        mini_dpo_train=mini_dpo_train,
        mini_dpo_validation=mini_dpo_validation,
        manual_review=manual_review,
        dpo_failures=_merge_counters(dpo_train_failures, dpo_validation_failures),
        smoke_test=smoke_test,
        run_id=run_id,
    )
    outputs = _stage2_outputs(output_paths, step_examples, sft_train, sft_validation, dpo_train, dpo_validation, manual_review, statistics)
    published = publish_json_outputs(
        outputs=outputs,
        manifest_name="manifest",
        overwrite=overwrite,
        run_id=run_id,
        manifest_builder=lambda manifest, paths, hashes, counts: _build_stage2_manifest(
            manifest,
            paths,
            hashes,
            counts,
            stage2_views,
            stage1_manifest_path,
            stage1_manifest_sha256,
            run_id,
            smoke_test,
        ),
    )
    return _result_payload(run_id, published, statistics)


def stage2_output_paths(config: Mapping[str, Any], output_dir: str | Path | None = None) -> dict[str, Path]:
    """Return Stage 2 final output paths."""

    data = config["data"]
    if output_dir is None:
        return {
            "step_train": Path(str(data["step_train_file"])),
            "step_validation": Path(str(data["step_validation_file"])),
            "step_evaluation": Path(str(data["step_eval_file"])),
            "sft_train": Path(str(data["sft_train_file"])),
            "sft_validation": Path(str(data["sft_validation_file"])),
            "dpo_train": Path(str(data["dpo_train_file"])),
            "dpo_validation": Path(str(data["dpo_validation_file"])),
            "manual_review": Path(str(data["manual_review_file"])),
            "statistics": Path(str(data["stage2_statistics_file"])),
            "manifest": Path(str(data["stage2_manifest_file"])),
        }
    root = Path(output_dir)
    return {
        "step_train": root / "step_train.jsonl",
        "step_validation": root / "step_validation.jsonl",
        "step_evaluation": root / "step_eval.jsonl",
        "sft_train": root / "sft_train.jsonl",
        "sft_validation": root / "sft_validation.jsonl",
        "dpo_train": root / "dpo_train.jsonl",
        "dpo_validation": root / "dpo_validation.jsonl",
        "manual_review": root / "manual_review_preferences.jsonl",
        "statistics": root / "stage2_statistics.json",
        "manifest": root / "stage2_manifest.json",
    }


def validate_mini_stage2_views(stage2_views: Mapping[str, Any], stage1_manifest: Mapping[str, Any]) -> None:
    """Validate that Mini Stage 2 source views come only from Stage 1 Mini IDs."""

    formal_stage1 = {split: set(stage1_manifest["views"]["formal"][split]) for split in SPLITS}
    mini_stage1 = {split: set(stage1_manifest["views"]["mini"][split]) for split in SPLITS}
    for split in SPLITS:
        _assert_subset(stage2_views["mini"]["step"][split], formal_stage1[split], mini_stage1[split], f"mini step {split}")
    for split in ("train", "validation"):
        _assert_subset(stage2_views["mini"]["sft_source_ids"][split], formal_stage1[split], mini_stage1[split], f"mini sft {split}")
        _assert_subset(stage2_views["mini"]["dpo_source_ids"][split], formal_stage1[split], mini_stage1[split], f"mini dpo {split}")


def _validate_stage2_configs(mini: Mapping[str, Any], formal: Mapping[str, Any]) -> None:
    for name, config in (("mini", mini), ("formal", formal)):
        negatives = config["negative_sampling"]
        if int(negatives["negatives_per_step"]) != 1:
            raise ValueError(f"{name}: Stage 2 only supports negative_sampling.negatives_per_step = 1")
        if negatives["strategy"] not in negatives["allowed_strategies"]:
            raise ValueError(f"{name}: negative_sampling.strategy must be listed in allowed_strategies")
        if "mixed" not in negatives["allowed_strategies"]:
            raise ValueError(f"{name}: allowed_strategies must include mixed")
    if mini["negative_sampling"]["strategy"] != formal["negative_sampling"]["strategy"]:
        raise ValueError("Mini and formal configs must share negative_sampling.strategy")
    if mini["preprocessing"]["minimum_steps"] != formal["preprocessing"]["minimum_steps"]:
        raise ValueError("Mini and formal configs must share preprocessing.minimum_steps")
    if mini["preprocessing"]["require_final_answer_for_dpo"] != formal["preprocessing"]["require_final_answer_for_dpo"]:
        raise ValueError("Mini and formal configs must share preprocessing.require_final_answer_for_dpo")
    if mini["negative_sampling"]["maximum_dpo_examples"] > formal["negative_sampling"]["maximum_dpo_examples"]:
        raise ValueError("Mini maximum_dpo_examples cannot exceed formal maximum_dpo_examples")


def _parse_steps(normalized: Mapping[str, list[dict[str, Any]]], minimum_steps: int) -> dict[str, list[dict[str, Any]]]:
    step_examples = {
        split: [parse_normalized_example(example, minimum_steps) for example in normalized[split]]
        for split in SPLITS
    }
    for rows in step_examples.values():
        for row in rows:
            validate_step_example(row)
    return step_examples


def _select_mini_step_examples(
    step_examples: Mapping[str, list[dict[str, Any]]],
    stage1_manifest: Mapping[str, Any],
    mini_config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    counts = sample_counts(dict(mini_config))
    selected: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows_by_id = {str(row["id"]): row for row in step_examples[split]}
        mini_ids = list(stage1_manifest["views"]["mini"][split])[: counts[split]]
        missing = [example_id for example_id in mini_ids if example_id not in rows_by_id]
        if missing:
            raise ValueError(f"Stage 2 step output missing Mini Stage 1 IDs for {split}: {missing[:3]}")
        selected[split] = [rows_by_id[example_id] for example_id in mini_ids]
    return selected


def _build_formal_and_mini_dpo(
    formal_steps: list[Mapping[str, Any]],
    mini_steps: list[Mapping[str, Any]],
    formal_config: Mapping[str, Any],
    mini_config: Mapping[str, Any],
    formal_maximum: int,
    mini_maximum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    formal_pairs, formal_failures = build_dpo_examples(
        list(formal_steps),
        formal_config,
        maximum=formal_maximum,
        max_pairs_per_source=MAX_DPO_PAIRS_PER_SOURCE,
        allowed_answer_confidences=DPO_CONFIDENCES,
    )
    mini_pairs, mini_failures = build_dpo_examples(
        list(mini_steps),
        mini_config,
        maximum=mini_maximum,
        max_pairs_per_source=MAX_DPO_PAIRS_PER_SOURCE,
        allowed_answer_confidences=DPO_CONFIDENCES,
    )
    mini_ids = {str(pair["id"]) for pair in mini_pairs}
    merged_by_id = {str(pair["id"]): pair for pair in mini_pairs}
    for pair in formal_pairs:
        merged_by_id.setdefault(str(pair["id"]), pair)
    ranked = sorted(merged_by_id.values(), key=lambda item: (item["metadata"]["sample_rank"], item["id"]))
    selected = [pair for pair in ranked if str(pair["id"]) in mini_ids]
    for pair in ranked:
        if len(selected) >= formal_maximum:
            break
        if str(pair["id"]) not in mini_ids:
            selected.append(pair)
    merged = sorted(selected, key=lambda item: (item["metadata"]["sample_rank"], item["id"]))
    return merged, mini_pairs, _merge_counters(formal_failures, mini_failures)


def _cap_normalized_for_run(
    normalized: dict[str, list[dict[str, Any]]],
    formal_config: Mapping[str, Any],
    smoke_test: bool,
) -> dict[str, list[dict[str, Any]]]:
    counts = sample_counts(dict(formal_config))
    if not smoke_test:
        return {split: normalized[split][: counts[split]] for split in SPLITS}
    return {split: normalized[split][: min(len(normalized[split]), counts[split] * 4)] for split in SPLITS}


def _validate_minimum_counts(
    mini: Mapping[str, Any],
    formal: Mapping[str, Any],
    mini_sft_train: list[Mapping[str, Any]],
    mini_sft_validation: list[Mapping[str, Any]],
    dpo_train: list[Mapping[str, Any]],
    mini_dpo_train: list[Mapping[str, Any]],
    mini_dpo_validation: list[Mapping[str, Any]],
) -> None:
    if not mini_sft_train:
        raise ValueError("No Mini SFT train examples were generated")
    if not mini_sft_validation:
        raise ValueError("No Mini SFT validation examples were generated")
    if len(mini_dpo_train) < int(mini["negative_sampling"]["minimum_dpo_examples"]):
        raise ValueError(f"Not enough Mini DPO train examples: {len(mini_dpo_train)}")
    if len(dpo_train) < int(formal["negative_sampling"]["minimum_dpo_examples"]):
        raise ValueError(f"Not enough formal DPO train examples: {len(dpo_train)}")
    if len(mini_dpo_validation) < int(mini["data"]["validation_samples"]):
        raise ValueError(f"Not enough Mini DPO validation examples: {len(mini_dpo_validation)}")


def _validate_examples(sft_examples: list[Mapping[str, Any]], dpo_examples: list[Mapping[str, Any]]) -> None:
    for row in sft_examples:
        validate_sft_example(row)
    for row in dpo_examples:
        validate_dpo_example(row)


def _build_stage2_views(
    stage1_manifest: Mapping[str, Any],
    mini: Mapping[str, Any],
    step_examples: Mapping[str, list[Mapping[str, Any]]],
    mini_step_examples: Mapping[str, list[Mapping[str, Any]]],
    sft_train: list[Mapping[str, Any]],
    sft_validation: list[Mapping[str, Any]],
    mini_sft_train: list[Mapping[str, Any]],
    mini_sft_validation: list[Mapping[str, Any]],
    dpo_train: list[Mapping[str, Any]],
    dpo_validation: list[Mapping[str, Any]],
    mini_dpo_train: list[Mapping[str, Any]],
    mini_dpo_validation: list[Mapping[str, Any]],
    manual_review: list[Mapping[str, Any]],
) -> dict[str, Any]:
    mini_counts = sample_counts(dict(mini))
    mini_review = int(mini["negative_sampling"]["save_manual_review_samples"])
    return {
        "formal": {
            "step": {split: [str(row["id"]) for row in step_examples[split]] for split in SPLITS},
            "sft": {
                "train": [str(row["id"]) for row in sft_train],
                "validation": [str(row["id"]) for row in sft_validation],
            },
            "dpo": {
                "train": [str(row["id"]) for row in dpo_train],
                "validation": [str(row["id"]) for row in dpo_validation],
            },
            "manual_review": [str(row["id"]) for row in manual_review],
        },
        "mini": {
            "step": {split: [str(row["id"]) for row in mini_step_examples[split][: mini_counts[split]]] for split in SPLITS},
            "sft": {
                "train": [str(row["id"]) for row in mini_sft_train],
                "validation": [str(row["id"]) for row in mini_sft_validation],
            },
            "sft_source_ids": {
                "train": [str(row["id"]).removesuffix("_sft") for row in mini_sft_train],
                "validation": [str(row["id"]).removesuffix("_sft") for row in mini_sft_validation],
            },
            "dpo": {
                "train": [str(row["id"]) for row in mini_dpo_train],
                "validation": [str(row["id"]) for row in mini_dpo_validation],
            },
            "dpo_source_ids": {
                "train": [str(row["metadata"]["normalized_id"]) for row in mini_dpo_train],
                "validation": [str(row["metadata"]["normalized_id"]) for row in mini_dpo_validation],
            },
            "manual_review": [str(row["id"]) for row in manual_review[:mini_review]],
        },
        "stage1_views": {
            "formal": stage1_manifest["views"]["formal"],
            "mini": stage1_manifest["views"]["mini"],
        },
    }


def _stage2_outputs(
    paths: Mapping[str, Path],
    step_examples: Mapping[str, list[dict[str, Any]]],
    sft_train: list[dict[str, Any]],
    sft_validation: list[dict[str, Any]],
    dpo_train: list[dict[str, Any]],
    dpo_validation: list[dict[str, Any]],
    manual_review: list[dict[str, Any]],
    statistics: Mapping[str, Any],
) -> dict[str, JsonOutput]:
    return {
        "step_train": JsonOutput(paths["step_train"], "jsonl", step_examples["train"], rows=len(step_examples["train"])),
        "step_validation": JsonOutput(paths["step_validation"], "jsonl", step_examples["validation"], rows=len(step_examples["validation"])),
        "step_evaluation": JsonOutput(paths["step_evaluation"], "jsonl", step_examples["evaluation"], rows=len(step_examples["evaluation"])),
        "sft_train": JsonOutput(paths["sft_train"], "jsonl", sft_train, rows=len(sft_train)),
        "sft_validation": JsonOutput(paths["sft_validation"], "jsonl", sft_validation, rows=len(sft_validation)),
        "dpo_train": JsonOutput(paths["dpo_train"], "jsonl", dpo_train, rows=len(dpo_train)),
        "dpo_validation": JsonOutput(paths["dpo_validation"], "jsonl", dpo_validation, rows=len(dpo_validation)),
        "manual_review": JsonOutput(paths["manual_review"], "jsonl", manual_review, rows=len(manual_review)),
        "statistics": JsonOutput(paths["statistics"], "json", statistics),
        "manifest": JsonOutput(paths["manifest"], "json", {"schema_version": "1.0", "stage": 2, "completed": False}),
    }


def _build_stage2_manifest(
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    counts: Mapping[str, int],
    stage2_views: Mapping[str, Any],
    stage1_manifest_path: Path,
    stage1_manifest_sha256: str,
    run_id: str,
    smoke_test: bool,
) -> dict[str, Any]:
    payload = dict(manifest)
    stage_files = {}
    for name in (
        "step_train",
        "step_validation",
        "step_evaluation",
        "sft_train",
        "sft_validation",
        "dpo_train",
        "dpo_validation",
        "manual_review",
    ):
        stage_files[name] = {"path": str(paths[name]), "rows": counts[name], "sha256": hashes[name]}
    payload.update(
        {
            "completed": True,
            "run_id": run_id,
            "smoke_test": smoke_test,
            "stage1_manifest_file": {"path": str(stage1_manifest_path), "sha256": stage1_manifest_sha256},
            "token_length_status": TOKEN_LENGTH_STATUS,
            "files": stage_files,
            "statistics_file": {"path": str(paths["statistics"]), "sha256": hashes["statistics"]},
            "manual_review_file": stage_files["manual_review"],
            "views": stage2_views,
        }
    )
    return payload


def _build_statistics(
    stage1_manifest_path: Path,
    stage1_manifest_sha256: str,
    stage1_manifest: Mapping[str, Any],
    step_examples: Mapping[str, list[Mapping[str, Any]]],
    mini_step_examples: Mapping[str, list[Mapping[str, Any]]],
    sft_train: list[Mapping[str, Any]],
    sft_validation: list[Mapping[str, Any]],
    mini_sft_train: list[Mapping[str, Any]],
    mini_sft_validation: list[Mapping[str, Any]],
    dpo_train: list[Mapping[str, Any]],
    dpo_validation: list[Mapping[str, Any]],
    mini_dpo_train: list[Mapping[str, Any]],
    mini_dpo_validation: list[Mapping[str, Any]],
    manual_review: list[Mapping[str, Any]],
    dpo_failures: Mapping[str, int],
    smoke_test: bool,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": 2,
        "completed": True,
        "run_id": run_id,
        "smoke_test": smoke_test,
        "dataset_name": stage1_manifest["dataset_name"],
        "dataset_revision": stage1_manifest["dataset_revision"],
        "source_split": stage1_manifest["source_split"],
        "seed": stage1_manifest["seed"],
        "stage1_manifest_file": {"path": str(stage1_manifest_path), "sha256": stage1_manifest_sha256},
        "token_length_status": TOKEN_LENGTH_STATUS,
        "step_counts_formal": {split: len(step_examples[split]) for split in SPLITS},
        "step_counts_mini": {split: len(mini_step_examples[split]) for split in SPLITS},
        "parse_status_counts": {split: parse_status_counts(list(step_examples[split])) for split in SPLITS},
        "parse_failure_reason_counts": {split: parse_failure_reason_counts(list(step_examples[split])) for split in SPLITS},
        "answer_extraction_method_counts": {split: answer_method_counts(list(step_examples[split])) for split in SPLITS},
        "answer_confidence_counts": {split: answer_confidence_counts(list(step_examples[split])) for split in SPLITS},
        "sft_counts_formal": {"train": len(sft_train), "validation": len(sft_validation)},
        "sft_counts_mini": {"train": len(mini_sft_train), "validation": len(mini_sft_validation)},
        "sft_source_status_counts": {
            "train": sft_status_counts(sft_train),
            "validation": sft_status_counts(sft_validation),
        },
        "dpo_counts_formal": {"train": len(dpo_train), "validation": len(dpo_validation)},
        "dpo_counts_mini": {"train": len(mini_dpo_train), "validation": len(mini_dpo_validation)},
        "dpo_max_pairs_per_source": MAX_DPO_PAIRS_PER_SOURCE,
        "dpo_allowed_answer_confidences": sorted(DPO_CONFIDENCES),
        "dpo_applied_strategy_counts": {
            "train": dpo_strategy_counts(dpo_train),
            "validation": dpo_strategy_counts(dpo_validation),
        },
        "dpo_mutation_failures": dict(sorted(dpo_failures.items())),
        "manual_review_count": len(manual_review),
    }


def _config_with_stage2_smoke_caps(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(config)
    copied["data"] = dict(config["data"])
    copied["negative_sampling"] = dict(config["negative_sampling"])
    copied["data"]["train_samples"] = int(config["smoke_test"]["train_samples"])
    copied["data"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
    copied["data"]["evaluation_samples"] = int(config["smoke_test"]["evaluation_samples"])
    copied["negative_sampling"]["maximum_dpo_examples"] = int(config["smoke_test"]["dpo_samples"])
    copied["negative_sampling"]["minimum_dpo_examples"] = min(
        int(config["negative_sampling"]["minimum_dpo_examples"]),
        int(config["smoke_test"]["dpo_samples"]),
    )
    copied["negative_sampling"]["save_manual_review_samples"] = min(
        int(config["negative_sampling"]["save_manual_review_samples"]),
        10,
    )
    return copied


def _merge_counters(*counters: Mapping[str, int]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update(counter)
    return dict(merged)


def _result_payload(run_id: str, published: PublishedOutputs, statistics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "counts": published.counts,
        "hashes": published.hashes,
        "paths": {name: str(path) for name, path in published.paths.items()},
        "stage2_statistics": statistics,
    }


def _assert_subset(candidate_ids: list[str], formal_ids: set[str], mini_ids: set[str], label: str) -> None:
    candidate_set = set(candidate_ids)
    outside_formal = sorted(candidate_set - formal_ids)
    outside_mini = sorted(candidate_set - mini_ids)
    if outside_formal:
        raise ValueError(f"{label} IDs are outside formal Stage 1 view: {outside_formal[:3]}")
    if outside_mini:
        raise ValueError(f"{label} IDs are outside Mini Stage 1 view: {outside_mini[:3]}")
