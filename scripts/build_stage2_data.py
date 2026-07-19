"""Stage 2 data construction entrypoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.config.load_config import load_stage1_configs, sample_counts
from mathalign_dpo.data.build_preferences import (
    build_dpo_examples,
    build_manual_review_examples,
    dpo_strategy_counts,
    validate_dpo_example,
)
from mathalign_dpo.data.build_sft import TOKEN_LENGTH_STATUS, build_sft_examples, sft_status_counts, validate_sft_example
from mathalign_dpo.data.parse_steps import (
    answer_method_counts,
    parse_normalized_example,
    parse_status_counts,
    validate_step_example,
)
from mathalign_dpo.data.select_views import load_normalized_views, load_stage1_manifest, mini_ids_for_stage1
from mathalign_dpo.data.write_outputs import JsonOutput, PublishedOutputs, publish_json_outputs


SPLITS = ("train", "validation", "evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2 step, SFT, and DPO data.")
    parser.add_argument("--mini-config", required=True, help="Path to the Mini YAML config.")
    parser.add_argument("--formal-config", required=True, help="Path to the formal YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Use deterministic smoke-test caps from config.")
    parser.add_argument("--output-dir", default=None, help="Override Stage 2 output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing Stage 2 outputs.")
    args = parser.parse_args()

    result = build_stage2_data(
        mini_config=args.mini_config,
        formal_config=args.formal_config,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def build_stage2_data(
    mini_config: str | Path,
    formal_config: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 2 and publish outputs transactionally."""

    configs = load_stage1_configs(mini_config, formal_config)
    mini = _config_with_stage2_smoke_caps(configs.mini) if smoke_test else configs.mini
    formal = _config_with_stage2_smoke_caps(configs.formal) if smoke_test else configs.formal
    _validate_stage2_configs(mini, formal)

    stage1_manifest = load_stage1_manifest(formal["data"]["split_manifest_file"], mini, formal)
    normalized = load_normalized_views(stage1_manifest)
    normalized = _cap_normalized_for_run(normalized, formal, smoke_test=smoke_test)

    minimum_steps = int(formal["preprocessing"]["minimum_steps"])
    step_examples = {
        split: [parse_normalized_example(example, minimum_steps) for example in normalized[split]]
        for split in SPLITS
    }
    for rows in step_examples.values():
        for row in rows:
            validate_step_example(row)

    sft_train = build_sft_examples(step_examples["train"], formal, int(formal["data"]["train_samples"]))
    sft_validation = build_sft_examples(
        step_examples["validation"],
        formal,
        int(formal["data"]["validation_samples"]),
    )
    dpo_train, dpo_train_failures = build_dpo_examples(
        step_examples["train"],
        formal,
        int(formal["negative_sampling"]["maximum_dpo_examples"]),
    )
    dpo_validation, dpo_validation_failures = build_dpo_examples(
        step_examples["validation"],
        formal,
        int(formal["data"]["validation_samples"]),
    )
    _validate_minimum_counts(mini, formal, sft_train, sft_validation, dpo_train, dpo_validation)
    for row in [*sft_train, *sft_validation]:
        validate_sft_example(row)
    for row in [*dpo_train, *dpo_validation]:
        validate_dpo_example(row)

    manual_review = build_manual_review_examples(
        [*dpo_train, *dpo_validation],
        sample_count=min(int(formal["negative_sampling"]["save_manual_review_samples"]), len(dpo_train) + len(dpo_validation)),
        seed=int(formal["project"]["seed"]),
    )

    output_paths = stage2_output_paths(formal, output_dir=output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_stage2")
    stage2_views = _build_stage2_views(mini, formal, step_examples, sft_train, sft_validation, dpo_train, dpo_validation, manual_review)
    statistics = _build_statistics(
        stage1_statistics_path=Path(str(stage1_manifest["statistics_file"]["path"])),
        stage1_manifest=stage1_manifest,
        step_examples=step_examples,
        sft_train=sft_train,
        sft_validation=sft_validation,
        dpo_train=dpo_train,
        dpo_validation=dpo_validation,
        manual_review=manual_review,
        dpo_failures=_merge_counters(dpo_train_failures, dpo_validation_failures),
        smoke_test=smoke_test,
        run_id=run_id,
    )
    outputs = _stage2_outputs(output_paths, step_examples, sft_train, sft_validation, dpo_train, dpo_validation, manual_review, statistics, stage1_manifest)
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
            "statistics": Path(str(data["statistics_file"])),
            "manifest": Path(str(data["split_manifest_file"])),
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
        "statistics": root / "data_statistics.json",
        "manifest": root / "split_manifest.json",
    }


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


def _cap_normalized_for_run(
    normalized: dict[str, list[dict[str, Any]]],
    formal_config: Mapping[str, Any],
    smoke_test: bool,
) -> dict[str, list[dict[str, Any]]]:
    counts = sample_counts(dict(formal_config))
    if not smoke_test:
        return {split: normalized[split][: counts[split]] for split in SPLITS}
    return {
        split: normalized[split][: min(len(normalized[split]), counts[split] * 4)]
        for split in SPLITS
    }


def _validate_minimum_counts(
    mini: Mapping[str, Any],
    formal: Mapping[str, Any],
    sft_train: list[Mapping[str, Any]],
    sft_validation: list[Mapping[str, Any]],
    dpo_train: list[Mapping[str, Any]],
    dpo_validation: list[Mapping[str, Any]],
) -> None:
    if len(sft_train) < int(mini["data"]["train_samples"]):
        raise ValueError(f"Not enough SFT train examples for Mini: {len(sft_train)}")
    if len(sft_validation) < int(mini["data"]["validation_samples"]):
        raise ValueError(f"Not enough SFT validation examples for Mini: {len(sft_validation)}")
    if len(dpo_train) < int(mini["negative_sampling"]["minimum_dpo_examples"]):
        raise ValueError(f"Not enough DPO train examples for Mini: {len(dpo_train)}")
    if len(dpo_train) < int(formal["negative_sampling"]["minimum_dpo_examples"]):
        raise ValueError(f"Not enough DPO train examples for formal: {len(dpo_train)}")
    if len(dpo_validation) < int(mini["data"]["validation_samples"]):
        raise ValueError(f"Not enough DPO validation examples for Mini: {len(dpo_validation)}")


def _build_stage2_views(
    mini: Mapping[str, Any],
    formal: Mapping[str, Any],
    step_examples: Mapping[str, list[Mapping[str, Any]]],
    sft_train: list[Mapping[str, Any]],
    sft_validation: list[Mapping[str, Any]],
    dpo_train: list[Mapping[str, Any]],
    dpo_validation: list[Mapping[str, Any]],
    manual_review: list[Mapping[str, Any]],
) -> dict[str, Any]:
    mini_counts = sample_counts(dict(mini))
    formal_counts = sample_counts(dict(formal))
    mini_dpo = int(mini["negative_sampling"]["maximum_dpo_examples"])
    formal_dpo = int(formal["negative_sampling"]["maximum_dpo_examples"])
    mini_review = int(mini["negative_sampling"]["save_manual_review_samples"])
    formal_review = int(formal["negative_sampling"]["save_manual_review_samples"])
    return {
        "formal": {
            "step": {split: [str(row["id"]) for row in step_examples[split][: formal_counts[split]]] for split in SPLITS},
            "sft": {
                "train": [str(row["id"]) for row in sft_train[: formal_counts["train"]]],
                "validation": [str(row["id"]) for row in sft_validation[: formal_counts["validation"]]],
            },
            "dpo": {
                "train": [str(row["id"]) for row in dpo_train[:formal_dpo]],
                "validation": [str(row["id"]) for row in dpo_validation[: formal_counts["validation"]]],
            },
            "manual_review": [str(row["id"]) for row in manual_review[:formal_review]],
        },
        "mini": {
            "step": {split: [str(row["id"]) for row in step_examples[split][: mini_counts[split]]] for split in SPLITS},
            "sft": {
                "train": [str(row["id"]) for row in sft_train[: mini_counts["train"]]],
                "validation": [str(row["id"]) for row in sft_validation[: mini_counts["validation"]]],
            },
            "dpo": {
                "train": [str(row["id"]) for row in dpo_train[:mini_dpo]],
                "validation": [str(row["id"]) for row in dpo_validation[: mini_counts["validation"]]],
            },
            "manual_review": [str(row["id"]) for row in manual_review[:mini_review]],
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
    stage1_manifest: Mapping[str, Any],
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
        "manifest": JsonOutput(paths["manifest"], "json", stage1_manifest),
    }


def _build_stage2_manifest(
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    counts: Mapping[str, int],
    stage2_views: Mapping[str, Any],
    run_id: str,
    smoke_test: bool,
) -> dict[str, Any]:
    payload = dict(manifest)
    payload["completed"] = True
    payload["statistics_file"] = {"path": str(paths["statistics"]), "sha256": hashes["statistics"]}
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
    payload["stage2"] = {
        "completed": True,
        "run_id": run_id,
        "smoke_test": smoke_test,
        "token_length_status": TOKEN_LENGTH_STATUS,
        "files": stage_files,
        "statistics_file": {"path": str(paths["statistics"]), "sha256": hashes["statistics"]},
        "manual_review_file": stage_files["manual_review"],
        "views": stage2_views,
    }
    return payload


def _build_statistics(
    stage1_statistics_path: Path,
    stage1_manifest: Mapping[str, Any],
    step_examples: Mapping[str, list[Mapping[str, Any]]],
    sft_train: list[Mapping[str, Any]],
    sft_validation: list[Mapping[str, Any]],
    dpo_train: list[Mapping[str, Any]],
    dpo_validation: list[Mapping[str, Any]],
    manual_review: list[Mapping[str, Any]],
    dpo_failures: Mapping[str, int],
    smoke_test: bool,
    run_id: str,
) -> dict[str, Any]:
    with stage1_statistics_path.open("r", encoding="utf-8") as handle:
        statistics = json.load(handle)
    statistics["stage"] = 2
    statistics["stage2"] = {
        "run_id": run_id,
        "smoke_test": smoke_test,
        "dataset_name": stage1_manifest["dataset_name"],
        "dataset_revision": stage1_manifest["dataset_revision"],
        "source_split": stage1_manifest["source_split"],
        "token_length_status": TOKEN_LENGTH_STATUS,
        "step_counts": {split: len(step_examples[split]) for split in SPLITS},
        "parse_status_counts": {split: parse_status_counts(list(step_examples[split])) for split in SPLITS},
        "answer_extraction_method_counts": {split: answer_method_counts(list(step_examples[split])) for split in SPLITS},
        "sft_counts": {"train": len(sft_train), "validation": len(sft_validation)},
        "sft_source_status_counts": {
            "train": sft_status_counts(sft_train),
            "validation": sft_status_counts(sft_validation),
        },
        "dpo_counts": {"train": len(dpo_train), "validation": len(dpo_validation)},
        "dpo_applied_strategy_counts": {
            "train": dpo_strategy_counts(dpo_train),
            "validation": dpo_strategy_counts(dpo_validation),
        },
        "dpo_mutation_failures": dict(sorted(dpo_failures.items())),
        "manual_review_count": len(manual_review),
    }
    return statistics


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
        "stage2_statistics": statistics["stage2"],
    }


if __name__ == "__main__":
    main()
