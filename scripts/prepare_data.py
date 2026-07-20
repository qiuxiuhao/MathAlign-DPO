"""Stage 1 preprocessing for Math-Step-DPO-10K.

This entrypoint builds the final Hugging Face Datasets consumed by SFT, DPO,
and Stage 4 evaluation. It intentionally stops using the old NuminaMath
solution-parsing and step-mutation data path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "2.0"
REQUIRED_FIELDS = ("prompt", "initial_reason_steps", "full_chosen", "full_rejected", "answer")
MODEL_INPUT_SUFFIX = "Let's think step by step."


@dataclass(frozen=True)
class ProjectConfigs:
    mini_path: Path
    formal_path: Path
    mini: dict[str, Any]
    formal: dict[str, Any]

    @property
    def dataset_name(self) -> str:
        return str(self.formal["data"]["dataset_name"])

    @property
    def dataset_revision(self) -> str:
        return str(self.formal["data"]["dataset_revision"])

    @property
    def source_split(self) -> str:
        return str(self.formal["data"]["source_split"])

    @property
    def seed(self) -> int:
        return int(self.formal["project"]["seed"])


@dataclass(frozen=True)
class Tokenizers:
    mini: Any
    formal: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedExample:
    raw_row_index: int
    source_id: str
    split_key: str
    normalized_prompt: str
    model_input: str
    full_positive: str
    full_negative: str
    answer: str
    source_dataset: str | None
    sft_token_count: int
    dpo_token_count: dict[str, int]
    evaluation_prompt_token_count: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Math-Step-DPO-10K Stage 1 final Datasets.")
    parser.add_argument("--mini-config", required=True, help="Path to the Mini YAML config.")
    parser.add_argument("--formal-config", required=True, help="Path to the formal YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Use deterministic smoke-test target counts.")
    parser.add_argument("--output-dir", default=None, help="Override processed output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing processed outputs.")
    parser.add_argument("--refresh-raw", action="store_true", help="Redownload and replace data/raw/Math-Step-DPO-10K.")
    args = parser.parse_args()

    result = prepare_data(
        mini_config=args.mini_config,
        formal_config=args.formal_config,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        refresh_raw=args.refresh_raw,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


def prepare_data(
    mini_config: str | Path,
    formal_config: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    refresh_raw: bool = False,
) -> dict[str, Any]:
    datasets = _import_datasets()
    configs = load_project_configs(mini_config, formal_config)
    mini = _config_with_smoke_counts(configs.mini) if smoke_test else configs.mini
    formal = _config_with_smoke_counts(configs.formal) if smoke_test else configs.formal
    raw_path = _raw_dataset_path(configs.formal)
    processed_root = Path(output_dir) if output_dir else Path(str(formal["data"]["processed_dir"]))
    _prepare_output_root(processed_root, overwrite)

    raw = load_or_download_raw_dataset(
        datasets=datasets,
        dataset_name=configs.dataset_name,
        dataset_revision=configs.dataset_revision,
        source_split=configs.source_split,
        raw_path=raw_path,
        refresh_raw=refresh_raw,
        overwrite=overwrite,
    )
    raw_rows = _dataset_rows(raw)
    tokenizers = load_tokenizers(mini, formal)
    build = build_final_datasets(datasets, raw_rows, configs, mini, formal, tokenizers)
    save_final_datasets(build["datasets"], processed_root)

    metadata = build_metadata(
        configs=configs,
        mini=mini,
        formal=formal,
        raw_path=raw_path,
        processed_root=processed_root,
        raw_source_rows=len(raw_rows),
        smoke_test=smoke_test,
        tokenizers=tokenizers,
        statistics=build["statistics"],
    )
    _write_json(processed_root / "metadata.json", metadata)
    _write_json(processed_root / "split_manifest.json", build["split_manifest"])
    return {
        "status": "completed",
        "stage": 1,
        "smoke_test": smoke_test,
        "raw_dataset_path": str(raw_path),
        "processed_dir": str(processed_root),
        "actual_counts": metadata["actual_counts"],
        "shortfall_counts": metadata["shortfall_counts"],
        "filter_counts_by_reason": metadata["filter_counts_by_reason"],
    }


def load_project_configs(mini_config: str | Path, formal_config: str | Path) -> ProjectConfigs:
    mini_path = Path(mini_config)
    formal_path = Path(formal_config)
    mini = _load_yaml(mini_path)
    formal = _load_yaml(formal_path)
    _validate_config_pair(mini, formal, mini_path, formal_path)
    return ProjectConfigs(mini_path=mini_path, formal_path=formal_path, mini=mini, formal=formal)


def load_or_download_raw_dataset(
    datasets: Any,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    raw_path: Path,
    refresh_raw: bool,
    overwrite: bool,
) -> Any:
    if raw_path.exists() and not refresh_raw:
        loaded = datasets.load_from_disk(str(raw_path))
    else:
        if raw_path.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to replace raw dataset without --overwrite: {raw_path}")
            shutil.rmtree(raw_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        loaded = datasets.load_dataset(dataset_name, revision=dataset_revision, split=source_split)
        loaded.save_to_disk(str(raw_path))
    if isinstance(loaded, datasets.DatasetDict):
        if source_split not in loaded:
            raise ValueError(f"Raw DatasetDict missing configured split {source_split!r}: {raw_path}")
        return loaded[source_split]
    return loaded


def build_final_datasets(
    datasets: Any,
    raw_rows: Sequence[Mapping[str, Any]],
    configs: ProjectConfigs,
    mini: Mapping[str, Any],
    formal: Mapping[str, Any],
    tokenizers: Tokenizers,
) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {
        "field_filter": Counter(),
        "dedupe": Counter(),
        "formal_length": Counter(),
        "mini_length": Counter(),
    }
    prepared = prepare_formal_examples(raw_rows, configs, formal, tokenizers.formal, counters)
    targets = target_counts(formal)
    train_count = min(int(targets["sft"]["train"]), len(prepared))
    validation_start = train_count
    validation_count = min(int(targets["sft"]["validation"]), max(0, len(prepared) - validation_start))
    evaluation_start = validation_start + validation_count
    evaluation_count = min(int(targets["evaluation"]), max(0, len(prepared) - evaluation_start))
    reserve_start = evaluation_start + evaluation_count
    formal_examples = {
        "train": prepared[:train_count],
        "validation": prepared[validation_start:evaluation_start],
        "evaluation": prepared[evaluation_start:reserve_start],
        "reserve": prepared[reserve_start:],
    }

    formal_rows = build_mode_rows(formal_examples, formal, tokenizers.formal)
    mini_rows = build_mini_rows(formal_examples, mini, tokenizers.mini, counters["mini_length"])
    final = {
        "formal": formal_rows,
        "mini": mini_rows,
    }
    assert_mini_prefix(final)
    split_manifest = build_split_manifest(formal_examples, configs, prepared_count=len(prepared))
    return {
        "datasets": {
            "formal": dataset_objects(datasets, formal_rows),
            "mini": dataset_objects(datasets, mini_rows),
        },
        "split_manifest": split_manifest,
        "statistics": {
            "actual_counts": dataset_counts(final),
            "shortfall_counts": shortfall_counts(final, mini, formal),
            "filter_counts_by_reason": {key: dict(sorted(value.items())) for key, value in counters.items()},
        },
    }


def prepare_formal_examples(
    raw_rows: Sequence[Mapping[str, Any]],
    configs: ProjectConfigs,
    formal: Mapping[str, Any],
    tokenizer: Any,
    counters: Mapping[str, Counter[str]],
) -> list[PreparedExample]:
    by_prompt: dict[str, list[tuple[str, int, Mapping[str, Any], dict[str, str]]]] = {}
    for raw_index, row in enumerate(progress(raw_rows, "normalize/filter raw rows", total=len(raw_rows))):
        fields, reason = normalize_raw_fields(row, formal["preprocessing"])
        if reason is not None:
            counters["field_filter"][reason] += 1
            continue
        normalized_prompt = normalize_prompt_for_dedupe(fields["prompt"])
        if not normalized_prompt:
            counters["field_filter"]["empty_normalized_prompt"] += 1
            continue
        sort_key = stable_row_key(configs, normalized_prompt, raw_index)
        by_prompt.setdefault(normalized_prompt, []).append((sort_key, raw_index, row, fields))

    representatives: list[tuple[str, int, Mapping[str, Any], dict[str, str], str]] = []
    for normalized_prompt, candidates in progress(by_prompt.items(), "dedupe prompts", total=len(by_prompt), unit="prompt"):
        candidates.sort(key=lambda item: item[0])
        counters["dedupe"]["duplicate_prompt_rows"] += max(0, len(candidates) - 1)
        sort_key, raw_index, row, fields = candidates[0]
        representatives.append((sort_key, raw_index, row, fields, normalized_prompt))

    prepared: list[PreparedExample] = []
    for _sort_key, raw_index, row, fields, normalized_prompt in progress(
        representatives,
        "token length filtering",
        total=len(representatives),
        unit="prompt",
    ):
        source_id = hash_text(normalized_prompt)
        model_input = f"{fields['prompt']}\n\n{MODEL_INPUT_SUFFIX}"
        full_positive = join_reasoning(fields["initial_reason_steps"], fields["full_chosen"])
        full_negative = join_reasoning(fields["initial_reason_steps"], fields["full_rejected"])
        prompt_messages = base_messages(model_input, formal)
        sft_messages = [*prompt_messages, assistant_message(full_positive)]
        dpo_pair = {
            "prompt": prompt_messages,
            "chosen": [assistant_message(full_positive)],
            "rejected": [assistant_message(full_negative)],
        }
        sft_token_count = count_chat_tokens(tokenizer, sft_messages, add_generation_prompt=False)
        if sft_token_count > int(formal["model"]["max_length"]):
            counters["formal_length"]["sft_too_long"] += 1
            continue
        dpo_token_count = count_dpo_tokens(tokenizer, dpo_pair)
        dpo_reason = dpo_length_filter_reason(dpo_token_count, formal)
        if dpo_reason is not None:
            counters["formal_length"][dpo_reason] += 1
            continue
        evaluation_prompt_token_count = count_chat_tokens(tokenizer, prompt_messages, add_generation_prompt=True)
        if evaluation_prompt_token_count > int(formal["model"]["max_length"]):
            counters["formal_length"]["evaluation_prompt_too_long"] += 1
            continue
        split_key = stable_split_key(configs, normalized_prompt)
        prepared.append(
            PreparedExample(
                raw_row_index=raw_index,
                source_id=source_id,
                split_key=split_key,
                normalized_prompt=normalized_prompt,
                model_input=model_input,
                full_positive=full_positive,
                full_negative=full_negative,
                answer=fields["answer"],
                source_dataset=str(row.get("dataset")) if row.get("dataset") is not None else None,
                sft_token_count=sft_token_count,
                dpo_token_count=dpo_token_count,
                evaluation_prompt_token_count=evaluation_prompt_token_count,
            )
        )
    prepared.sort(key=lambda item: item.split_key)
    return prepared


def normalize_raw_fields(row: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    fields: dict[str, str] = {}
    for key in REQUIRED_FIELDS:
        value = row.get(key)
        if not isinstance(value, str):
            return {}, f"{key}_not_string"
        normalized = normalize_text(value, preprocessing)
        if not normalized:
            return {}, f"{key}_empty"
        fields[key] = normalized
    initial = fields["initial_reason_steps"]
    for key in ("full_chosen", "full_rejected"):
        if starts_with_normalized(fields[key], initial):
            return {}, f"{key}_duplicated_initial_reason_steps"
    positive = join_reasoning(initial, fields["full_chosen"])
    negative = join_reasoning(initial, fields["full_rejected"])
    if normalize_for_comparison(positive) == normalize_for_comparison(negative):
        return {}, "full_positive_equals_full_negative"
    return fields, None


def build_mode_rows(
    examples: Mapping[str, Sequence[PreparedExample]],
    config: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    sft = {
        split: [build_sft_row(example, split, index, tokenizer, config) for index, example in enumerate(examples[split])]
        for split in ("train", "validation")
    }
    dpo = {
        split: [build_dpo_row(example, split, index, tokenizer, config) for index, example in enumerate(examples[split])]
        for split in ("train", "validation")
    }
    evaluation = [build_evaluation_row(example, index, tokenizer, config) for index, example in enumerate(examples["evaluation"])]
    reserve = [build_reserve_row(example, index, tokenizer) for index, example in enumerate(examples.get("reserve", []))]
    return {"sft": sft, "dpo": dpo, "evaluation": evaluation, "reserve": reserve}


def build_mini_rows(
    formal_examples: Mapping[str, Sequence[PreparedExample]],
    mini: Mapping[str, Any],
    tokenizer: Any,
    counters: Counter[str],
) -> dict[str, Any]:
    targets = target_counts(mini)
    mini_examples = {
        "train": list(formal_examples["train"][: min(int(targets["sft"]["train"]), len(formal_examples["train"]))]),
        "validation": list(formal_examples["validation"][: min(int(targets["sft"]["validation"]), len(formal_examples["validation"]))]),
        "evaluation": list(formal_examples["evaluation"][: min(int(targets["evaluation"]), len(formal_examples["evaluation"]))]),
    }
    for split in ("train", "validation"):
        for example in progress(
            mini_examples[split],
            f"mini {split} prefix length check",
            total=len(mini_examples[split]),
            unit="row",
        ):
            sft_count = count_chat_tokens(tokenizer, [*base_messages(example.model_input, mini), assistant_message(example.full_positive)], False)
            if sft_count > int(mini["model"]["max_length"]):
                counters[f"sft_{split}_prefix_too_long"] += 1
                raise ValueError(f"Mini {split} prefix row exceeds SFT max_length: {example.source_id}")
            dpo_count = count_dpo_tokens(
                tokenizer,
                {
                    "prompt": base_messages(example.model_input, mini),
                    "chosen": [assistant_message(example.full_positive)],
                    "rejected": [assistant_message(example.full_negative)],
                },
            )
            reason = dpo_length_filter_reason(dpo_count, mini)
            if reason is not None:
                counters[f"dpo_{split}_{reason}"] += 1
                raise ValueError(f"Mini {split} prefix row fails DPO length filter {reason}: {example.source_id}")
    for example in progress(
        mini_examples["evaluation"],
        "mini evaluation prefix length check",
        total=len(mini_examples["evaluation"]),
        unit="row",
    ):
        prompt_tokens = count_chat_tokens(tokenizer, base_messages(example.model_input, mini), add_generation_prompt=True)
        if prompt_tokens > int(mini["model"]["max_length"]):
            counters["evaluation_prompt_too_long"] += 1
            raise ValueError(f"Mini evaluation prefix row exceeds prompt max_length: {example.source_id}")
    return build_mode_rows(mini_examples, mini, tokenizer)


def build_sft_row(example: PreparedExample, split: str, index: int, tokenizer: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    prompt = base_messages(example.model_input, config)
    completion = [assistant_message(example.full_positive)]
    messages = [*prompt, *completion]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"math_step_dpo_{split}_{index:05d}_sft",
        "source_id": example.source_id,
        "prompt": prompt,
        "completion": completion,
        "messages": messages,
        "token_count": count_chat_tokens(tokenizer, messages, add_generation_prompt=False),
        "split": split,
        "metadata": common_metadata(example),
    }


def build_dpo_row(example: PreparedExample, split: str, index: int, tokenizer: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    prompt = base_messages(example.model_input, config)
    chosen = [assistant_message(example.full_positive)]
    rejected = [assistant_message(example.full_negative)]
    pair = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"math_step_dpo_{split}_{index:05d}_dpo",
        "source_id": example.source_id,
        "step_index": 0,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "token_count": count_dpo_tokens(tokenizer, pair),
        "split": split,
        "metadata": {**common_metadata(example), "dpo_type": "full_response"},
    }


def build_evaluation_row(example: PreparedExample, index: int, tokenizer: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    prompt_messages = base_messages(example.model_input, config)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"math_step_dpo_evaluation_{index:05d}",
        "source_id": example.source_id,
        "problem": example.model_input,
        "reference_answer": example.full_positive,
        "prompt_messages": prompt_messages,
        "prompt_token_count": count_chat_tokens(tokenizer, prompt_messages, add_generation_prompt=True),
        "split": "evaluation",
        "metadata": {**common_metadata(example), "answer": example.answer},
    }


def build_reserve_row(example: PreparedExample, index: int, tokenizer: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"math_step_dpo_reserve_{index:05d}",
        "source_id": example.source_id,
        "problem": example.model_input,
        "full_positive": example.full_positive,
        "full_negative": example.full_negative,
        "answer": example.answer,
        "split": "reserve",
        "metadata": common_metadata(example),
    }


def common_metadata(example: PreparedExample) -> dict[str, Any]:
    return {
        "raw_row_index": example.raw_row_index,
        "normalized_prompt_sha256": example.source_id,
        "source_dataset": example.source_dataset,
        "reference_answer_policy": "full_positive",
    }


def dataset_objects(datasets: Any, rows: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sft": datasets.DatasetDict({split: datasets.Dataset.from_list(rows["sft"][split]) for split in ("train", "validation")}),
        "dpo": datasets.DatasetDict({split: datasets.Dataset.from_list(rows["dpo"][split]) for split in ("train", "validation")}),
        "evaluation": datasets.Dataset.from_list(rows["evaluation"]),
        "reserve": datasets.Dataset.from_list(rows["reserve"]),
    }


def save_final_datasets(final: Mapping[str, Mapping[str, Any]], processed_root: Path) -> None:
    for mode in ("formal", "mini"):
        for kind in ("sft", "dpo", "evaluation"):
            path = processed_root / mode / kind
            path.parent.mkdir(parents=True, exist_ok=True)
            final[mode][kind].save_to_disk(str(path))
    reserve_path = processed_root / "formal" / "reserve"
    reserve_path.parent.mkdir(parents=True, exist_ok=True)
    final["formal"]["reserve"].save_to_disk(str(reserve_path))


def build_metadata(
    configs: ProjectConfigs,
    mini: Mapping[str, Any],
    formal: Mapping[str, Any],
    raw_path: Path,
    processed_root: Path,
    raw_source_rows: int,
    smoke_test: bool,
    tokenizers: Tokenizers,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "completed": True,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke_test": smoke_test,
        "dataset_name": configs.dataset_name,
        "dataset_revision": configs.dataset_revision,
        "source_split": configs.source_split,
        "seed": configs.seed,
        "raw_dataset_path": str(raw_path),
        "raw_source_rows": raw_source_rows,
        "processed_dataset_paths": {
            "formal": {
                "sft": str(processed_root / "formal" / "sft"),
                "dpo": str(processed_root / "formal" / "dpo"),
                "evaluation": str(processed_root / "formal" / "evaluation"),
                "reserve": str(processed_root / "formal" / "reserve"),
            },
            "mini": {
                "sft": str(processed_root / "mini" / "sft"),
                "dpo": str(processed_root / "mini" / "dpo"),
                "evaluation": str(processed_root / "mini" / "evaluation"),
            },
        },
        "config_paths": {"mini": str(configs.mini_path), "formal": str(configs.formal_path)},
        "tokenizers": tokenizers.metadata,
        "target_counts": {"mini": target_counts(mini), "formal": target_counts(formal)},
        "actual_counts": statistics["actual_counts"],
        "shortfall_counts": statistics["shortfall_counts"],
        "filter_counts_by_reason": statistics["filter_counts_by_reason"],
        "split_method": "sha256_math_step_dpo_10k_prompt_seed_revision_v1",
        "dedupe_method": "normalized_prompt_sha256_group_first_stable_raw_row_v1",
        "selection_method": "formal_sorted_then_mini_prefix_v1",
        "reference_answer_policy": "full_positive",
    }


def build_split_manifest(
    examples: Mapping[str, Sequence[PreparedExample]],
    configs: ProjectConfigs,
    prepared_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": configs.dataset_name,
        "dataset_revision": configs.dataset_revision,
        "source_split": configs.source_split,
        "seed": configs.seed,
        "prepared_unique_prompt_count": prepared_count,
        "splits": {
            split: [
                {
                    "rank": index,
                    "source_id": example.source_id,
                    "normalized_prompt_sha256": example.source_id,
                    "raw_row_index": example.raw_row_index,
                    "split_key": example.split_key,
                }
                for index, example in enumerate(examples[split])
            ]
            for split in ("train", "validation", "evaluation", "reserve")
        },
    }


def assert_mini_prefix(final: Mapping[str, Mapping[str, Any]]) -> None:
    for kind in ("sft", "dpo"):
        for split in ("train", "validation"):
            formal_ids = list(row["id"] for row in final["formal"][kind][split])
            mini_ids = list(row["id"] for row in final["mini"][kind][split])
            expected = formal_ids[: len(mini_ids)]
            expected_mini = [item.replace("math_step_dpo_", "math_step_dpo_") for item in expected]
            if mini_ids != expected_mini:
                formal_sources = [row["source_id"] for row in final["formal"][kind][split]]
                mini_sources = [row["source_id"] for row in final["mini"][kind][split]]
                if mini_sources != formal_sources[: len(mini_sources)]:
                    raise ValueError(f"Mini {kind}/{split} is not a formal prefix subset")
    formal_eval_sources = [row["source_id"] for row in final["formal"]["evaluation"]]
    mini_eval_sources = [row["source_id"] for row in final["mini"]["evaluation"]]
    if mini_eval_sources != formal_eval_sources[: len(mini_eval_sources)]:
        raise ValueError("Mini evaluation is not a formal prefix subset")


def dataset_counts(final: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        mode: {
            "sft": {split: len(final[mode]["sft"][split]) for split in ("train", "validation")},
            "dpo": {split: len(final[mode]["dpo"][split]) for split in ("train", "validation")},
            "evaluation": len(final[mode]["evaluation"]),
            **({"reserve": len(final[mode]["reserve"])} if "reserve" in final[mode] else {}),
        }
        for mode in ("formal", "mini")
    }


def shortfall_counts(final: Mapping[str, Mapping[str, Any]], mini: Mapping[str, Any], formal: Mapping[str, Any]) -> dict[str, Any]:
    counts = dataset_counts(final)
    targets = {"mini": target_counts(mini), "formal": target_counts(formal)}
    result: dict[str, Any] = {}
    for mode in ("formal", "mini"):
        result[mode] = {
            "sft": {
                split: max(0, int(targets[mode]["sft"][split]) - int(counts[mode]["sft"][split]))
                for split in ("train", "validation")
            },
            "dpo": {
                split: max(0, int(targets[mode]["dpo"][split]) - int(counts[mode]["dpo"][split]))
                for split in ("train", "validation")
            },
            "evaluation": max(0, int(targets[mode]["evaluation"]) - int(counts[mode]["evaluation"])),
        }
    return result


def load_tokenizers(mini: Mapping[str, Any], formal: Mapping[str, Any]) -> Tokenizers:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Stage 1 tokenizer length filtering requires the 'transformers' package") from exc

    loaded: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for mode, config in (("mini", mini), ("formal", formal)):
        model_dir = ensure_local_tokenizer_model(config)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=bool(config["model"].get("trust_remote_code", False)))
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(f"{mode}: tokenizer must provide a chat_template")
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is None:
                raise ValueError(f"{mode}: tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        loaded[mode] = tokenizer
        metadata[mode] = {
            "name_or_path": str(model_dir),
            "modelscope_name_or_path": str(config["model"].get("modelscope_name_or_path") or ""),
            "modelscope_revision": str(config["model"].get("modelscope_revision") or ""),
            "revision": str(config["model"]["revision"]),
            "max_length": int(config["model"]["max_length"]),
            "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest(),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    return Tokenizers(mini=loaded["mini"], formal=loaded["formal"], metadata=metadata)


def ensure_local_tokenizer_model(config: Mapping[str, Any]) -> Path:
    local_dir = Path(str(config["model"]["name_or_path"]))
    if _has_tokenizer_files(local_dir):
        return local_dir
    remote = str(config["model"].get("modelscope_name_or_path") or config["model"].get("remote_name_or_path") or "")
    if not remote:
        raise ValueError(f"model.modelscope_name_or_path is required when tokenizer files are missing: {local_dir}")
    revision = str(config["model"].get("modelscope_revision") or "master")
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Stage 1 tokenizer setup requires the 'modelscope' package when local tokenizer files are missing") from exc
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id=remote, revision=revision, local_dir=str(local_dir))
    if not _has_tokenizer_files(local_dir):
        raise FileNotFoundError(f"ModelScope download did not create tokenizer files in {local_dir}")
    return local_dir


def base_messages(model_input: str, config: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    system_prompt = "You are a careful mathematical reasoning assistant."
    if config is not None:
        system_prompt = str(config["preprocessing"]["system_prompt"]).strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": model_input.strip()},
    ]


def assistant_message(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content.strip()}


def count_chat_tokens(tokenizer: Any, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool) -> int:
    rendered = tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(rendered, "shape"):
        return int(rendered.shape[-1])
    return len(rendered)


def count_dpo_tokens(tokenizer: Any, row: Mapping[str, Any]) -> dict[str, int]:
    prompt = list(row["prompt"])
    chosen = list(row["chosen"])
    rejected = list(row["rejected"])
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
    chosen_ids = tokenizer.apply_chat_template(prompt + chosen, tokenize=True, return_dict=True)["input_ids"]
    rejected_ids = tokenizer.apply_chat_template(prompt + rejected, tokenize=True, return_dict=True)["input_ids"]
    prompt_len = len(_flatten_ids(prompt_ids))
    chosen_len = len(_flatten_ids(chosen_ids))
    rejected_len = len(_flatten_ids(rejected_ids))
    return {
        "prompt": prompt_len,
        "chosen_total": chosen_len,
        "rejected_total": rejected_len,
        "chosen_completion": chosen_len - prompt_len,
        "rejected_completion": rejected_len - prompt_len,
    }


def dpo_length_filter_reason(token_count: Mapping[str, int], config: Mapping[str, Any]) -> str | None:
    if int(token_count["prompt"]) > int(config["dpo"]["max_prompt_length"]):
        return "prompt_too_long"
    if int(token_count["chosen_total"]) > int(config["dpo"]["max_length"]):
        return "chosen_too_long"
    if int(token_count["rejected_total"]) > int(config["dpo"]["max_length"]):
        return "rejected_too_long"
    if int(token_count["chosen_completion"]) <= 0:
        return "chosen_completion_empty"
    if int(token_count["rejected_completion"]) <= 0:
        return "rejected_completion_empty"
    return None


def target_counts(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sft": {"train": int(config["data"]["train_samples"]), "validation": int(config["data"]["validation_samples"])},
        "dpo": {"train": int(config["dpo"]["train_samples"]), "validation": int(config["dpo"]["validation_samples"])},
        "evaluation": int(config["evaluation"]["samples"]),
    }


def _config_with_smoke_counts(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    train = int(config["smoke_test"]["train_samples"])
    validation = int(config["smoke_test"]["validation_samples"])
    copied["data"]["train_samples"] = train
    copied["data"]["validation_samples"] = validation
    copied["dpo"]["train_samples"] = train
    copied["dpo"]["validation_samples"] = validation
    copied["evaluation"]["samples"] = int(config["smoke_test"]["evaluation_samples"])
    return copied


def stable_row_key(configs: ProjectConfigs, normalized_prompt: str, raw_index: int) -> str:
    return hashlib.sha256(
        "|".join(["math_step_dpo_row_v1", str(configs.seed), configs.dataset_revision, normalized_prompt, str(raw_index)]).encode("utf-8")
    ).hexdigest()


def stable_split_key(configs: ProjectConfigs, normalized_prompt: str) -> str:
    return hashlib.sha256(
        "|".join(["math_step_dpo_10k_v1", str(configs.seed), configs.dataset_revision, normalized_prompt]).encode("utf-8")
    ).hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def join_reasoning(initial_reason_steps: str, tail: str) -> str:
    return f"{initial_reason_steps.strip()}\n\n{tail.strip()}"


def normalize_prompt_for_dedupe(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def normalize_for_comparison(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def starts_with_normalized(text: str, prefix: str) -> bool:
    return normalize_for_comparison(text).startswith(normalize_for_comparison(prefix))


def normalize_text(text: str, preprocessing: Mapping[str, Any]) -> str:
    normalized = text
    if preprocessing.get("normalize_line_endings", True):
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if preprocessing.get("strip_outer_whitespace", True):
        normalized = normalized.strip()
    max_blank_lines = int(preprocessing.get("max_consecutive_blank_lines", 2))
    if max_blank_lines >= 0:
        pattern = r"\n{" + str(max_blank_lines + 2) + r",}"
        normalized = re.sub(pattern, "\n" * (max_blank_lines + 1), normalized)
    return normalized


def _dataset_rows(raw: Any) -> list[Mapping[str, Any]]:
    return [raw[index] for index in progress(range(len(raw)), "read raw rows", total=len(raw), unit="row")]


def progress(iterable: Iterable[Any], desc: str, total: int | None = None, unit: str = "row") -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit)


def _flatten_ids(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        flattened: list[int] = []
        for item in ids:
            flattened.extend(int(value) for value in item)
        return flattened
    return [int(value) for value in ids]


def _raw_dataset_path(config: Mapping[str, Any]) -> Path:
    return Path(str(config["data"]["raw_dir"]))


def _prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty processed directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _has_tokenizer_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists() or (path / "tokenizer_config.json").exists()
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    return has_config and has_tokenizer


def _validate_config_pair(mini: Mapping[str, Any], formal: Mapping[str, Any], mini_path: Path, formal_path: Path) -> None:
    if mini.get("project", {}).get("run_mode") != "mini":
        raise ValueError(f"{mini_path}: project.run_mode must be 'mini'")
    if formal.get("project", {}).get("run_mode") != "formal":
        raise ValueError(f"{formal_path}: project.run_mode must be 'formal'")
    shared = ("dataset_name", "dataset_revision", "source_split", "processed_dir", "raw_dir", "mini_dir", "formal_dir")
    for key in shared:
        if mini["data"].get(key) != formal["data"].get(key):
            raise ValueError(f"Mini and formal configs must share data.{key}")
    if mini["project"]["seed"] != formal["project"]["seed"]:
        raise ValueError("Mini and formal configs must share project.seed")
    for path, config in ((mini_path, mini), (formal_path, formal)):
        if int(config["dpo"]["max_prompt_length"]) >= int(config["dpo"]["max_length"]):
            raise ValueError(f"{path}: dpo.max_prompt_length must be less than dpo.max_length")
        if int(config["data"]["train_samples"]) != int(config["dpo"]["train_samples"]):
            raise ValueError(f"{path}: data.train_samples and dpo.train_samples must match for shared SFT/DPO splits")
        if int(config["data"]["validation_samples"]) != int(config["dpo"]["validation_samples"]):
            raise ValueError(f"{path}: data.validation_samples and dpo.validation_samples must match for shared SFT/DPO splits")
    if int(mini["data"]["train_samples"]) > int(formal["data"]["train_samples"]):
        raise ValueError("Mini SFT train count cannot exceed formal")
    if int(mini["dpo"]["train_samples"]) > int(formal["dpo"]["train_samples"]):
        raise ValueError("Mini DPO train count cannot exceed formal")
    if int(mini["evaluation"]["samples"]) > int(formal["evaluation"]["samples"]):
        raise ValueError("Mini evaluation count cannot exceed formal")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Stage 1 config files") from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _import_datasets() -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Stage 1 requires the 'datasets' package") from exc
    return datasets


if __name__ == "__main__":
    main()
