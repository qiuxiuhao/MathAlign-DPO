"""Stage 1 all-in-one data preprocessing entrypoint.

This script intentionally owns the Stage 1 data path during the refactor. It
does not import ``mathalign_dpo.data`` because that old Stage 1/2 JSONL pipeline
has been removed in this stage.
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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0"
SOURCE_ID_FIELDS = ("id", "source_id", "problem_id", "question_id", "uuid", "uid")
PROBLEM_FIELDS = ("problem", "question", "prompt")
SOLUTION_FIELDS = ("solution", "answer", "response")
SPLITS = ("train", "validation", "evaluation")
NUMBER_MUTATION = "number_mutation"
OPERATOR_MUTATION = "operator_mutation"
MIXED_MUTATION = "mixed"
DPO_CONFIDENCES = {"high", "medium"}
ANSWER_METHOD_NONE = "none"
_NUMBER_RE = re.compile(r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?")
_OPERATOR_RE = re.compile(r"\\times|\\cdot|\\div|\\le|\\ge|[+\-*/=<>]")


@dataclass(frozen=True)
class ProjectConfigs:
    """The two approved configs plus their file paths."""

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
    """The tokenizer pair used for final Stage 1 length filtering."""

    mini: Any
    formal: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MutationResult:
    """A deterministic rule-based mutation result."""

    strategy: str
    text: str
    changed_span: tuple[int, int] | None
    replacement: str | None
    success: bool
    reason: str


@dataclass(frozen=True)
class AnswerExtraction:
    """Final answer extraction result."""

    answer: str | None
    method: str
    confidence: str


@dataclass(frozen=True)
class ParsedSolution:
    """Parsed reasoning steps and answer metadata."""

    steps: list[str]
    final_answer: str | None
    answer_method: str
    answer_confidence: str
    answer_candidate: str | None
    parse_status: str
    failure_reason: str | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 final Hugging Face Datasets.")
    parser.add_argument("--mini-config", required=True, help="Path to the Mini YAML config.")
    parser.add_argument("--formal-config", required=True, help="Path to the formal YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Use deterministic smoke-test target counts.")
    parser.add_argument("--output-dir", default=None, help="Override processed output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing processed outputs.")
    parser.add_argument("--refresh-raw", action="store_true", help="Redownload and replace data/raw/numina_math.")
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
    """Run Stage 1 preprocessing and publish final Dataset directories."""

    datasets = _import_datasets()
    configs = load_project_configs(mini_config, formal_config)
    mini = _config_with_smoke_counts(configs.mini) if smoke_test else configs.mini
    formal = _config_with_smoke_counts(configs.formal) if smoke_test else configs.formal
    raw_path = _raw_dataset_path(configs.formal)
    processed_root = Path(output_dir) if output_dir else Path(str(configs.formal["data"]["processed_dir"]))
    _prepare_processed_root(processed_root, overwrite)

    raw = load_or_download_raw_dataset(
        datasets=datasets,
        dataset_name=configs.dataset_name,
        dataset_revision=configs.dataset_revision,
        source_split=configs.source_split,
        raw_path=raw_path,
        refresh_raw=refresh_raw,
        overwrite=overwrite,
    )
    raw_source_rows = len(raw)
    if smoke_test:
        raw = raw.select(range(min(len(raw), _smoke_source_rows(formal))))

    normalized = normalize_dataset(raw, configs)
    tokenizers = load_tokenizers(mini, formal)
    build = build_final_datasets(datasets, normalized, configs, mini, formal, tokenizers)
    save_final_datasets(build["datasets"], processed_root)
    metadata = build_metadata(
        configs=configs,
        mini=mini,
        formal=formal,
        raw_path=raw_path,
        processed_root=processed_root,
        raw_source_rows=raw_source_rows,
        smoke_test=smoke_test,
        tokenizers=tokenizers,
        statistics=build["statistics"],
    )
    _write_json(processed_root / "metadata.json", metadata)
    return {
        "status": "completed",
        "stage": 1,
        "smoke_test": smoke_test,
        "raw_dataset_path": str(raw_path),
        "processed_dir": str(processed_root),
        "actual_counts": metadata["actual_counts"],
        "filter_counts_by_reason": metadata["filter_counts_by_reason"],
    }


def load_project_configs(mini_config: str | Path, formal_config: str | Path) -> ProjectConfigs:
    """Load the Mini/formal YAML pair without using the old config module."""

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
    """Load local raw Dataset, or download and persist it once."""

    if raw_path.exists() and not refresh_raw:
        return datasets.load_from_disk(str(raw_path))
    if raw_path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to replace raw dataset without --overwrite: {raw_path}")
        shutil.rmtree(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.load_dataset(dataset_name, revision=dataset_revision, split=source_split)
    dataset.save_to_disk(str(raw_path))
    return dataset


def normalize_dataset(raw: Any, configs: ProjectConfigs) -> Any:
    """Normalize raw fields, assign deterministic split/rank, and filter invalid rows."""

    problem_field = _first_present(raw.column_names, PROBLEM_FIELDS, "problem")
    solution_field = _first_present(raw.column_names, SOLUTION_FIELDS, "solution")
    id_field = next((field for field in SOURCE_ID_FIELDS if field in raw.column_names), None)
    preprocessing = configs.formal["preprocessing"]
    ratios = split_ratios(configs.formal)

    def normalize_batch(batch: Mapping[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        output: dict[str, list[Any]] = {
            "schema_version": [],
            "id": [],
            "source_id": [],
            "problem": [],
            "solution": [],
            "split": [],
            "rank": [],
            "is_valid": [],
            "filter_reason": [],
            "metadata": [],
        }
        for offset, row_index in enumerate(indices):
            problem = batch[problem_field][offset]
            solution = batch[solution_field][offset]
            raw_source_id = build_source_id(batch, offset, row_index, id_field)
            source_id = f"{raw_source_id}_{row_index:08d}" if id_field else raw_source_id
            stable_id = f"numina_{configs.source_split}_{source_id}"
            normalized_problem = normalize_text(problem, preprocessing) if isinstance(problem, str) else ""
            normalized_solution = normalize_text(solution, preprocessing) if isinstance(solution, str) else ""
            reason = _normalization_filter_reason(problem, solution, normalized_problem, normalized_solution)
            split = assign_split(
                source_id=source_id,
                dataset_name=configs.dataset_name,
                dataset_revision=configs.dataset_revision,
                source_split=configs.source_split,
                seed=configs.seed,
                ratios=ratios,
            )
            output["schema_version"].append(SCHEMA_VERSION)
            output["id"].append(stable_id)
            output["source_id"].append(source_id)
            output["problem"].append(normalized_problem)
            output["solution"].append(normalized_solution)
            output["split"].append(split)
            output["rank"].append(stable_rank(configs, source_id, split))
            output["is_valid"].append(reason is None)
            output["filter_reason"].append(reason or "")
            output["metadata"].append(
                {
                    "source": configs.dataset_name,
                    "source_split": configs.source_split,
                    "raw_source_id": raw_source_id,
                    "source_subset": _batch_value(batch, "source", offset),
                    "original_fields": list(raw.column_names),
                }
            )
        return output

    normalized = raw.map(
        normalize_batch,
        with_indices=True,
        batched=True,
        batch_size=1000,
        remove_columns=raw.column_names,
        desc="normalize fields and assign deterministic splits",
    )
    return normalized.filter(lambda row: bool(row["is_valid"]), desc="filter invalid normalized rows")


def build_final_datasets(
    datasets: Any,
    normalized: Any,
    configs: ProjectConfigs,
    mini: Mapping[str, Any],
    formal: Mapping[str, Any],
    tokenizers: Tokenizers,
) -> dict[str, Any]:
    """Build formal datasets first, then deterministic Mini subsets."""

    counters: dict[str, Counter[str]] = {
        "normalization": Counter(),
        "sft_mini": Counter(),
        "sft_formal": Counter(),
        "dpo_mini": Counter(),
        "dpo_formal": Counter(),
        "evaluation_mini": Counter(),
        "evaluation_formal": Counter(),
    }
    formal_sft = {
        "train": collect_sft_rows(
            normalized,
            "train",
            formal,
            tokenizers.formal,
            int(formal["data"]["train_samples"]),
            counters["sft_formal"],
            prefix_config=mini,
            prefix_tokenizer=tokenizers.mini,
            prefix_count=int(mini["data"]["train_samples"]),
        ),
        "validation": collect_sft_rows(
            normalized,
            "validation",
            formal,
            tokenizers.formal,
            int(formal["data"]["validation_samples"]),
            counters["sft_formal"],
            prefix_config=mini,
            prefix_tokenizer=tokenizers.mini,
            prefix_count=int(mini["data"]["validation_samples"]),
        ),
    }
    formal_dpo = {
        "train": collect_dpo_rows(
            normalized,
            "train",
            formal,
            tokenizers.formal,
            int(formal["dpo"]["train_samples"]),
            counters["dpo_formal"],
            prefix_config=mini,
            prefix_tokenizer=tokenizers.mini,
            prefix_count=int(mini["dpo"]["train_samples"]),
        ),
        "validation": collect_dpo_rows(
            normalized,
            "validation",
            formal,
            tokenizers.formal,
            int(formal["dpo"]["validation_samples"]),
            counters["dpo_formal"],
            prefix_config=mini,
            prefix_tokenizer=tokenizers.mini,
            prefix_count=int(mini["dpo"]["validation_samples"]),
        ),
    }
    formal_eval = collect_evaluation_rows(
        normalized,
        formal,
        tokenizers.formal,
        int(formal["evaluation"]["samples"]),
        counters["evaluation_formal"],
        prefix_config=mini,
        prefix_tokenizer=tokenizers.mini,
        prefix_count=int(mini["evaluation"]["samples"]),
    )

    mini_sft = {
        "train": refilter_mode_rows(formal_sft["train"], "sft", mini, tokenizers.mini, int(mini["data"]["train_samples"]), counters["sft_mini"]),
        "validation": refilter_mode_rows(
            formal_sft["validation"],
            "sft",
            mini,
            tokenizers.mini,
            int(mini["data"]["validation_samples"]),
            counters["sft_mini"],
        ),
    }
    mini_dpo = {
        "train": refilter_mode_rows(formal_dpo["train"], "dpo", mini, tokenizers.mini, int(mini["dpo"]["train_samples"]), counters["dpo_mini"]),
        "validation": refilter_mode_rows(
            formal_dpo["validation"],
            "dpo",
            mini,
            tokenizers.mini,
            int(mini["dpo"]["validation_samples"]),
            counters["dpo_mini"],
        ),
    }
    mini_eval = refilter_mode_rows(
        formal_eval,
        "evaluation",
        mini,
        tokenizers.mini,
        int(mini["evaluation"]["samples"]),
        counters["evaluation_mini"],
    )

    final = {
        "formal": {
            "sft": datasets.DatasetDict({split: datasets.Dataset.from_list(rows) for split, rows in formal_sft.items()}),
            "dpo": datasets.DatasetDict({split: datasets.Dataset.from_list(rows) for split, rows in formal_dpo.items()}),
            "evaluation": datasets.Dataset.from_list(formal_eval),
        },
        "mini": {
            "sft": datasets.DatasetDict({split: datasets.Dataset.from_list(rows) for split, rows in mini_sft.items()}),
            "dpo": datasets.DatasetDict({split: datasets.Dataset.from_list(rows) for split, rows in mini_dpo.items()}),
            "evaluation": datasets.Dataset.from_list(mini_eval),
        },
    }
    assert_mini_prefix(final)
    return {
        "datasets": final,
        "statistics": {
            "actual_counts": dataset_counts(final),
            "filter_counts_by_reason": {key: dict(sorted(value.items())) for key, value in counters.items()},
        },
    }


def collect_sft_rows(
    normalized: Any,
    split: str,
    config: Mapping[str, Any],
    tokenizer: Any,
    target_count: int,
    counters: Counter[str],
    prefix_config: Mapping[str, Any] | None = None,
    prefix_tokenizer: Any | None = None,
    prefix_count: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_split_rows(normalized, split):
        parsed = parse_solution(str(row["solution"]), int(config["preprocessing"]["minimum_steps"]))
        if parsed.parse_status not in {"success", "partial"}:
            counters[f"parse_{parsed.parse_status}"] += 1
            continue
        messages = base_messages(str(row["problem"]), config)
        messages.append(assistant_message(str(row["solution"])))
        token_count = count_chat_tokens(tokenizer, messages, add_generation_prompt=False)
        if token_count > int(config["model"]["max_length"]):
            counters["token_too_long"] += 1
            continue
        if prefix_tokenizer is not None and prefix_config is not None and len(rows) < prefix_count:
            prefix_token_count = count_chat_tokens(prefix_tokenizer, messages, add_generation_prompt=False)
            if prefix_token_count > int(prefix_config["model"]["max_length"]):
                counters["mini_prefix_token_too_long"] += 1
                continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"{row['id']}_sft",
                "source_id": str(row["id"]),
                "prompt": messages[:2],
                "completion": [messages[2]],
                "messages": messages,
                "token_count": token_count,
                "split": split,
                "metadata": {
                    "final_answer": parsed.final_answer,
                    "parse_status": parsed.parse_status,
                    "answer_confidence": parsed.answer_confidence,
                    "raw_source_id": row["metadata"]["raw_source_id"],
                },
            }
        )
        if len(rows) >= target_count:
            return rows
    raise ValueError(f"Not enough {split} SFT rows after Stage 1 filtering: need {target_count}, got {len(rows)}")


def collect_dpo_rows(
    normalized: Any,
    split: str,
    config: Mapping[str, Any],
    tokenizer: Any,
    target_count: int,
    counters: Counter[str],
    prefix_config: Mapping[str, Any] | None = None,
    prefix_tokenizer: Any | None = None,
    prefix_count: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(config["project"]["seed"])
    offsets = [int(offset) for offset in config["negative_sampling"]["number_offset_choices"]]
    strategy = str(config["negative_sampling"]["strategy"])
    for row in iter_split_rows(normalized, split):
        parsed = parse_solution(str(row["solution"]), int(config["preprocessing"]["minimum_steps"]))
        if parsed.parse_status != "success":
            counters[f"parse_{parsed.parse_status}"] += 1
            continue
        if not parsed.final_answer:
            counters["missing_final_answer"] += 1
            continue
        if parsed.answer_confidence not in DPO_CONFIDENCES:
            counters[f"answer_confidence_{parsed.answer_confidence}"] += 1
            continue
        for step_index, chosen_step in enumerate(parsed.steps):
            result = mutate_step(chosen_step, str(row["source_id"]), step_index, strategy, seed, offsets)
            if not result.success:
                counters[result.reason] += 1
                continue
            rejected_step = result.text
            if chosen_step.strip() == rejected_step.strip():
                counters["unchanged_output"] += 1
                continue
            prompt = base_messages(str(row["problem"]), config)
            prompt.extend(assistant_message(step) for step in parsed.steps[:step_index])
            if rejected_step in "\n".join(message["content"] for message in prompt):
                counters["rejected_in_prompt_history"] += 1
                continue
            pair = {
                "schema_version": SCHEMA_VERSION,
                "id": f"{row['id']}_step_{step_index:03d}_{result.strategy}",
                "source_id": str(row["id"]),
                "step_index": step_index,
                "prompt": prompt,
                "chosen": [assistant_message(chosen_step)],
                "rejected": [assistant_message(rejected_step)],
                "split": split,
                "metadata": {
                    "final_answer": parsed.final_answer,
                    "answer_confidence": parsed.answer_confidence,
                    "negative_strategy": strategy,
                    "mutation": mutation_metadata(result, strategy),
                    "raw_source_id": row["metadata"]["raw_source_id"],
                },
            }
            token_count = count_dpo_tokens(tokenizer, pair)
            reason = dpo_length_filter_reason(token_count, config)
            if reason is not None:
                counters[reason] += 1
                continue
            if prefix_tokenizer is not None and prefix_config is not None and len(rows) < prefix_count:
                prefix_token_count = count_dpo_tokens(prefix_tokenizer, pair)
                prefix_reason = dpo_length_filter_reason(prefix_token_count, prefix_config)
                if prefix_reason is not None:
                    counters[f"mini_prefix_{prefix_reason}"] += 1
                    continue
            pair["token_count"] = token_count
            rows.append(pair)
            if len(rows) >= target_count:
                return rows
    raise ValueError(f"Not enough {split} DPO rows after Stage 1 filtering: need {target_count}, got {len(rows)}")


def collect_evaluation_rows(
    normalized: Any,
    config: Mapping[str, Any],
    tokenizer: Any,
    target_count: int,
    counters: Counter[str],
    prefix_config: Mapping[str, Any] | None = None,
    prefix_tokenizer: Any | None = None,
    prefix_count: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_split_rows(normalized, "evaluation"):
        parsed = parse_solution(str(row["solution"]), int(config["preprocessing"]["minimum_steps"]))
        if parsed.parse_status != "success" or not parsed.final_answer:
            counters[f"parse_{parsed.parse_status}"] += 1
            continue
        prompt_messages = base_messages(str(row["problem"]), config)
        prompt_token_count = count_chat_tokens(tokenizer, prompt_messages, add_generation_prompt=True)
        if prompt_token_count + int(config["evaluation"]["max_new_tokens"]) > int(config["model"]["max_length"]):
            counters["prompt_plus_generation_too_long"] += 1
            continue
        if prefix_tokenizer is not None and prefix_config is not None and len(rows) < prefix_count:
            prefix_token_count = count_chat_tokens(prefix_tokenizer, prompt_messages, add_generation_prompt=True)
            if prefix_token_count + int(prefix_config["evaluation"]["max_new_tokens"]) > int(prefix_config["model"]["max_length"]):
                counters["mini_prefix_prompt_plus_generation_too_long"] += 1
                continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": str(row["id"]),
                "source_id": str(row["id"]),
                "problem": str(row["problem"]),
                "reference_answer": str(parsed.final_answer),
                "prompt_messages": prompt_messages,
                "prompt_token_count": prompt_token_count,
                "split": "evaluation",
                "metadata": {
                    "answer_confidence": parsed.answer_confidence,
                    "answer_extraction_method": parsed.answer_method,
                    "raw_source_id": row["metadata"]["raw_source_id"],
                },
            }
        )
        if len(rows) >= target_count:
            return rows
    raise ValueError(f"Not enough evaluation rows after Stage 1 filtering: need {target_count}, got {len(rows)}")


def refilter_mode_rows(
    rows: Sequence[Mapping[str, Any]],
    kind: str,
    config: Mapping[str, Any],
    tokenizer: Any,
    target_count: int,
    counters: Counter[str],
) -> list[dict[str, Any]]:
    """Recompute Mini token counts against the Mini tokenizer while preserving formal order."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        copied = json.loads(json.dumps(row, ensure_ascii=False))
        if kind == "sft":
            token_count = count_chat_tokens(tokenizer, copied["messages"], add_generation_prompt=False)
            if token_count > int(config["model"]["max_length"]):
                counters["token_too_long"] += 1
                continue
            copied["token_count"] = token_count
        elif kind == "dpo":
            token_count = count_dpo_tokens(tokenizer, copied)
            reason = dpo_length_filter_reason(token_count, config)
            if reason is not None:
                counters[reason] += 1
                continue
            copied["token_count"] = token_count
        elif kind == "evaluation":
            token_count = count_chat_tokens(tokenizer, copied["prompt_messages"], add_generation_prompt=True)
            if token_count + int(config["evaluation"]["max_new_tokens"]) > int(config["model"]["max_length"]):
                counters["prompt_plus_generation_too_long"] += 1
                continue
            copied["prompt_token_count"] = token_count
        else:
            raise ValueError(f"Unsupported final dataset kind: {kind}")
        selected.append(copied)
        if len(selected) >= target_count:
            return selected
    raise ValueError(f"Not enough Mini {kind} rows from formal prefix after token filtering: need {target_count}, got {len(selected)}")


def iter_split_rows(normalized: Any, split: str) -> Iterable[Mapping[str, Any]]:
    """Yield valid normalized rows for one split in deterministic order."""

    split_rows = normalized.filter(lambda row: row["split"] == split, desc=f"select {split} rows").sort("rank")
    for row in split_rows:
        yield row


def save_final_datasets(final: Mapping[str, Mapping[str, Any]], processed_root: Path) -> None:
    """Persist final Mini/formal Hugging Face Dataset directories."""

    for mode in ("mini", "formal"):
        for kind in ("sft", "dpo", "evaluation"):
            path = processed_root / mode / kind
            path.parent.mkdir(parents=True, exist_ok=True)
            final[mode][kind].save_to_disk(str(path))


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
            mode: {kind: str(processed_root / mode / kind) for kind in ("sft", "dpo", "evaluation")}
            for mode in ("mini", "formal")
        },
        "config_paths": {"mini": str(configs.mini_path), "formal": str(configs.formal_path)},
        "tokenizers": tokenizers.metadata,
        "target_counts": {"mini": target_counts(mini), "formal": target_counts(formal)},
        "actual_counts": statistics["actual_counts"],
        "filter_counts_by_reason": statistics["filter_counts_by_reason"],
        "split_method": "sha256_dataset_revision_source_id_seed_bucket_v1",
        "selection_method": "formal_first_stable_rank_then_mini_prefix_v1",
    }


def assert_mini_prefix(final: Mapping[str, Mapping[str, Any]]) -> None:
    """Ensure Mini rows are deterministic prefixes of formal rows by ID."""

    for kind in ("sft", "dpo"):
        for split in ("train", "validation"):
            formal_ids = list(final["formal"][kind][split]["id"])
            mini_ids = list(final["mini"][kind][split]["id"])
            if mini_ids != formal_ids[: len(mini_ids)]:
                raise ValueError(f"Mini {kind}/{split} is not a formal prefix subset")
    formal_eval_ids = list(final["formal"]["evaluation"]["id"])
    mini_eval_ids = list(final["mini"]["evaluation"]["id"])
    if mini_eval_ids != formal_eval_ids[: len(mini_eval_ids)]:
        raise ValueError("Mini evaluation is not a formal prefix subset")


def dataset_counts(final: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        mode: {
            "sft": {split: len(final[mode]["sft"][split]) for split in ("train", "validation")},
            "dpo": {split: len(final[mode]["dpo"][split]) for split in ("train", "validation")},
            "evaluation": len(final[mode]["evaluation"]),
        }
        for mode in ("mini", "formal")
    }


def load_tokenizers(mini: Mapping[str, Any], formal: Mapping[str, Any]) -> Tokenizers:
    """Load the exact tokenizer pair configured for Mini/formal length filtering."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Stage 1 tokenizer length filtering requires the 'transformers' package") from exc

    loaded: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for mode, config in (("mini", mini), ("formal", formal)):
        model_dir = ensure_local_tokenizer_model(config)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        )
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
    """Ensure the configured local model directory can provide a tokenizer."""

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


def _has_tokenizer_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists() or (path / "tokenizer_config.json").exists()
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    return has_config and has_tokenizer


def count_chat_tokens(tokenizer: Any, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool) -> int:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
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


def parse_solution(solution: str, minimum_steps: int) -> ParsedSolution:
    answer = extract_final_answer(solution)
    candidates = [
        _split_numbered_or_markdown(solution),
        _split_paragraphs(solution),
        _split_sentences_conservatively(solution),
    ]
    steps: list[str] = []
    for candidate in candidates:
        cleaned = _clean_steps(candidate)
        if len(cleaned) >= minimum_steps:
            steps = cleaned
            break
    if not steps:
        return ParsedSolution([], None, answer.method, answer.confidence, answer.answer, "failed", "insufficient_steps")
    if answer.answer is None or answer.confidence == "low":
        return ParsedSolution(steps, None, answer.method, answer.confidence, answer.answer, "partial", None)
    return ParsedSolution(steps, answer.answer, answer.method, answer.confidence, answer.answer, "success", None)


def extract_final_answer(solution: str) -> AnswerExtraction:
    boxed = _extract_last_boxed(solution)
    if boxed is not None:
        return AnswerExtraction(boxed, "boxed", "high")
    hash_answers = re.findall(r"####\s*([^\n]+)", solution)
    if hash_answers:
        answer = _strip_answer(hash_answers[-1])
        if answer:
            return AnswerExtraction(answer, "hash_answer", "high")
    label_pattern = re.compile(
        r"(?is)(?:final\s+answer|correct\s+answer|answer)\s*(?:is|=|:)?\s*(?:\\boxed\{)?\s*([A-Za-z]|\(?[A-E]\)?|[-+]?\d+(?:\.\d+)?|[-+]?\d+\s*/\s*[-+]?\d+|\\frac\{[-+]?\d+\}\{[-+]?\d+\})"
    )
    label_answers = label_pattern.findall(solution)
    if label_answers:
        answer = _strip_answer(label_answers[-1])
        if answer:
            return AnswerExtraction(answer, "answer_label", "high")
    tail = solution[-500:]
    choices = re.findall(r"(?<![A-Za-z])\(([A-E])\)|(?:option|choice)\s+([A-E])", tail, flags=re.IGNORECASE)
    if choices:
        letter = choices[-1][0] or choices[-1][1]
        return AnswerExtraction(letter.upper(), "multiple_choice", "medium")
    line_answer = _extract_final_line_answer(solution)
    if line_answer is not None:
        return AnswerExtraction(line_answer, "last_line_answer", "medium")
    numbers = _NUMBER_RE.findall(tail)
    if numbers:
        answer = _strip_answer(numbers[-1])
        if answer:
            return AnswerExtraction(answer, "numeric_fallback", "low")
    return AnswerExtraction(None, ANSWER_METHOD_NONE, "none")


def mutate_step(
    step: str,
    source_id: str,
    step_index: int,
    strategy: str,
    seed: int,
    number_offsets: Sequence[int],
) -> MutationResult:
    if strategy == NUMBER_MUTATION:
        return mutate_number(step, source_id, step_index, seed, number_offsets)
    if strategy == OPERATOR_MUTATION:
        return mutate_operator(step, source_id, step_index, seed)
    if strategy == MIXED_MUTATION:
        first_number = _stable_index(2, seed, source_id, step_index, MIXED_MUTATION, "order") == 0
        strategies = [NUMBER_MUTATION, OPERATOR_MUTATION] if first_number else [OPERATOR_MUTATION, NUMBER_MUTATION]
        failures: list[str] = []
        for candidate in strategies:
            result = mutate_step(step, source_id, step_index, candidate, seed, number_offsets)
            if result.success:
                return result
            failures.append(f"{candidate}:{result.reason}")
        return MutationResult(MIXED_MUTATION, step, None, None, False, ";".join(failures))
    raise ValueError(f"Unsupported mutation strategy: {strategy}")


def mutate_number(step: str, source_id: str, step_index: int, seed: int, number_offsets: Sequence[int]) -> MutationResult:
    offsets = [int(offset) for offset in number_offsets if int(offset) != 0]
    if not offsets:
        raise ValueError("number_offset_choices must contain at least one non-zero offset")
    candidates = [match for match in _NUMBER_RE.finditer(step) if match.start() >= _content_start(step)]
    if not candidates:
        return MutationResult(NUMBER_MUTATION, step, None, None, False, "no_number_target")
    match = candidates[_stable_index(len(candidates), seed, source_id, step_index, NUMBER_MUTATION, "target")]
    offset = offsets[_stable_index(len(offsets), seed, source_id, step_index, NUMBER_MUTATION, "offset")]
    replacement = _offset_number(match.group(0), offset)
    mutated = f"{step[:match.start()]}{replacement}{step[match.end():]}"
    if mutated.strip() == step.strip():
        return MutationResult(NUMBER_MUTATION, step, None, None, False, "unchanged_output")
    return MutationResult(NUMBER_MUTATION, mutated, (match.start(), match.end()), replacement, True, "applied")


def mutate_operator(step: str, source_id: str, step_index: int, seed: int) -> MutationResult:
    candidates = [match for match in _OPERATOR_RE.finditer(step) if _is_binary_operator_target(step, match)]
    if not candidates:
        return MutationResult(OPERATOR_MUTATION, step, None, None, False, "no_operator_target")
    match = candidates[_stable_index(len(candidates), seed, source_id, step_index, OPERATOR_MUTATION, "target")]
    replacement = _operator_replacement(match.group(0))
    mutated = f"{step[:match.start()]}{replacement}{step[match.end():]}"
    if mutated.strip() == step.strip():
        return MutationResult(OPERATOR_MUTATION, step, None, None, False, "unchanged_output")
    return MutationResult(OPERATOR_MUTATION, mutated, (match.start(), match.end()), replacement, True, "applied")


def mutation_metadata(result: MutationResult, configured_strategy: str) -> dict[str, Any]:
    return {
        "configured_strategy": configured_strategy,
        "strategy": result.strategy,
        "changed_span": list(result.changed_span) if result.changed_span is not None else None,
        "replacement": result.replacement,
        "success": result.success,
        "reason": result.reason,
    }


def base_messages(problem: str, config: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": str(config["preprocessing"]["system_prompt"]).strip()},
        {
            "role": "user",
            "content": f"{str(config['preprocessing']['user_instruction']).strip()}\n\nProblem:\n{problem.strip()}",
        },
    ]


def assistant_message(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content.strip()}


def split_ratios(config: Mapping[str, Any]) -> dict[str, float]:
    data = config["data"]
    return {
        "train": float(data["train_ratio"]),
        "validation": float(data["validation_ratio"]),
        "evaluation": float(data["evaluation_ratio"]),
    }


def target_counts(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sft": {"train": int(config["data"]["train_samples"]), "validation": int(config["data"]["validation_samples"])},
        "dpo": {"train": int(config["dpo"]["train_samples"]), "validation": int(config["dpo"]["validation_samples"])},
        "evaluation": int(config["evaluation"]["samples"]),
    }


def assign_split(
    source_id: str,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    seed: int,
    ratios: Mapping[str, float],
) -> str:
    bucket = _bucket("split", dataset_name, dataset_revision, source_split, source_id, str(seed))
    train_cutoff = float(ratios["train"])
    validation_cutoff = train_cutoff + float(ratios["validation"])
    if bucket < train_cutoff:
        return "train"
    if bucket < validation_cutoff:
        return "validation"
    return "evaluation"


def stable_rank(configs: ProjectConfigs, source_id: str, split: str) -> str:
    return hashlib.sha256(
        "|".join(["rank", configs.dataset_name, configs.dataset_revision, configs.source_split, split, source_id, str(configs.seed)]).encode(
            "utf-8"
        )
    ).hexdigest()


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


def build_source_id(batch: Mapping[str, list[Any]], offset: int, row_index: int, id_field: str | None) -> str:
    if id_field is None:
        return f"{row_index:08d}"
    raw_value = str(batch[id_field][offset]).strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_value).strip("_")
    return cleaned[:96] if cleaned else f"{row_index:08d}"


def _validate_config_pair(mini: Mapping[str, Any], formal: Mapping[str, Any], mini_path: Path, formal_path: Path) -> None:
    if mini.get("project", {}).get("run_mode") != "mini":
        raise ValueError(f"{mini_path}: project.run_mode must be 'mini'")
    if formal.get("project", {}).get("run_mode") != "formal":
        raise ValueError(f"{formal_path}: project.run_mode must be 'formal'")
    shared = ("dataset_name", "dataset_revision", "source_split", "processed_dir", "raw_dir")
    for key in shared:
        if mini["data"].get(key) != formal["data"].get(key):
            raise ValueError(f"Mini and formal configs must share data.{key}")
    if mini["project"]["seed"] != formal["project"]["seed"]:
        raise ValueError("Mini and formal configs must share project.seed")
    for path, config in ((mini_path, mini), (formal_path, formal)):
        ratios = split_ratios(config)
        if abs(sum(ratios.values()) - 1.0) > 1e-9:
            raise ValueError(f"{path}: split ratios must sum to 1.0")
        if int(config["dpo"]["max_prompt_length"]) >= int(config["dpo"]["max_length"]):
            raise ValueError(f"{path}: dpo.max_prompt_length must be less than dpo.max_length")
    if int(mini["data"]["train_samples"]) > int(formal["data"]["train_samples"]):
        raise ValueError("Mini SFT train count cannot exceed formal")
    if int(mini["dpo"]["train_samples"]) > int(formal["dpo"]["train_samples"]):
        raise ValueError("Mini DPO train count cannot exceed formal")
    if int(mini["evaluation"]["samples"]) > int(formal["evaluation"]["samples"]):
        raise ValueError("Mini evaluation count cannot exceed formal")


def _config_with_smoke_counts(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    copied["data"]["train_samples"] = int(config["smoke_test"]["train_samples"])
    copied["data"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
    copied["evaluation"]["samples"] = int(config["smoke_test"]["evaluation_samples"])
    copied["dpo"]["train_samples"] = int(config["smoke_test"]["dpo_samples"])
    copied["dpo"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
    return copied


def _smoke_source_rows(config: Mapping[str, Any]) -> int:
    counts = target_counts(config)
    total = counts["sft"]["train"] + counts["sft"]["validation"] + counts["dpo"]["train"] + counts["dpo"]["validation"] + counts["evaluation"]
    return max(2000, int(total) * 50)


def _raw_dataset_path(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["data"]["raw_dir"]))
    return path if path.name == "numina_math" else path / "numina_math"


def _prepare_processed_root(processed_root: Path, overwrite: bool) -> None:
    if processed_root.exists() and any(processed_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty processed directory: {processed_root}")
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)


def _normalization_filter_reason(problem: Any, solution: Any, normalized_problem: str, normalized_solution: str) -> str | None:
    if not isinstance(problem, str):
        return "problem_not_string"
    if not isinstance(solution, str):
        return "solution_not_string"
    if not normalized_problem:
        return "empty_problem"
    if not normalized_solution:
        return "empty_solution"
    if normalized_problem == normalized_solution:
        return "problem_equals_solution"
    return None


def _first_present(fields: Sequence[str], candidates: Sequence[str], semantic_name: str) -> str:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    raise ValueError(f"Could not find a {semantic_name} field in raw dataset fields: {list(fields)}")


def _batch_value(batch: Mapping[str, list[Any]], key: str, offset: int) -> Any:
    if key not in batch:
        return None
    value = batch[key][offset]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_final_line_answer(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        cleaned = _strip_answer(line)
        if re.fullmatch(r"\(?[A-E]\)?|\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?", cleaned):
            return _strip_answer(cleaned)
    return None


def _extract_last_boxed(text: str) -> str | None:
    commands = ("\\boxed{", "\\fbox{")
    results: list[tuple[int, str]] = []
    for command in commands:
        start = 0
        while True:
            index = text.find(command, start)
            if index == -1:
                break
            content = _balanced_brace_content(text, index + len(command) - 1)
            if content is not None:
                results.append((index, content))
            start = index + len(command)
    if not results:
        return None
    answer = _strip_answer(max(results, key=lambda item: item[0])[1])
    return answer or None


def _balanced_brace_content(text: str, open_brace_index: int) -> str | None:
    depth = 0
    content_start = open_brace_index + 1
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return None


def _split_numbered_or_markdown(solution: str) -> list[str]:
    pattern = re.compile(r"(?m)(?=^\s*(?:\d+[\).]|[-*]\s+\*\*|Step\s+\d+[:.)]))")
    parts = [part.strip() for part in pattern.split(solution) if part.strip()]
    return parts if len(parts) >= 2 else []


def _split_paragraphs(solution: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", solution) if part.strip()]


def _split_sentences_conservatively(solution: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", solution.strip())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。])\s+(?=[A-Z$\\(]|Therefore|Thus|So|Hence)", normalized)
    return [part.strip() for part in parts if part.strip()]


def _clean_steps(candidates: Sequence[str]) -> list[str]:
    steps: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned:
            continue
        if len(cleaned) < 3 and steps:
            steps[-1] = f"{steps[-1]} {cleaned}".strip()
            continue
        if steps and cleaned == steps[-1]:
            continue
        steps.append(cleaned)
    return steps


def _strip_answer(answer: str) -> str:
    stripped = answer.strip().rstrip(".。,:;")
    if stripped.startswith("(") and stripped.endswith(")") and len(stripped) == 3:
        return stripped[1].upper()
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()
    return stripped


def _offset_number(text: str, offset: int) -> str:
    if text.startswith("\\frac"):
        match = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", text)
        if match is None:
            return text
        return f"\\frac{{{int(match.group(1)) + offset}}}{{{match.group(2)}}}"
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return f"{int(numerator.strip()) + offset}/{denominator.strip()}"
    try:
        value = Decimal(text)
    except InvalidOperation:
        return text
    mutated = value + Decimal(offset)
    if "." in text:
        decimal_places = len(text.rsplit(".", 1)[1])
        return f"{mutated:.{decimal_places}f}"
    return str(int(mutated))


def _operator_replacement(operator: str) -> str:
    replacements = {
        "+": "-",
        "-": "+",
        "*": "+",
        "/": "*",
        "\\times": "+",
        "\\cdot": "+",
        "\\div": "\\times",
        "=": "\\ne",
        "<": ">",
        ">": "<",
        "\\le": "\\ge",
        "\\ge": "\\le",
    }
    return replacements[operator]


def _is_binary_operator_target(step: str, match: re.Match[str]) -> bool:
    if match.start() < _content_start(step):
        return False
    operator = match.group(0)
    if operator == "-":
        if match.end() < len(step) and step[match.end()].isdigit():
            return False
        before = step[: match.start()].rstrip()
        if not before or before[-1] in "([={+-*/<>":
            return False
    if operator in {"+", "*", "/", "=", "<", ">"}:
        before = step[: match.start()].rstrip()
        after = step[match.end() :].lstrip()
        if not before or not after:
            return False
    return True


def _content_start(step: str) -> int:
    match = re.match(r"\s*(?:step\s+\d+[:.)]|\d+[\).])\s*", step, flags=re.IGNORECASE)
    return match.end() if match else 0


def _stable_index(size: int, seed: int, source_id: str, step_index: int, *parts: str) -> int:
    payload = "|".join([str(seed), source_id, str(step_index), *parts])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % size


def _bucket(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _flatten_ids(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        return [int(value) for value in ids[0]]
    return [int(value) for value in ids]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Stage 1 preprocessing requires PyYAML to read YAML configs") from exc
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _import_datasets() -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Stage 1 preprocessing requires the 'datasets' package") from exc
    return datasets


if __name__ == "__main__":
    main()
