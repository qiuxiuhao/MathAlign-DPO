"""Build DPO v2 data from real SFT model mistakes.

This script replaces the old step-level mutation path for the new DPO data
lineage. It reads Stage 1 SFT train/validation rows, samples full assistant
answers from the best SFT adapter, and keeps one valid wrong completion per
source_id.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.load_config import load_config
from evaluation.common import (
    _generation_stop_token_ids,
    _real_generated_token_ids,
    _synchronize_cuda,
    compare_answers,
    extract_answer,
    json_safe,
)
from sft.modeling import load_sft_for_generation, load_tokenizer


SCHEMA_VERSION = "2.0"
DEFAULT_FORMAL_CONFIG = "configs/qwen25_3b_4090.yaml"
DEFAULT_MINI_CONFIG = "configs/qwen25_0_5b_m5_24gb_mini.yaml"
DEFAULT_SFT_DIR = "outputs/formal/sft"
DEFAULT_CANDIDATE_DIR = "data/generated/formal/sft_candidates"
DEFAULT_FORMAL_DPO_DIR = "data/processed/formal/dpo_v2"
DEFAULT_MINI_DPO_DIR = "data/processed/mini/dpo_v2"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_dpo_v2(
        formal_config_path=args.formal_config,
        mini_config_path=args.mini_config,
        sft_dir=args.sft_dir,
        candidates_dir=args.candidates_dir,
        formal_output_dir=args.formal_output_dir,
        mini_output_dir=args.mini_output_dir,
        candidates_per_prompt=args.candidates_per_prompt,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        audit_samples=args.audit_samples,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DPO v2 data from best SFT model sampled mistakes.")
    parser.add_argument("--formal-config", default=DEFAULT_FORMAL_CONFIG)
    parser.add_argument("--mini-config", default=DEFAULT_MINI_CONFIG)
    parser.add_argument("--sft-dir", default=DEFAULT_SFT_DIR)
    parser.add_argument("--candidates-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--formal-output-dir", default=DEFAULT_FORMAL_DPO_DIR)
    parser.add_argument("--mini-output-dir", default=DEFAULT_MINI_DPO_DIR)
    parser.add_argument("--candidates-per-prompt", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--audit-samples", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_dpo_v2(
    formal_config_path: str | Path,
    mini_config_path: str | Path,
    sft_dir: str | Path,
    candidates_dir: str | Path,
    formal_output_dir: str | Path,
    mini_output_dir: str | Path,
    candidates_per_prompt: int,
    generation_batch_size: int,
    max_new_tokens: int | None,
    temperature: float,
    top_p: float,
    seed: int | None,
    audit_samples: int,
    train_limit: int | None,
    validation_limit: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    from datasets import load_from_disk

    if candidates_per_prompt <= 0:
        raise ValueError("--candidates-per-prompt must be positive")
    if generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    if temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    formal_config = load_config(formal_config_path)
    mini_config = load_config(mini_config_path)
    seed = int(seed if seed is not None else formal_config["project"]["seed"])
    max_new_tokens = int(max_new_tokens)

    sft_root = Path(sft_dir)
    adapter_dir = sft_root / "best_adapter"
    tokenizer_dir = sft_root / "tokenizer"
    _require_adapter(adapter_dir, label="SFT best adapter")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"SFT tokenizer directory is missing: {tokenizer_dir}")

    candidates_path = Path(candidates_dir)
    formal_output_path = Path(formal_output_dir)
    mini_output_path = Path(mini_output_dir)
    for path in (candidates_path, formal_output_path, mini_output_path):
        _prepare_output_dir(path, overwrite=overwrite)

    sft_path = Path(str(formal_config["data"]["formal_dir"])) / "sft"
    mini_sft_path = Path(str(mini_config["data"]["mini_dir"])) / "sft"
    sft_data = load_from_disk(str(sft_path))
    mini_sft_data = load_from_disk(str(mini_sft_path))
    formal_splits = {
        "train": _limit_rows(sft_data["train"], train_limit),
        "validation": _limit_rows(sft_data["validation"], validation_limit),
    }
    mini_source_ids = {
        split: {str(row["source_id"]) for row in mini_sft_data[split]}
        for split in ("train", "validation")
    }

    loaded = load_sft_for_generation(formal_config, adapter_dir=adapter_dir, tokenizer_dir=tokenizer_dir)
    try:
        candidate_splits: dict[str, Any] = {}
        formal_pair_splits: dict[str, Any] = {}
        split_reports: dict[str, Any] = {}
        started_at = time.perf_counter()
        for split, rows in formal_splits.items():
            candidates = generate_candidates_for_split(
                config=formal_config,
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                rows=rows,
                split=split,
                candidates_per_prompt=candidates_per_prompt,
                generation_batch_size=generation_batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            pairs, report = select_pairs(
                config=formal_config,
                tokenizer=loaded.tokenizer,
                source_rows=rows,
                candidates=candidates,
                split=split,
            )
            candidate_splits[split] = candidates
            formal_pair_splits[split] = pairs
            split_reports[split] = report
    finally:
        del loaded

    from datasets import Dataset, DatasetDict

    candidate_dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in candidate_splits.items()})
    formal_dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in formal_pair_splits.items()})
    mini_tokenizer = load_tokenizer(mini_config)
    mini_pair_splits, mini_reports = build_mini_pairs(
        mini_config=mini_config,
        mini_tokenizer=mini_tokenizer,
        formal_pair_splits=formal_pair_splits,
        mini_source_ids=mini_source_ids,
    )
    mini_dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in mini_pair_splits.items()})

    candidate_dataset.save_to_disk(str(candidates_path))
    formal_dataset.save_to_disk(str(formal_output_path))
    mini_dataset.save_to_disk(str(mini_output_path))

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "formal_config": str(formal_config_path),
        "mini_config": str(mini_config_path),
        "sft_adapter_dir": str(adapter_dir),
        "sft_tokenizer_dir": str(tokenizer_dir),
        "candidate_generation": {
            "candidates_per_prompt": candidates_per_prompt,
            "generation_batch_size": generation_batch_size,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": 1,
            "seed": seed,
        },
        "outputs": {
            "candidates": str(candidates_path),
            "formal_dpo_v2": str(formal_output_path),
            "mini_dpo_v2": str(mini_output_path),
        },
        "splits": split_reports,
        "mini": mini_reports,
        "totals": _total_report(split_reports, mini_reports),
        "elapsed_seconds": round(time.perf_counter() - started_at, 6),
    }
    _write_json(candidates_path / "generation_report.json", report)
    _write_json(formal_output_path / "generation_report.json", report)
    _write_json(mini_output_path / "generation_report.json", report)
    _write_jsonl(candidates_path / "candidates.jsonl", [row for rows in candidate_splits.values() for row in rows])
    _write_jsonl(formal_output_path / "audit_pairs_100.jsonl", _audit_sample(formal_pair_splits, audit_samples, seed))
    _write_jsonl(mini_output_path / "audit_pairs_100.jsonl", _audit_sample(mini_pair_splits, audit_samples, seed))
    return report


def generate_candidates_for_split(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    candidates_per_prompt: int,
    generation_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict[str, Any]]:
    import torch

    stop_token_ids = _generation_stop_token_ids(tokenizer)
    candidates: list[dict[str, Any]] = []
    device = str(config["runtime"]["device"])
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        starts = range(0, len(rows), generation_batch_size)
        progress = _progress(starts, description=f"Generating {split} SFT candidates", total=len(starts))
        for start in progress:
            batch = list(rows[start : start + generation_batch_size])
            prompts = [
                tokenizer.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            prompt_width = int(encoded["input_ids"].shape[-1])
            _synchronize_cuda(torch, device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=1,
                    num_return_sequences=candidates_per_prompt,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=sorted(stop_token_ids),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            _synchronize_cuda(torch, device)
            seconds_per_candidate = (time.perf_counter() - t0) / (len(batch) * candidates_per_prompt)
            transition_scores = model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                normalize_logits=True,
            )
            for sequence_index, generated_ids in enumerate(generated.sequences):
                row = batch[sequence_index // candidates_per_prompt]
                candidate_index = sequence_index % candidates_per_prompt
                raw_new_tokens = generated_ids[prompt_width:].detach().cpu().tolist()
                new_token_ids, finish_reason = _real_generated_token_ids(
                    raw_new_tokens,
                    stop_token_ids=stop_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                )
                generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
                predicted, extraction_method = extract_answer(generated_text, finish_reason=finish_reason)
                reference_answer = str(row["metadata"].get("final_answer") or "")
                strict_reference = str(row["metadata"].get("final_answer") or "")
                strict_exact = predicted is not None and predicted.strip() == strict_reference.strip()
                math_equivalent, match_method = compare_answers(
                    predicted,
                    reference_answer,
                    prompt_messages=list(row["prompt"]),
                    strict_exact_match=strict_exact,
                )
                scores = transition_scores[sequence_index].detach().float().cpu().tolist()
                kept_scores = scores[: len(new_token_ids)]
                average_logprob = sum(kept_scores) / len(kept_scores) if kept_scores else None
                candidates.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": f"{row['source_id']}_{split}_cand_{candidate_index:02d}",
                        "source_id": str(row["source_id"]),
                        "sft_row_id": str(row["id"]),
                        "split": split,
                        "candidate_index": candidate_index,
                        "prompt": list(row["prompt"]),
                        "generated_text": generated_text,
                        "predicted_answer": predicted,
                        "reference_answer": reference_answer,
                        "answer_extracted": predicted is not None,
                        "extraction_method": extraction_method,
                        "math_equivalent": math_equivalent,
                        "match_method": match_method,
                        "finish_reason": finish_reason,
                        "hit_max_new_tokens": finish_reason == "length" and len(new_token_ids) >= max_new_tokens,
                        "output_tokens": len(new_token_ids),
                        "average_logprob": average_logprob,
                        "generation_seconds": round(seconds_per_candidate, 6),
                        "metadata": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                        },
                    }
                )
            del encoded, generated, transition_scores
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    return candidates


def select_pairs(
    config: Mapping[str, Any],
    tokenizer: Any,
    source_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_source[str(candidate["source_id"])].append(candidate)

    candidate_stats = Counter()
    rejection_reasons = Counter()
    pairs: list[dict[str, Any]] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    seen_sources: set[str] = set()
    progress = _progress(source_rows, description=f"Selecting {split} DPO pairs", total=len(source_rows))
    for row in progress:
        source_id = str(row["source_id"])
        if source_id in seen_sources:
            rejection_reasons["duplicate_source_row"] += 1
            continue
        seen_sources.add(source_id)
        valid: list[tuple[float, Mapping[str, Any], dict[str, int]]] = []
        seen_texts: set[str] = set()
        for candidate in by_source.get(source_id, []):
            _count_candidate(candidate, candidate_stats)
            reason = rejected_filter_reason(candidate, seen_texts)
            if reason is not None:
                rejection_reasons[reason] += 1
                continue
            pair = {
                "prompt": list(row["prompt"]),
                "chosen": list(row["completion"]),
                "rejected": [{"role": "assistant", "content": str(candidate["generated_text"]).strip()}],
            }
            if _normalized_text(pair["chosen"][0]["content"]) == _normalized_text(pair["rejected"][0]["content"]):
                rejection_reasons["same_as_chosen"] += 1
                continue
            token_count = count_dpo_tokens(tokenizer, pair)
            length_reason = dpo_length_filter_reason(token_count, config)
            if length_reason is not None:
                rejection_reasons[length_reason] += 1
                continue
            score = candidate_quality_score(candidate)
            valid.append((score, candidate, token_count))
        if not valid:
            continue
        valid.sort(key=lambda item: item[0], reverse=True)
        _, selected, token_count = valid[0]
        pair_row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"{source_id}_dpo_v2",
            "source_id": source_id,
            "step_index": 0,
            "prompt": list(row["prompt"]),
            "chosen": list(row["completion"]),
            "rejected": [{"role": "assistant", "content": str(selected["generated_text"]).strip()}],
            "token_count": token_count,
            "split": split,
            "metadata": {
                "negative_source": "sft_best_adapter_sample",
                "candidate_id": str(selected["id"]),
                "candidate_index": int(selected["candidate_index"]),
                "reference_answer": str(selected["reference_answer"]),
                "predicted_answer": selected["predicted_answer"],
                "extraction_method": str(selected["extraction_method"]),
                "finish_reason": str(selected["finish_reason"]),
                "average_logprob": selected["average_logprob"],
                "match_method": str(selected["match_method"]),
                "original_sft_row_id": str(row["id"]),
                "raw_source_id": row.get("metadata", {}).get("raw_source_id"),
            },
        }
        pairs.append(pair_row)
        chosen_lengths.append(int(token_count["chosen_completion"]))
        rejected_lengths.append(int(token_count["rejected_completion"]))

    report = {
        "source_questions": len(source_rows),
        "unique_source_id": len(seen_sources),
        "candidate_count": len(candidates),
        "final_preference_pairs": len(pairs),
        "no_valid_rejected": len(seen_sources) - len(pairs),
        "candidate_stats": _ratio_report(candidate_stats, len(candidates)),
        "filter_counts": dict(sorted(rejection_reasons.items())),
        "chosen_average_tokens": _average(chosen_lengths),
        "rejected_average_tokens": _average(rejected_lengths),
    }
    return pairs, report


def build_mini_pairs(
    mini_config: Mapping[str, Any],
    mini_tokenizer: Any,
    formal_pair_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    mini_source_ids: Mapping[str, set[str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    mini: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    for split, rows in formal_pair_splits.items():
        selected: list[dict[str, Any]] = []
        counters = Counter()
        progress = _progress(rows, description=f"Filtering mini {split} DPO pairs", total=len(rows))
        for row in progress:
            if str(row["source_id"]) not in mini_source_ids[split]:
                continue
            copied = deepcopy(dict(row))
            token_count = count_dpo_tokens(mini_tokenizer, copied)
            reason = dpo_length_filter_reason(token_count, mini_config)
            if reason is not None:
                counters[reason] += 1
                continue
            copied["token_count"] = token_count
            selected.append(copied)
        mini[split] = selected
        reports[split] = {
            "source_questions": len(mini_source_ids[split]),
            "final_preference_pairs": len(selected),
            "filter_counts": dict(sorted(counters.items())),
        }
    return mini, reports


def rejected_filter_reason(candidate: Mapping[str, Any], seen_texts: set[str]) -> str | None:
    text = str(candidate.get("generated_text") or "").strip()
    if not text:
        return "empty_text"
    normalized = _normalized_text(text)
    if normalized in seen_texts:
        return "duplicate_text"
    seen_texts.add(normalized)
    if candidate.get("finish_reason") != "eos":
        return "not_eos"
    if bool(candidate.get("hit_max_new_tokens")):
        return "hit_max_new_tokens"
    if not bool(candidate.get("answer_extracted")):
        return "no_answer"
    if str(candidate.get("extraction_method")) in {"tail_number", "not_found", "truncated_no_final_answer"}:
        return "no_explicit_final_answer"
    if bool(candidate.get("math_equivalent")):
        return "correct_answer"
    if _is_obvious_abnormal(text):
        return "abnormal_output"
    return None


def candidate_quality_score(candidate: Mapping[str, Any]) -> float:
    average_logprob = candidate.get("average_logprob")
    confidence = float(average_logprob) if average_logprob is not None else -100.0
    length_bonus = min(int(candidate.get("output_tokens") or 0), 256) / 10000.0
    method_bonus = {
        "boxed": 0.03,
        "labeled_answer": 0.02,
        "hash_answer": 0.02,
        "final_line_choice": 0.01,
        "final_line_numeric": 0.005,
    }.get(str(candidate.get("extraction_method")), 0.0)
    return confidence + length_bonus + method_bonus


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


def _count_candidate(candidate: Mapping[str, Any], stats: Counter[str]) -> None:
    stats["eos" if candidate.get("finish_reason") == "eos" else "length"] += 1
    if not bool(candidate.get("answer_extracted")):
        stats["no_answer"] += 1
    elif bool(candidate.get("math_equivalent")):
        stats["correct"] += 1
    else:
        stats["wrong"] += 1


def _is_obvious_abnormal(text: str) -> bool:
    if len(text.strip()) < 5:
        return True
    if "<|" in text or "\ufffd" in text:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 6:
        most_common = Counter(lines).most_common(1)[0][1]
        if most_common / len(lines) >= 0.5:
            return True
    words = re_split_words(text)
    if len(words) >= 60:
        trigrams = list(zip(words, words[1:], words[2:]))
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < 0.35:
                return True
    return False


def re_split_words(text: str) -> list[str]:
    import re

    return re.findall(r"\w+|[^\w\s]", text.lower())


def _normalized_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _flatten_ids(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        flattened: list[int] = []
        for item in ids:
            flattened.extend(int(value) for value in item)
        return flattened
    return [int(value) for value in ids]


def _limit_rows(dataset: Any, limit: int | None) -> list[Mapping[str, Any]]:
    count = len(dataset) if limit is None else min(len(dataset), int(limit))
    return [dataset[index] for index in range(count)]


def _require_adapter(adapter_dir: Path, label: str) -> None:
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        path = adapter_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{label} artifact is missing: {path}")


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")


def _audit_sample(splits: Mapping[str, Sequence[Mapping[str, Any]]], limit: int, seed: int) -> list[dict[str, Any]]:
    rows = [dict(row) for split_rows in splits.values() for row in split_rows]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(limit, len(rows))]


def _ratio_report(counts: Counter[str], total: int) -> dict[str, Any]:
    return {
        key: {
            "count": int(counts.get(key, 0)),
            "ratio": (int(counts.get(key, 0)) / total if total else 0.0),
        }
        for key in ("eos", "length", "correct", "wrong", "no_answer")
    }


def _total_report(split_reports: Mapping[str, Mapping[str, Any]], mini_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source_questions = sum(int(report["source_questions"]) for report in split_reports.values())
    preference_pairs = sum(int(report["final_preference_pairs"]) for report in split_reports.values())
    candidates = sum(int(report["candidate_count"]) for report in split_reports.values())
    no_valid = sum(int(report["no_valid_rejected"]) for report in split_reports.values())
    mini_pairs = sum(int(report["final_preference_pairs"]) for report in mini_reports.values())
    return {
        "formal_source_questions": source_questions,
        "formal_candidate_count": candidates,
        "formal_preference_pairs": preference_pairs,
        "formal_no_valid_rejected": no_valid,
        "mini_preference_pairs": mini_pairs,
    }


def _average(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _progress(rows: Any, description: str, total: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return rows
    return tqdm(rows, total=total, desc=description, dynamic_ncols=True)


if __name__ == "__main__":
    main()
