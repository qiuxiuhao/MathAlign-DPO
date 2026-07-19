"""Build step-level DPO preferences from parsed math examples."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping, Sequence

from mathalign_dpo.data.mutate_steps import mutate_step, mutation_metadata
from mathalign_dpo.data.prompts import assistant_message, base_messages


def build_dpo_examples(
    step_examples: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build DPO examples in deterministic source and step order."""

    strategy = str(config["negative_sampling"]["strategy"])
    offsets = [int(offset) for offset in config["negative_sampling"]["number_offset_choices"]]
    seed = int(config["project"]["seed"])
    require_answer = bool(config["preprocessing"].get("require_final_answer_for_dpo", True))
    failures: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []

    for step_example in step_examples:
        if step_example["parse_status"] != "success":
            failures[f"skipped_parse_{step_example['parse_status']}"] += 1
            continue
        if require_answer and not step_example.get("final_answer"):
            failures["skipped_missing_final_answer"] += 1
            continue
        for step_index, chosen_step in enumerate(step_example["steps"]):
            result = mutate_step(
                step=str(chosen_step),
                source_id=str(step_example["source_id"]),
                step_index=step_index,
                strategy=strategy,
                seed=seed,
                number_offsets=offsets,
            )
            if not result.success:
                failures[result.reason] += 1
                continue
            rejected_step = result.text
            if chosen_step.strip() == rejected_step.strip():
                failures["unchanged_output"] += 1
                continue
            if _text_in_prompt_history(rejected_step, step_example, step_index, config):
                failures["rejected_in_prompt_history"] += 1
                continue
            pair = _build_pair(step_example, step_index, chosen_step, rejected_step, config, result, strategy)
            validate_dpo_example(pair)
            pairs.append(pair)
            if len(pairs) >= maximum:
                return pairs, dict(failures)
    return pairs, dict(failures)


def build_manual_review_examples(
    dpo_examples: list[Mapping[str, Any]],
    sample_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select deterministic hash-ranked DPO examples for manual inspection."""

    ranked = sorted(dpo_examples, key=lambda example: _manual_review_rank(str(example["id"]), seed))
    selected = ranked[:sample_count]
    return [
        {
            "schema_version": "1.0",
            "id": f"{example['id']}_review",
            "source_id": example["source_id"],
            "dpo_id": example["id"],
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "metadata": example["metadata"],
        }
        for example in selected
    ]


def validate_dpo_example(example: Mapping[str, Any]) -> None:
    """Validate the Stage 2 DPO schema."""

    required = {"schema_version", "id", "source_id", "prompt", "chosen", "rejected", "metadata", "token_count"}
    missing = sorted(required - set(example))
    if missing:
        raise ValueError(f"DPO example missing fields: {missing}")
    prompt = example["prompt"]
    if not isinstance(prompt, list) or len(prompt) < 2:
        raise ValueError(f"DPO prompt must contain at least system and user: {example['id']}")
    if [message.get("role") for message in prompt[:2]] != ["system", "user"]:
        raise ValueError(f"DPO prompt must begin with system/user: {example['id']}")
    chosen = _single_assistant_text(example["chosen"], "chosen", str(example["id"]))
    rejected = _single_assistant_text(example["rejected"], "rejected", str(example["id"]))
    if chosen.strip() == rejected.strip():
        raise ValueError(f"DPO chosen and rejected are identical: {example['id']}")
    prompt_text = "\n".join(str(message.get("content", "")) for message in prompt)
    if rejected in prompt_text:
        raise ValueError(f"DPO prompt leaks rejected step text: {example['id']}")
    if example["token_count"] is not None:
        raise ValueError(f"Stage 2 must not populate DPO token_count: {example['id']}")


def dpo_strategy_counts(examples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count applied mutation strategies in DPO examples."""

    counts = Counter(str(example["metadata"]["mutation"]["strategy"]) for example in examples)
    return {strategy: int(count) for strategy, count in sorted(counts.items())}


def _build_pair(
    step_example: Mapping[str, Any],
    step_index: int,
    chosen_step: str,
    rejected_step: str,
    config: Mapping[str, Any],
    result: Any,
    configured_strategy: str,
) -> dict[str, Any]:
    prompt = base_messages(str(step_example["problem"]), config)
    prompt.extend(assistant_message(step) for step in step_example["steps"][:step_index])
    return {
        "schema_version": "1.0",
        "id": f"{step_example['source_id']}_step_{step_index:03d}_{configured_strategy}",
        "source_id": step_example["source_id"],
        "step_index": step_index,
        "prompt": prompt,
        "chosen": [assistant_message(chosen_step)],
        "rejected": [assistant_message(rejected_step)],
        "token_count": None,
        "metadata": {
            "negative_strategy": configured_strategy,
            "parse_status": step_example["parse_status"],
            "final_answer": step_example["final_answer"],
            "prompt_history_step_count": step_index,
            "mutation": mutation_metadata(result, configured_strategy),
            "token_length_status": "not_checked_no_tokenizer",
        },
    }


def _text_in_prompt_history(text: str, step_example: Mapping[str, Any], step_index: int, config: Mapping[str, Any]) -> bool:
    prompt = base_messages(str(step_example["problem"]), config)
    prompt.extend(assistant_message(step) for step in step_example["steps"][:step_index])
    prompt_text = "\n".join(str(message.get("content", "")) for message in prompt)
    return text in prompt_text


def _single_assistant_text(messages: Any, field: str, example_id: str) -> str:
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "assistant":
        raise ValueError(f"DPO {field} must be exactly one assistant message: {example_id}")
    content = messages[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"DPO {field} content must be non-empty: {example_id}")
    return content


def _manual_review_rank(example_id: str, seed: int) -> str:
    return hashlib.sha256(f"manual_review|{seed}|{example_id}".encode("utf-8")).hexdigest()
