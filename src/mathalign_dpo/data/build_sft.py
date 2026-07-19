"""Build Stage 2 SFT chat samples from parsed math examples."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from mathalign_dpo.data.prompts import assistant_message, base_messages


TOKEN_LENGTH_STATUS = "not_checked_no_tokenizer"


def build_sft_example(step_example: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Build one SFT sample preserving the original full solution."""

    status = str(step_example["parse_status"])
    if status not in {"success", "partial"}:
        raise ValueError(f"SFT requires success or partial parse_status: {step_example['id']}")
    messages = base_messages(str(step_example["problem"]), config)
    messages.append(assistant_message(str(step_example["solution"])))
    example = {
        "schema_version": "1.0",
        "id": f"{step_example['id']}_sft",
        "source_id": step_example["source_id"],
        "messages": messages,
        "token_count": None,
        "metadata": {
            "parse_status": status,
            "final_answer": step_example["final_answer"],
            "token_length_status": TOKEN_LENGTH_STATUS,
        },
    }
    validate_sft_example(example)
    return example


def build_sft_examples(
    step_examples: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    maximum: int,
) -> list[dict[str, Any]]:
    """Build SFT examples in input order up to a configured cap."""

    usable = [example for example in step_examples if example["parse_status"] in {"success", "partial"}]
    return [build_sft_example(example, config) for example in usable[:maximum]]


def validate_sft_example(example: Mapping[str, Any]) -> None:
    """Validate the Stage 2 SFT schema."""

    required = {"schema_version", "id", "source_id", "messages", "token_count", "metadata"}
    missing = sorted(required - set(example))
    if missing:
        raise ValueError(f"SFT example missing fields: {missing}")
    messages = example["messages"]
    if not isinstance(messages, list) or [msg.get("role") for msg in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"SFT messages must be exactly system/user/assistant: {example['id']}")
    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"SFT message content must be non-empty: {example['id']}")
    if example["token_count"] is not None:
        raise ValueError(f"Stage 2 must not populate token_count without a tokenizer: {example['id']}")


def sft_status_counts(examples: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count SFT source parse statuses."""

    counts = Counter(str(example["metadata"]["parse_status"]) for example in examples)
    return {status: int(counts.get(status, 0)) for status in ("success", "partial")}
