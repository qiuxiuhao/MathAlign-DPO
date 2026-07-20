"""Base/SFT/DPO evaluation for the standalone DPO stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from dpo.modeling import load_dpo_for_generation
from sft.data import load_evaluation_dataset
from sft.evaluate import case_samples, generate_predictions, release_accelerator_memory, write_json, write_jsonl
from sft.modeling import load_base_for_generation, load_sft_for_generation


MODEL_STAGES = ("base", "sft", "dpo")


def evaluate_base_sft_dpo(
    config: Mapping[str, Any],
    sft_adapter_dir: Path,
    dpo_adapter_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate Base, SFT, and DPO with identical prompts and generation settings."""

    evaluation = load_evaluation_dataset(config, limit=sample_count or int(config["evaluation"]["samples"]))
    predictions: list[dict[str, Any]] = []
    for stage in MODEL_STAGES:
        if stage == "base":
            loaded = load_base_for_generation(config, tokenizer_dir=tokenizer_dir)
        elif stage == "sft":
            loaded = load_sft_for_generation(config, adapter_dir=sft_adapter_dir, tokenizer_dir=tokenizer_dir)
        else:
            loaded = load_dpo_for_generation(config, adapter_dir=dpo_adapter_dir, tokenizer_dir=tokenizer_dir)
        try:
            predictions.extend(generate_predictions(config, loaded.model, loaded.tokenizer, evaluation, stage))
        finally:
            del loaded
            release_accelerator_memory()
    summary = summarize_predictions(predictions)
    correct, errors = case_samples(predictions)
    write_jsonl(output_dir / "base_sft_dpo_predictions.jsonl", predictions)
    write_json(output_dir / "base_sft_dpo_summary.json", summary)
    write_jsonl(output_dir / "correct_cases.jsonl", correct)
    write_jsonl(output_dir / "error_cases.jsonl", errors)
    return summary


def summarize_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize exact match and generation speed by model stage."""

    summary: dict[str, Any] = {}
    for stage in MODEL_STAGES:
        rows = [row for row in predictions if row["model_stage"] == stage]
        if not rows:
            raise ValueError(f"No predictions for {stage}")
        extracted = sum(1 for row in rows if bool(row["answer_extracted"]))
        exact = sum(1 for row in rows if bool(row["exact_match"]))
        tokens = [int(row["output_tokens"]) for row in rows]
        seconds = [float(row["generation_seconds"]) for row in rows]
        summary[stage] = {
            "num_examples": len(rows),
            "answer_extraction_rate": extracted / len(rows),
            "exact_match": exact / len(rows),
            "average_output_tokens": sum(tokens) / len(tokens),
            "average_generation_seconds": sum(seconds) / len(seconds),
        }
    return summary
