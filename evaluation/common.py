"""Shared generation, scoring, and JSON helpers for evaluation."""

from __future__ import annotations

import json
import math
import re
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import Dataset


def generate_predictions(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    rows: Dataset,
    model_stage: str,
) -> list[dict[str, Any]]:
    """Generate deterministic predictions for one model stage."""

    import torch

    output: list[dict[str, Any]] = []
    generation_config = {
        "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
        "do_sample": bool(config["evaluation"].get("do_sample", False)),
        "num_beams": int(config["evaluation"]["num_beams"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_config["do_sample"]:
        generation_config["temperature"] = float(config["evaluation"]["temperature"])
        generation_config["top_p"] = float(config["evaluation"]["top_p"])
    progress = progress_rows(rows, description=f"Evaluating {model_stage}")
    for row in progress:
        prompt_messages = list(row["prompt_messages"])
        encoded = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(str(config["runtime"]["device"]))
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(encoded, **generation_config)
        seconds = time.perf_counter() - start
        new_tokens = generated[0][encoded.shape[-1] :]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predicted = extract_answer(generated_text)
        reference = normalize_answer(str(row["reference_answer"]))
        normalized_predicted = normalize_answer(predicted)
        output.append(
            {
                "schema_version": "1.0",
                "id": str(row["id"]),
                "source_id": str(row["source_id"]),
                "model_stage": model_stage,
                "prompt_messages": prompt_messages,
                "generated_text": generated_text,
                "predicted_answer": predicted,
                "normalized_predicted_answer": normalized_predicted,
                "reference_answer": str(row["reference_answer"]),
                "normalized_reference_answer": reference,
                "answer_extracted": normalized_predicted is not None,
                "exact_match": normalized_predicted is not None and reference is not None and normalized_predicted == reference,
                "output_tokens": int(new_tokens.shape[-1]),
                "generation_seconds": round(seconds, 6),
            }
        )
        del encoded, generated
    return output


def progress_rows(rows: Dataset, description: str) -> Any:
    """Wrap evaluation rows with tqdm when it is available."""

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return rows
    return tqdm(rows, total=len(rows), desc=description, dynamic_ncols=True)


def summarize_predictions(predictions: Sequence[Mapping[str, Any]], model_stages: Sequence[str]) -> dict[str, Any]:
    """Summarize exact match and generation speed by model stage."""

    summary: dict[str, Any] = {}
    for stage in model_stages:
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


def case_samples(predictions: Sequence[Mapping[str, Any]], limit: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a few correct and incorrect examples for review."""

    correct = [dict(row) for row in predictions if bool(row["exact_match"])][:limit]
    errors = [dict(row) for row in predictions if not bool(row["exact_match"])][:limit]
    return correct, errors


def extract_answer(text: str) -> str | None:
    """Extract a final answer from generated text with deterministic simple rules."""

    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return boxed
    hash_answers = re.findall(r"####\s*([^\n]+)", text)
    if hash_answers:
        answer = _strip_answer(hash_answers[-1])
        if answer:
            return answer
    label_answers = re.findall(
        r"(?is)(?:final\s+answer|correct\s+answer|answer)\s*(?:is|=|:)?\s*(?:\\boxed\{)?\s*([A-Za-z]|\(?[A-E]\)?|[-+]?\d+(?:\.\d+)?|[-+]?\d+\s*/\s*[-+]?\d+|\\frac\{[-+]?\d+\}\{[-+]?\d+\})",
        text,
    )
    if label_answers:
        answer = _strip_answer(label_answers[-1])
        if answer:
            return answer
    tail = text[-500:]
    line_answer = _extract_final_line_answer(text)
    if line_answer is not None:
        return line_answer
    numbers = re.findall(r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?", tail)
    if numbers:
        return _strip_answer(numbers[-1])
    return None


def normalize_answer(answer: str | None) -> str | None:
    """Normalize simple final answers for exact match."""

    if answer is None:
        return None
    cleaned = answer.strip().replace("\\(", "").replace("\\)", "").replace("$", "").rstrip(".。,:;")
    boxed = _extract_last_boxed(cleaned)
    if boxed is not None:
        cleaned = boxed
    if not cleaned:
        return None
    if re.fullmatch(r"\(?[A-Ea-e]\)?", cleaned):
        return cleaned.strip("()").upper()
    fraction = _normalize_fraction(cleaned)
    if fraction is not None:
        return fraction
    decimal = _normalize_decimal(cleaned)
    if decimal is not None:
        return decimal
    return re.sub(r"\s+", "", cleaned).replace("\\left", "").replace("\\right", "")


def release_accelerator_memory() -> None:
    """Release accelerator cache if torch is available."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_is_available = bool(mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)())
        if mps_is_available and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except ImportError:
        return


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")


def json_safe(value: Any) -> Any:
    """Convert non-finite floats to JSON-compliant null values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _extract_final_line_answer(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        cleaned = _strip_answer(line)
        if re.fullmatch(r"\(?[A-E]\)?|\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?", cleaned):
            return cleaned
    return None


def _extract_last_boxed(text: str) -> str | None:
    results: list[tuple[int, str]] = []
    for command in ("\\boxed{", "\\fbox{"):
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


def _strip_answer(answer: str) -> str:
    stripped = answer.strip().rstrip(".。,:;")
    if stripped.startswith("(") and stripped.endswith(")") and len(stripped) == 3:
        return stripped[1].upper()
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()
    return stripped


def _normalize_fraction(value: str) -> str | None:
    compact = re.sub(r"\s+", "", value)
    latex = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", compact)
    slash = re.fullmatch(r"([-+]?\d+)/([-+]?\d+)", compact)
    if latex:
        numerator, denominator = int(latex.group(1)), int(latex.group(2))
    elif slash:
        numerator, denominator = int(slash.group(1)), int(slash.group(2))
    else:
        return None
    if denominator == 0:
        return None
    reduced = Fraction(numerator, denominator)
    return str(reduced.numerator) if reduced.denominator == 1 else f"{reduced.numerator}/{reduced.denominator}"


def _normalize_decimal(value: str) -> str | None:
    compact = value.strip().replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
        return None
    try:
        decimal = Decimal(compact)
    except InvalidOperation:
        return None
    if not decimal.is_finite():
        return None
    if decimal == decimal.to_integral_value():
        return str(int(decimal))
    return format(decimal.normalize(), "f")

