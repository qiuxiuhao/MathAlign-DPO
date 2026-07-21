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
    batch_size = int(config["evaluation"].get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("evaluation.batch_size must be positive")
    if bool(config["evaluation"].get("do_sample", False)):
        raise ValueError("Stage 4 unified evaluation requires evaluation.do_sample=false")
    if int(config["evaluation"]["num_beams"]) != 1:
        raise ValueError("Stage 4 unified evaluation requires evaluation.num_beams=1")
    max_new_tokens = int(config["evaluation"]["max_new_tokens"])
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": sorted(stop_token_ids),
    }
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    try:
        progress = progress_rows(range(0, len(rows), batch_size), description=f"Evaluating {model_stage}")
        for start_index in progress:
            batch = [rows[index] for index in range(start_index, min(start_index + batch_size, len(rows)))]
            prompt_messages_batch = [list(row["prompt_messages"]) for row in batch]
            prompts = [
                tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt_messages in prompt_messages_batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(str(config["runtime"]["device"]))
            _synchronize_cuda(torch, str(config["runtime"]["device"]))
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_config)
            _synchronize_cuda(torch, str(config["runtime"]["device"]))
            seconds_per_example = (time.perf_counter() - start) / len(batch)
            prompt_width = int(encoded["input_ids"].shape[-1])
            for row, prompt_messages, generated_ids in zip(batch, prompt_messages_batch, generated):
                raw_new_tokens = generated_ids[prompt_width:].detach().cpu().tolist()
                new_token_ids, finish_reason = _real_generated_token_ids(
                    raw_new_tokens,
                    stop_token_ids=stop_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                )
                hit_max_new_tokens = finish_reason == "length" and len(new_token_ids) >= max_new_tokens
                generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
                predicted, extraction_method = extract_answer(generated_text, finish_reason=finish_reason)
                reference_answer = str(row["reference_answer"])
                reference_final_answer, reference_source, reference_extraction_method = gold_reference_answer(row)
                reference = normalize_answer(reference_final_answer)
                normalized_predicted = normalize_answer(predicted)
                strict_exact_match = (
                    normalized_predicted is not None and reference is not None and normalized_predicted == reference
                )
                math_equivalent, match_method = compare_answers(
                    predicted,
                    reference_final_answer,
                    prompt_messages=prompt_messages,
                    strict_exact_match=strict_exact_match,
                )
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
                        "reference_answer": reference_answer,
                        "reference_final_answer": reference_final_answer,
                        "reference_answer_source": reference_source,
                        "normalized_reference_answer": reference,
                        "reference_extraction_method": reference_extraction_method,
                        "answer_extracted": normalized_predicted is not None,
                        "reference_answer_extracted": reference_final_answer is not None,
                        "extraction_method": extraction_method,
                        "strict_exact_match": strict_exact_match,
                        "math_equivalent": math_equivalent,
                        "match_method": match_method,
                        "exact_match": strict_exact_match,
                        "finish_reason": finish_reason,
                        "hit_max_new_tokens": hit_max_new_tokens,
                        "output_tokens": len(new_token_ids),
                        "generation_seconds": round(seconds_per_example, 6),
                    }
                )
            del encoded, generated
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
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
        reference_extracted = sum(1 for row in rows if bool(row.get("reference_answer_extracted", True)))
        strict_exact = sum(1 for row in rows if bool(row["strict_exact_match"]))
        math_matches = sum(1 for row in rows if bool(row["math_equivalent"]))
        eos_finishes = sum(1 for row in rows if row["finish_reason"] == "eos")
        length_finishes = sum(1 for row in rows if row["finish_reason"] == "length")
        max_token_hits = sum(1 for row in rows if bool(row["hit_max_new_tokens"]))
        tokens = [int(row["output_tokens"]) for row in rows]
        seconds = [float(row["generation_seconds"]) for row in rows]
        summary[stage] = {
            "num_examples": len(rows),
            "answer_extraction_rate": extracted / len(rows),
            "reference_answer_extraction_rate": reference_extracted / len(rows),
            "strict_exact_match": strict_exact / len(rows),
            "math_equivalent": math_matches / len(rows),
            "exact_match": strict_exact / len(rows),
            "eos_finish_rate": eos_finishes / len(rows),
            "length_truncation_rate": length_finishes / len(rows),
            "hit_max_new_tokens_rate": max_token_hits / len(rows),
            "average_real_output_tokens": sum(tokens) / len(tokens),
            "average_output_tokens": sum(tokens) / len(tokens),
            "average_generation_seconds": sum(seconds) / len(seconds),
        }
    return summary


def case_samples(predictions: Sequence[Mapping[str, Any]], limit: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a few correct and incorrect examples for review."""

    correct = [dict(row) for row in predictions if bool(row["math_equivalent"])][:limit]
    errors = [dict(row) for row in predictions if not bool(row["math_equivalent"])][:limit]
    return correct, errors


def extract_answer(text: str, finish_reason: str = "eos") -> tuple[str | None, str]:
    """Extract a final answer from generated text with deterministic simple rules."""

    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return boxed, "boxed"
    hash_answers = re.findall(r"####\s*([^\n]+)", text)
    if hash_answers:
        answer = _strip_answer(hash_answers[-1])
        if answer:
            return answer, "hash_answer"
    answer_line = _extract_labeled_answer_line(text)
    if answer_line is not None:
        return answer_line, "labeled_answer_line"
    label_answers = re.findall(
        r"(?is)(?:final\s+answer|correct\s+answer|answer)\s*(?:(?:is)?\s*:|is|=)?\s*(?:\\boxed\{)?\s*(\(?[A-E]\)?|[-+]?\d+(?:\.\d+)?(?:\\?%)?|[-+]?\d+\s*/\s*[-+]?\d+|\\d?frac\{[-+]?\d+\}\{[-+]?\d+\})",
        text,
    )
    if label_answers:
        answer = _strip_answer(label_answers[-1])
        if answer:
            return answer, "labeled_answer"
    line_answer, line_method = _extract_final_line_answer(text)
    if line_answer is not None:
        if finish_reason == "length" and line_method != "final_line_choice":
            return None, "truncated_no_final_answer"
        return line_answer, line_method
    if finish_reason == "length":
        return None, "truncated_no_final_answer"
    tail = text[-500:]
    numbers = re.findall(r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?", tail)
    if numbers:
        return _strip_answer(numbers[-1]), "tail_number"
    return None, "not_found"


def extract_reference_answer(text: str) -> tuple[str | None, str]:
    """Extract the final answer from a gold full solution trajectory."""

    answer, method = extract_answer(text, finish_reason="eos")
    if answer is not None:
        return answer, method
    return None, method


def gold_reference_answer(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the gold final answer, preferring Stage 1 metadata.answer."""

    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        answer = metadata.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip(), "metadata.answer", "metadata_answer"
    reference_answer = str(row["reference_answer"])
    extracted, method = extract_reference_answer(reference_answer)
    if extracted is not None:
        return extracted, "reference_answer", method
    return reference_answer, "reference_answer", "full_reference_fallback"


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
    assignment = re.fullmatch(r"[A-Za-z]\s*=\s*(.+)", cleaned)
    if assignment is not None:
        cleaned = assignment.group(1).strip()
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


def compare_answers(
    predicted: str | None,
    reference: str,
    prompt_messages: Sequence[Mapping[str, Any]] | Sequence[Any],
    strict_exact_match: bool,
) -> tuple[bool, str]:
    """Return strict-or-math answer match status and the method that matched."""

    if predicted is None:
        return False, "no_prediction"
    if strict_exact_match:
        return True, "strict_exact"

    predicted_numeric = _answer_decimal(predicted)
    reference_numeric = _answer_decimal(reference)
    if predicted_numeric is not None and reference_numeric is not None and predicted_numeric == reference_numeric:
        return True, "numeric_equivalent"
    if _percent_context_equivalent(predicted, reference, prompt_messages):
        return True, "percent_context_equivalent"
    embedded_predicted = _single_embedded_numeric(predicted)
    if embedded_predicted is not None and reference_numeric is not None and embedded_predicted == reference_numeric:
        return True, "embedded_numeric_equivalent"

    predicted_unitless = _strip_trailing_unit(predicted)
    reference_unitless = _strip_trailing_unit(reference)
    if predicted_unitless and reference_unitless:
        normalized_predicted = normalize_answer(predicted_unitless)
        normalized_reference = normalize_answer(reference_unitless)
        if normalized_predicted is not None and normalized_predicted == normalized_reference:
            return True, "unit_stripped_exact"

    choice_match = _choice_content_match(predicted, reference, prompt_messages)
    if choice_match is not None:
        return choice_match
    return False, "none"


def _generation_stop_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        ids.add(int(eos_token_id))
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if im_end_id is not None and im_end_id != unk_id and int(im_end_id) >= 0:
            ids.add(int(im_end_id))
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError("Tokenizer must provide an EOS or <|im_end|> stop token")
    return ids


def _real_generated_token_ids(
    token_ids: Sequence[int],
    stop_token_ids: set[int],
    pad_token_id: int | None,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    limited = [int(token_id) for token_id in token_ids[:max_new_tokens]]
    for index, token_id in enumerate(limited):
        if token_id in stop_token_ids:
            return limited[:index], "eos"
    if pad_token_id is not None and int(pad_token_id) not in stop_token_ids:
        while limited and limited[-1] == int(pad_token_id):
            limited.pop()
    return limited, "length"


def _synchronize_cuda(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _extract_final_line_answer(text: str) -> tuple[str | None, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        cleaned = _strip_answer(line)
        if re.fullmatch(r"\(?[A-E]\)?", cleaned):
            return cleaned, "final_line_choice"
        if re.fullmatch(r"\\d?frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?(?:\\?%)?", cleaned):
            return cleaned, "final_line_numeric"
    return None, "not_found"


def _extract_labeled_answer_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-5:]):
        match = re.search(r"(?i)(?:the\s+)?(?:final\s+answer|correct\s+answer|answer)\s*(?:(?:is)?\s*:|is|=)\s*(.+)$", line)
        if not match:
            continue
        answer = _strip_answer(match.group(1))
        boxed = _extract_last_boxed(answer)
        if boxed is not None:
            answer = boxed
        answer = re.sub(r"^\\boxed\{(.+)\}$", r"\1", answer).strip()
        if answer:
            return answer
    return None


def _answer_decimal(value: str | None) -> Fraction | None:
    if value is None:
        return None
    compact = _strip_trailing_unit(value)
    compact = _clean_math_text(compact)
    assignment = re.fullmatch(r"[A-Za-z]\s*=\s*(.+)", compact)
    if assignment is not None:
        compact = assignment.group(1).strip()
    if not compact:
        return None
    percent = compact.endswith("%")
    if percent:
        compact = compact[:-1].strip()
    latex = re.fullmatch(r"\\d?frac\{([-+]?\d+)\}\{([-+]?\d+)\}", compact)
    slash = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", compact)
    try:
        if latex:
            denominator = int(latex.group(2))
            if denominator == 0:
                return None
            number = Fraction(int(latex.group(1)), denominator)
        elif slash:
            denominator = int(slash.group(2))
            if denominator == 0:
                return None
            number = Fraction(int(slash.group(1)), denominator)
        elif re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", compact):
            number = Fraction(Decimal(compact))
        else:
            return None
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return number / 100 if percent else number


def _strip_trailing_unit(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = _clean_math_text(value)
    numeric = r"(?:\\d?frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?)"
    match = re.fullmatch(rf"\s*({numeric})\s*(?:[A-Za-z°²³^/\\{{}}\-\s]+)?\s*", cleaned)
    if match:
        return re.sub(r"\s+%", "%", match.group(1).strip())
    return cleaned


def _clean_math_text(value: str) -> str:
    cleaned = value.strip()
    boxed = _extract_last_boxed(cleaned)
    if boxed is not None:
        cleaned = boxed
    cleaned = re.sub(r"\\text\{[^{}]*\}", "", cleaned)
    cleaned = cleaned.replace("\\%", "%").replace("\\$", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "").replace("$", "")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace(",", "").strip().rstrip(".。,:;")
    return cleaned


def _single_embedded_numeric(value: str | None) -> Fraction | None:
    if value is None:
        return None
    cleaned = _clean_math_text(value)
    numeric = r"\\d?frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?"
    matches = re.findall(numeric, cleaned)
    if len(matches) != 1:
        return None
    return _answer_decimal(matches[0])


def _percent_context_equivalent(
    predicted: str | None,
    reference: str,
    prompt_messages: Sequence[Mapping[str, Any]] | Sequence[Any],
) -> bool:
    prompt_text = "\n".join(_message_content(message) for message in prompt_messages).lower()
    if "percent" not in prompt_text and "percentage" not in prompt_text and "%" not in prompt_text:
        return False
    predicted_percent = _percent_display_number(predicted)
    reference_percent = _percent_display_number(reference)
    predicted_numeric = _answer_decimal(predicted)
    reference_numeric = _answer_decimal(reference)
    if predicted_percent is not None and reference_numeric is not None and predicted_percent == reference_numeric:
        return True
    if reference_percent is not None and predicted_numeric is not None and reference_percent == predicted_numeric:
        return True
    return False


def _percent_display_number(value: str | None) -> Fraction | None:
    if value is None:
        return None
    compact = _clean_math_text(value)
    if not compact.endswith("%"):
        return None
    compact = compact[:-1].strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", compact):
        return None
    try:
        return Fraction(Decimal(compact))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _choice_content_match(
    predicted: str,
    reference: str,
    prompt_messages: Sequence[Mapping[str, Any]] | Sequence[Any],
) -> tuple[bool, str] | None:
    options = _choice_options(prompt_messages)
    if not options:
        return None
    predicted_letter = _choice_letter(predicted)
    reference_letter = _choice_letter(reference)
    if predicted_letter is not None and predicted_letter in options:
        if _answers_equivalent_basic(options[predicted_letter], reference):
            return True, "choice_letter_to_content"
    if reference_letter is not None and reference_letter in options:
        if _answers_equivalent_basic(predicted, options[reference_letter]):
            return True, "choice_content_to_letter"
    return None


def _choice_options(prompt_messages: Sequence[Mapping[str, Any]] | Sequence[Any]) -> dict[str, str]:
    text = "\n".join(_message_content(message) for message in prompt_messages)
    options: dict[str, str] = {}
    multiline = re.finditer(
        r"(?ims)(?:^|\n)\s*\(?([A-E])\)?[\.\):]\s*(.+?)(?=(?:\n\s*\(?[A-E]\)?[\.\):])|\Z)",
        text,
    )
    for match in multiline:
        content = _clean_choice_content(match.group(2))
        if content:
            options[match.group(1).upper()] = content
    if options:
        return options
    inline = re.finditer(r"(?i)\b([A-E])[\.\)]\s*([^A-E\n]+?)(?=\s+[A-E][\.\)]|\Z)", text)
    for match in inline:
        content = _clean_choice_content(match.group(2))
        if content:
            options[match.group(1).upper()] = content
    return options


def _message_content(message: Mapping[str, Any] | Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("content", ""))
    return str(message)


def _clean_choice_content(value: str) -> str:
    return value.strip().rstrip(".。,:;")


def _choice_letter(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"\(?([A-Ea-e])\)?", value.strip())
    return match.group(1).upper() if match else None


def _answers_equivalent_basic(left: str, right: str) -> bool:
    normalized_left = normalize_answer(left)
    normalized_right = normalize_answer(right)
    if normalized_left is not None and normalized_left == normalized_right:
        return True
    left_number = _answer_decimal(left)
    right_number = _answer_decimal(right)
    if left_number is not None and right_number is not None and left_number == right_number:
        return True
    unitless_left = normalize_answer(_strip_trailing_unit(left))
    unitless_right = normalize_answer(_strip_trailing_unit(right))
    return unitless_left is not None and unitless_left == unitless_right


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
    latex = re.fullmatch(r"\\d?frac\{([-+]?\d+)\}\{([-+]?\d+)\}", compact)
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
