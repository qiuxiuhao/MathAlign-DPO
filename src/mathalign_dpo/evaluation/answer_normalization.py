"""Deterministic answer extraction and normalization for Stage 5 evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from mathalign_dpo.data.parse_steps import extract_final_answer


@dataclass(frozen=True)
class NormalizedAnswer:
    """Extracted and normalized answer for exact-match scoring."""

    raw_answer: str | None
    normalized_answer: str | None
    extracted: bool
    method: str
    confidence: str


def extract_and_normalize_answer(text: str) -> NormalizedAnswer:
    """Extract a final answer from generated text and normalize it."""

    extracted = extract_final_answer(text)
    normalized = normalize_answer(extracted.answer) if extracted.answer is not None else None
    return NormalizedAnswer(
        raw_answer=extracted.answer,
        normalized_answer=normalized,
        extracted=normalized is not None,
        method=extracted.method,
        confidence=extracted.confidence,
    )


def normalize_answer(answer: str | None) -> str | None:
    """Normalize simple math answers without symbolic equivalence."""

    if answer is None:
        return None
    cleaned = _strip_wrappers(answer)
    if not cleaned:
        return None
    if re.fullmatch(r"\(?[A-Ea-e]\)?", cleaned):
        return cleaned.strip("()").upper()
    choice = re.fullmatch(r"(?i)(?:option|choice)\s+([A-E])", cleaned)
    if choice:
        return choice.group(1).upper()
    fraction = _normalize_fraction(cleaned)
    if fraction is not None:
        return fraction
    decimal = _normalize_decimal(cleaned)
    if decimal is not None:
        return decimal
    return _normalize_latex_surface(cleaned)


def exact_match(predicted: str | None, reference: str | None) -> bool:
    """Return exact match after deterministic normalization."""

    pred = normalize_answer(predicted)
    ref = normalize_answer(reference)
    return pred is not None and ref is not None and pred == ref


def _strip_wrappers(answer: str) -> str:
    stripped = answer.strip()
    stripped = stripped.replace("\\(", "").replace("\\)", "")
    stripped = stripped.replace("$", "")
    stripped = stripped.rstrip(".。,:;")
    boxed = _extract_boxed(stripped)
    if boxed is not None:
        stripped = boxed
    if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 3:
        stripped = stripped[1:-1]
    return stripped.strip()


def _extract_boxed(text: str) -> str | None:
    for command in ("\\boxed{", "\\fbox{"):
        index = text.rfind(command)
        if index == -1:
            continue
        start = index + len(command)
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:pos].strip()
    return None


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
    if reduced.denominator == 1:
        return str(reduced.numerator)
    return f"{reduced.numerator}/{reduced.denominator}"


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
    normalized = decimal.normalize()
    if abs(normalized.as_tuple().exponent) > 1000 or math.isnan(float(normalized)):
        return None
    return format(normalized, "f")


def _normalize_latex_surface(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    compact = compact.replace("\\left", "").replace("\\right", "")
    compact = compact.rstrip(".。,:;")
    return compact
