"""Deterministic rule-based negative step mutations for Stage 2."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


NUMBER_MUTATION = "number_mutation"
OPERATOR_MUTATION = "operator_mutation"
MIXED_MUTATION = "mixed"
SUPPORTED_STRATEGIES = {NUMBER_MUTATION, OPERATOR_MUTATION, MIXED_MUTATION}

_NUMBER_RE = re.compile(r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?")
_OPERATOR_RE = re.compile(r"\\times|\\cdot|\\div|\\le|\\ge|[+\-*/=<>]")


@dataclass(frozen=True)
class MutationResult:
    """Mutation output and metadata."""

    strategy: str
    text: str
    changed_span: tuple[int, int] | None
    replacement: str | None
    success: bool
    reason: str


def mutate_step(
    step: str,
    source_id: str,
    step_index: int,
    strategy: str,
    seed: int,
    number_offsets: Sequence[int],
) -> MutationResult:
    """Mutate one reasoning step deterministically."""

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported mutation strategy: {strategy}")
    if len(step.strip()) < 3:
        return MutationResult(strategy, step, None, None, False, "step_too_short")
    if strategy == NUMBER_MUTATION:
        return mutate_number(step, source_id, step_index, seed, number_offsets)
    if strategy == OPERATOR_MUTATION:
        return mutate_operator(step, source_id, step_index, seed)
    return _mutate_mixed(step, source_id, step_index, seed, number_offsets)


def mutate_number(
    step: str,
    source_id: str,
    step_index: int,
    seed: int,
    number_offsets: Sequence[int],
) -> MutationResult:
    """Apply a deterministic non-zero numeric offset."""

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
    """Apply a deterministic binary arithmetic or comparison operator change."""

    candidates = [match for match in _OPERATOR_RE.finditer(step) if _is_binary_operator_target(step, match)]
    if not candidates:
        return MutationResult(OPERATOR_MUTATION, step, None, None, False, "no_operator_target")
    match = candidates[_stable_index(len(candidates), seed, source_id, step_index, OPERATOR_MUTATION, "target")]
    replacement = _operator_replacement(match.group(0))
    mutated = f"{step[:match.start()]}{replacement}{step[match.end():]}"
    if mutated.strip() == step.strip():
        return MutationResult(OPERATOR_MUTATION, step, None, None, False, "unchanged_output")
    return MutationResult(OPERATOR_MUTATION, mutated, (match.start(), match.end()), replacement, True, "applied")


def _mutate_mixed(
    step: str,
    source_id: str,
    step_index: int,
    seed: int,
    number_offsets: Sequence[int],
) -> MutationResult:
    first_number = _stable_index(2, seed, source_id, step_index, MIXED_MUTATION, "order") == 0
    strategies = [NUMBER_MUTATION, OPERATOR_MUTATION] if first_number else [OPERATOR_MUTATION, NUMBER_MUTATION]
    failures: list[str] = []
    for strategy in strategies:
        result = mutate_step(step, source_id, step_index, strategy, seed, number_offsets)
        if result.success:
            return result
        failures.append(f"{strategy}:{result.reason}")
    return MutationResult(MIXED_MUTATION, step, None, None, False, ";".join(failures))


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


def mutation_metadata(result: MutationResult, configured_strategy: str) -> dict[str, Any]:
    """Return contract metadata for a mutation result."""

    return {
        "configured_strategy": configured_strategy,
        "strategy": result.strategy,
        "changed_span": list(result.changed_span) if result.changed_span is not None else None,
        "replacement": result.replacement,
        "success": result.success,
        "reason": result.reason,
    }
