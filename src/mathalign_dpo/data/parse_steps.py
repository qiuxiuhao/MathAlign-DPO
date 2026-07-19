"""Parse normalized mathematical solutions into steps and final answers."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping


ANSWER_METHOD_NONE = "none"


@dataclass(frozen=True)
class AnswerExtraction:
    """Extracted final answer text and extraction method."""

    answer: str | None
    method: str


@dataclass(frozen=True)
class StepParseResult:
    """Parsed steps and status for one normalized example."""

    steps: list[str]
    final_answer: str | None
    answer_method: str
    parse_status: str
    failure_reason: str | None


def parse_normalized_example(example: Mapping[str, Any], minimum_steps: int) -> dict[str, Any]:
    """Convert one normalized example into the step schema."""

    parsed = parse_solution(str(example["solution"]), minimum_steps=minimum_steps)
    return {
        "schema_version": "1.0",
        "id": example["id"],
        "source_id": example["source_id"],
        "problem": example["problem"],
        "solution": example["solution"],
        "steps": parsed.steps,
        "final_answer": parsed.final_answer,
        "parse_status": parsed.parse_status,
        "metadata": {
            "step_count": len(parsed.steps),
            "answer_extraction_method": parsed.answer_method,
            "parse_failure_reason": parsed.failure_reason,
        },
    }


def parse_solution(solution: str, minimum_steps: int) -> StepParseResult:
    """Parse one solution into ordered steps and final answer status."""

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
        return StepParseResult([], None, answer.method, "failed", "insufficient_steps")
    if answer.answer is None:
        return StepParseResult(steps, None, answer.method, "partial", None)
    return StepParseResult(steps, answer.answer, answer.method, "success", None)


def extract_final_answer(solution: str) -> AnswerExtraction:
    """Extract final answer text with deterministic priority rules."""

    boxed = _extract_last_boxed(solution)
    if boxed is not None:
        return AnswerExtraction(boxed, "boxed")

    hash_answers = re.findall(r"####\s*([^\n]+)", solution)
    if hash_answers:
        answer = _strip_answer(hash_answers[-1])
        if answer:
            return AnswerExtraction(answer, "hash_answer")

    label_pattern = re.compile(
        r"(?is)(?:final\s+answer|correct\s+answer|answer)\s*(?:is|=|:)?\s*(?:\\boxed\{)?\s*([A-Za-z]|\(?[A-E]\)?|[-+]?\d+(?:\.\d+)?|[-+]?\d+\s*/\s*[-+]?\d+|\\frac\{[-+]?\d+\}\{[-+]?\d+\})"
    )
    label_answers = label_pattern.findall(solution)
    if label_answers:
        answer = _strip_answer(label_answers[-1])
        if answer:
            return AnswerExtraction(answer, "answer_label")

    tail = solution[-500:]
    choices = re.findall(r"(?<![A-Za-z])\(([A-E])\)|(?:option|choice)\s+([A-E])", tail, flags=re.IGNORECASE)
    if choices:
        letter = choices[-1][0] or choices[-1][1]
        return AnswerExtraction(letter.upper(), "multiple_choice")

    numeric_pattern = re.compile(r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+(?:\.\d+)?")
    numbers = numeric_pattern.findall(tail)
    if numbers:
        answer = _strip_answer(numbers[-1])
        if answer:
            return AnswerExtraction(answer, "numeric")
    return AnswerExtraction(None, ANSWER_METHOD_NONE)


def validate_step_example(example: Mapping[str, Any]) -> None:
    """Validate the Stage 2 step schema."""

    required = {"schema_version", "id", "source_id", "problem", "solution", "steps", "final_answer", "parse_status", "metadata"}
    missing = sorted(required - set(example))
    if missing:
        raise ValueError(f"Step example missing fields: {missing}")
    status = example["parse_status"]
    if status not in {"success", "partial", "failed"}:
        raise ValueError(f"Invalid parse_status for {example['id']}: {status}")
    steps = example["steps"]
    if not isinstance(steps, list) or any(not isinstance(step, str) or not step.strip() for step in steps):
        raise ValueError(f"Invalid steps for {example['id']}")
    if status == "success" and not example["final_answer"]:
        raise ValueError(f"Successful step example lacks final_answer: {example['id']}")
    if status in {"partial", "failed"} and example["final_answer"] is not None:
        raise ValueError(f"{status} step example must have final_answer null: {example['id']}")
    if status == "failed" and steps:
        raise ValueError(f"Failed step example must have no steps: {example['id']}")


def parse_status_counts(examples: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count parse statuses."""

    counts = Counter(str(example["parse_status"]) for example in examples)
    return {status: int(counts.get(status, 0)) for status in ("success", "partial", "failed")}


def answer_method_counts(examples: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count answer extraction methods."""

    counts = Counter(str(example["metadata"]["answer_extraction_method"]) for example in examples)
    return {method: int(count) for method, count in sorted(counts.items())}


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
    if not answer:
        return None
    return answer


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
    if len(parts) < 2:
        return []
    return parts


def _split_paragraphs(solution: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", solution) if part.strip()]


def _split_sentences_conservatively(solution: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", solution.strip())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。])\s+(?=[A-Z$\\(]|Therefore|Thus|So|Hence)", normalized)
    return [part.strip() for part in parts if part.strip()]


def _clean_steps(candidates: list[str]) -> list[str]:
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
