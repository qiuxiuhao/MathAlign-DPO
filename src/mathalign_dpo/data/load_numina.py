"""Load, audit, and normalize NuminaMath-CoT rows for Stage 1."""

from __future__ import annotations

import re
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.0"
SOURCE_ID_FIELDS = ("id", "source_id", "problem_id", "question_id", "uuid", "uid")
PROBLEM_FIELDS = ("problem", "question", "prompt")
SOLUTION_FIELDS = ("solution", "answer", "response")


@dataclass(frozen=True)
class FieldAudit:
    """Raw dataset field audit used to pick normalization policy."""

    row_count: int
    fields: list[str]
    field_types: dict[str, list[str]]
    empty_counts: dict[str, int]
    id_field: str | None
    id_strategy: str
    problem_field: str
    solution_field: str
    source_rows_sha256: str


@dataclass(frozen=True)
class NormalizationResult:
    """Normalized examples plus counters from rejected source rows."""

    examples: list[dict[str, Any]]
    rejected: dict[str, int]
    audit: FieldAudit


def load_numina_dataset(dataset_name: str, revision: str, source_split: str) -> Iterable[Mapping[str, Any]]:
    """Load NuminaMath through Hugging Face datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Stage 1 data loading requires the 'datasets' package") from exc
    return load_dataset(dataset_name, revision=revision, split=source_split)


def audit_rows(rows: Iterable[Mapping[str, Any]]) -> FieldAudit:
    """Inspect raw rows and choose fields for Stage 1 normalization."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot audit an empty dataset")

    fields = sorted({key for row in materialized for key in row.keys()})
    field_types: dict[str, set[str]] = {field: set() for field in fields}
    empty_counts: Counter[str] = Counter()
    for row in materialized:
        for field in fields:
            value = row.get(field)
            field_types[field].add(type(value).__name__)
            if value is None or (isinstance(value, str) and not value.strip()):
                empty_counts[field] += 1

    problem_field = _first_present(fields, PROBLEM_FIELDS, "problem")
    solution_field = _first_present(fields, SOLUTION_FIELDS, "solution")
    id_field = _choose_id_field(materialized, fields)
    return FieldAudit(
        row_count=len(materialized),
        fields=fields,
        field_types={field: sorted(types) for field, types in field_types.items()},
        empty_counts={field: int(empty_counts[field]) for field in fields},
        id_field=id_field,
        id_strategy="native_field" if id_field else "row_index_fallback",
        problem_field=problem_field,
        solution_field=solution_field,
        source_rows_sha256=hash_source_rows(materialized),
    )


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    preprocessing: Mapping[str, Any],
) -> NormalizationResult:
    """Normalize raw NuminaMath rows into `NormalizedMathExample` dictionaries."""

    materialized = list(rows)
    audit = audit_rows(materialized)
    rejected: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row_index, row in enumerate(materialized):
        source_id = build_source_id(row, row_index, audit.id_field)
        stable_id = f"numina_{source_split}_{source_id}"
        problem = row.get(audit.problem_field)
        solution = row.get(audit.solution_field)
        if not isinstance(problem, str):
            rejected["problem_not_string"] += 1
            continue
        if not isinstance(solution, str):
            rejected["solution_not_string"] += 1
            continue

        normalized_problem = normalize_text(problem, preprocessing)
        normalized_solution = normalize_text(solution, preprocessing)
        if not normalized_problem:
            rejected["empty_problem"] += 1
            continue
        if not normalized_solution:
            rejected["empty_solution"] += 1
            continue
        if normalized_problem == normalized_solution:
            rejected["problem_equals_solution"] += 1
            continue

        if stable_id in seen_ids:
            rejected["duplicate_id"] += 1
            continue
        seen_ids.add(stable_id)
        examples.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": stable_id,
                "source": dataset_name,
                "source_split": source_split,
                "source_id": source_id,
                "problem": normalized_problem,
                "solution": normalized_solution,
                "metadata": {
                    "source_subset": row.get("source"),
                    "original_fields": sorted(row.keys()),
                },
            }
        )

    return NormalizationResult(examples=examples, rejected=dict(rejected), audit=audit)


def hash_source_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash raw rows in source order for fixed-revision drift checks."""

    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, default=str)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_text(text: str, preprocessing: Mapping[str, Any]) -> str:
    """Apply contract-approved text normalization."""

    normalized = text
    if preprocessing.get("normalize_line_endings", True):
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if preprocessing.get("strip_outer_whitespace", True):
        normalized = normalized.strip()

    max_blank_lines = int(preprocessing.get("max_consecutive_blank_lines", 2))
    if max_blank_lines >= 0:
        pattern = r"\n{" + str(max_blank_lines + 2) + r",}"
        replacement = "\n" * (max_blank_lines + 1)
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def build_source_id(row: Mapping[str, Any], row_index: int, id_field: str | None) -> str:
    """Build a contract-safe source ID from a native field or source row index."""

    if id_field is None:
        return f"{row_index:08d}"
    raw_value = str(row[id_field]).strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_value).strip("_")
    if not cleaned:
        return f"{row_index:08d}"
    if len(cleaned) > 96:
        return cleaned[:96]
    return cleaned


def validate_normalized_example(example: Mapping[str, Any]) -> None:
    """Raise when a normalized example does not match the Stage 1 contract."""

    required = {"schema_version", "id", "source", "source_split", "source_id", "problem", "solution", "metadata"}
    missing = sorted(required - set(example))
    if missing:
        raise ValueError(f"Normalized example missing fields: {missing}")
    for field in ("id", "source", "source_split", "source_id", "problem", "solution"):
        if not isinstance(example[field], str) or not example[field].strip():
            raise ValueError(f"Normalized example field must be non-empty string: {field}")
    if example["problem"] == example["solution"]:
        raise ValueError(f"Normalized example has identical problem and solution: {example['id']}")
    metadata = example["metadata"]
    if not isinstance(metadata, dict) or "original_fields" not in metadata:
        raise ValueError(f"Normalized example has invalid metadata: {example['id']}")


def _first_present(fields: list[str], candidates: tuple[str, ...], semantic_name: str) -> str:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    raise ValueError(f"Could not find a {semantic_name} field in raw dataset fields: {fields}")


def _choose_id_field(rows: list[Mapping[str, Any]], fields: list[str]) -> str | None:
    for candidate in SOURCE_ID_FIELDS:
        if candidate not in fields:
            continue
        values: list[str] = []
        valid = True
        for row in rows:
            value = row.get(candidate)
            if not isinstance(value, (str, int)) or not str(value).strip():
                valid = False
                break
            values.append(str(value).strip())
        if valid and len(values) == len(set(values)):
            return candidate
    return None
