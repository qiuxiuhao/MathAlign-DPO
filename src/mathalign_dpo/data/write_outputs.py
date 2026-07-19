"""Transactional writers for Stage 1 data outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.data.load_numina import validate_normalized_example


@dataclass(frozen=True)
class PublishedOutputs:
    """Final paths and hashes from a successful Stage 1 publish."""

    paths: dict[str, Path]
    hashes: dict[str, str]
    counts: dict[str, int]
    staging_dir: Path


def publish_stage1_outputs(
    canonical: Mapping[str, list[dict[str, Any]]],
    statistics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    overwrite: bool,
    run_id: str | None = None,
) -> PublishedOutputs:
    """Write Stage 1 outputs through a staging directory and publish atomically."""

    run_name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_manifest = output_paths["manifest"]
    output_root = final_manifest.parent
    staging_dir = output_root / f".stage_{run_name}"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")

    final_paths = {
        "train": output_paths["train"],
        "validation": output_paths["validation"],
        "evaluation": output_paths["evaluation"],
        "statistics": output_paths["statistics"],
        "manifest": output_paths["manifest"],
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Stage 1 outputs: {existing}")

    staging_dir.mkdir(parents=True)
    try:
        staged_paths = {name: staging_dir / path.name for name, path in final_paths.items()}
        counts: dict[str, int] = {}
        hashes: dict[str, str] = {}

        for split in ("train", "validation", "evaluation"):
            rows = list(canonical[split])
            for example in rows:
                validate_normalized_example(example)
            counts[split] = len(rows)
            _write_jsonl(staged_paths[split], rows)
            hashes[split] = sha256_file(staged_paths[split])

        stats_payload = dict(statistics)
        _write_json(staged_paths["statistics"], stats_payload)
        hashes["statistics"] = sha256_file(staged_paths["statistics"])

        manifest_payload = dict(manifest)
        manifest_payload["completed"] = True
        manifest_payload["files"] = {
            split: {
                "path": str(final_paths[split]),
                "rows": counts[split],
                "sha256": hashes[split],
            }
            for split in ("train", "validation", "evaluation")
        }
        manifest_payload["statistics_file"] = {
            "path": str(final_paths["statistics"]),
            "sha256": hashes["statistics"],
        }
        _write_json(staged_paths["manifest"], manifest_payload)
        hashes["manifest"] = sha256_file(staged_paths["manifest"])

        output_root.mkdir(parents=True, exist_ok=True)
        for name, final_path in final_paths.items():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_paths[name], final_path)
        shutil.rmtree(staging_dir)
        return PublishedOutputs(paths=final_paths, hashes=hashes, counts=counts, staging_dir=staging_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def sha256_file(path: Path) -> str:
    """Calculate a file sha256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
