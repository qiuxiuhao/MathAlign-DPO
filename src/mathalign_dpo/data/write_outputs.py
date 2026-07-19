"""Transactional writers for staged data outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from mathalign_dpo.data.load_numina import validate_normalized_example


Publisher = Any
ManifestBuilder = Callable[[Mapping[str, Any], Mapping[str, Path], Mapping[str, str], Mapping[str, int]], Mapping[str, Any]]


@dataclass(frozen=True)
class PublishedOutputs:
    """Final paths and hashes from a successful Stage 1 publish."""

    paths: dict[str, Path]
    hashes: dict[str, str]
    counts: dict[str, int]
    staging_dir: Path


@dataclass(frozen=True)
class JsonOutput:
    """One JSON or JSONL output file to publish transactionally."""

    path: Path
    kind: str
    payload: Any
    rows: int | None = None


def publish_stage1_outputs(
    canonical: Mapping[str, list[dict[str, Any]]],
    statistics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    overwrite: bool,
    run_id: str | None = None,
    replace_file: Publisher = os.replace,
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

        manifest_payload = _build_manifest_payload(manifest, final_paths, counts, hashes)
        _write_json(staged_paths["manifest"], manifest_payload)
        hashes["manifest"] = sha256_file(staged_paths["manifest"])
        _validate_staged_outputs(staged_paths, manifest_payload)

        output_root.mkdir(parents=True, exist_ok=True)
        _publish_with_rollback(
            staged_paths,
            final_paths,
            staging_dir,
            replace_file,
            publish_order=("train", "validation", "evaluation", "statistics", "manifest"),
        )
        shutil.rmtree(staging_dir)
        return PublishedOutputs(paths=final_paths, hashes=hashes, counts=counts, staging_dir=staging_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def publish_json_outputs(
    outputs: Mapping[str, JsonOutput],
    manifest_name: str,
    overwrite: bool,
    run_id: str | None = None,
    replace_file: Publisher = os.replace,
    manifest_builder: ManifestBuilder | None = None,
) -> PublishedOutputs:
    """Publish arbitrary JSON/JSONL outputs through one staged transaction."""

    run_name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_manifest = outputs[manifest_name].path
    output_root = final_manifest.parent
    staging_dir = output_root / f".stage_{run_name}"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    final_paths = {name: output.path for name, output in outputs.items()}
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    staging_dir.mkdir(parents=True)
    try:
        staged_paths = {name: staging_dir / output.path.name for name, output in outputs.items()}
        hashes: dict[str, str] = {}
        counts: dict[str, int] = {}
        for name, output in outputs.items():
            if name == manifest_name:
                continue
            if output.kind == "jsonl":
                _write_jsonl(staged_paths[name], list(output.payload))
                actual_rows = _count_jsonl_rows(staged_paths[name])
                if output.rows is not None and actual_rows != output.rows:
                    raise ValueError(f"Staged row count mismatch for {name}: expected {output.rows}, got {actual_rows}")
                counts[name] = actual_rows
            elif output.kind == "json":
                _write_json(staged_paths[name], output.payload)
            else:
                raise ValueError(f"Unsupported output kind for {name}: {output.kind}")
            hashes[name] = sha256_file(staged_paths[name])

        manifest_output = outputs[manifest_name]
        if manifest_output.kind != "json":
            raise ValueError(f"Manifest output must be JSON: {manifest_name}")
        manifest_payload = manifest_output.payload
        if manifest_builder is not None:
            manifest_payload = manifest_builder(manifest_output.payload, final_paths, hashes, counts)
        _write_json(staged_paths[manifest_name], manifest_payload)
        hashes[manifest_name] = sha256_file(staged_paths[manifest_name])

        _publish_with_rollback(
            staged_paths,
            final_paths,
            staging_dir,
            replace_file,
            publish_order=tuple(name for name in outputs if name != manifest_name) + (manifest_name,),
        )
        shutil.rmtree(staging_dir)
        return PublishedOutputs(paths=final_paths, hashes=hashes, counts=counts, staging_dir=staging_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def validate_completed_manifest(manifest_path: Path) -> None:
    """Validate that a published Stage 1 manifest points to complete outputs."""

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("completed") is not True:
        raise ValueError(f"Manifest is not completed: {manifest_path}")
    for split, file_info in manifest.get("files", {}).items():
        path = Path(str(file_info["path"]))
        if not path.exists():
            raise ValueError(f"Manifest file is missing for {split}: {path}")
        rows = _count_jsonl_rows(path)
        if rows != int(file_info["rows"]):
            raise ValueError(f"Manifest row count mismatch for {split}: expected {file_info['rows']}, got {rows}")
        digest = sha256_file(path)
        if digest != file_info["sha256"]:
            raise ValueError(f"Manifest sha256 mismatch for {split}: {path}")

    statistics_info = manifest.get("statistics_file", {})
    statistics_path = Path(str(statistics_info.get("path", "")))
    if not statistics_path.exists():
        raise ValueError(f"Manifest statistics file is missing: {statistics_path}")
    if sha256_file(statistics_path) != statistics_info.get("sha256"):
        raise ValueError(f"Manifest statistics sha256 mismatch: {statistics_path}")


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


def _build_manifest_payload(
    manifest: Mapping[str, Any],
    final_paths: Mapping[str, Path],
    counts: Mapping[str, int],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = dict(manifest)
    payload["completed"] = True
    payload["files"] = {
        split: {
            "path": str(final_paths[split]),
            "rows": counts[split],
            "sha256": hashes[split],
        }
        for split in ("train", "validation", "evaluation")
    }
    payload["statistics_file"] = {
        "path": str(final_paths["statistics"]),
        "sha256": hashes["statistics"],
    }
    return payload


def _validate_staged_outputs(staged_paths: Mapping[str, Path], manifest: Mapping[str, Any]) -> None:
    for split, file_info in manifest["files"].items():
        staged_path = staged_paths[split]
        rows = _count_jsonl_rows(staged_path)
        if rows != int(file_info["rows"]):
            raise ValueError(f"Staged row count mismatch for {split}: expected {file_info['rows']}, got {rows}")
        digest = sha256_file(staged_path)
        if digest != file_info["sha256"]:
            raise ValueError(f"Staged sha256 mismatch for {split}: {staged_path}")
    statistics_path = staged_paths["statistics"]
    if sha256_file(statistics_path) != manifest["statistics_file"]["sha256"]:
        raise ValueError(f"Staged statistics sha256 mismatch: {statistics_path}")


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line in {path}: line {line_number}")
            json.loads(line)
            count += 1
    return count


def _publish_with_rollback(
    staged_paths: Mapping[str, Path],
    final_paths: Mapping[str, Path],
    staging_dir: Path,
    replace_file: Publisher,
    publish_order: tuple[str, ...],
) -> None:
    backup_dir = staging_dir / "backup"
    backup_dir.mkdir()
    backups: dict[str, Path] = {}
    existed: dict[str, bool] = {}
    for name in publish_order:
        final_path = final_paths[name]
        final_path.parent.mkdir(parents=True, exist_ok=True)
        existed[name] = final_path.exists()
        if final_path.exists():
            backup_path = backup_dir / final_path.name
            shutil.copy2(final_path, backup_path)
            backups[name] = backup_path

    try:
        for name in publish_order:
            replace_file(staged_paths[name], final_paths[name])
    except BaseException:
        _rollback_publish(final_paths, backups, existed, publish_order, replace_file)
        raise


def _rollback_publish(
    final_paths: Mapping[str, Path],
    backups: Mapping[str, Path],
    existed: Mapping[str, bool],
    publish_order: tuple[str, ...],
    replace_file: Publisher,
) -> None:
    for name in reversed(publish_order):
        final_path = final_paths[name]
        if existed[name]:
            replace_file(backups[name], final_path)
        elif final_path.exists():
            final_path.unlink()
