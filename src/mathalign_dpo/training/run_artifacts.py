"""Transactional training run output directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RunDirectories:
    """Final and staging directories for one training run."""

    final_dir: Path
    staging_dir: Path


def resolve_stage_output_dir(
    config: Mapping[str, Any],
    run_id: str,
    output_dir: str | Path | None,
    stage_key: str,
) -> Path:
    """Resolve the final run directory for one configured training stage."""

    if output_dir is not None:
        return Path(output_dir)
    return Path(str(config[stage_key]["output_dir"])) / run_id


def prepare_staged_output_dir(
    config: Mapping[str, Any],
    run_id: str,
    output_dir: str | Path | None,
    overwrite: bool,
    stage_key: str,
    stage_label: str,
) -> RunDirectories:
    """Prepare a hidden staging directory without replacing old successful runs."""

    final_dir = resolve_stage_output_dir(config, run_id, output_dir, stage_key)
    if final_dir.exists() and any(final_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {stage_label} output directory: {final_dir}")
    staging_dir = final_dir.parent / f".{final_dir.name}.{run_id}.staging"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    return RunDirectories(final_dir=final_dir, staging_dir=staging_dir)


def publish_staged_output(run_dirs: RunDirectories, overwrite: bool, stage_label: str) -> None:
    """Atomically publish a completed staging directory and preserve old output on failure."""

    final_dir = run_dirs.final_dir
    staging_dir = run_dirs.staging_dir
    backup_dir = final_dir.parent / f".{final_dir.name}.backup"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    if final_dir.exists() and any(final_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty {stage_label} output directory: {final_dir}")
        final_dir.rename(backup_dir)
    elif final_dir.exists():
        final_dir.rmdir()
    try:
        staging_dir.rename(final_dir)
    except BaseException:
        if final_dir.exists() and final_dir != staging_dir:
            shutil.rmtree(final_dir, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.rename(final_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
