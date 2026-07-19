"""Runtime metadata helpers for training runs."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


VERSION_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "safetensors",
    "psutil",
)


@dataclass
class RunClock:
    """Simple wall-clock timer."""

    start_time: float
    start_iso: str

    @classmethod
    def start(cls) -> "RunClock":
        return cls(
            start_time=time.monotonic(),
            start_iso=datetime.now(timezone.utc).isoformat(),
        )

    def finish(self) -> dict[str, Any]:
        return {
            "start_time": self.start_iso,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.start_time, 3),
        }


def build_run_id(stage: str, smoke_test: bool, stage_number: int = 3) -> str:
    """Build a timestamped run ID."""

    suffix = "smoke" if smoke_test else "mini"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_stage{stage_number}_{stage}_{suffix}_{uuid.uuid4().hex[:8]}"


def collect_base_metadata(
    config: Mapping[str, Any],
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    stage_number: int,
    training_stage: str,
    run_mode: str,
    smoke_test: bool,
    runtime_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect metadata available before training starts."""

    return {
        "schema_version": "1.0",
        "stage": int(stage_number),
        "training_stage": str(training_stage),
        "status": "running",
        "run_id": run_id,
        "run_mode": str(run_mode),
        "smoke_test": bool(smoke_test),
        "original_config_path": str(config_path),
        "output_dir": str(output_dir),
        "effective_config": dict(config),
        "runtime_overrides": dict(runtime_overrides),
        "project": dict(config["project"]),
        "model": dict(config["model"]),
        "runtime": dict(config["runtime"]),
        "sft": dict(config["sft"]),
        "dpo": dict(config.get("dpo", {})),
        "seed": int(config["project"]["seed"]),
        "git_commit": git_commit(),
        "system": system_metadata(),
        "device": device_metadata(config),
        "software_versions": software_versions(),
    }


def finalize_metadata(metadata: Mapping[str, Any], clock: RunClock, status: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    """Return a completed metadata payload."""

    payload = dict(metadata)
    payload.update(clock.finish())
    payload["status"] = status
    payload["peak_process_memory_mb"] = peak_process_memory_mb()
    payload.update(extra)
    return payload


def software_versions() -> dict[str, str | None]:
    """Collect installed package versions without importing heavy libraries."""

    versions: dict[str, str | None] = {}
    for package in VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def system_metadata() -> dict[str, Any]:
    """Collect OS and Python details."""

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "macos_version": platform.mac_ver()[0] or None,
    }


def device_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect backend-specific device metadata without changing execution state."""

    backend = str(config["runtime"]["backend"])
    info: dict[str, Any] = {
        "backend": backend,
        "configured_device": str(config["runtime"]["device"]),
        "allow_cpu_fallback": bool(config["runtime"]["allow_cpu_fallback"]),
    }
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        info["torch_importable"] = False
        return info

    info["torch_importable"] = True
    if backend == "mps":
        info["mps_is_built"] = bool(torch.backends.mps.is_built())
        info["mps_is_available"] = bool(torch.backends.mps.is_available())
        info["mps_memory_api_skipped"] = "process_peak_memory_is_used_to_avoid_mps_counter_crashes"
    elif backend == "cuda":
        info["cuda_is_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            info["cuda_device_name"] = torch.cuda.get_device_name(index)
            info["cuda_device_index"] = int(index)
    return info


def peak_process_memory_mb() -> float:
    """Return peak resident memory in MB for the current process."""

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(usage / (1024 * 1024), 3)
    return round(usage / 1024, 3)


def git_commit() -> str | None:
    """Return the current git commit if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write JSONL rows with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")


def json_safe(value: Any) -> Any:
    """Return a JSON-compliant value, replacing non-finite floats with null."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value
