"""Load and validate the two approved project configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SPLITS = ("train", "validation", "evaluation")


@dataclass(frozen=True)
class ProjectConfigs:
    """Validated Mini/formal configuration pair."""

    mini_path: Path
    formal_path: Path
    mini: dict[str, Any]
    formal: dict[str, Any]

    @property
    def dataset_name(self) -> str:
        return str(self.formal["data"]["dataset_name"])

    @property
    def dataset_revision(self) -> str:
        return str(self.formal["data"]["dataset_revision"])

    @property
    def source_split(self) -> str:
        return str(self.formal["data"]["source_split"])

    @property
    def seed(self) -> int:
        return int(self.formal["project"]["seed"])

    @property
    def processed_dir(self) -> Path:
        return Path(str(self.formal["data"]["processed_dir"]))


def load_project_configs(mini_config: str | Path, formal_config: str | Path) -> ProjectConfigs:
    """Load and validate Mini and formal configs for the shared project pipeline."""

    mini_path = Path(mini_config)
    formal_path = Path(formal_config)
    mini = _load_yaml(mini_path)
    formal = _load_yaml(formal_path)
    _validate_single_config(mini, mini_path, expected_mode="mini")
    _validate_single_config(formal, formal_path, expected_mode="formal")
    _validate_shared_data_config(mini, formal)
    return ProjectConfigs(mini_path=mini_path, formal_path=formal_path, mini=mini, formal=formal)


def load_single_config(config_path: str | Path, expected_mode: str | None = None) -> dict[str, Any]:
    """Load and validate one approved run config."""

    path = Path(config_path)
    config = _load_yaml(path)
    mode = expected_mode or str(config.get("project", {}).get("run_mode", ""))
    _validate_single_config(config, path, expected_mode=mode)
    return config


def split_ratios(config: dict[str, Any]) -> dict[str, float]:
    """Return configured split ratios keyed by contract split name."""

    data = config["data"]
    return {
        "train": float(data["train_ratio"]),
        "validation": float(data["validation_ratio"]),
        "evaluation": float(data["evaluation_ratio"]),
    }


def sample_counts(config: dict[str, Any]) -> dict[str, int]:
    """Return configured sample counts keyed by contract split name."""

    data = config["data"]
    return {
        "train": int(data["train_samples"]),
        "validation": int(data["validation_samples"]),
        "evaluation": int(data["evaluation_samples"]),
    }


def output_paths(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Path]:
    """Return Stage 1 final output paths, optionally rooted at a debug output dir."""

    data = config["data"]
    if output_dir is None:
        return {
            "train": Path(str(data["normalized_train_file"])),
            "validation": Path(str(data["normalized_validation_file"])),
            "evaluation": Path(str(data["normalized_eval_file"])),
            "statistics": Path(str(data["statistics_file"])),
            "manifest": Path(str(data["split_manifest_file"])),
        }
    root = Path(output_dir)
    return {
        "train": root / "normalized_train.jsonl",
        "validation": root / "normalized_validation.jsonl",
        "evaluation": root / "normalized_eval.jsonl",
        "statistics": root / "data_statistics.json",
        "manifest": root / "split_manifest.json",
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _validate_single_config(config: dict[str, Any], path: Path, expected_mode: str) -> None:
    required_top_keys = {
        "schema_version",
        "project",
        "model",
        "quantization",
        "lora",
        "data",
        "preprocessing",
        "negative_sampling",
        "sft",
        "dpo",
        "evaluation",
        "runtime",
        "output",
        "smoke_test",
    }
    missing = sorted(required_top_keys - set(config))
    if missing:
        raise ValueError(f"{path}: missing top-level config keys: {missing}")

    project = config["project"]
    if "stage" in project:
        raise ValueError(f"{path}: project.stage must be removed; stage belongs in docs/reports")
    if project.get("run_mode") != expected_mode:
        raise ValueError(f"{path}: project.run_mode must be {expected_mode!r}")
    if not isinstance(project.get("seed"), int):
        raise ValueError(f"{path}: project.seed must be an integer")

    data = config["data"]
    if not data.get("dataset_revision"):
        raise ValueError(f"{path}: data.dataset_revision must be pinned to a commit hash")
    ratios = split_ratios(config)
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"{path}: split ratios must sum to 1.0, got {ratios}")
    for split, count in sample_counts(config).items():
        if count <= 0:
            raise ValueError(f"{path}: data.{split}_samples must be positive")

    backend = config["runtime"].get("backend")
    quantization = config["quantization"]
    model = config["model"]
    lora = config["lora"]
    sft = config["sft"]
    runtime = config["runtime"]
    optimizer = config["sft"].get("optimizer")
    if not bool(sft.get("enabled")):
        raise ValueError(f"{path}: sft.enabled must be true")
    if not bool(lora.get("enabled")):
        raise ValueError(f"{path}: lora.enabled must be true")
    if int(lora.get("rank", 0)) <= 0:
        raise ValueError(f"{path}: lora.rank must be positive")
    if int(lora.get("alpha", 0)) <= 0:
        raise ValueError(f"{path}: lora.alpha must be positive")
    if not isinstance(lora.get("target_modules"), list) or not lora["target_modules"]:
        raise ValueError(f"{path}: lora.target_modules must be a non-empty list")
    if int(model.get("max_length", 0)) <= 0:
        raise ValueError(f"{path}: model.max_length must be positive")
    if not isinstance(model.get("revision"), str) or not model["revision"].strip():
        raise ValueError(f"{path}: model.revision must pin a non-empty commit revision")
    if int(sft.get("max_steps", 0)) <= 0:
        raise ValueError(f"{path}: sft.max_steps must be positive")
    if int(sft.get("adapter_reload_samples", 0)) <= 0:
        raise ValueError(f"{path}: sft.adapter_reload_samples must be positive")
    if int(sft.get("adapter_reload_max_new_tokens", 0)) <= 0:
        raise ValueError(f"{path}: sft.adapter_reload_max_new_tokens must be positive")
    if runtime.get("allow_cpu_fallback") is not False:
        raise ValueError(f"{path}: runtime.allow_cpu_fallback must be false")
    if backend == "mps":
        if quantization.get("enabled") or quantization.get("load_in_4bit"):
            raise ValueError(f"{path}: MPS config must not enable BitsAndBytes or 4-bit loading")
        if optimizer != "adamw_torch":
            raise ValueError(f"{path}: MPS SFT optimizer must be adamw_torch")
        if model.get("torch_dtype") != "float16":
            raise ValueError(f"{path}: MPS config must use model.torch_dtype = float16")
    elif backend == "cuda":
        if not quantization.get("enabled") or not quantization.get("load_in_4bit"):
            raise ValueError(f"{path}: CUDA config must enable 4-bit quantization")
        if quantization.get("quant_type") != "nf4":
            raise ValueError(f"{path}: CUDA config must use quantization.quant_type = nf4")
        if quantization.get("compute_dtype") != "bfloat16":
            raise ValueError(f"{path}: CUDA config must use quantization.compute_dtype = bfloat16")
    else:
        raise ValueError(f"{path}: runtime.backend must be mps or cuda")


def _validate_shared_data_config(mini: dict[str, Any], formal: dict[str, Any]) -> None:
    shared_data_keys = [
        "dataset_name",
        "dataset_revision",
        "source_split",
        "train_ratio",
        "validation_ratio",
        "evaluation_ratio",
        "processed_dir",
        "normalized_train_file",
        "normalized_validation_file",
        "normalized_eval_file",
        "statistics_file",
        "split_manifest_file",
        "stage2_statistics_file",
        "stage2_manifest_file",
        "manual_review_file",
    ]
    for key in shared_data_keys:
        if mini["data"].get(key) != formal["data"].get(key):
            raise ValueError(f"Mini and formal configs must share data.{key}")
    if mini["project"].get("seed") != formal["project"].get("seed"):
        raise ValueError("Mini and formal configs must share project.seed")

    mini_counts = sample_counts(mini)
    formal_counts = sample_counts(formal)
    for split in SPLITS:
        if mini_counts[split] > formal_counts[split]:
            raise ValueError(f"Mini {split} count cannot exceed formal {split} count")
