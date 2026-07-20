"""Load the YAML configuration used by the standalone SFT stage."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one YAML config."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    validate_config(config, path)
    return config


def validate_config(config: Mapping[str, Any], path: Path | None = None) -> None:
    """Validate only the fields required by the standalone SFT stage."""

    label = str(path) if path else "config"
    for key in ("project", "model", "data", "lora", "sft", "evaluation", "runtime", "smoke_test"):
        if key not in config or not isinstance(config[key], Mapping):
            raise ValueError(f"{label}: missing mapping section {key!r}")
    run_mode = str(config["project"].get("run_mode"))
    if run_mode not in {"mini", "formal"}:
        raise ValueError(f"{label}: project.run_mode must be mini or formal")
    data = config["data"]
    for key in ("processed_dir", f"{run_mode}_dir"):
        if not data.get(key):
            raise ValueError(f"{label}: data.{key} is required")
    model = config["model"]
    if not model.get("name_or_path"):
        raise ValueError(f"{label}: model.name_or_path is required")
    if not model.get("revision"):
        raise ValueError(f"{label}: model.revision is required")
    if int(model.get("max_length", 0)) <= 0:
        raise ValueError(f"{label}: model.max_length must be positive")
    runtime = config["runtime"]
    if runtime.get("backend") not in {"mps", "cuda"}:
        raise ValueError(f"{label}: runtime.backend must be mps or cuda")
    if runtime.get("allow_cpu_fallback") is not False:
        raise ValueError(f"{label}: runtime.allow_cpu_fallback must be false")
    if runtime.get("backend") == "mps" and str(model.get("torch_dtype")) != "float16":
        raise ValueError(f"{label}: MPS SFT requires model.torch_dtype = float16")
    if runtime.get("backend") == "cuda" and config.get("quantization", {}).get("load_in_4bit"):
        quantization = config["quantization"]
        if quantization.get("quant_type") != "nf4":
            raise ValueError(f"{label}: CUDA 4-bit SFT requires quantization.quant_type = nf4")
        if quantization.get("compute_dtype") != "bfloat16":
            raise ValueError(f"{label}: CUDA 4-bit SFT requires quantization.compute_dtype = bfloat16")
    sft = config["sft"]
    if not sft.get("enabled", False):
        raise ValueError(f"{label}: sft.enabled must be true")
    if int(sft.get("max_steps", 0)) <= 0:
        raise ValueError(f"{label}: sft.max_steps must be positive")
    if int(config["evaluation"].get("samples", 0)) <= 0:
        raise ValueError(f"{label}: evaluation.samples must be positive")


def apply_runtime_overrides(
    config: Mapping[str, Any],
    smoke_test: bool,
    train_samples: int | None,
    validation_samples: int | None,
    eval_samples: int | None,
    max_steps: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a config copy with CLI debug overrides applied."""

    copied = deepcopy(dict(config))
    overrides: dict[str, Any] = {
        "smoke_test": bool(smoke_test),
        "cli": {
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "eval_samples": eval_samples,
            "max_steps": max_steps,
        },
        "applied": {},
    }
    if smoke_test:
        copied["sft"]["max_steps"] = int(config["smoke_test"]["max_steps"])
        copied["data"]["train_samples"] = int(config["smoke_test"]["train_samples"])
        copied["data"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
        copied["evaluation"]["samples"] = int(config["smoke_test"]["evaluation_samples"])
        overrides["applied"].update(
            {
                "sft.max_steps": copied["sft"]["max_steps"],
                "data.train_samples": copied["data"]["train_samples"],
                "data.validation_samples": copied["data"]["validation_samples"],
                "evaluation.samples": copied["evaluation"]["samples"],
            }
        )
    if train_samples is not None:
        copied["data"]["train_samples"] = int(train_samples)
        overrides["applied"]["data.train_samples"] = int(train_samples)
    if validation_samples is not None:
        copied["data"]["validation_samples"] = int(validation_samples)
        overrides["applied"]["data.validation_samples"] = int(validation_samples)
    if eval_samples is not None:
        copied["evaluation"]["samples"] = int(eval_samples)
        overrides["applied"]["evaluation.samples"] = int(eval_samples)
    if max_steps is not None:
        copied["sft"]["max_steps"] = int(max_steps)
        overrides["applied"]["sft.max_steps"] = int(max_steps)
    return copied, overrides
