"""Model loading helpers for standalone DPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from sft.modeling import (
    LoadedModel,
    _import_peft,
    _import_torch,
    _import_transformers,
    _load_base_model,
    assert_trainable_parameters,
    configure_deterministic_generation,
    ensure_local_model,
    runtime_metadata,
    validate_runtime,
    validate_tokenizer,
)


def validate_sft_dir(config: Mapping[str, Any], sft_dir: str | Path, smoke_test: bool) -> dict[str, Any]:
    """Validate that an SFT run can initialize DPO."""

    root = Path(sft_dir)
    run_config_path = root / "run_config.json"
    adapter_dir = root / "adapter"
    tokenizer_dir = root / "tokenizer"
    if not run_config_path.exists():
        raise FileNotFoundError(f"SFT run_config.json is missing: {run_config_path}")
    with run_config_path.open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    if run_config.get("status") != "completed":
        raise ValueError(f"SFT run is not completed: {root}")
    if run_config.get("training_stage") != "sft":
        raise ValueError(f"SFT run directory has wrong training_stage: {root}")
    _validate_sft_run_mode(config, run_config, root)
    _validate_sft_model_identity(config, run_config, root)
    if run_config.get("smoke_test") is True and not smoke_test:
        raise ValueError("Non-smoke DPO run cannot initialize from a smoke SFT adapter")
    for path in (adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_config.json"):
        if not path.exists():
            raise FileNotFoundError(f"SFT adapter artifact is missing: {path}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"SFT tokenizer directory is missing: {tokenizer_dir}")
    return {
        "run_dir": str(root),
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "run_config_path": str(run_config_path),
        "smoke_test": bool(run_config.get("smoke_test")),
        "run_mode": run_config.get("run_mode"),
        "model": run_config.get("model", {}),
        "dataset_counts": run_config.get("dataset_counts", {}),
    }


def _validate_sft_run_mode(config: Mapping[str, Any], run_config: Mapping[str, Any], root: Path) -> None:
    expected_mode = str(config["project"]["run_mode"])
    recorded_mode = run_config.get("run_mode")
    if recorded_mode is not None:
        if str(recorded_mode) != expected_mode:
            raise ValueError(f"SFT run_mode={recorded_mode!r} does not match DPO run_mode={expected_mode!r}: {root}")
        return

    recorded_dataset = str(run_config.get("dataset_paths", {}).get("sft", ""))
    if f"/{expected_mode}/" not in recorded_dataset:
        raise ValueError(f"SFT run does not appear to match run_mode={expected_mode}: {root}")


def _validate_sft_model_identity(config: Mapping[str, Any], run_config: Mapping[str, Any], root: Path) -> None:
    expected = config["model"]
    recorded = run_config.get("model") if isinstance(run_config.get("model"), Mapping) else {}
    loader = run_config.get("model_loader") if isinstance(run_config.get("model_loader"), Mapping) else {}
    recorded_revision = recorded.get("revision") or loader.get("revision")
    if recorded_revision and str(recorded_revision) != str(expected["revision"]):
        raise ValueError(f"SFT revision={recorded_revision!r} does not match DPO revision={expected['revision']!r}: {root}")

    expected_remote = str(expected.get("modelscope_name_or_path") or expected.get("remote_name_or_path") or "")
    recorded_remote = str(recorded.get("modelscope_name_or_path") or loader.get("modelscope_name_or_path") or loader.get("remote_name_or_path") or "")
    if recorded_remote and expected_remote and recorded_remote != expected_remote:
        raise ValueError(f"SFT model={recorded_remote!r} does not match DPO model={expected_remote!r}: {root}")


def load_policy_from_sft_adapter(config: Mapping[str, Any], sft_adapter_dir: str | Path, tokenizer_dir: str | Path) -> LoadedModel:
    """Load Base + trainable SFT adapter as the initial DPO policy."""

    runtime = validate_runtime(config)
    torch = _import_torch()
    transformers = _import_transformers()
    peft = _import_peft()
    model_dir = ensure_local_model(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))
    validate_tokenizer(tokenizer)
    model = _load_base_model(config, model_dir, torch, transformers)
    if hasattr(model, "config"):
        model.config.use_cache = bool(config["model"]["use_cache"])
    if bool(config["model"]["gradient_checkpointing"]) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if runtime["backend"] == "cuda" and bool(config.get("quantization", {}).get("load_in_4bit")):
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config["model"]["gradient_checkpointing"]),
        )
    model = peft.PeftModel.from_pretrained(model, str(sft_adapter_dir), is_trainable=True)
    if runtime["backend"] != "cuda":
        model.to(runtime["device"])
    assert_trainable_parameters(model, expected_device_type=str(runtime["backend"]))
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        metadata={
            "model_dir": str(model_dir),
            "sft_adapter_dir": str(sft_adapter_dir),
            "tokenizer_dir": str(tokenizer_dir),
            "backend": runtime["backend"],
            "device": runtime["device"],
            "reference_policy": "trl_peft_ref_adapter",
        },
    )


def load_dpo_for_generation(
    config: Mapping[str, Any],
    adapter_dir: str | Path,
    tokenizer_dir: str | Path,
) -> LoadedModel:
    """Load Base + DPO adapter for deterministic generation."""

    torch = _import_torch()
    transformers = _import_transformers()
    peft = _import_peft()
    runtime = runtime_metadata(config)
    model_dir = ensure_local_model(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))
    validate_tokenizer(tokenizer)
    model = _load_base_model(config, model_dir, torch, transformers)
    if hasattr(model, "config"):
        model.config.use_cache = True
    model = peft.PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    if runtime["backend"] != "cuda":
        model.to(runtime["device"])
    configure_deterministic_generation(model)
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        metadata={"model_dir": str(model_dir), "adapter_dir": str(adapter_dir), **runtime},
    )
