"""Shared model and tokenizer loading for SFT/DPO training."""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LoadedModelAndTokenizer:
    """Model, tokenizer, and loader metadata."""

    model: Any
    tokenizer: Any
    metadata: dict[str, Any]


def load_tokenizer(config: Mapping[str, Any]) -> Any:
    """Load the configured tokenizer."""

    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(config["model"]["name_or_path"]),
        revision=str(config["model"]["revision"]),
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    return tokenizer


def model_revision_metadata(config: Mapping[str, Any]) -> dict[str, str]:
    """Return configured and resolved model revision metadata."""

    configured = str(config["model"]["revision"])
    resolved = configured if re.fullmatch(r"[0-9a-f]{40}", configured) else configured
    return {"configured_revision": configured, "resolved_revision": resolved}


def load_model_and_tokenizer(config: Mapping[str, Any], training_stage: str = "sft") -> LoadedModelAndTokenizer:
    """Load base model, tokenizer, and LoRA/QLoRA adapter modules."""

    if training_stage not in {"sft", "dpo"}:
        raise ValueError(f"Unsupported training_stage: {training_stage!r}")
    validate_runtime_backend(config)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")

    backend = str(config["runtime"]["backend"])
    tokenizer = load_tokenizer(config)
    if backend == "mps":
        model, metadata = _load_mps_model(config, torch, transformers, peft)
    elif backend == "cuda":
        model, metadata = _load_cuda_model(config, torch, transformers, peft)
    else:
        raise ValueError(f"runtime.backend must be mps or cuda, got {backend!r}")
    return LoadedModelAndTokenizer(model=model, tokenizer=tokenizer, metadata=metadata)


def validate_runtime_backend(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that the configured backend is available before loading model assets."""

    torch = importlib.import_module("torch")
    backend = str(config["runtime"]["backend"])
    if backend == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is not allowed; Stage 3 must not silently fall back to CPU")
        built = bool(torch.backends.mps.is_built())
        available = bool(torch.backends.mps.is_available())
        if not built or not available:
            raise RuntimeError(f"MPS backend unavailable: torch.backends.mps.is_built()={built}, is_available()={available}")
        return {"backend": "mps", "mps_is_built": built, "mps_is_available": available}
    if backend == "cuda":
        available = bool(torch.cuda.is_available())
        if not available:
            raise RuntimeError("CUDA backend unavailable: torch.cuda.is_available()=False")
        return {"backend": "cuda", "cuda_is_available": available}
    raise ValueError(f"runtime.backend must be mps or cuda, got {backend!r}")


def _load_mps_model(config: Mapping[str, Any], torch: Any, transformers: Any, peft: Any) -> tuple[Any, dict[str, Any]]:
    if bool(config["quantization"]["enabled"]) or bool(config["quantization"]["load_in_4bit"]):
        raise ValueError("MPS config must not enable BitsAndBytes or 4-bit loading")

    dtype = _torch_dtype(torch, str(config["model"]["torch_dtype"]))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(config["model"]["name_or_path"]),
        revision=str(config["model"]["revision"]),
        dtype=dtype,
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        low_cpu_mem_usage=True,
    )
    _configure_model_for_training(model, config)
    model = _apply_lora(model, config, peft)
    model.to("mps")
    _assert_trainable_parameters(model, expected_device_type="mps")
    return model, {
        "backend": "mps",
        "torch_dtype": str(config["model"]["torch_dtype"]),
        "quantization": "none",
        "lora": _lora_metadata(config),
        "device": "mps",
        "model_revision": model_revision_metadata(config),
    }


def _load_cuda_model(config: Mapping[str, Any], torch: Any, transformers: Any, peft: Any) -> tuple[Any, dict[str, Any]]:
    if not bool(config["quantization"]["enabled"]) or not bool(config["quantization"]["load_in_4bit"]):
        raise ValueError("CUDA config must enable 4-bit quantization")
    if str(config["quantization"]["quant_type"]) != "nf4":
        raise ValueError("CUDA config must use NF4 quantization")

    dtype = _torch_dtype(torch, str(config["quantization"]["compute_dtype"]))
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(config["quantization"]["quant_type"]),
        bnb_4bit_use_double_quant=bool(config["quantization"]["use_double_quant"]),
        bnb_4bit_compute_dtype=dtype,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(config["model"]["name_or_path"]),
        revision=str(config["model"]["revision"]),
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    _configure_model_for_training(model, config)
    if hasattr(peft, "prepare_model_for_kbit_training"):
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config["model"]["gradient_checkpointing"]),
        )
    model = _apply_lora(model, config, peft)
    _assert_trainable_parameters(model, expected_device_type="cuda")
    return model, {
        "backend": "cuda",
        "torch_dtype": str(config["model"]["torch_dtype"]),
        "quantization": "nf4_4bit",
        "lora": _lora_metadata(config),
        "device": "cuda",
        "model_revision": model_revision_metadata(config),
    }


def _configure_model_for_training(model: Any, config: Mapping[str, Any]) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = bool(config["model"]["use_cache"])
    if bool(config["model"]["gradient_checkpointing"]) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def _apply_lora(model: Any, config: Mapping[str, Any], peft: Any) -> Any:
    if not bool(config["lora"]["enabled"]):
        raise ValueError("Stage 3 requires lora.enabled = true")
    lora_config = peft.LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora"]["rank"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        bias=str(config["lora"]["bias"]),
        target_modules=list(config["lora"]["target_modules"]),
    )
    return peft.get_peft_model(model, lora_config)


def _assert_trainable_parameters(model: Any, expected_device_type: str) -> None:
    trainable = [(name, param) for name, param in model.named_parameters() if getattr(param, "requires_grad", False)]
    if not trainable:
        raise ValueError("LoRA model has zero trainable parameters")
    wrong_device = [
        name
        for name, param in trainable
        if getattr(getattr(param, "device", None), "type", None) != expected_device_type
    ]
    if wrong_device:
        raise ValueError(f"Trainable parameters are not on {expected_device_type}: {wrong_device[:3]}")


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name!r}")
    return mapping[dtype_name]


def _lora_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(config["lora"]["rank"]),
        "alpha": int(config["lora"]["alpha"]),
        "dropout": float(config["lora"]["dropout"]),
        "bias": str(config["lora"]["bias"]),
        "target_modules": list(config["lora"]["target_modules"]),
    }
