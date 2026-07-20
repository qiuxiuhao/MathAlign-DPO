"""Model, tokenizer, and LoRA helpers for SFT."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class LoadedModel:
    """Loaded model/tokenizer pair and metadata."""

    model: Any
    tokenizer: Any
    metadata: dict[str, Any]


def ensure_local_model(config: Mapping[str, Any]) -> Path:
    """Ensure the configured local model directory exists and has core files."""

    local_dir = Path(str(config["model"]["name_or_path"]))
    if _has_model_files(local_dir):
        return local_dir
    remote = str(config["model"].get("modelscope_name_or_path") or config["model"].get("remote_name_or_path") or "Qwen/Qwen2.5-0.5B-Instruct")
    revision = str(config["model"].get("modelscope_revision") or config["model"].get("revision") or "master")
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Downloading the local model requires the 'modelscope' package") from exc
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id=remote,
        revision=revision,
        local_dir=str(local_dir),
    )
    if not _has_model_files(local_dir):
        raise FileNotFoundError(f"Model download did not create required files in {local_dir}")
    return local_dir


def load_tokenizer(config: Mapping[str, Any]) -> Any:
    """Load the tokenizer from the local model directory."""

    transformers = _import_transformers()
    model_dir = ensure_local_model(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    validate_tokenizer(tokenizer)
    return tokenizer


def load_lora_model_and_tokenizer(config: Mapping[str, Any]) -> LoadedModel:
    """Load the configured base model and attach a fresh LoRA adapter."""

    runtime = validate_runtime(config)
    torch = _import_torch()
    transformers = _import_transformers()
    peft = _import_peft()
    model_dir = ensure_local_model(config)
    tokenizer = load_tokenizer(config)
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
    model = peft.get_peft_model(model, _lora_config(peft, config))
    if runtime["backend"] != "cuda":
        model.to(runtime["device"])
    assert_trainable_parameters(model, expected_device_type=str(runtime["backend"]))
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        metadata={
            "model_dir": str(model_dir),
            "modelscope_name_or_path": str(config["model"].get("modelscope_name_or_path") or ""),
            "remote_name_or_path": str(config["model"].get("remote_name_or_path") or ""),
            "revision": str(config["model"]["revision"]),
            "backend": runtime["backend"],
            "device": runtime["device"],
            "torch_dtype": str(config["model"]["torch_dtype"]),
            "lora": {
                "rank": int(config["lora"]["rank"]),
                "alpha": int(config["lora"]["alpha"]),
                "dropout": float(config["lora"]["dropout"]),
                "target_modules": list(config["lora"]["target_modules"]),
            },
        },
    )


def load_base_for_generation(config: Mapping[str, Any], tokenizer_dir: str | os.PathLike[str] | None = None) -> LoadedModel:
    """Load the base model for deterministic generation."""

    runtime = validate_runtime(config)
    torch = _import_torch()
    transformers = _import_transformers()
    model_dir = ensure_local_model(config)
    tokenizer_source = tokenizer_dir if tokenizer_dir is not None else model_dir
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source)
    validate_tokenizer(tokenizer)
    model = _load_base_model(config, model_dir, torch, transformers)
    if hasattr(model, "config"):
        model.config.use_cache = True
    if runtime["backend"] != "cuda":
        model.to(runtime["device"])
    configure_deterministic_generation(model)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, metadata={"model_dir": str(model_dir), **runtime})


def load_sft_for_generation(
    config: Mapping[str, Any],
    adapter_dir: str | os.PathLike[str],
    tokenizer_dir: str | os.PathLike[str],
) -> LoadedModel:
    """Load base + SFT adapter for deterministic generation."""

    peft = _import_peft()
    runtime = runtime_metadata(config)
    loaded = load_base_for_generation(config, tokenizer_dir=tokenizer_dir)
    model = peft.PeftModel.from_pretrained(loaded.model, adapter_dir, is_trainable=False)
    if runtime["backend"] != "cuda":
        model.to(runtime["device"])
    configure_deterministic_generation(model)
    model.eval()
    metadata = dict(loaded.metadata)
    metadata["adapter_dir"] = str(adapter_dir)
    return LoadedModel(model=model, tokenizer=loaded.tokenizer, metadata=metadata)


def validate_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail fast if the configured runtime is unavailable."""

    torch = _import_torch()
    backend = str(config["runtime"]["backend"])
    device = str(config["runtime"].get("device", backend))
    if backend == "mps":
        built = bool(torch.backends.mps.is_built())
        available = bool(torch.backends.mps.is_available())
        if not built or not available:
            raise RuntimeError(f"MPS backend unavailable: is_built={built}, is_available={available}")
        return {
            "backend": "mps",
            "device": device,
            "mps_is_built": built,
            "mps_is_available": available,
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
        }
    if backend == "cuda":
        available = bool(torch.cuda.is_available())
        if not available:
            raise RuntimeError("CUDA backend unavailable: torch.cuda.is_available()=False")
        return {"backend": "cuda", "device": device, "cuda_is_available": available}
    raise ValueError(f"runtime.backend must be mps or cuda, got {backend!r}")


def runtime_metadata(config: Mapping[str, Any]) -> dict[str, str]:
    """Return configured backend/device without checking hardware availability."""

    backend = str(config["runtime"]["backend"])
    return {"backend": backend, "device": str(config["runtime"].get("device", backend))}


def validate_tokenizer(tokenizer: Any) -> dict[str, Any]:
    """Require a chat template and usable pad token."""

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer must provide a chat_template")
    pad_token_before = getattr(tokenizer, "pad_token", None)
    if pad_token_before is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = eos_token
    return {
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest(),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def configure_deterministic_generation(model: Any) -> None:
    """Remove sampling defaults from model generation config for greedy decoding."""

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = False
    for key in ("temperature", "top_p", "top_k"):
        if hasattr(generation_config, key):
            setattr(generation_config, key, None)


def assert_trainable_parameters(model: Any, expected_device_type: str) -> None:
    """Fail if the LoRA model has no trainable parameters on the expected device."""

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


def _load_base_model(config: Mapping[str, Any], model_dir: Path, torch: Any, transformers: Any) -> Any:
    backend = str(config["runtime"]["backend"])
    if backend == "cuda" and bool(config.get("quantization", {}).get("load_in_4bit")):
        quantization = config["quantization"]
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization["quant_type"]),
            bnb_4bit_use_double_quant=bool(quantization["use_double_quant"]),
            bnb_4bit_compute_dtype=_torch_dtype(torch, str(quantization["compute_dtype"])),
        )
        return transformers.AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=quantization_config,
            device_map={"": 0},
            trust_remote_code=bool(config["model"]["trust_remote_code"]),
        )
    return transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=_torch_dtype(torch, str(config["model"]["torch_dtype"])),
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        low_cpu_mem_usage=True,
    )


def _lora_config(peft: Any, config: Mapping[str, Any]) -> Any:
    return peft.LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora"]["rank"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        bias=str(config["lora"]["bias"]),
        target_modules=list(config["lora"]["target_modules"]),
    )


def _has_model_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    config = path / "config.json"
    tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")) or (path / "model.safetensors.index.json").exists()
    return config.exists() and tokenizer and weights


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _import_torch() -> Any:
    import torch

    return torch


def _import_transformers() -> Any:
    import transformers

    return transformers


def _import_peft() -> Any:
    import peft

    return peft
