"""Stage 3 SFT training orchestration."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.training.model_loader import load_model_and_tokenizer, load_tokenizer, validate_runtime_backend
from mathalign_dpo.training.runtime_metadata import (
    RunClock,
    build_run_id,
    collect_base_metadata,
    finalize_metadata,
    write_json,
    write_jsonl,
)
from mathalign_dpo.training.sft_data import load_sft_data, tokenize_and_filter_sft_data, validate_tokenizer_chat_template


def train_sft_from_config(
    config_path: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    train_samples: int | None = None,
    validation_samples: int | None = None,
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 3 Mini SFT from one YAML config."""

    config = load_single_config(config_path)
    if config["project"]["run_mode"] != "mini":
        raise ValueError("Stage 3 actual SFT training only supports project.run_mode = mini")
    run_id = build_run_id("sft", smoke_test)
    run_dir = resolve_output_dir(config, run_id, output_dir)
    prepare_output_dir(run_dir, overwrite=overwrite)
    clock = RunClock.start()
    metadata = collect_base_metadata(config, config_path, run_dir, run_id, smoke_test)
    progress_metadata: dict[str, Any] = {}

    try:
        run_config = _apply_runtime_overrides(config, smoke_test, train_samples, validation_samples, max_steps)
        sft_data = load_sft_data(
            run_config,
            train_limit=_effective_train_limit(config, smoke_test, train_samples),
            validation_limit=_effective_validation_limit(config, smoke_test, validation_samples),
        )
        progress_metadata["dataset_counts"] = {"selected": sft_data.selected_counts}
        backend_metadata = validate_runtime_backend(run_config)
        progress_metadata["backend_preflight"] = backend_metadata
        tokenizer = load_tokenizer(run_config)
        tokenizer_metadata = validate_tokenizer_chat_template(tokenizer)
        tokenized = tokenize_and_filter_sft_data(
            sft_data.train_rows,
            sft_data.validation_rows,
            tokenizer,
            max_length=int(run_config["model"]["max_length"]),
        )
        loaded = load_model_and_tokenizer(run_config, training_stage="sft")
        validate_tokenizer_chat_template(loaded.tokenizer)
        metrics = _train_with_trl(run_config, loaded.model, loaded.tokenizer, tokenized, run_dir)
        final_adapter_dir = run_dir / "final_adapter"
        tokenizer_dir = run_dir / "tokenizer"
        loaded.model.save_pretrained(final_adapter_dir)
        loaded.tokenizer.save_pretrained(tokenizer_dir)
        reload_samples = reload_adapter_and_generate(
            run_config,
            final_adapter_dir,
            tokenized.validation_rows,
            sample_count=int(run_config["sft"]["adapter_reload_samples"]),
            max_new_tokens=int(run_config["sft"]["adapter_reload_max_new_tokens"]),
        )
        write_jsonl(run_dir / "adapter_reload_samples.jsonl", reload_samples)

        completed = finalize_metadata(
            metadata,
            clock,
            "completed",
            {
                "dataset_counts": {
                    "selected": sft_data.selected_counts,
                    "after_token_filter": {
                        "train": len(tokenized.train_rows),
                        "validation": len(tokenized.validation_rows),
                    },
                },
                "tokenizer": tokenizer_metadata,
                "token_statistics": tokenized.token_statistics,
                "backend_preflight": backend_metadata,
                "model_loader": loaded.metadata,
                "metrics": metrics,
                "artifacts": {
                    "final_adapter": str(final_adapter_dir),
                    "tokenizer": str(tokenizer_dir),
                    "trainer_state": str(run_dir / "trainer_state.json"),
                    "train_metrics": str(run_dir / "train_metrics.json"),
                    "loss_history": str(run_dir / "loss_history.jsonl"),
                    "adapter_reload_samples": str(run_dir / "adapter_reload_samples.jsonl"),
                },
            },
        )
        write_json(run_dir / "run_metadata.json", completed)
        return completed
    except BaseException as exc:
        failed = finalize_metadata(
            metadata,
            clock,
            "failed",
            {
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                **progress_metadata,
            },
        )
        write_json(run_dir / "run_metadata.json", failed)
        raise


def resolve_output_dir(config: Mapping[str, Any], run_id: str, output_dir: str | Path | None) -> Path:
    """Resolve the run output directory."""

    if output_dir is not None:
        return Path(output_dir)
    return Path(str(config["sft"]["output_dir"])) / run_id


def prepare_output_dir(run_dir: Path, overwrite: bool) -> None:
    """Create an output directory with collision protection."""

    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty SFT output directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def reload_adapter_and_generate(
    config: Mapping[str, Any],
    adapter_dir: Path,
    validation_rows: list[Mapping[str, Any]],
    sample_count: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Reload the saved adapter and generate deterministic validation samples."""

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    backend = str(config["runtime"]["backend"])
    dtype = torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.bfloat16
    if backend != "mps":
        raise ValueError("Stage 3 adapter reload validation only runs for Mini MPS")

    tokenizer = load_tokenizer(config)
    validate_tokenizer_chat_template(tokenizer)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(config["model"]["name_or_path"]),
        torch_dtype=dtype,
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = True
    model = peft.PeftModel.from_pretrained(base_model, adapter_dir)
    model.to("mps")
    model.eval()
    _configure_deterministic_generation(model)

    outputs: list[dict[str, Any]] = []
    for row in validation_rows[:sample_count]:
        prompt = list(row["prompt"])
        encoded = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        encoded = encoded.to("mps")
        with torch.no_grad():
            generated = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][encoded.shape[-1] :]
        outputs.append(
            {
                "id": str(row["id"]),
                "source_id": str(row["source_id"]),
                "prompt": prompt,
                "generated_text": tokenizer.decode(new_tokens, skip_special_tokens=True),
                "adapter_path": str(adapter_dir),
                "max_new_tokens": int(max_new_tokens),
            }
        )
    return outputs


def _configure_deterministic_generation(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = False
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(generation_config, name):
            setattr(generation_config, name, None)


def _train_with_trl(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    tokenized: Any,
    run_dir: Path,
) -> dict[str, Any]:
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")
    train_dataset = datasets.Dataset.from_list(tokenized.train_rows)
    eval_dataset = datasets.Dataset.from_list(tokenized.validation_rows)
    args = _sft_config(trl, config, run_dir)
    trainer = trl.SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    trainer.save_state()
    _write_trainer_artifacts(trainer, metrics, run_dir)
    return metrics


def _sft_config(trl: Any, config: Mapping[str, Any], run_dir: Path) -> Any:
    sft = config["sft"]
    runtime = config["runtime"]
    backend = str(runtime["backend"])
    eos_token = "<|im_end|>" if "Qwen" in str(config["model"]["name_or_path"]) else None
    kwargs = {
        "output_dir": str(run_dir),
        "max_length": int(config["model"]["max_length"]),
        "packing": bool(sft["packing"]),
        "completion_only_loss": True,
        "learning_rate": float(sft["learning_rate"]),
        "num_train_epochs": float(sft["num_train_epochs"]),
        "max_steps": int(sft["max_steps"]),
        "per_device_train_batch_size": int(sft["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(sft["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(sft["gradient_accumulation_steps"]),
        "warmup_ratio": float(sft["warmup_ratio"]),
        "weight_decay": float(sft["weight_decay"]),
        "max_grad_norm": float(sft["max_grad_norm"]),
        "optim": str(sft["optimizer"]),
        "lr_scheduler_type": str(sft["lr_scheduler_type"]),
        "eval_strategy": str(sft["eval_strategy"]),
        "eval_steps": int(sft["eval_steps"]),
        "save_strategy": str(sft["save_strategy"]),
        "save_steps": int(sft["save_steps"]),
        "save_total_limit": int(sft["save_total_limit"]),
        "logging_steps": int(sft["logging_steps"]),
        "report_to": str(runtime["report_to"]),
        "dataloader_num_workers": int(runtime["dataloader_num_workers"]),
        "dataloader_pin_memory": bool(runtime["pin_memory"]),
        "seed": int(config["project"]["seed"]),
        "fp16": backend == "mps" and str(runtime["mixed_precision"]) == "fp16",
        "bf16": backend == "cuda" and str(runtime["mixed_precision"]) == "bf16",
    }
    if eos_token is not None:
        kwargs["eos_token"] = eos_token
    return trl.SFTConfig(**kwargs)


def _write_trainer_artifacts(trainer: Any, metrics: Mapping[str, Any], run_dir: Path) -> None:
    state_path = run_dir / "trainer_state.json"
    if hasattr(trainer.state, "save_to_json"):
        trainer.state.save_to_json(str(state_path))
    elif hasattr(trainer.state, "to_json_string"):
        state_path.write_text(trainer.state.to_json_string(), encoding="utf-8")
    else:
        write_json(state_path, dict(getattr(trainer.state, "__dict__", {})))
    write_json(run_dir / "train_metrics.json", dict(metrics))
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    loss_rows = [row for row in log_history if "loss" in row or "eval_loss" in row]
    write_jsonl(run_dir / "loss_history.jsonl", loss_rows)


def _apply_runtime_overrides(
    config: Mapping[str, Any],
    smoke_test: bool,
    train_samples: int | None,
    validation_samples: int | None,
    max_steps: int | None,
) -> dict[str, Any]:
    copied = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    if smoke_test:
        copied["sft"]["max_steps"] = int(config["smoke_test"]["max_steps"])
    if max_steps is not None:
        copied["sft"]["max_steps"] = int(max_steps)
    return copied


def _effective_train_limit(config: Mapping[str, Any], smoke_test: bool, train_samples: int | None) -> int | None:
    if train_samples is not None:
        return int(train_samples)
    if smoke_test:
        return int(config["smoke_test"]["train_samples"])
    return None


def _effective_validation_limit(config: Mapping[str, Any], smoke_test: bool, validation_samples: int | None) -> int | None:
    if validation_samples is not None:
        return int(validation_samples)
    if smoke_test:
        return int(config["smoke_test"]["validation_samples"])
    return None


def cli_payload(result: Mapping[str, Any]) -> str:
    """Serialize the compact CLI result."""

    summary = {
        "status": result["status"],
        "run_id": result["run_id"],
        "output_dir": result["output_dir"],
        "elapsed_seconds": result["elapsed_seconds"],
        "dataset_counts": result.get("dataset_counts"),
        "artifacts": result.get("artifacts"),
        "error": result.get("error"),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
