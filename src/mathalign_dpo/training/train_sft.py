"""Stage 3 SFT training orchestration."""

from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.training.model_loader import load_model_and_tokenizer, load_tokenizer, model_revision_metadata, validate_runtime_backend
from mathalign_dpo.training.runtime_metadata import (
    RunClock,
    build_run_id,
    collect_base_metadata,
    finalize_metadata,
    write_json,
    write_jsonl,
)
from mathalign_dpo.training.sft_data import (
    assert_rows_within_max_length,
    load_sft_candidate_pools,
    select_tokenized_sft_data,
    validate_tokenizer_chat_template,
)


@dataclass(frozen=True)
class RunDirectories:
    """Final and staging directories for one SFT run."""

    final_dir: Path
    staging_dir: Path


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

    original_config = load_single_config(config_path)
    if original_config["project"]["run_mode"] != "mini":
        raise ValueError("Stage 3 actual SFT training only supports project.run_mode = mini")
    run_config, runtime_overrides = _apply_runtime_overrides(
        original_config,
        smoke_test,
        train_samples,
        validation_samples,
        max_steps,
    )
    run_id = build_run_id("sft", smoke_test)
    run_dirs = prepare_staged_output_dir(run_config, run_id, output_dir, overwrite=overwrite)
    clock = RunClock.start()
    metadata = collect_base_metadata(
        config=run_config,
        config_path=config_path,
        output_dir=run_dirs.final_dir,
        run_id=run_id,
        stage_number=3,
        training_stage="sft",
        run_mode=str(run_config["project"]["run_mode"]),
        smoke_test=smoke_test,
        runtime_overrides=runtime_overrides,
    )
    progress_metadata: dict[str, Any] = {}

    try:
        candidate_pools = load_sft_candidate_pools(run_config)
        progress_metadata["candidate_counts"] = candidate_pools.candidate_counts
        backend_metadata = validate_runtime_backend(run_config)
        progress_metadata["backend_preflight"] = backend_metadata
        tokenizer = load_tokenizer(run_config)
        tokenizer_metadata = validate_tokenizer_chat_template(tokenizer)
        target_counts = _target_counts(run_config)
        tokenized = select_tokenized_sft_data(
            candidate_pools,
            tokenizer,
            max_length=int(run_config["model"]["max_length"]),
            seed=int(run_config["project"]["seed"]),
            target_train_count=target_counts["train"],
            target_validation_count=target_counts["validation"],
        )
        progress_metadata["dataset_counts"] = _dataset_counts(candidate_pools, tokenized)
        loaded = load_model_and_tokenizer(run_config, training_stage="sft")
        validate_tokenizer_chat_template(loaded.tokenizer)
        train_metrics, eval_metrics = _train_with_trl(run_config, loaded.model, loaded.tokenizer, tokenized, run_dirs.staging_dir)
        final_adapter_dir = run_dirs.staging_dir / "final_adapter"
        tokenizer_dir = run_dirs.staging_dir / "tokenizer"
        loaded.model.save_pretrained(final_adapter_dir)
        loaded.tokenizer.save_pretrained(tokenizer_dir)
        reload_samples = reload_adapter_and_generate(
            run_config,
            final_adapter_dir,
            tokenizer_dir,
            tokenized.validation_rows,
            sample_count=int(run_config["sft"]["adapter_reload_samples"]),
            max_new_tokens=int(run_config["sft"]["adapter_reload_max_new_tokens"]),
        )
        if not reload_samples:
            raise ValueError("Adapter reload validation produced no samples")
        write_jsonl(run_dirs.staging_dir / "adapter_reload_samples.jsonl", reload_samples)

        completed = finalize_metadata(
            metadata,
            clock,
            "completed",
            {
                "dataset_counts": _dataset_counts(candidate_pools, tokenized),
                "tokenizer": tokenizer_metadata,
                "token_statistics": tokenized.token_statistics,
                "backend_preflight": backend_metadata,
                "model_loader": loaded.metadata,
                "model_revision": model_revision_metadata(run_config),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "metrics": {"train": train_metrics, "eval": eval_metrics},
                "artifacts": {
                    "final_adapter": str(run_dirs.final_dir / "final_adapter"),
                    "tokenizer": str(run_dirs.final_dir / "tokenizer"),
                    "trainer_state": str(run_dirs.final_dir / "trainer_state.json"),
                    "train_metrics": str(run_dirs.final_dir / "train_metrics.json"),
                    "eval_metrics": str(run_dirs.final_dir / "eval_metrics.json"),
                    "loss_history": str(run_dirs.final_dir / "loss_history.jsonl"),
                    "adapter_reload_samples": str(run_dirs.final_dir / "adapter_reload_samples.jsonl"),
                },
            },
        )
        write_json(run_dirs.staging_dir / "run_metadata.json", completed)
        publish_staged_output(run_dirs, overwrite=overwrite)
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
        write_json(run_dirs.staging_dir / "run_metadata.json", failed)
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
        return
    run_dir.mkdir(parents=True, exist_ok=True)


def prepare_staged_output_dir(
    config: Mapping[str, Any],
    run_id: str,
    output_dir: str | Path | None,
    overwrite: bool,
) -> RunDirectories:
    """Prepare a staging directory without deleting old outputs."""

    final_dir = resolve_output_dir(config, run_id, output_dir)
    if final_dir.exists() and any(final_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty SFT output directory: {final_dir}")
    staging_dir = final_dir.parent / f".{final_dir.name}.{run_id}.staging"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    return RunDirectories(final_dir=final_dir, staging_dir=staging_dir)


def publish_staged_output(run_dirs: RunDirectories, overwrite: bool) -> None:
    """Publish a completed staging directory while preserving old output on failure."""

    final_dir = run_dirs.final_dir
    staging_dir = run_dirs.staging_dir
    backup_dir = final_dir.parent / f".{final_dir.name}.backup"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    if final_dir.exists() and any(final_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty SFT output directory: {final_dir}")
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


def reload_adapter_and_generate(
    config: Mapping[str, Any],
    adapter_dir: Path,
    tokenizer_dir: Path,
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
    validate_tokenizer_chat_template(tokenizer)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(config["model"]["name_or_path"]),
        revision=str(config["model"]["revision"]),
        dtype=dtype,
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = True
    model = peft.PeftModel.from_pretrained(base_model, adapter_dir)
    model.to("mps")
    model.eval()

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


def _train_with_trl(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    tokenized: Any,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")
    assert_rows_within_max_length(tokenized.train_rows, int(config["model"]["max_length"]), "train")
    assert_rows_within_max_length(tokenized.validation_rows, int(config["model"]["max_length"]), "validation")
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
    assert_trainer_input_lengths(trainer, int(config["model"]["max_length"]))
    train_result = trainer.train()
    train_metrics = dict(getattr(train_result, "metrics", {}) or {})
    eval_metrics = dict(trainer.evaluate())
    trainer.save_state()
    _write_trainer_artifacts(trainer, train_metrics, eval_metrics, run_dir)
    return train_metrics, eval_metrics


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


def assert_trainer_input_lengths(trainer: Any, max_length: int) -> None:
    """Verify tokenized Trainer datasets respect max_length."""

    for label, dataset in (("train", trainer.train_dataset), ("eval", trainer.eval_dataset)):
        too_long: list[int] = []
        for index in range(len(dataset)):
            row = dataset[index]
            input_ids = row.get("input_ids") if isinstance(row, Mapping) else None
            if input_ids is not None and len(input_ids) > max_length:
                too_long.append(index)
        if too_long:
            raise ValueError(f"Trainer {label} dataset contains rows longer than max_length={max_length}: {too_long[:3]}")


def _write_trainer_artifacts(
    trainer: Any,
    train_metrics: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
    run_dir: Path,
) -> None:
    state_path = run_dir / "trainer_state.json"
    if hasattr(trainer.state, "save_to_json"):
        trainer.state.save_to_json(str(state_path))
    elif hasattr(trainer.state, "to_json_string"):
        state_path.write_text(trainer.state.to_json_string(), encoding="utf-8")
    else:
        write_json(state_path, dict(getattr(trainer.state, "__dict__", {})))
    write_json(run_dir / "train_metrics.json", dict(train_metrics))
    write_json(run_dir / "eval_metrics.json", dict(eval_metrics))
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    loss_rows = [row for row in log_history if "loss" in row or "eval_loss" in row]
    write_jsonl(run_dir / "loss_history.jsonl", loss_rows)


def _apply_runtime_overrides(
    config: Mapping[str, Any],
    smoke_test: bool,
    train_samples: int | None,
    validation_samples: int | None,
    max_steps: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    copied = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    overrides: dict[str, Any] = {
        "smoke_test": bool(smoke_test),
        "cli": {
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "max_steps": max_steps,
        },
        "applied": {},
    }
    if smoke_test:
        copied["sft"]["max_steps"] = int(config["smoke_test"]["max_steps"])
        copied["data"]["train_samples"] = int(config["smoke_test"]["train_samples"])
        copied["data"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
        overrides["applied"]["sft.max_steps"] = copied["sft"]["max_steps"]
        overrides["applied"]["data.train_samples"] = copied["data"]["train_samples"]
        overrides["applied"]["data.validation_samples"] = copied["data"]["validation_samples"]
    if train_samples is not None:
        copied["data"]["train_samples"] = int(train_samples)
        overrides["applied"]["data.train_samples"] = int(train_samples)
    if validation_samples is not None:
        copied["data"]["validation_samples"] = int(validation_samples)
        overrides["applied"]["data.validation_samples"] = int(validation_samples)
    if max_steps is not None:
        copied["sft"]["max_steps"] = int(max_steps)
        overrides["applied"]["sft.max_steps"] = int(max_steps)
    return copied, overrides


def _target_counts(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        "train": int(config["data"]["train_samples"]),
        "validation": int(config["data"]["validation_samples"]),
    }


def _dataset_counts(candidate_pools: Any, tokenized: Any) -> dict[str, Any]:
    return {
        "candidate_pools": candidate_pools.candidate_counts,
        "final_actual": {
            "train": len(tokenized.train_rows),
            "validation": len(tokenized.validation_rows),
        },
        "after_token_filter": {
            "train": int(tokenized.token_statistics["train"]["kept_count"]),
            "validation": int(tokenized.token_statistics["validation"]["kept_count"]),
        },
    }


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
