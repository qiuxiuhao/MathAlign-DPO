"""Stage 4 DPO training orchestration."""

from __future__ import annotations

import importlib
import gc
import json
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.training.dpo_data import (
    TokenizedDPOData,
    assert_dpo_rows_within_limits,
    load_dpo_candidate_pools,
    select_tokenized_dpo_data,
)
from mathalign_dpo.training.model_loader import (
    load_tokenizer,
    load_policy_model_from_sft_adapter,
    model_revision_metadata,
    validate_runtime_backend,
)
from mathalign_dpo.training.runtime_metadata import (
    RunClock,
    build_run_id,
    collect_base_metadata,
    finalize_metadata,
    json_safe,
    write_json,
    write_jsonl,
)
from mathalign_dpo.training.sft_data import validate_tokenizer_chat_template
from mathalign_dpo.training.train_sft import RunDirectories, publish_staged_output


def train_dpo_from_config(
    config_path: str | Path,
    sft_run_dir: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    train_samples: int | None = None,
    validation_samples: int | None = None,
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 4 Mini DPO from one YAML config and one Stage 3 SFT output."""

    original_config = load_single_config(config_path)
    if original_config["project"]["run_mode"] != "mini":
        raise ValueError("Stage 4 actual DPO training only supports project.run_mode = mini")
    sft_metadata = validate_sft_run_dir(sft_run_dir, original_config, smoke_test=smoke_test)
    run_config, runtime_overrides = _apply_runtime_overrides(
        original_config,
        smoke_test,
        train_samples,
        validation_samples,
        max_steps,
    )
    run_id = build_run_id("dpo", smoke_test, stage_number=4)
    run_dirs = prepare_dpo_staged_output_dir(run_config, run_id, output_dir, overwrite=overwrite)
    clock = RunClock.start()
    metadata = collect_base_metadata(
        config=run_config,
        config_path=config_path,
        output_dir=run_dirs.final_dir,
        run_id=run_id,
        stage_number=4,
        training_stage="dpo",
        run_mode=str(run_config["project"]["run_mode"]),
        smoke_test=smoke_test,
        runtime_overrides=runtime_overrides,
    )
    progress_metadata: dict[str, Any] = {"sft_source": sft_metadata}

    try:
        candidate_pools = load_dpo_candidate_pools(run_config)
        progress_metadata["candidate_counts"] = candidate_pools.candidate_counts
        backend_metadata = validate_runtime_backend(run_config)
        progress_metadata["backend_preflight"] = backend_metadata
        tokenizer = load_tokenizer(run_config)
        tokenizer_metadata = validate_tokenizer_chat_template(tokenizer)
        target_counts = _target_counts(run_config)
        tokenized = select_tokenized_dpo_data(
            candidate_pools,
            tokenizer,
            max_length=int(run_config["dpo"]["max_length"]),
            max_prompt_length=int(run_config["dpo"]["max_prompt_length"]),
            seed=int(run_config["project"]["seed"]),
            target_train_count=target_counts["train"],
            target_validation_count=target_counts["validation"],
        )
        progress_metadata["dataset_counts"] = _dataset_counts(candidate_pools, tokenized)

        loaded = load_policy_model_from_sft_adapter(run_config, Path(sft_run_dir) / "final_adapter")
        validate_tokenizer_chat_template(loaded.tokenizer)
        model_loader_metadata = dict(loaded.metadata)
        train_metrics, eval_metrics, preference_rows = _train_with_trl(
            run_config,
            loaded.model,
            loaded.tokenizer,
            tokenized,
            run_dirs.staging_dir,
        )
        final_adapter_dir = run_dirs.staging_dir / "final_adapter"
        tokenizer_dir = run_dirs.staging_dir / "tokenizer"
        loaded.model.save_pretrained(final_adapter_dir)
        loaded.tokenizer.save_pretrained(tokenizer_dir)
        del loaded
        _release_device_memory(run_config)
        reload_samples = reload_dpo_adapter_and_generate(
            run_config,
            final_adapter_dir,
            tokenizer_dir,
            tokenized.validation_rows,
            sample_count=int(run_config["dpo"]["adapter_reload_samples"]),
            max_new_tokens=int(run_config["dpo"]["adapter_reload_max_new_tokens"]),
        )
        if not reload_samples:
            raise ValueError("DPO adapter reload validation produced no samples")
        write_jsonl(run_dirs.staging_dir / "adapter_reload_samples.jsonl", reload_samples)
        write_jsonl(run_dirs.staging_dir / "preference_validation.jsonl", preference_rows)

        completed = finalize_metadata(
            metadata,
            clock,
            "completed",
            {
                "sft_source": sft_metadata,
                "dataset_counts": _dataset_counts(candidate_pools, tokenized),
                "tokenizer": tokenizer_metadata,
                "token_statistics": tokenized.token_statistics,
                "backend_preflight": backend_metadata,
                "model_loader": model_loader_metadata,
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
                    "preference_validation": str(run_dirs.final_dir / "preference_validation.jsonl"),
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


def validate_sft_run_dir(sft_run_dir: str | Path, config: Mapping[str, Any], smoke_test: bool) -> dict[str, Any]:
    """Validate that a Stage 3 output can initialize Stage 4 DPO."""

    root = Path(sft_run_dir)
    metadata_path = root / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage 3 SFT metadata is missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "completed":
        raise ValueError(f"Stage 3 SFT run is not completed: {root}")
    if int(metadata.get("stage", -1)) != 3 or metadata.get("training_stage") != "sft":
        raise ValueError(f"SFT run directory must contain a Stage 3 SFT run: {root}")
    run_mode = metadata.get("run_mode") or metadata.get("project", {}).get("run_mode")
    if run_mode != "mini":
        raise ValueError(f"Stage 4 Mini DPO requires a Mini SFT adapter: {root}")
    if not smoke_test and metadata.get("smoke_test") is True:
        raise ValueError("Normal Stage 4 DPO cannot initialize from a smoke SFT adapter")

    _assert_metadata_matches_config(metadata, config)
    final_train_count = _sft_final_train_count(metadata)
    if not smoke_test and final_train_count != 256:
        raise ValueError(f"Normal Stage 4 DPO requires a 256-row Stage 3 SFT run, got {final_train_count}")
    if smoke_test and final_train_count <= 0:
        raise ValueError("Smoke Stage 4 DPO requires a Stage 3 SFT run with at least one selected train row")

    adapter_dir = root / "final_adapter"
    tokenizer_dir = root / "tokenizer"
    for path in [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_config.json"]:
        if not path.exists():
            raise FileNotFoundError(f"Stage 3 SFT adapter artifact is missing: {path}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Stage 3 tokenizer directory is missing: {tokenizer_dir}")
    return {
        "run_dir": str(root),
        "run_id": metadata.get("run_id"),
        "git_commit": metadata.get("git_commit"),
        "smoke_test": bool(metadata.get("smoke_test")),
        "final_train_count": final_train_count,
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
    }


def resolve_dpo_output_dir(config: Mapping[str, Any], run_id: str, output_dir: str | Path | None) -> Path:
    """Resolve the DPO run output directory."""

    if output_dir is not None:
        return Path(output_dir)
    return Path(str(config["dpo"]["output_dir"])) / run_id


def prepare_dpo_staged_output_dir(
    config: Mapping[str, Any],
    run_id: str,
    output_dir: str | Path | None,
    overwrite: bool,
) -> RunDirectories:
    """Prepare a staging directory without deleting old DPO outputs."""

    final_dir = resolve_dpo_output_dir(config, run_id, output_dir)
    if final_dir.exists() and any(final_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty DPO output directory: {final_dir}")
    staging_dir = final_dir.parent / f".{final_dir.name}.{run_id}.staging"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    return RunDirectories(final_dir=final_dir, staging_dir=staging_dir)


def assert_dpo_trainer_input_lengths(trainer: Any, max_length: int) -> None:
    """Verify tokenized DPO Trainer datasets respect max_length."""

    for label, dataset in (("train", trainer.train_dataset), ("eval", trainer.eval_dataset)):
        too_long: list[int] = []
        for index in range(len(dataset)):
            row = dataset[index]
            prompt_ids = row.get("prompt_ids") if isinstance(row, Mapping) else None
            chosen_ids = row.get("chosen_ids") if isinstance(row, Mapping) else None
            rejected_ids = row.get("rejected_ids") if isinstance(row, Mapping) else None
            if prompt_ids is None or chosen_ids is None or rejected_ids is None:
                continue
            if len(prompt_ids) + len(chosen_ids) > max_length or len(prompt_ids) + len(rejected_ids) > max_length:
                too_long.append(index)
        if too_long:
            raise ValueError(f"Trainer {label} DPO dataset contains rows longer than max_length={max_length}: {too_long[:3]}")


def reload_dpo_adapter_and_generate(
    config: Mapping[str, Any],
    adapter_dir: Path,
    tokenizer_dir: Path,
    validation_rows: list[Mapping[str, Any]],
    sample_count: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Reload the saved DPO adapter and generate deterministic validation samples."""

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    backend = str(config["runtime"]["backend"])
    if backend != "mps":
        raise ValueError("Stage 4 adapter reload validation only runs for Mini MPS")
    dtype = torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.bfloat16
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
        del encoded, generated
        _release_device_memory(config)
    return outputs


def _release_device_memory(config: Mapping[str, Any]) -> None:
    """Release Python and backend caches before reload/generation on constrained devices."""

    gc.collect()
    if str(config["runtime"]["backend"]) != "mps":
        return
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return
    empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _train_with_trl(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    tokenized: TokenizedDPOData,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")
    max_length = int(config["dpo"]["max_length"])
    max_prompt_length = int(config["dpo"]["max_prompt_length"])
    assert_dpo_rows_within_limits(tokenized.train_rows, max_length, max_prompt_length, "train")
    assert_dpo_rows_within_limits(tokenized.validation_rows, max_length, max_prompt_length, "validation")
    train_dataset = datasets.Dataset.from_list(_trainer_rows(tokenized.train_rows))
    eval_dataset = datasets.Dataset.from_list(_trainer_rows(tokenized.validation_rows))
    args = _dpo_config(trl, config, run_dir)
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,
    )
    _assert_reference_adapter_not_trainable(trainer.model)
    assert_dpo_trainer_input_lengths(trainer, max_length)
    train_result = trainer.train()
    train_metrics = dict(getattr(train_result, "metrics", {}) or {})
    eval_metrics = dict(trainer.evaluate())
    trainer.save_state()
    _write_trainer_artifacts(trainer, train_metrics, eval_metrics, run_dir)
    preference_rows = _preference_validation_rows(trainer, tokenized.validation_rows, sample_count=3)
    return train_metrics, eval_metrics, preference_rows


def _dpo_config(trl: Any, config: Mapping[str, Any], run_dir: Path) -> Any:
    dpo = config["dpo"]
    runtime = config["runtime"]
    backend = str(runtime["backend"])
    return trl.DPOConfig(
        output_dir=str(run_dir),
        max_length=int(dpo["max_length"]),
        truncation_mode="keep_start",
        precompute_ref_log_probs=False,
        beta=float(dpo["beta"]),
        loss_type=[str(dpo["loss_type"])],
        learning_rate=float(dpo["learning_rate"]),
        num_train_epochs=float(dpo["num_train_epochs"]),
        max_steps=int(dpo["max_steps"]),
        per_device_train_batch_size=int(dpo["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(dpo["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(dpo["gradient_accumulation_steps"]),
        warmup_ratio=float(dpo["warmup_ratio"]),
        weight_decay=float(dpo["weight_decay"]),
        max_grad_norm=float(dpo["max_grad_norm"]),
        optim=str(dpo["optimizer"]),
        lr_scheduler_type=str(dpo["lr_scheduler_type"]),
        eval_strategy=str(dpo["eval_strategy"]),
        eval_steps=int(dpo["eval_steps"]),
        save_strategy=str(dpo["save_strategy"]),
        save_steps=int(dpo["save_steps"]),
        save_total_limit=int(dpo["save_total_limit"]),
        logging_steps=int(dpo["logging_steps"]),
        report_to=str(runtime["report_to"]),
        dataloader_num_workers=int(runtime["dataloader_num_workers"]),
        dataloader_pin_memory=bool(runtime["pin_memory"]),
        seed=int(config["project"]["seed"]),
        fp16=backend == "mps" and str(runtime["mixed_precision"]) == "fp16",
        bf16=backend == "cuda" and str(runtime["mixed_precision"]) == "bf16",
        gradient_checkpointing=bool(config["model"]["gradient_checkpointing"]),
    )


def _trainer_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": list(row["prompt"]),
            "chosen": list(row["chosen"]),
            "rejected": list(row["rejected"]),
        }
        for row in rows
    ]


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
    loss_rows = [row for row in log_history if "loss" in row or "eval_loss" in row or "rewards/accuracies" in row]
    write_jsonl(run_dir / "loss_history.jsonl", loss_rows)


def _preference_validation_rows(trainer: Any, rows: list[Mapping[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    selected = rows[:sample_count]
    output: list[dict[str, Any]] = []
    for row in selected:
        output.append(
            {
                "id": str(row["id"]),
                "source_id": str(row["source_id"]),
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "token_count": dict(row["token_count"]),
                "metrics_source": "trl_eval_metrics",
                "eval_rewards_accuracies": _last_metric(trainer, "eval", "rewards/accuracies"),
                "eval_rewards_margins": _last_metric(trainer, "eval", "rewards/margins"),
            }
        )
    return output


def _last_metric(trainer: Any, mode: str, key: str) -> Any:
    metrics = getattr(trainer, "_metrics", {})
    values = metrics.get(mode, {}).get(key, []) if isinstance(metrics, Mapping) else []
    return values[-1] if values else None


def _assert_reference_adapter_not_trainable(model: Any) -> None:
    bad = [name for name, param in model.named_parameters() if ".ref." in name and getattr(param, "requires_grad", False)]
    if bad:
        raise ValueError(f"Reference adapter parameters must not be trainable: {bad[:3]}")


def _assert_metadata_matches_config(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    pairs = [
        ("model.name_or_path", metadata.get("model", {}).get("name_or_path"), config["model"]["name_or_path"]),
        ("model.revision", metadata.get("model", {}).get("revision"), config["model"]["revision"]),
        ("model.torch_dtype", metadata.get("model", {}).get("torch_dtype"), config["model"]["torch_dtype"]),
        ("runtime.backend", metadata.get("runtime", {}).get("backend"), config["runtime"]["backend"]),
        ("project.seed", metadata.get("seed"), config["project"]["seed"]),
    ]
    for label, actual, expected in pairs:
        if actual != expected:
            raise ValueError(f"Stage 3 SFT metadata mismatch for {label}: expected {expected!r}, got {actual!r}")
    actual_modules = metadata.get("model_loader", {}).get("lora", {}).get("target_modules")
    expected_modules = config["lora"]["target_modules"]
    if actual_modules != expected_modules:
        raise ValueError(f"Stage 3 SFT LoRA target modules mismatch: expected {expected_modules!r}, got {actual_modules!r}")


def _sft_final_train_count(metadata: Mapping[str, Any]) -> int:
    counts = metadata.get("dataset_counts", {})
    if "final_actual" in counts:
        return int(counts["final_actual"].get("train", 0))
    if "after_token_filter" in counts:
        return int(counts["after_token_filter"].get("train", 0))
    if "selected" in counts:
        return int(counts["selected"].get("train", 0))
    return 0


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
        copied["dpo"]["max_steps"] = int(config["smoke_test"]["max_steps"])
        copied["data"]["train_samples"] = int(config["smoke_test"]["dpo_samples"])
        copied["data"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
        overrides["applied"]["dpo.max_steps"] = copied["dpo"]["max_steps"]
        overrides["applied"]["data.train_samples"] = copied["data"]["train_samples"]
        overrides["applied"]["data.validation_samples"] = copied["data"]["validation_samples"]
    if train_samples is not None:
        copied["data"]["train_samples"] = int(train_samples)
        overrides["applied"]["data.train_samples"] = int(train_samples)
    if validation_samples is not None:
        copied["data"]["validation_samples"] = int(validation_samples)
        overrides["applied"]["data.validation_samples"] = int(validation_samples)
    if max_steps is not None:
        copied["dpo"]["max_steps"] = int(max_steps)
        overrides["applied"]["dpo.max_steps"] = int(max_steps)
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
        "sft_source": result.get("sft_source"),
        "artifacts": result.get("artifacts"),
        "error": result.get("error"),
    }
    return json.dumps(json_safe(summary), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
