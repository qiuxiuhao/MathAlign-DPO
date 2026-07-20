"""Stage 4 DPO training orchestration."""

from __future__ import annotations

import importlib
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mathalign_dpo.data.write_outputs import sha256_file
from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.training.dpo_data import (
    TokenizedDPOData,
    assert_dpo_rows_within_limits,
    load_dpo_candidate_pools,
    select_tokenized_dpo_data,
)
from mathalign_dpo.training.model_loader import (
    load_tokenizer_from_dir,
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
from mathalign_dpo.training.run_artifacts import (
    RunDirectories,
    prepare_staged_output_dir as prepare_stage_staged_output_dir,
    publish_staged_output as publish_stage_staged_output,
    resolve_stage_output_dir,
)
from mathalign_dpo.training.sft_data import validate_tokenizer_chat_template


PINNED_DPO_PACKAGE_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "4.57.6",
    "trl": "0.29.1",
    "peft": "0.17.1",
    "accelerate": "1.14.0",
}
REFERENCE_ADAPTER_ATOL = 5e-5


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
        _assert_pinned_dpo_dependency_versions(metadata["software_versions"])
        candidate_pools = load_dpo_candidate_pools(run_config)
        progress_metadata["candidate_counts"] = candidate_pools.candidate_counts
        backend_metadata = validate_runtime_backend(run_config)
        progress_metadata["backend_preflight"] = backend_metadata
        tokenizer = load_tokenizer_from_dir(sft_metadata["tokenizer_dir"])
        tokenizer_metadata = validate_tokenizer_chat_template(tokenizer)
        tokenizer_validation = _validate_sft_tokenizer_metadata(tokenizer_metadata, sft_metadata)
        progress_metadata["tokenizer_validation"] = tokenizer_validation
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
        data_lineage = _data_lineage(run_config, tokenized)
        data_lineage_validation = _validate_data_lineage(data_lineage, sft_metadata)
        progress_metadata["data_lineage"] = data_lineage
        progress_metadata["data_lineage_validation"] = data_lineage_validation

        loaded = load_policy_model_from_sft_adapter(
            run_config,
            Path(sft_metadata["adapter_dir"]),
            tokenizer_dir=Path(sft_metadata["tokenizer_dir"]),
        )
        loaded_tokenizer_metadata = validate_tokenizer_chat_template(loaded.tokenizer)
        _assert_matching_tokenizer_metadata(tokenizer_metadata, loaded_tokenizer_metadata)
        model_loader_metadata = dict(loaded.metadata)
        model_loader_metadata["lora"] = dict(sft_metadata["adapter_config"])
        train_metrics, eval_metrics, preference_rows, stability_report, reference_adapter_validation = _train_with_trl(
            run_config,
            loaded.model,
            loaded.tokenizer,
            tokenized,
            run_dirs.staging_dir,
            Path(sft_metadata["adapter_dir"]),
        )
        final_adapter_dir = run_dirs.staging_dir / "final_adapter"
        tokenizer_dir = run_dirs.staging_dir / "tokenizer"
        loaded.model.save_pretrained(final_adapter_dir, selected_adapters=["default"])
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
                "tokenizer_validation": tokenizer_validation,
                "token_statistics": tokenized.token_statistics,
                "data_lineage": data_lineage,
                "data_lineage_validation": data_lineage_validation,
                "backend_preflight": backend_metadata,
                "model_loader": model_loader_metadata,
                "model_revision": model_revision_metadata(run_config),
                "dpo_trainer_runtime": _dpo_trainer_runtime_metadata(run_config),
                "numerical_stability": stability_report,
                "reference_adapter_validation": reference_adapter_validation,
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
        publish_stage_staged_output(run_dirs, overwrite=overwrite, stage_label="DPO")
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
    adapter_config = _load_and_validate_sft_adapter_config(adapter_dir / "adapter_config.json", metadata, config)
    return {
        "run_dir": str(root),
        "run_id": metadata.get("run_id"),
        "git_commit": metadata.get("git_commit"),
        "smoke_test": bool(metadata.get("smoke_test")),
        "final_train_count": final_train_count,
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "adapter_config": adapter_config,
        "tokenizer": metadata.get("tokenizer", {}),
        "data_lineage": metadata.get("data_lineage", {}),
        "effective_stage2_manifest_file": metadata.get("effective_config", {}).get("data", {}).get("stage2_manifest_file"),
    }


def _load_and_validate_sft_adapter_config(adapter_config_path: Path, metadata: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    with adapter_config_path.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    actual = {
        "rank": int(adapter_config.get("r", 0)),
        "alpha": int(adapter_config.get("lora_alpha", 0)),
        "dropout": float(adapter_config.get("lora_dropout", -1.0)),
        "bias": adapter_config.get("bias"),
        "target_modules": sorted(str(module) for module in adapter_config.get("target_modules", [])),
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        "base_model_revision": adapter_config.get("revision") or metadata.get("model", {}).get("revision"),
        "peft_type": adapter_config.get("peft_type"),
        "task_type": adapter_config.get("task_type"),
    }
    expected = {
        "rank": int(config["lora"]["rank"]),
        "alpha": int(config["lora"]["alpha"]),
        "dropout": float(config["lora"]["dropout"]),
        "bias": str(config["lora"]["bias"]),
        "target_modules": sorted(str(module) for module in config["lora"]["target_modules"]),
        "base_model_name_or_path": str(config["model"]["name_or_path"]),
        "base_model_revision": str(config["model"]["revision"]),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"Stage 3 SFT adapter config mismatch: {mismatches}")
    if actual["peft_type"] != "LORA" or actual["task_type"] != "CAUSAL_LM":
        raise ValueError(f"Stage 3 SFT adapter must be a CAUSAL_LM LoRA adapter: {actual}")
    return actual


def _validate_sft_tokenizer_metadata(current: Mapping[str, Any], sft_source: Mapping[str, Any]) -> dict[str, Any]:
    recorded = sft_source.get("tokenizer", {})
    if not isinstance(recorded, Mapping):
        recorded = {}
    compared: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key in ("vocab_size", "pad_token_after", "pad_token_id", "eos_token", "eos_token_id", "chat_template_sha256"):
        if key not in recorded:
            missing.append(key)
            continue
        compared[key] = {"stage3": recorded[key], "stage4": current.get(key)}
        if recorded[key] != current.get(key):
            raise ValueError(f"Stage 4 tokenizer mismatch for {key}: expected Stage 3 {recorded[key]!r}, got {current.get(key)!r}")
    return {
        "source": "stage3_saved_tokenizer",
        "compared": compared,
        "legacy_missing_stage3_fields": missing,
        "passed": True,
    }


def _assert_matching_tokenizer_metadata(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for key in ("vocab_size", "pad_token_after", "pad_token_id", "eos_token", "eos_token_id", "chat_template_sha256"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Tokenizer changed between DPO filtering and model loading for {key}")


def _data_lineage(config: Mapping[str, Any], tokenized: TokenizedDPOData) -> dict[str, Any]:
    stage2_manifest = Path(str(config["data"]["stage2_manifest_file"]))
    return {
        "stage2_manifest_file": str(stage2_manifest),
        "stage2_manifest_sha256": sha256_file(stage2_manifest),
        "selection_hashes": {
            "train": tokenized.token_statistics["train"].get("selection_hash"),
            "validation": tokenized.token_statistics["validation"].get("selection_hash"),
        },
    }


def _validate_data_lineage(current: Mapping[str, Any], sft_source: Mapping[str, Any]) -> dict[str, Any]:
    sft_lineage = sft_source.get("data_lineage", {})
    if not isinstance(sft_lineage, Mapping):
        sft_lineage = {}
    if isinstance(sft_lineage, Mapping) and sft_lineage.get("stage2_manifest_sha256"):
        if sft_lineage.get("stage2_manifest_sha256") != current.get("stage2_manifest_sha256"):
            raise ValueError("Stage 4 DPO data and Stage 3 SFT adapter use different Stage 2 manifest hashes")
        if sft_lineage.get("stage2_manifest_file") != current.get("stage2_manifest_file"):
            raise ValueError("Stage 4 DPO data and Stage 3 SFT adapter use different Stage 2 manifest paths")
        status = "matched_stage3_hash"
    else:
        sft_path = sft_source.get("effective_stage2_manifest_file")
        if sft_path != current.get("stage2_manifest_file"):
            raise ValueError(
                "Legacy Stage 3 metadata lacks Stage 2 manifest hash and its manifest path does not match current DPO config"
            )
        status = "legacy_stage3_hash_missing_path_matched"
    return {
        "status": status,
        "stage4_stage2_manifest_sha256": current.get("stage2_manifest_sha256"),
        "stage3_stage2_manifest_sha256": sft_lineage.get("stage2_manifest_sha256"),
        "passed": True,
    }


def _assert_pinned_dpo_dependency_versions(versions: Mapping[str, Any]) -> None:
    mismatches = {
        package: {"expected": expected, "actual": versions.get(package)}
        for package, expected in PINNED_DPO_PACKAGE_VERSIONS.items()
        if versions.get(package) != expected
    }
    if mismatches:
        raise RuntimeError(f"Stage 4 DPO requires pinned package versions: {mismatches}")


def resolve_dpo_output_dir(config: Mapping[str, Any], run_id: str, output_dir: str | Path | None) -> Path:
    """Resolve the DPO run output directory."""

    return resolve_stage_output_dir(config, run_id, output_dir, "dpo")


def prepare_dpo_staged_output_dir(
    config: Mapping[str, Any],
    run_id: str,
    output_dir: str | Path | None,
    overwrite: bool,
) -> RunDirectories:
    """Prepare a staging directory without deleting old DPO outputs."""

    return prepare_stage_staged_output_dir(config, run_id, output_dir, overwrite, "dpo", "DPO")


def assert_dpo_trainer_input_lengths(trainer: Any, max_length: int, max_prompt_length: int) -> None:
    """Verify tokenized DPO Trainer datasets respect max_length."""

    for label, dataset in (("train", trainer.train_dataset), ("eval", trainer.eval_dataset)):
        too_long: list[int] = []
        prompt_too_long: list[int] = []
        bad_completion: list[int] = []
        for index in range(len(dataset)):
            row = dataset[index]
            prompt_ids = row.get("prompt_ids") if isinstance(row, Mapping) else None
            chosen_ids = row.get("chosen_ids") if isinstance(row, Mapping) else None
            rejected_ids = row.get("rejected_ids") if isinstance(row, Mapping) else None
            if prompt_ids is None or chosen_ids is None or rejected_ids is None:
                raise ValueError(f"Trainer {label} DPO dataset row {index} is missing prompt_ids/chosen_ids/rejected_ids")
            if len(prompt_ids) > max_prompt_length:
                prompt_too_long.append(index)
            if len(chosen_ids) <= 0 or len(rejected_ids) <= 0:
                bad_completion.append(index)
            if len(prompt_ids) + len(chosen_ids) > max_length or len(prompt_ids) + len(rejected_ids) > max_length:
                too_long.append(index)
        if prompt_too_long:
            raise ValueError(
                f"Trainer {label} DPO dataset contains prompt rows longer than max_prompt_length={max_prompt_length}: "
                f"{prompt_too_long[:3]}"
            )
        if bad_completion:
            raise ValueError(f"Trainer {label} DPO dataset contains empty chosen/rejected completions: {bad_completion[:3]}")
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
    sft_adapter_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")
    max_length = int(config["dpo"]["max_length"])
    max_prompt_length = int(config["dpo"]["max_prompt_length"])
    assert_dpo_rows_within_limits(tokenized.train_rows, max_length, max_prompt_length, "train")
    assert_dpo_rows_within_limits(tokenized.validation_rows, max_length, max_prompt_length, "validation")
    train_dataset = datasets.Dataset.from_list(_trainer_rows(tokenized.train_rows))
    eval_dataset = datasets.Dataset.from_list(_trainer_rows(tokenized.validation_rows))
    args = _dpo_config(trl, config, run_dir)
    callbacks = _dpo_callbacks(config)
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=None,
    )
    _reload_ref_adapter_from_sft(trainer.model, sft_adapter_dir)
    reference_adapter_validation = _validate_reference_adapters(trainer.model)
    assert_dpo_trainer_input_lengths(trainer, max_length, max_prompt_length)
    ref_precompute_report = _precompute_mps_reference_logps(config, trainer)
    reference_adapter_validation["reference_logps"] = ref_precompute_report
    train_result = trainer.train()
    train_metrics = dict(getattr(train_result, "metrics", {}) or {})
    eval_metrics = dict(trainer.evaluate())
    trainer.save_state()
    preference_rows = _preference_validation_rows(trainer, tokenized.validation_rows, sample_count=3)
    stability_report = _assert_numerical_stability(trainer, train_metrics, eval_metrics, preference_rows)
    _write_trainer_artifacts(trainer, train_metrics, eval_metrics, run_dir)
    return train_metrics, eval_metrics, preference_rows, stability_report, reference_adapter_validation


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
        max_grad_norm=0.0 if backend == "mps" else float(dpo["max_grad_norm"]),
        optim=str(dpo["optimizer"]),
        lr_scheduler_type=str(dpo["lr_scheduler_type"]),
        eval_strategy=str(dpo["eval_strategy"]),
        eval_steps=int(dpo["eval_steps"]),
        save_strategy=str(dpo["save_strategy"]),
        save_steps=int(dpo["save_steps"]),
        save_total_limit=int(dpo["save_total_limit"]),
        logging_steps=int(dpo["logging_steps"]),
        logging_nan_inf_filter=False,
        report_to=str(runtime["report_to"]),
        dataloader_num_workers=int(runtime["dataloader_num_workers"]),
        dataloader_pin_memory=bool(runtime["pin_memory"]),
        seed=int(config["project"]["seed"]),
        fp16=backend == "mps" and str(runtime["mixed_precision"]) == "fp16",
        bf16=backend == "cuda" and str(runtime["mixed_precision"]) == "bf16",
        gradient_checkpointing=bool(config["model"]["gradient_checkpointing"]),
    )


def _dpo_callbacks(config: Mapping[str, Any]) -> list[Any]:
    if str(config["runtime"]["backend"]) != "mps":
        return []
    transformers = importlib.import_module("transformers")
    return [_MPSGradNormCallback(transformers.TrainerCallback, float(config["dpo"]["max_grad_norm"]))]


def _dpo_trainer_runtime_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    if str(config["runtime"]["backend"]) != "mps":
        return {"trainer_max_grad_norm": float(config["dpo"]["max_grad_norm"]), "grad_norm_strategy": "trainer_default"}
    return {
        "trainer_max_grad_norm": 0.0,
        "configured_max_grad_norm": float(config["dpo"]["max_grad_norm"]),
        "grad_norm_strategy": "mps_lora_cpu_norm_callback",
        "reason": "avoid non-finite Accelerate/MPS global grad norm on PEFT DPO",
    }


class _MPSGradNormCallback:
    """Compute and clip trainable LoRA gradients without MPS global norm kernels."""

    def __new__(cls, trainer_callback_cls: Any, max_grad_norm: float) -> Any:
        class Callback(trainer_callback_cls):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                self.max_grad_norm = max_grad_norm
                self.last_grad_norm: float | None = None

            def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                model = kwargs.get("model")
                if model is None:
                    self.last_grad_norm = None
                    return None
                norm = _compute_trainable_grad_norm(model)
                self.last_grad_norm = norm
                if norm is not None and math.isfinite(norm) and self.max_grad_norm > 0.0 and norm > self.max_grad_norm:
                    _scale_trainable_gradients(model, self.max_grad_norm / (norm + 1e-12))
                return None

            def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                if self.last_grad_norm is None:
                    return None
                logs = kwargs.get("logs")
                if isinstance(logs, dict) and "grad_norm" in logs:
                    logs["grad_norm"] = self.last_grad_norm
                history = getattr(state, "log_history", None)
                if history and isinstance(history[-1], dict) and "grad_norm" in history[-1]:
                    history[-1]["grad_norm"] = self.last_grad_norm
                return None

        return Callback()


def _compute_trainable_grad_norm(model: Any) -> float | None:
    torch = importlib.import_module("torch")
    total = 0.0
    seen = 0
    for _, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_cpu = grad.detach().float().cpu()
        if not bool(torch.isfinite(grad_cpu).all().item()):
            return float("nan")
        total += float(grad_cpu.pow(2).sum().item())
        seen += 1
    if seen == 0:
        return None
    return math.sqrt(total)


def _scale_trainable_gradients(model: Any, scale: float) -> None:
    for _, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        grad = getattr(param, "grad", None)
        if grad is not None:
            grad.mul_(scale)


def _trainer_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": list(row["prompt"]),
            "chosen": list(row["chosen"]),
            "rejected": list(row["rejected"]),
        }
        for row in rows
    ]


def _precompute_mps_reference_logps(config: Mapping[str, Any], trainer: Any) -> dict[str, Any]:
    """Precompute reference log-probs on MPS after the ref adapter has been validated."""

    if str(config["runtime"]["backend"]) != "mps":
        return {"enabled": False, "reason": "backend_not_mps"}
    precompute = getattr(trainer, "_precompute_ref_logps", None)
    if not callable(precompute):
        raise ValueError("TRL DPOTrainer does not expose _precompute_ref_logps required for MPS DPO stability")
    batch_size = int(getattr(trainer.args, "precompute_ref_batch_size", 0) or trainer.args.per_device_train_batch_size)
    trainer.train_dataset = precompute(trainer.train_dataset, "train", batch_size)
    eval_batch_size = int(getattr(trainer.args, "precompute_ref_batch_size", 0) or trainer.args.per_device_eval_batch_size)
    if trainer.eval_dataset is not None:
        trainer.eval_dataset = precompute(trainer.eval_dataset, "eval", eval_batch_size)
    trainer.precompute_ref_logps = True
    report = {
        "enabled": True,
        "train": _reference_logp_column_report(trainer.train_dataset, "train"),
        "eval": _reference_logp_column_report(trainer.eval_dataset, "eval") if trainer.eval_dataset is not None else None,
    }
    _assert_reference_logp_report_sane(report)
    return report


def _reference_logp_column_report(dataset: Any, label: str) -> dict[str, Any]:
    chosen: list[float] = []
    rejected: list[float] = []
    for index in range(len(dataset)):
        row = dataset[index]
        if not isinstance(row, Mapping) or "ref_chosen_logps" not in row or "ref_rejected_logps" not in row:
            raise ValueError(f"Precomputed {label} DPO row {index} is missing ref log-prob columns")
        chosen.append(float(row["ref_chosen_logps"]))
        rejected.append(float(row["ref_rejected_logps"]))
    return {
        "rows": len(dataset),
        "ref_chosen_finite": all(math.isfinite(value) for value in chosen),
        "ref_rejected_finite": all(math.isfinite(value) for value in rejected),
        "ref_chosen_abs_max": max((abs(value) for value in chosen), default=0.0),
        "ref_rejected_abs_max": max((abs(value) for value in rejected), default=0.0),
        "ref_margin_mean": (sum(c - r for c, r in zip(chosen, rejected)) / len(chosen)) if chosen else None,
    }


def _assert_reference_logp_report_sane(report: Mapping[str, Any]) -> None:
    for split in ("train", "eval"):
        details = report.get(split)
        if not isinstance(details, Mapping):
            continue
        if int(details.get("rows", 0)) <= 0:
            raise ValueError(f"Precomputed DPO reference log-probs for {split} are empty")
        if not details.get("ref_chosen_finite") or not details.get("ref_rejected_finite"):
            raise ValueError(f"Precomputed DPO reference log-probs for {split} contain NaN/Inf")
        chosen_abs = float(details.get("ref_chosen_abs_max", 0.0))
        rejected_abs = float(details.get("ref_rejected_abs_max", 0.0))
        if chosen_abs == 0.0 and rejected_abs == 0.0:
            raise ValueError(f"Precomputed DPO reference log-probs for {split} are all zero")


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
    loss_rows = [row for row in log_history if any(_is_metric_path(str(key)) for key in row)]
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
                "eval_logps_chosen": _last_metric(trainer, "eval", "logps/chosen"),
                "eval_logps_rejected": _last_metric(trainer, "eval", "logps/rejected"),
            }
        )
    return output


def _last_metric(trainer: Any, mode: str, key: str) -> Any:
    metrics = getattr(trainer, "_metrics", {})
    values = metrics.get(mode, {}).get(key, []) if isinstance(metrics, Mapping) else []
    return values[-1] if values else None


def _validate_reference_adapters(model: Any) -> dict[str, Any]:
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping):
        raise ValueError("DPO policy model must expose PEFT peft_config")
    adapters = set(str(name) for name in peft_config)
    if "default" not in adapters or "ref" not in adapters:
        raise ValueError(f"DPOTrainer must create both default and ref adapters, got {sorted(adapters)}")

    params = dict(model.named_parameters())
    default_params = {name: param for name, param in params.items() if ".default." in name}
    ref_params = {name: param for name, param in params.items() if ".ref." in name}
    if not default_params:
        raise ValueError("DPO policy model has no default adapter parameters")
    if not ref_params:
        raise ValueError("DPO policy model has no ref adapter parameters")
    trainable_default = [name for name, param in default_params.items() if getattr(param, "requires_grad", False)]
    trainable_ref = [name for name, param in ref_params.items() if getattr(param, "requires_grad", False)]
    if not trainable_default:
        raise ValueError("DPO default adapter must have trainable parameters")
    if trainable_ref:
        raise ValueError(f"DPO ref adapter parameters must be frozen: {trainable_ref[:3]}")

    missing_ref: list[str] = []
    mismatched: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for name, param in default_params.items():
        ref_name = name.replace(".default.", ".ref.")
        ref_param = ref_params.get(ref_name)
        if ref_param is None:
            missing_ref.append(ref_name)
            continue
        detail = _parameter_difference_summary(param, ref_param)
        diff = detail.get("max_abs_diff")
        if isinstance(diff, (float, int)):
            max_abs_diff = max(max_abs_diff, float(diff))
        if not _parameters_close(param, ref_param):
            detail = {"parameter": name, **detail}
            mismatched.append(detail)
    if missing_ref:
        raise ValueError(f"DPO ref adapter is missing parameters copied from default: {missing_ref[:3]}")
    if mismatched:
        raise ValueError(f"DPO default/ref adapter weights differ at initialization: {mismatched[:3]}")
    return {
        "adapters": sorted(adapters),
        "default_trainable_parameter_count": len(trainable_default),
        "ref_parameter_count": len(ref_params),
        "ref_trainable_parameter_count": 0,
        "initial_weights_equal": max_abs_diff == 0.0,
        "initial_weights_allclose": True,
        "initial_weight_atol": REFERENCE_ADAPTER_ATOL,
        "initial_max_abs_diff": max_abs_diff,
    }


def _copy_default_adapter_to_ref(model: Any) -> None:
    """Make the DPO reference adapter an exact frozen copy of the active policy adapter."""

    torch = importlib.import_module("torch")
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping) or "default" not in peft_config or "ref" not in peft_config:
        raise ValueError("DPOTrainer must create default and ref adapters before reference weight copy")
    params = dict(model.named_parameters())
    copied = 0
    for name, param in params.items():
        if ".default." not in name:
            continue
        ref_name = name.replace(".default.", ".ref.")
        ref_param = _get_model_parameter(model, ref_name) or params.get(ref_name)
        if ref_param is None:
            raise ValueError(f"DPO ref adapter is missing parameter copied from default: {ref_name}")
        _copy_parameter(param, ref_param)
        ref_param.requires_grad_(False)
        copied += 1
    if copied == 0:
        raise ValueError("DPO reference adapter copy found no default adapter parameters")
    _synchronize_torch_device(torch)


def _reload_ref_adapter_from_sft(model: Any, sft_adapter_dir: Path) -> None:
    """Load the frozen reference adapter directly from the Stage 3 SFT adapter artifact."""

    delete_adapter = getattr(model, "delete_adapter", None)
    load_adapter = getattr(model, "load_adapter", None)
    set_adapter = getattr(model, "set_adapter", None)
    if not callable(delete_adapter) or not callable(load_adapter) or not callable(set_adapter):
        _copy_default_adapter_to_ref(model)
        return

    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, Mapping) and "ref" in peft_config:
        delete_adapter("ref")
    load_adapter(
        sft_adapter_dir,
        adapter_name="ref",
        is_trainable=False,
        torch_device=_default_adapter_device(model),
    )
    _freeze_adapter_parameters(model, "ref")
    set_adapter("default")
    _synchronize_torch_device(importlib.import_module("torch"))


def _default_adapter_device(model: Any) -> str | None:
    for name, param in model.named_parameters():
        if ".default." not in name:
            continue
        device = getattr(param, "device", None)
        device_type = getattr(device, "type", None)
        if device_type:
            return str(device_type)
    return None


def _freeze_adapter_parameters(model: Any, adapter_name: str) -> None:
    marker = f".{adapter_name}."
    for name, param in model.named_parameters():
        if marker in name:
            param.requires_grad_(False)


def _get_model_parameter(model: Any, name: str) -> Any | None:
    get_parameter = getattr(model, "get_parameter", None)
    if not callable(get_parameter):
        return None
    try:
        return get_parameter(name)
    except (AttributeError, KeyError, ValueError):
        return None


def _copy_parameter(source: Any, target: Any) -> None:
    try:
        torch = importlib.import_module("torch")
        with torch.no_grad():
            source_data = getattr(source, "data", source)
            target_data = getattr(target, "data", target)
            target_data.copy_(source_data)
    except (AttributeError, TypeError):
        if hasattr(target, "value") and hasattr(source, "value"):
            target.value = source.value
        else:
            raise


def _synchronize_torch_device(torch: Any) -> None:
    mps = getattr(torch, "mps", None)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if mps_available and mps is not None and hasattr(mps, "synchronize"):
        mps.synchronize()
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available():
        cuda.synchronize()


def _parameters_close(left: Any, right: Any) -> bool:
    left_detached = left.detach() if hasattr(left, "detach") else left
    right_detached = right.detach() if hasattr(right, "detach") else right
    try:
        torch = importlib.import_module("torch")
        if hasattr(left_detached, "cpu") and hasattr(right_detached, "cpu"):
            return bool(torch.allclose(left_detached.cpu(), right_detached.cpu(), rtol=0.0, atol=REFERENCE_ADAPTER_ATOL))
        comparison = left_detached == right_detached
        if hasattr(comparison, "all"):
            comparison = comparison.all()
        if hasattr(comparison, "item"):
            return bool(comparison.item())
        return bool(comparison)
    except (TypeError, ValueError):
        return left_detached == right_detached


def _parameter_difference_summary(left: Any, right: Any) -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        left_detached = left.detach() if hasattr(left, "detach") else left
        right_detached = right.detach() if hasattr(right, "detach") else right
        if hasattr(left_detached, "cpu") and hasattr(right_detached, "cpu"):
            left_cpu = left_detached.float().cpu()
            right_cpu = right_detached.float().cpu()
            return {
                "left_device": str(getattr(left, "device", None)),
                "right_device": str(getattr(right, "device", None)),
                "left_dtype": str(getattr(left, "dtype", None)),
                "right_dtype": str(getattr(right, "dtype", None)),
                "max_abs_diff": float((left_cpu - right_cpu).abs().max().item()),
                "torch_equal": bool(torch.equal(left_detached.cpu(), right_detached.cpu())),
                "torch_allclose": bool(torch.allclose(left_detached.cpu(), right_detached.cpu(), rtol=0.0, atol=REFERENCE_ADAPTER_ATOL)),
                "atol": REFERENCE_ADAPTER_ATOL,
            }
    except (RuntimeError, TypeError, ValueError):
        pass
    return {
        "left_device": str(getattr(left, "device", None)),
        "right_device": str(getattr(right, "device", None)),
        "left_dtype": str(getattr(left, "dtype", None)),
        "right_dtype": str(getattr(right, "dtype", None)),
        "max_abs_diff": None,
        "torch_equal": False,
        "torch_allclose": False,
        "atol": REFERENCE_ADAPTER_ATOL,
    }


def _assert_numerical_stability(
    trainer: Any,
    train_metrics: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
    preference_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    trainer_metrics = getattr(trainer, "_metrics", {})
    payloads = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "loss_history": log_history,
        "preference_validation": preference_rows,
        "trainer_metrics": trainer_metrics if isinstance(trainer_metrics, Mapping) else {},
    }
    nonfinite = _nonfinite_metric_paths(payloads)
    if nonfinite:
        raise ValueError(f"Non-finite DPO metrics detected; refusing to publish completed run: {nonfinite[:10]}")
    zero_signal = _zero_dpo_signal_paths(payloads)
    if zero_signal:
        raise ValueError(f"DPO metrics are all zero for core signals; refusing to publish completed run: {zero_signal}")
    return {
        "checked_metric_families": ["loss", "reward", "margin", "log-prob", "grad_norm"],
        "nonfinite_count": 0,
        "zero_signal_count": 0,
        "passed": True,
    }


def _nonfinite_metric_paths(value: Any, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            paths.extend(_nonfinite_metric_paths(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_nonfinite_metric_paths(item, f"{path}[{index}]"))
    elif _is_metric_path(path):
        scalar = _as_float(value)
        if scalar is not None and not math.isfinite(scalar):
            paths.append(path)
    return paths


def _is_metric_path(path: str) -> bool:
    metric_name = path.rsplit(".", 1)[-1]
    lowered = metric_name.lower().replace("_", "/")
    return any(token in lowered for token in ("loss", "reward", "margin", "logp", "log/prob", "grad/norm"))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        try:
            item = value.item()
        except (RuntimeError, ValueError, TypeError):
            return None
        if isinstance(item, (float, int)) and not isinstance(item, bool):
            return float(item)
    return None


def _zero_dpo_signal_paths(payloads: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for metric_name in ("loss", "logps/chosen", "logps/rejected"):
        values = _metric_values_for_name(payloads, metric_name)
        if values and all(value == 0.0 for value in values):
            paths.append(metric_name)
    return paths


def _metric_values_for_name(value: Any, metric_name: str, path: str = "") -> list[float]:
    values: list[float] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            values.extend(_metric_values_for_name(item, metric_name, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            values.extend(_metric_values_for_name(item, metric_name, f"{path}[{index}]"))
    elif path.endswith(metric_name):
        scalar = _as_float(value)
        if scalar is not None:
            values.append(scalar)
    return values


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
        copied["dpo"]["train_samples"] = int(config["smoke_test"]["dpo_samples"])
        copied["dpo"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
        overrides["applied"]["dpo.max_steps"] = copied["dpo"]["max_steps"]
        overrides["applied"]["dpo.train_samples"] = copied["dpo"]["train_samples"]
        overrides["applied"]["dpo.validation_samples"] = copied["dpo"]["validation_samples"]
    if train_samples is not None:
        copied["dpo"]["train_samples"] = int(train_samples)
        overrides["applied"]["dpo.train_samples"] = int(train_samples)
    if validation_samples is not None:
        copied["dpo"]["validation_samples"] = int(validation_samples)
        overrides["applied"]["dpo.validation_samples"] = int(validation_samples)
    if max_steps is not None:
        copied["dpo"]["max_steps"] = int(max_steps)
        overrides["applied"]["dpo.max_steps"] = int(max_steps)
    return copied, overrides


def _target_counts(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        "train": int(config["dpo"]["train_samples"]),
        "validation": int(config["dpo"]["validation_samples"]),
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
