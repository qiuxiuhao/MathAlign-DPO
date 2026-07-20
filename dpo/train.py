"""Standalone Stage 3 DPO training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from configs.load_config import apply_runtime_overrides, load_config
from dpo.data import DPODatasets, load_dpo_datasets
from dpo.evaluate import evaluate_base_sft_dpo
from dpo.modeling import load_dpo_for_generation, load_policy_from_sft_adapter, validate_sft_dir
from sft.evaluate import release_accelerator_memory, write_json, write_jsonl
from sft.modeling import validate_runtime, validate_tokenizer
from transformers import TrainerCallback


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = train_dpo(
        config_path=args.config,
        sft_dir=args.sft_dir,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        eval_samples=args.eval_samples,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 3 DPO CLI parser."""

    parser = argparse.ArgumentParser(description="Train DPO from Stage 1 Hugging Face Datasets.")
    parser.add_argument("--config", required=True, help="Path to one YAML config.")
    parser.add_argument("--sft-dir", required=True, help="Completed SFT output directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke training/evaluation limits from config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory. Defaults to config.dpo.output_dir.")
    parser.add_argument("--train-samples", type=int, default=None, help="Use the first N train rows for debugging.")
    parser.add_argument("--validation-samples", type=int, default=None, help="Use the first N validation rows for debugging.")
    parser.add_argument("--eval-samples", type=int, default=None, help="Use the first N evaluation rows for debugging.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override DPO max_steps for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser


def train_dpo(
    config_path: str | Path,
    sft_dir: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    train_samples: int | None = None,
    validation_samples: int | None = None,
    eval_samples: int | None = None,
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run DPO training, adapter reload validation, and Base/SFT/DPO evaluation."""

    original_config = load_config(config_path)
    config, overrides = apply_runtime_overrides(
        original_config,
        smoke_test=smoke_test,
        train_samples=train_samples,
        validation_samples=validation_samples,
        eval_samples=eval_samples,
        max_steps=max_steps,
        training_stage="dpo",
    )
    out_dir = Path(output_dir) if output_dir is not None else Path(str(config["dpo"]["output_dir"]))
    prepare_output_dir(out_dir, overwrite=overwrite)
    start = time.perf_counter()
    try:
        runtime = validate_runtime(config)
        sft_source = validate_sft_dir(config, sft_dir=sft_dir, smoke_test=smoke_test)
        datasets = load_dpo_datasets(
            config,
            train_limit=int(config["dpo"]["train_samples"]),
            validation_limit=int(config["dpo"]["validation_samples"]),
        )
        loaded = load_policy_from_sft_adapter(
            config,
            sft_adapter_dir=Path(sft_source["adapter_dir"]),
            tokenizer_dir=Path(sft_source["tokenizer_dir"]),
        )
        policy_metadata = dict(loaded.metadata)
        tokenizer_metadata = validate_tokenizer(loaded.tokenizer)
        train_metrics, eval_metrics = train_with_trl(config, loaded.model, loaded.tokenizer, datasets, out_dir)
        adapter_dir = out_dir / "adapter"
        tokenizer_dir = out_dir / "tokenizer"
        save_adapter(loaded.model, adapter_dir)
        loaded.tokenizer.save_pretrained(tokenizer_dir)
        del loaded
        release_accelerator_memory()
        reload_samples = reload_dpo_adapter_samples(
            config,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            validation_rows=datasets.validation,
            sample_count=int(config["dpo"]["adapter_reload_samples"]),
            max_new_tokens=int(config["dpo"]["adapter_reload_max_new_tokens"]),
        )
        write_jsonl(out_dir / "adapter_reload_samples.jsonl", reload_samples)
        evaluation_summary = evaluate_base_sft_dpo(
            config,
            sft_adapter_dir=Path(sft_source["adapter_dir"]),
            dpo_adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            output_dir=out_dir,
            sample_count=int(config["evaluation"]["samples"]),
        )
        run_config = {
            "status": "completed",
            "stage": 3,
            "training_stage": "dpo",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_path": str(config_path),
            "output_dir": str(out_dir),
            "smoke_test": smoke_test,
            "runtime_overrides": overrides,
            "runtime": runtime,
            "sft_source": sft_source,
            "model_loader": policy_metadata,
            "tokenizer": tokenizer_metadata,
            "dataset_paths": {"dpo": str(datasets.path), "evaluation": str(Path(str(config["data"][f"{config['project']['run_mode']}_dir"])) / "evaluation")},
            "dataset_counts": {"train": len(datasets.train), "validation": len(datasets.validation)},
            "dpo_config": dpo_runtime_metadata(config),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "base_sft_dpo_evaluation": evaluation_summary,
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
        write_json(out_dir / "run_config.json", run_config)
        return run_config
    except BaseException as exc:
        failed = {
            "status": "failed",
            "stage": 3,
            "training_stage": "dpo",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "run_config.json", failed)
        raise


def train_with_trl(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    datasets: DPODatasets,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train with TRL DPOTrainer using Stage 1 prompt/chosen/rejected rows."""

    import trl

    args = dpo_config(trl, config, output_dir)
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=datasets.train,
        eval_dataset=datasets.validation,
        processing_class=tokenizer,
        callbacks=[FiniteLossCallback()],
    )
    train_result = trainer.train()
    train_metrics = dict(getattr(train_result, "metrics", {}) or {})
    eval_metrics = dict(trainer.evaluate())
    trainer.save_state()
    assert_finite_metrics([train_metrics, eval_metrics], trainer.state.log_history)
    write_json(output_dir / "train_metrics.json", train_metrics)
    write_json(output_dir / "eval_metrics.json", eval_metrics)
    write_jsonl(output_dir / "loss_history.jsonl", [row for row in trainer.state.log_history if "loss" in row or "eval_loss" in row])
    return train_metrics, eval_metrics


def dpo_config(trl: Any, config: Mapping[str, Any], output_dir: Path) -> Any:
    """Create a TRL DPOConfig from YAML values."""

    dpo = config["dpo"]
    runtime = config["runtime"]
    precision = str(runtime.get("mixed_precision") or config["model"].get("torch_dtype"))
    use_bf16 = precision in {"bf16", "bfloat16"}
    use_fp16 = precision in {"fp16", "float16"} or (str(runtime["backend"]) == "mps" and not use_bf16)
    kwargs = {
        "output_dir": str(output_dir),
        "max_length": int(dpo["max_length"]),
        "beta": float(dpo["beta"]),
        "loss_type": [str(dpo["loss_type"])],
        "learning_rate": float(dpo["learning_rate"]),
        "num_train_epochs": float(dpo["num_train_epochs"]),
        "max_steps": int(dpo["max_steps"]),
        "per_device_train_batch_size": int(dpo["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(dpo["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(dpo["gradient_accumulation_steps"]),
        "warmup_ratio": float(dpo["warmup_ratio"]),
        "weight_decay": float(dpo["weight_decay"]),
        "max_grad_norm": float(dpo["max_grad_norm"]),
        "optim": str(dpo["optimizer"]),
        "lr_scheduler_type": str(dpo["lr_scheduler_type"]),
        "eval_strategy": str(dpo["eval_strategy"]),
        "eval_steps": int(dpo["eval_steps"]),
        "save_strategy": str(dpo["save_strategy"]),
        "save_steps": int(dpo["save_steps"]),
        "save_total_limit": int(dpo["save_total_limit"]),
        "logging_steps": int(dpo["logging_steps"]),
        "report_to": str(runtime["report_to"]),
        "dataloader_num_workers": int(runtime["dataloader_num_workers"]),
        "dataloader_pin_memory": bool(runtime["pin_memory"]),
        "seed": int(config["project"]["seed"]),
        "fp16": bool(use_fp16),
        "bf16": bool(use_bf16),
        "gradient_checkpointing": bool(config["model"]["gradient_checkpointing"]),
        "precompute_ref_log_probs": False,
    }
    return trl.DPOConfig(**kwargs)


def reload_dpo_adapter_samples(
    config: Mapping[str, Any],
    adapter_dir: Path,
    tokenizer_dir: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    sample_count: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Reload saved DPO adapter and generate a few validation samples."""

    import torch

    loaded = load_dpo_for_generation(config, adapter_dir=adapter_dir, tokenizer_dir=tokenizer_dir)
    output: list[dict[str, Any]] = []
    for row in validation_rows.select(range(min(len(validation_rows), sample_count))):
        encoded = loaded.tokenizer.apply_chat_template(
            list(row["prompt"]),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(str(config["runtime"]["device"]))
        with torch.no_grad():
            generated = loaded.model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][encoded.shape[-1] :]
        output.append(
            {
                "id": str(row["id"]),
                "source_id": str(row["source_id"]),
                "step_index": int(row["step_index"]),
                "prompt": list(row["prompt"]),
                "generated_text": loaded.tokenizer.decode(new_tokens, skip_special_tokens=True),
                "max_new_tokens": int(max_new_tokens),
            }
        )
    return output


def save_adapter(model: Any, adapter_dir: Path) -> None:
    """Save only the trainable DPO adapter when PEFT exposes adapter selection."""

    try:
        model.save_pretrained(adapter_dir, selected_adapters=["default"])
    except TypeError:
        model.save_pretrained(adapter_dir)


def dpo_runtime_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the DPO trainer settings that affect behavior."""

    dpo = config["dpo"]
    return {
        "beta": float(dpo["beta"]),
        "loss_type": str(dpo["loss_type"]),
        "max_length": int(dpo["max_length"]),
        "max_prompt_length_checked": int(dpo["max_prompt_length"]),
        "reference_policy": "trl_peft_ref_adapter",
    }


class FiniteLossCallback(TrainerCallback):
    """Stop training as soon as logged loss becomes NaN or Inf."""

    def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        for key in ("loss", "eval_loss"):
            if logs and key in logs and not math.isfinite(float(logs[key])):
                raise FloatingPointError(f"Non-finite {key}: {logs[key]}")


def assert_finite_metrics(metrics: Sequence[Mapping[str, Any]], log_history: Sequence[Mapping[str, Any]]) -> None:
    """Validate train/eval metrics and trainer loss history."""

    for payload in list(metrics) + list(log_history):
        for key, value in payload.items():
            if key.endswith("loss") or key == "loss":
                if value is not None and not math.isfinite(float(value)):
                    raise FloatingPointError(f"Non-finite metric {key}: {value}")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory with collision protection."""

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty DPO output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
