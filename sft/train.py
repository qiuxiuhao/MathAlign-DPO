"""Standalone Stage 2 Mini SFT training entrypoint."""

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
from sft.data import SFTDatasets, load_sft_datasets
from sft.evaluate import evaluate_base_and_sft, write_json, write_jsonl
from sft.modeling import load_lora_model_and_tokenizer, load_sft_for_generation, validate_runtime, validate_tokenizer
from transformers import TrainerCallback


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = train_sft(
        config_path=args.config,
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
    """Build the Stage 2 SFT CLI parser."""

    parser = argparse.ArgumentParser(description="Train SFT from Stage 1 Hugging Face Datasets.")
    parser.add_argument("--config", required=True, help="Path to the Mini YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke training/evaluation limits from config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory. Defaults to config.sft.output_dir.")
    parser.add_argument("--train-samples", type=int, default=None, help="Use the first N train rows for debugging.")
    parser.add_argument("--validation-samples", type=int, default=None, help="Use the first N validation rows for debugging.")
    parser.add_argument("--eval-samples", type=int, default=None, help="Use the first N evaluation rows for debugging.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override SFT max_steps for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser


def train_sft(
    config_path: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    train_samples: int | None = None,
    validation_samples: int | None = None,
    eval_samples: int | None = None,
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Mini SFT training, adapter reload validation, and Base/SFT evaluation."""

    original_config = load_config(config_path)
    config, overrides = apply_runtime_overrides(
        original_config,
        smoke_test=smoke_test,
        train_samples=train_samples,
        validation_samples=validation_samples,
        eval_samples=eval_samples,
        max_steps=max_steps,
    )
    out_dir = Path(output_dir) if output_dir is not None else Path(str(config["sft"]["output_dir"]))
    prepare_output_dir(out_dir, overwrite=overwrite)
    start = time.perf_counter()
    status_payload: dict[str, Any] = {}
    try:
        runtime = validate_runtime(config)
        datasets = load_sft_datasets(
            config,
            train_limit=int(config["data"]["train_samples"]),
            validation_limit=int(config["data"]["validation_samples"]),
        )
        loaded = load_lora_model_and_tokenizer(config)
        tokenizer_metadata = validate_tokenizer(loaded.tokenizer)
        train_metrics, eval_metrics = train_with_trl(config, loaded.model, loaded.tokenizer, datasets, out_dir)
        adapter_dir = out_dir / "adapter"
        tokenizer_dir = out_dir / "tokenizer"
        loaded.model.save_pretrained(adapter_dir)
        loaded.tokenizer.save_pretrained(tokenizer_dir)
        reload_samples = reload_adapter_samples(
            config,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            validation_rows=datasets.validation,
            sample_count=int(config["sft"]["adapter_reload_samples"]),
            max_new_tokens=int(config["sft"]["adapter_reload_max_new_tokens"]),
        )
        write_jsonl(out_dir / "adapter_reload_samples.jsonl", reload_samples)
        evaluation_summary = evaluate_base_and_sft(
            config,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            output_dir=out_dir,
            sample_count=int(config["evaluation"]["samples"]),
        )
        run_config = {
            "status": "completed",
            "stage": 2,
            "training_stage": "sft",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_path": str(config_path),
            "output_dir": str(out_dir),
            "smoke_test": smoke_test,
            "runtime_overrides": overrides,
            "runtime": runtime,
            "model_loader": loaded.metadata,
            "tokenizer": tokenizer_metadata,
            "dataset_paths": {"sft": str(datasets.path), "evaluation": str(Path(str(config["data"]["mini_dir"])) / "evaluation")},
            "dataset_counts": {"train": len(datasets.train), "validation": len(datasets.validation)},
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "base_sft_evaluation": evaluation_summary,
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
        write_json(out_dir / "run_config.json", run_config)
        return run_config
    except BaseException as exc:
        status_payload = {
            "status": "failed",
            "stage": 2,
            "training_stage": "sft",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "run_config.json", status_payload)
        raise


def train_with_trl(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    datasets: SFTDatasets,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train with TRL SFTTrainer using Stage 1 prompt/completion rows."""

    import trl

    args = sft_config(trl, config, output_dir)
    trainer = trl.SFTTrainer(
        model=model,
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


def sft_config(trl: Any, config: Mapping[str, Any], output_dir: Path) -> Any:
    """Create a TRL SFTConfig for Mini SFT."""

    sft = config["sft"]
    runtime = config["runtime"]
    precision = str(runtime.get("mixed_precision") or config["model"].get("torch_dtype"))
    use_bf16 = precision in {"bf16", "bfloat16"}
    use_fp16 = precision in {"fp16", "float16"} or (str(runtime["backend"]) == "mps" and not use_bf16)
    kwargs = {
        "output_dir": str(output_dir),
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
        "fp16": bool(use_fp16),
        "bf16": bool(use_bf16),
        "eos_token": "<|im_end|>",
    }
    return trl.SFTConfig(**kwargs)


def reload_adapter_samples(
    config: Mapping[str, Any],
    adapter_dir: Path,
    tokenizer_dir: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    sample_count: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Reload saved adapter and generate a few validation samples."""

    import torch

    loaded = load_sft_for_generation(config, adapter_dir=adapter_dir, tokenizer_dir=tokenizer_dir)
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
                "prompt": list(row["prompt"]),
                "generated_text": loaded.tokenizer.decode(new_tokens, skip_special_tokens=True),
                "max_new_tokens": int(max_new_tokens),
            }
        )
    return output


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
            raise FileExistsError(f"Refusing to overwrite non-empty SFT output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
