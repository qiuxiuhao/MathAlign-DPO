"""Stage 5 unified Mini evaluation orchestration."""

from __future__ import annotations

import gc
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from mathalign_dpo.config.load_config import load_single_config
from mathalign_dpo.evaluation.answer_normalization import extract_and_normalize_answer, normalize_answer
from mathalign_dpo.evaluation.eval_data import (
    load_stage5_eval_dataset,
    validate_dpo_eval_source,
    validate_no_training_leakage,
    validate_sft_eval_source,
)
from mathalign_dpo.evaluation.preference_eval import (
    load_preference_validation_rows,
    preference_summary,
    score_preference_rows,
)
from mathalign_dpo.training.model_loader import load_base_model_and_tokenizer, load_tokenizer_from_dir, model_revision_metadata, validate_runtime_backend
from mathalign_dpo.training.run_artifacts import prepare_staged_output_dir, publish_staged_output
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


MODEL_STAGES = ("base", "sft", "dpo")


def evaluate_math_from_config(
    config_path: str | Path,
    sft_run_dir: str | Path,
    dpo_run_dir: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 5 Mini Base/SFT/DPO evaluation."""

    original_config = load_single_config(config_path)
    if original_config["project"]["run_mode"] != "mini":
        raise ValueError("Stage 5 actual evaluation only supports project.run_mode = mini")
    run_config, overrides = _apply_runtime_overrides(original_config, smoke_test, samples)
    run_id = build_run_id("eval", smoke_test, stage_number=5)
    run_dirs = prepare_staged_output_dir(run_config, run_id, output_dir, overwrite, "evaluation", "EVAL")
    clock = RunClock.start()
    metadata = collect_base_metadata(
        config=run_config,
        config_path=config_path,
        output_dir=run_dirs.final_dir,
        run_id=run_id,
        stage_number=5,
        training_stage="evaluation",
        run_mode=str(run_config["project"]["run_mode"]),
        smoke_test=smoke_test,
        runtime_overrides=overrides,
    )
    progress: dict[str, Any] = {}

    try:
        sft_source = validate_sft_eval_source(sft_run_dir, run_config)
        dpo_source = validate_dpo_eval_source(dpo_run_dir, sft_source, run_config)
        progress["sft_source"] = _source_summary(sft_source)
        progress["dpo_source"] = _source_summary(dpo_source)

        eval_rows, data_lineage = load_stage5_eval_dataset(run_config, int(run_config["evaluation"]["samples"]))
        leakage = validate_no_training_leakage(
            eval_rows,
            sft_source["metadata"],
            dpo_source["metadata"],
            run_config["data"]["stage2_manifest_file"],
        )
        progress["data_lineage"] = data_lineage
        progress["leakage_validation"] = leakage

        preference_rows = load_preference_validation_rows(
            run_config,
            dpo_source["metadata"],
            _preference_sample_count(run_config, smoke_test),
        )
        tokenizer = load_tokenizer_from_dir(sft_source["tokenizer_dir"])
        tokenizer_metadata = validate_tokenizer_chat_template(tokenizer)
        backend = validate_runtime_backend(run_config)
        progress["backend_preflight"] = backend

        predictions: list[dict[str, Any]] = []
        preference_predictions: list[dict[str, Any]] = []
        for stage in MODEL_STAGES:
            _progress(f"loading {stage} model")
            model = _load_eval_model(run_config, stage, sft_source, dpo_source)
            try:
                _progress(f"generating {stage} predictions on {len(eval_rows)} examples")
                predictions.extend(_generate_predictions(run_config, model, tokenizer, eval_rows, stage))
                _progress(f"scoring {stage} preference pairs on {len(preference_rows)} examples")
                preference_predictions.extend(
                    score_preference_rows(
                        model,
                        tokenizer,
                        preference_rows,
                        stage,
                        str(run_config["runtime"]["device"]),
                    )
                )
                _progress(f"finished {stage}")
            finally:
                del model
                _release_device_memory(run_config)

        summaries = _summaries(predictions, preference_predictions)
        comparison_rows = _comparison_examples(predictions, limit=5)
        error_cases = [row for row in predictions if not row["exact_match"]]
        report_text = _markdown_report(summaries)

        write_jsonl(run_dirs.staging_dir / "evaluation_dataset.jsonl", eval_rows)
        write_jsonl(run_dirs.staging_dir / "predictions.jsonl", predictions)
        write_json(run_dirs.staging_dir / "summary.json", summaries)
        write_jsonl(run_dirs.staging_dir / "preference_predictions.jsonl", preference_predictions)
        write_json(run_dirs.staging_dir / "preference_summary.json", summaries["preference"])
        write_jsonl(run_dirs.staging_dir / "comparison_examples.jsonl", comparison_rows)
        write_jsonl(run_dirs.staging_dir / "error_cases.jsonl", error_cases)
        (run_dirs.staging_dir / "report.md").write_text(report_text, encoding="utf-8")

        completed = finalize_metadata(
            metadata,
            clock,
            "completed",
            {
                "sft_source": _source_summary(sft_source),
                "dpo_source": _source_summary(dpo_source),
                "data_lineage": data_lineage,
                "leakage_validation": leakage,
                "tokenizer": tokenizer_metadata,
                "backend_preflight": backend,
                "model_revision": model_revision_metadata(run_config),
                "metrics": summaries,
                "artifacts": _artifact_paths(run_dirs.final_dir),
            },
        )
        write_json(run_dirs.staging_dir / "run_metadata.json", completed)
        publish_staged_output(run_dirs, overwrite, "EVAL")
        return completed
    except BaseException as exc:
        failed = finalize_metadata(
            metadata,
            clock,
            "failed",
            {
                "error": {"type": type(exc).__name__, "message": str(exc)},
                **progress,
            },
        )
        write_json(run_dirs.staging_dir / "run_metadata.json", failed)
        raise


def cli_payload(result: Mapping[str, Any]) -> str:
    """Serialize a compact Stage 5 CLI result."""

    summary = {
        "status": result["status"],
        "run_id": result["run_id"],
        "output_dir": result["output_dir"],
        "elapsed_seconds": result["elapsed_seconds"],
        "metrics": result.get("metrics"),
        "artifacts": result.get("artifacts"),
        "error": result.get("error"),
    }
    return json.dumps(json_safe(summary), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


def _apply_runtime_overrides(config: Mapping[str, Any], smoke_test: bool, samples: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    copied = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    overrides: dict[str, Any] = {"smoke_test": bool(smoke_test), "cli": {"samples": samples}, "applied": {}}
    if smoke_test:
        copied["evaluation"]["samples"] = int(config["smoke_test"]["evaluation_samples"])
        overrides["applied"]["evaluation.samples"] = copied["evaluation"]["samples"]
    if samples is not None:
        copied["evaluation"]["samples"] = int(samples)
        overrides["applied"]["evaluation.samples"] = int(samples)
    return copied, overrides


def _preference_sample_count(config: Mapping[str, Any], smoke_test: bool) -> int:
    if smoke_test:
        return min(int(config["smoke_test"]["validation_samples"]), int(config["dpo"]["validation_samples"]))
    return int(config["dpo"]["validation_samples"])


def _load_eval_model(config: Mapping[str, Any], stage: str, sft_source: Mapping[str, Any], dpo_source: Mapping[str, Any]) -> Any:
    loaded = load_base_model_and_tokenizer(config, training_stage="reload", tokenizer_dir=sft_source["tokenizer_dir"])
    model = loaded.model
    if stage == "base":
        _configure_eval_model(model)
        return model
    peft = importlib.import_module("peft")
    adapter_dir = sft_source["adapter_dir"] if stage == "sft" else dpo_source["adapter_dir"]
    model = peft.PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    if str(config["runtime"]["backend"]) == "mps":
        model.to("mps")
    _configure_eval_model(model)
    return model


def _configure_eval_model(model: Any) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = True
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    if hasattr(model, "eval"):
        model.eval()


def _generate_predictions(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    model_stage: str,
) -> list[dict[str, Any]]:
    torch = importlib.import_module("torch")
    device = str(config["runtime"]["device"])
    predictions: list[dict[str, Any]] = []
    generation_config = _generation_config(config, tokenizer)
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        _progress(f"{model_stage} generate {index}/{total}")
        encoded = tokenizer.apply_chat_template(
            list(row["prompt_messages"]),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        start = time.monotonic()
        with torch.no_grad():
            generated = model.generate(encoded, **generation_config)
        generation_seconds = time.monotonic() - start
        new_tokens = generated[0][encoded.shape[-1] :]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        answer = extract_and_normalize_answer(generated_text)
        reference = normalize_answer(str(row["reference_answer"]))
        predictions.append(
            {
                "schema_version": "1.0",
                "id": str(row["id"]),
                "source_id": str(row["source_id"]),
                "model_stage": model_stage,
                "prompt_messages": list(row["prompt_messages"]),
                "generated_text": generated_text,
                "predicted_answer": answer.raw_answer,
                "normalized_predicted_answer": answer.normalized_answer,
                "reference_answer": str(row["reference_answer"]),
                "normalized_reference_answer": reference,
                "answer_extracted": answer.extracted,
                "answer_method": answer.method,
                "exact_match": bool(answer.extracted and answer.normalized_answer == reference),
                "output_tokens": int(new_tokens.shape[-1]),
                "generation_seconds": round(generation_seconds, 6),
            }
        )
        del encoded, generated
    return predictions


def _progress(message: str) -> None:
    print(f"[stage5] {message}", file=sys.stderr, flush=True)


def _generation_config(config: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    return {
        "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
        "do_sample": False,
        "num_beams": int(config["evaluation"]["num_beams"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _summaries(predictions: Sequence[Mapping[str, Any]], preference_predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_summaries = {stage: _stage_summary(predictions, stage) for stage in MODEL_STAGES}
    preference = {stage: preference_summary(preference_predictions, stage) for stage in MODEL_STAGES}
    return {"models": model_summaries, "preference": preference}


def _stage_summary(predictions: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    rows = [row for row in predictions if row["model_stage"] == stage]
    if not rows:
        raise ValueError(f"No predictions for {stage}")
    extracted = sum(1 for row in rows if row["answer_extracted"])
    exact = sum(1 for row in rows if row["exact_match"])
    output_tokens = [int(row["output_tokens"]) for row in rows]
    seconds = [float(row["generation_seconds"]) for row in rows]
    return {
        "model_stage": stage,
        "num_examples": len(rows),
        "answer_extraction_rate": extracted / len(rows),
        "exact_match_accuracy": exact / len(rows),
        "invalid_output_rate": 1.0 - (extracted / len(rows)),
        "average_output_tokens": sum(output_tokens) / len(output_tokens),
        "total_generation_seconds": sum(seconds),
        "average_generation_seconds": sum(seconds) / len(seconds),
    }


def _comparison_examples(predictions: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    ids = []
    for row in predictions:
        if row["id"] not in ids:
            ids.append(row["id"])
    output: list[dict[str, Any]] = []
    for row_id in ids[:limit]:
        group = [row for row in predictions if row["id"] == row_id]
        output.append(
            {
                "id": row_id,
                "source_id": group[0]["source_id"],
                "reference_answer": group[0]["reference_answer"],
                "models": {
                    row["model_stage"]: {
                        "generated_text": row["generated_text"],
                        "predicted_answer": row["predicted_answer"],
                        "exact_match": row["exact_match"],
                    }
                    for row in group
                },
            }
        )
    return output


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5 Mini Evaluation Report",
        "",
        "| Model | Exact Match | Extraction | Avg Tokens | Avg Seconds | Preference Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in MODEL_STAGES:
        model = summary["models"][stage]
        preference = summary["preference"][stage]
        lines.append(
            "| {stage} | {exact:.4f} | {extract:.4f} | {tokens:.2f} | {seconds:.3f} | {pref:.4f} |".format(
                stage=stage,
                exact=float(model["exact_match_accuracy"]),
                extract=float(model["answer_extraction_rate"]),
                tokens=float(model["average_output_tokens"]),
                seconds=float(model["average_generation_seconds"]),
                pref=float(preference["preference_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "Preference accuracy is diagnostic. Mini evaluation is not a formal model-quality conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(final_dir: Path) -> dict[str, str]:
    return {
        "run_metadata": str(final_dir / "run_metadata.json"),
        "evaluation_dataset": str(final_dir / "evaluation_dataset.jsonl"),
        "predictions": str(final_dir / "predictions.jsonl"),
        "summary": str(final_dir / "summary.json"),
        "preference_predictions": str(final_dir / "preference_predictions.jsonl"),
        "preference_summary": str(final_dir / "preference_summary.json"),
        "comparison_examples": str(final_dir / "comparison_examples.jsonl"),
        "error_cases": str(final_dir / "error_cases.jsonl"),
        "report": str(final_dir / "report.md"),
    }


def _source_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": source["run_dir"],
        "run_id": source["run_id"],
        "adapter_dir": source["adapter_dir"],
        "adapter_sha256": source["adapter_sha256"],
    }


def _release_device_memory(config: Mapping[str, Any]) -> None:
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
