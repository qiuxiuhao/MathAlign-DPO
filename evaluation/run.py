"""Standalone Stage 4 evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from configs.load_config import load_config
from dpo.modeling import validate_sft_dir
from evaluation.common import write_json
from evaluation.evaluate import evaluate_base_sft_dpo
from sft.modeling import validate_runtime


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_evaluation(
        config_path=args.config,
        sft_dir=args.sft_dir,
        dpo_dir=args.dpo_dir,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        eval_samples=args.eval_samples,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 4 evaluation CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate Base, SFT, and DPO from Stage 1 Hugging Face Datasets.")
    parser.add_argument("--config", required=True, help="Path to one YAML config.")
    parser.add_argument("--sft-dir", default=None, help="Completed SFT output directory. Defaults to config.sft.output_dir.")
    parser.add_argument("--dpo-dir", default=None, help="Completed DPO output directory. Defaults to config.dpo.output_dir.")
    parser.add_argument("--output-dir", default=None, help="Override output directory. Defaults to config.evaluation.output_dir.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke evaluation limits from config.")
    parser.add_argument("--eval-samples", type=int, default=None, help="Use the first N evaluation rows for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser


def run_evaluation(
    config_path: str | Path,
    sft_dir: str | Path | None = None,
    dpo_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    smoke_test: bool = False,
    eval_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 4 Base/SFT/DPO evaluation."""

    original_config = load_config(config_path)
    config, overrides = apply_evaluation_overrides(original_config, smoke_test=smoke_test, eval_samples=eval_samples)
    sft_root = Path(sft_dir) if sft_dir is not None else Path(str(config["sft"]["output_dir"]))
    dpo_root = Path(dpo_dir) if dpo_dir is not None else Path(str(config["dpo"]["output_dir"]))
    out_dir = Path(output_dir) if output_dir is not None else Path(str(config["evaluation"]["output_dir"]))
    prepare_output_dir(out_dir, overwrite=overwrite)
    start = time.perf_counter()
    try:
        runtime = validate_runtime(config)
        sft_source = validate_sft_dir(config, sft_dir=sft_root, smoke_test=smoke_test)
        dpo_source = validate_dpo_dir(config, dpo_dir=dpo_root, smoke_test=smoke_test)
        evaluation_summary = evaluate_base_sft_dpo(
            config,
            sft_adapter_dir=Path(sft_source["adapter_dir"]),
            dpo_adapter_dir=Path(dpo_source["adapter_dir"]),
            tokenizer_dir=Path(dpo_source["tokenizer_dir"]),
            output_dir=out_dir,
            sample_count=int(config["evaluation"]["samples"]),
        )
        run_config = {
            "status": "completed",
            "stage": 4,
            "evaluation_stage": "base_sft_dpo",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_path": str(config_path),
            "run_mode": str(config["project"]["run_mode"]),
            "output_dir": str(out_dir),
            "smoke_test": smoke_test,
            "runtime_overrides": overrides,
            "runtime": runtime,
            "sft_source": sft_source,
            "dpo_source": dpo_source,
            "dataset_paths": {"evaluation": str(Path(str(config["data"][f"{config['project']['run_mode']}_dir"])) / "evaluation")},
            "evaluation_config": evaluation_runtime_metadata(config),
            "base_sft_dpo_evaluation": evaluation_summary,
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
        write_json(out_dir / "run_config.json", run_config)
        return run_config
    except BaseException as exc:
        failed = {
            "status": "failed",
            "stage": 4,
            "evaluation_stage": "base_sft_dpo",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "run_config.json", failed)
        raise


def apply_evaluation_overrides(
    config: Mapping[str, Any],
    smoke_test: bool,
    eval_samples: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a config copy with Stage 4 debug overrides applied."""

    copied = deepcopy(dict(config))
    overrides: dict[str, Any] = {
        "smoke_test": bool(smoke_test),
        "cli": {"eval_samples": eval_samples},
        "applied": {},
    }
    if smoke_test:
        copied["evaluation"]["samples"] = int(config["smoke_test"]["evaluation_samples"])
        overrides["applied"]["evaluation.samples"] = copied["evaluation"]["samples"]
    if eval_samples is not None:
        copied["evaluation"]["samples"] = int(eval_samples)
        overrides["applied"]["evaluation.samples"] = int(eval_samples)
    return copied, overrides


def validate_dpo_dir(config: Mapping[str, Any], dpo_dir: str | Path, smoke_test: bool) -> dict[str, Any]:
    """Validate that a completed DPO run can be evaluated."""

    root = Path(dpo_dir)
    run_config_path = root / "run_config.json"
    adapter_dir = root / "adapter"
    tokenizer_dir = root / "tokenizer"
    if not run_config_path.exists():
        raise FileNotFoundError(f"DPO run_config.json is missing: {run_config_path}")
    with run_config_path.open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    if run_config.get("status") != "completed":
        raise ValueError(f"DPO run is not completed: {root}")
    if run_config.get("training_stage") != "dpo":
        raise ValueError(f"DPO run directory has wrong training_stage: {root}")
    _validate_run_mode(config, run_config, root, label="DPO")
    _validate_model_identity(config, run_config, root, label="DPO")
    if run_config.get("smoke_test") is True and not smoke_test:
        raise ValueError("Non-smoke Stage 4 evaluation cannot use a smoke DPO adapter")
    for path in (adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_config.json"):
        if not path.exists():
            raise FileNotFoundError(f"DPO adapter artifact is missing: {path}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"DPO tokenizer directory is missing: {tokenizer_dir}")
    return {
        "run_dir": str(root),
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "run_config_path": str(run_config_path),
        "smoke_test": bool(run_config.get("smoke_test")),
        "run_mode": run_config.get("run_mode"),
        "model": run_config.get("model", {}),
        "dataset_counts": run_config.get("dataset_counts", {}),
        "sft_source": run_config.get("sft_source", {}),
    }


def _validate_run_mode(config: Mapping[str, Any], run_config: Mapping[str, Any], root: Path, label: str) -> None:
    expected_mode = str(config["project"]["run_mode"])
    recorded_mode = run_config.get("run_mode")
    if recorded_mode is not None:
        if str(recorded_mode) != expected_mode:
            raise ValueError(f"{label} run_mode={recorded_mode!r} does not match config run_mode={expected_mode!r}: {root}")
        return
    recorded_dataset = str(run_config.get("dataset_paths", {}).get("dpo", ""))
    if f"/{expected_mode}/" not in recorded_dataset:
        raise ValueError(f"{label} run does not appear to match run_mode={expected_mode}: {root}")


def _validate_model_identity(config: Mapping[str, Any], run_config: Mapping[str, Any], root: Path, label: str) -> None:
    expected = config["model"]
    recorded = run_config.get("model") if isinstance(run_config.get("model"), Mapping) else {}
    loader = run_config.get("model_loader") if isinstance(run_config.get("model_loader"), Mapping) else {}
    recorded_revision = recorded.get("revision") or loader.get("revision")
    if recorded_revision and str(recorded_revision) != str(expected["revision"]):
        raise ValueError(f"{label} revision={recorded_revision!r} does not match config revision={expected['revision']!r}: {root}")

    expected_remote = str(expected.get("modelscope_name_or_path") or expected.get("remote_name_or_path") or "")
    recorded_remote = str(recorded.get("modelscope_name_or_path") or loader.get("modelscope_name_or_path") or loader.get("remote_name_or_path") or "")
    if recorded_remote and expected_remote and recorded_remote != expected_remote:
        raise ValueError(f"{label} model={recorded_remote!r} does not match config model={expected_remote!r}: {root}")


def evaluation_runtime_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Stage 4 generation settings."""

    evaluation = config["evaluation"]
    return {
        "output_dir": str(evaluation["output_dir"]),
        "samples": int(evaluation["samples"]),
        "max_new_tokens": int(evaluation["max_new_tokens"]),
        "do_sample": bool(evaluation.get("do_sample", False)),
        "temperature": float(evaluation["temperature"]),
        "top_p": float(evaluation["top_p"]),
        "num_beams": int(evaluation["num_beams"]),
        "model_stages": list(evaluation.get("model_stages", [])),
        "metrics": list(evaluation.get("metrics", [])),
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory with collision protection."""

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty evaluation output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()

