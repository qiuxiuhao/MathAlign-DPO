"""Stage 1 data preparation entrypoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mathalign_dpo.config.load_config import load_stage1_configs, output_paths, sample_counts, split_ratios
from mathalign_dpo.data.load_numina import load_numina_dataset, normalize_rows
from mathalign_dpo.data.split_normalized import split_examples
from mathalign_dpo.data.write_outputs import publish_stage1_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 normalized NuminaMath data.")
    parser.add_argument("--mini-config", required=True, help="Path to the Mini YAML config.")
    parser.add_argument("--formal-config", required=True, help="Path to the formal YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a small deterministic row prefix.")
    parser.add_argument("--output-dir", default=None, help="Override Stage 1 output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing Stage 1 outputs.")
    args = parser.parse_args()

    result = prepare_data(
        mini_config=args.mini_config,
        formal_config=args.formal_config,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def prepare_data(
    mini_config: str | Path,
    formal_config: str | Path,
    smoke_test: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 1 preparation and publish normalized outputs."""

    configs = load_stage1_configs(mini_config, formal_config)
    rows = list(load_numina_dataset(configs.dataset_name, configs.dataset_revision, configs.source_split))
    if smoke_test:
        needed = _smoke_source_rows(configs.formal)
        rows = rows[:needed]

    normalized = normalize_rows(
        rows=rows,
        dataset_name=configs.dataset_name,
        dataset_revision=configs.dataset_revision,
        source_split=configs.source_split,
        preprocessing=configs.formal["preprocessing"],
    )
    split = split_examples(
        examples=normalized.examples,
        dataset_name=configs.dataset_name,
        dataset_revision=configs.dataset_revision,
        source_split=configs.source_split,
        seed=configs.seed,
        ratios=split_ratios(configs.formal),
        mini_config=_config_with_smoke_counts(configs.mini) if smoke_test else configs.mini,
        formal_config=_config_with_smoke_counts(configs.formal) if smoke_test else configs.formal,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_stage1")
    stats = build_statistics(configs, normalized, split.formal_ids, split.mini_ids, smoke_test)
    manifest = build_manifest(configs, normalized.audit, split.formal_ids, split.mini_ids, smoke_test, run_id)
    published = publish_stage1_outputs(
        canonical=split.canonical,
        statistics=stats,
        manifest=manifest,
        output_paths=output_paths(configs.formal, output_dir=output_dir),
        overwrite=overwrite,
        run_id=run_id,
    )
    return {
        "run_id": run_id,
        "id_strategy": normalized.audit.id_strategy,
        "counts": published.counts,
        "hashes": published.hashes,
        "paths": {name: str(path) for name, path in published.paths.items()},
    }


def build_statistics(
    configs: Any,
    normalized: Any,
    formal_ids: dict[str, list[str]],
    mini_ids: dict[str, list[str]],
    smoke_test: bool,
) -> dict[str, Any]:
    """Build Stage 1 statistics only."""

    return {
        "schema_version": "1.0",
        "stage": 1,
        "seed": configs.seed,
        "dataset_name": configs.dataset_name,
        "dataset_revision": configs.dataset_revision,
        "source_split": configs.source_split,
        "smoke_test": smoke_test,
        "source_rows": normalized.audit.row_count,
        "normalized_rows": len(normalized.examples),
        "normalization_rejected": sum(normalized.rejected.values()),
        "normalization_rejected_by_reason": normalized.rejected,
        "id_strategy": normalized.audit.id_strategy,
        "id_field": normalized.audit.id_field,
        "split_counts_formal": {split: len(ids) for split, ids in formal_ids.items()},
        "split_counts_mini": {split: len(ids) for split, ids in mini_ids.items()},
        "field_audit": {
            "fields": normalized.audit.fields,
            "field_types": normalized.audit.field_types,
            "empty_counts": normalized.audit.empty_counts,
            "problem_field": normalized.audit.problem_field,
            "solution_field": normalized.audit.solution_field,
            "source_rows_sha256": normalized.audit.source_rows_sha256,
        },
    }


def build_manifest(
    configs: Any,
    audit: Any,
    formal_ids: dict[str, list[str]],
    mini_ids: dict[str, list[str]],
    smoke_test: bool,
    run_id: str,
) -> dict[str, Any]:
    """Build split manifest before file hashes are attached by the writer."""

    return {
        "schema_version": "1.0",
        "stage": 1,
        "completed": False,
        "run_id": run_id,
        "dataset_name": configs.dataset_name,
        "dataset_revision": configs.dataset_revision,
        "source_split": configs.source_split,
        "seed": configs.seed,
        "smoke_test": smoke_test,
        "split_method": "sha256_source_id_bucket_v1",
        "split_ratios": split_ratios(configs.formal),
        "id_strategy": audit.id_strategy,
        "id_field": audit.id_field,
        "source_rows_sha256": audit.source_rows_sha256,
        "configs": {
            "mini": str(configs.mini_path),
            "formal": str(configs.formal_path),
        },
        "views": {
            "formal": formal_ids,
            "mini": mini_ids,
        },
    }


def _config_with_smoke_counts(config: dict[str, Any]) -> dict[str, Any]:
    copied = dict(config)
    copied["data"] = dict(config["data"])
    copied["data"]["train_samples"] = int(config["smoke_test"]["train_samples"])
    copied["data"]["validation_samples"] = int(config["smoke_test"]["validation_samples"])
    copied["data"]["evaluation_samples"] = int(config["smoke_test"]["evaluation_samples"])
    return copied


def _smoke_source_rows(config: dict[str, Any]) -> int:
    counts = sample_counts(_config_with_smoke_counts(config))
    # Ratios include 2.5% validation/evaluation tails, so read enough rows for
    # stable smoke counts without touching the full dataset.
    return max(2000, sum(counts.values()) * 25)


if __name__ == "__main__":
    main()
