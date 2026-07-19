# Stage 1 Plan: NuminaMath Normalization And Deterministic Splits

## Goals

Stage 1 implements the data preparation foundation for `AI-MO/NuminaMath-CoT`.
It loads both approved project configs, audits the real source fields, pins the
dataset revision, normalizes raw samples into the data contract, assigns stable
IDs, creates deterministic source-level splits, and writes normalized JSONL,
split manifest, and statistics through a complete output transaction.

## Scope

- Read Mini and formal YAML configs together.
- Validate shared data settings across both configs.
- Keep `project.stage` out of config files.
- Require a non-empty fixed `data.dataset_revision`.
- Load NuminaMath from the configured dataset name, revision, and source split.
- Audit source fields, candidate ID fields, field types, empty values, and row
  count.
- Prefer a native stable ID field when one exists; otherwise use the fixed
  revision plus source split row index as the fallback `source_id`.
- Normalize only the Stage 1 schema fields: `problem`, `solution`, `source`,
  `source_split`, `source_id`, `id`, `schema_version`, and `metadata`.
- Split by `source_id` before any later step-level processing.
- Make Mini views deterministic prefix subsets of the formal canonical views.
- Write outputs through `data/processed/.stage_<run_id>/` and publish only after
  schema, count, and sha256 checks pass.

## Non-Goals

- No reasoning step splitting.
- No final answer extraction.
- No SFT or DPO example construction.
- No negative sampling.
- No tokenizer length filtering.
- No model loading, training, inference, or evaluation.
- No multi-dataset registry, factory, plugin, workflow engine, or future-facing
  abstraction.

## Files

Add:

- `plans/stage_1.md`
- `pyproject.toml`
- `.gitignore`
- `src/mathalign_dpo/__init__.py`
- `src/mathalign_dpo/config/__init__.py`
- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/data/__init__.py`
- `src/mathalign_dpo/data/load_numina.py`
- `src/mathalign_dpo/data/split_normalized.py`
- `src/mathalign_dpo/data/write_outputs.py`
- `scripts/__init__.py`
- `scripts/prepare_data.py`
- `tests/test_config.py`
- `tests/test_normalize_numina.py`
- `tests/test_split_normalized.py`
- `tests/test_write_outputs.py`
- `reports/stage_1_report.md`

Modify:

- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `README.md`
- `docs/design.md`
- `docs/data_contract.md`

## Data Flow

1. Load Mini and formal config files in one call.
2. Validate matching `dataset_name`, `dataset_revision`, `source_split`, seed,
   split ratios, and canonical output paths.
3. Load the fixed Hugging Face dataset revision with `datasets.load_dataset`.
4. Audit raw rows and select `source_id` policy: native stable ID first, row
   index fallback.
5. Normalize rows, reject contract violations, and record rejection reasons.
6. Assign split by stable sha256 bucket over dataset, revision, source split,
   source ID, and seed.
7. Sort each split by a separate stable sha256 rank.
8. Select canonical formal counts and record Mini as deterministic prefix
   subsets.
9. Write JSONL, statistics, and manifest to a staging directory.
10. Validate staging outputs and publish final files only when complete.

## Key Design Decisions

- `train_ratio`, `validation_ratio`, and `evaluation_ratio` control split
  membership only. Per-split sample counts control how many already-assigned
  examples are materialized for each run mode.
- Canonical normalized files are shared. They are sized from the formal config,
  while Mini membership is recorded in `split_manifest.json`.
- `source_id` is selected after field audit. Native unique string/integer ID
  fields are preferred; if NuminaMath has no such field, row index fallback is
  deterministic because the dataset commit revision is fixed.
- Output paths are not partially updated. A run succeeds only after all staged
  files are valid and the manifest has `completed: true`.
- Stage 1 statistics include only Stage 1 facts. Step/SFT/DPO counters remain
  absent until those stages create them.

## Test Plan

- Config tests: both configs parse, `project.stage` is absent, revisions match
  and are non-empty, split ratios sum to 1, Mini sample counts do not exceed
  formal counts, and MPS/CUDA quantization rules hold.
- Normalization tests: native ID is preferred, row index fallback is stable,
  text cleanup follows the contract, and invalid rows are rejected with counted
  reasons.
- Split tests: repeated runs are deterministic, split membership is disjoint,
  Mini views are prefix subsets of formal views, and sample counts do not
  affect split membership.
- Transaction tests: no final manifest is published on staged failure, existing
  outputs fail without overwrite, successful publish writes matching counts and
  sha256 values.

## Commands

```bash
python -m pytest
```

```bash
python -m scripts.prepare_data \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --smoke-test \
  --output-dir /tmp/mathalign_stage1_smoke \
  --overwrite
```

```bash
python -m scripts.prepare_data \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml
```

## Risks

- Hugging Face network or cache availability may block real dataset loading.
- NuminaMath may not contain a native stable ID field; in that case row-index
  fallback is expected and recorded.
- Full formal output depends on enough normalized rows after rejection.
- The local environment must install Stage 1 dependencies before running tests
  or data preparation.

## Acceptance Criteria

- `plans/stage_1.md` exists.
- Both YAML configs have no `project.stage` and pin the same dataset revision.
- Stage 1 code lives under `src/mathalign_dpo/`.
- Unit tests pass.
- Smoke data preparation can run on Mac CPU without model downloads.
- Normalized JSONL, split manifest, and data statistics obey
  `docs/data_contract.md`.
- Mini IDs are a traceable subset of formal IDs.
- `reports/stage_1_report.md` records commands, results, revision, ID policy,
  output hashes, limitations, deviations, and recommended next stage.
- No Stage 2+ functionality is implemented.
