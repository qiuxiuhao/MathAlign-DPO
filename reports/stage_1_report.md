# Stage 1 Report

## Implemented

- Created the Stage 1 plan in `plans/stage_1.md`.
- Added the formal package structure under `src/mathalign_dpo/`.
- Added dual-config loading and validation for Mini/formal shared data settings.
- Removed `project.stage` from both YAML configs.
- Pinned `AI-MO/NuminaMath-CoT` to revision `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`.
- Implemented NuminaMath field audit and normalized JSONL construction.
- Implemented stable ID selection with native ID preference and row-index fallback.
- Implemented deterministic source-level split assignment and stable split ordering.
- Implemented Mini/formal split views where Mini is a formal prefix subset.
- Implemented transactional output publishing through `.stage_<run_id>/`.
- Added `requirements.txt` and a conda environment named `mathalign-dpo`.
- Generated Stage 1 normalized outputs in `data/processed/`.
- Removed the temporary script-side Python search-path bootstrap.
- Split dependency responsibilities: `requirements.txt` is the only dependency source,
  while `pyproject.toml` only keeps project metadata, Python version, package
  discovery, and pytest configuration.
- Strengthened overwrite transactions with old-output backup and rollback.
- Confirmed row-index fallback IDs are assigned from original row positions before
  validation, filtering, splitting, or shuffling.
- Added source row content hashing to Stage 1 audit metadata.

## Files Added

- `.gitignore`
- `plans/stage_1.md`
- `pyproject.toml`
- `requirements.txt`
- `scripts/__init__.py`
- `scripts/prepare_data.py`
- `src/mathalign_dpo/__init__.py`
- `src/mathalign_dpo/config/__init__.py`
- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/data/__init__.py`
- `src/mathalign_dpo/data/load_numina.py`
- `src/mathalign_dpo/data/split_normalized.py`
- `src/mathalign_dpo/data/write_outputs.py`
- `tests/test_config.py`
- `tests/test_normalize_numina.py`
- `tests/test_packaging.py`
- `tests/test_split_normalized.py`
- `tests/test_write_outputs.py`
- `reports/stage_1_report.md`

## Files Modified

- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `pyproject.toml`
- `requirements.txt`
- `scripts/prepare_data.py`
- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/data/load_numina.py`
- `src/mathalign_dpo/data/write_outputs.py`
- `tests/test_normalize_numina.py`
- `tests/test_packaging.py`
- `tests/test_write_outputs.py`

## Commands Executed

- `conda create -n mathalign-dpo python=3.11 -y`
- `conda run -n mathalign-dpo python -m pip install -r requirements.txt`
- `conda run -n mathalign-dpo python -m pytest`
- `conda run -n mathalign-dpo python -m scripts.prepare_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml --smoke-test --output-dir /tmp/mathalign_stage1_smoke --overwrite`
- `conda run -n mathalign-dpo python -m scripts.prepare_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml`
- `wc -l data/processed/normalized_train.jsonl data/processed/normalized_validation.jsonl data/processed/normalized_eval.jsonl`
- `conda run -n mathalign-dpo python -m pip install -r requirements.txt`
- `conda run -n mathalign-dpo python -m pip install -e .`
- `conda run -n mathalign-dpo python -c "import mathalign_dpo"`
- `conda run -n mathalign-dpo python -m scripts.prepare_data --help`
- `conda run -n mathalign-dpo python -m pip check`
- `conda run -n mathalign-dpo python -c "from pathlib import Path; from mathalign_dpo.data.write_outputs import validate_completed_manifest; validate_completed_manifest(Path('data/processed/split_manifest.json')); print('manifest ok')"`

## Test Results

- Unit tests after initial Stage 1 implementation: `15 passed`.
- Unit tests after closeout repairs: `26 passed`.
- Editable install: succeeded.
- Package import from outside repo: succeeded.
- `python -m scripts.prepare_data --help` after editable install: succeeded.
- Dependency check: no broken requirements found.
- Static search-path bootstrap check: no forbidden script/source path mutation found.
- Smoke data preparation:
  - train: 64
  - validation: 16
  - evaluation: 16
  - output dir: `/tmp/mathalign_stage1_smoke`
- Formal Stage 1 data preparation:
  - train: 5000
  - validation: 200
  - evaluation: 200
  - output dir: `data/processed`
- Output line counts:
  - `data/processed/normalized_train.jsonl`: 5000
  - `data/processed/normalized_validation.jsonl`: 200
  - `data/processed/normalized_eval.jsonl`: 200
- Manifest check:
  - `completed`: true
  - Mini view counts: train 256, validation 32, evaluation 32
  - Mini views are formal prefix subsets: true

## Data Audit

- Dataset: `AI-MO/NuminaMath-CoT`
- Revision: `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`
- Source split: `train`
- Source rows: 859494
- Raw fields: `messages`, `problem`, `solution`, `source`
- Problem field: `problem`
- Solution field: `solution`
- Native ID field: none
- ID strategy: `row_index_fallback`
- Source rows sha256: `2ef15ea841934eeb5009e3c72d287bc8d47655d0fe2b668673d93657485a441e`
- Normalized rows: 859493
- Rejected rows: 1
- Rejection reasons:
  - `problem_equals_solution`: 1

## Output Hashes

- `data/processed/normalized_train.jsonl`: `3e30cd1a5e48e7fae47be28b511df4797ae8230e4c461ebadc6b86bb8d2409c4`
- `data/processed/normalized_validation.jsonl`: `05978c026409b10ea3d31f59a3e83cbdc8914774673b872086ed9102e64ff660`
- `data/processed/normalized_eval.jsonl`: `006b6e991ed1d63fc7b4608f16ae19060ff7b251df45ac7bfd7148d44a81f21a`
- `data/processed/data_statistics.json`: `d042d31527dfbe789372ef7180f471365eab200d3cabf422af94dd5ce627b599`
- `data/processed/split_manifest.json`: `53d1a7d94adb7388274bd4f496e0005fd5878f1d3780217ea5089d239de43e26`

## Closeout Repair Verification

- Search-path bootstrap removal:
  - Removed the script-side `sys` import and local `src` injection from `scripts/prepare_data.py`.
  - No script or package module now mutates Python module search paths.
- Dependency ownership:
  - `requirements.txt` contains bounded Stage 1 dependencies only: `datasets`,
    `huggingface_hub`, `pyarrow`, `PyYAML`, and `pytest`.
  - `pyproject.toml` contains no runtime dependency list and no optional dependency
    list, avoiding duplicate dependency maintenance.
- Transaction tests added:
  - Existing output is preserved when a new overwrite run fails during staged JSONL writing.
  - Existing output is preserved when a new overwrite run fails during staged count/hash validation.
  - Existing output is restored when final publish fails after at least one replacement attempt.
  - Completed manifests are validated by checking file existence, row counts, and sha256 values.
- ID assignment tests added:
  - Rejected middle rows do not cause later row-index fallback IDs to move forward.
  - Repeated normalization produces identical IDs.
  - Relaxing a filter for an earlier row does not renumber later rows.
  - Row-index IDs are fixed before split or shuffle logic sees the examples.

## Known Limitations

- The local directory is not a git repository, so no git commit could be recorded.
- Hugging Face access was unauthenticated, so runs may be subject to public rate limits.
- Stage 1 does not validate step, SFT, DPO, tokenizer length, model, or training behavior.

## Deviations From Plan

- `requirements.txt` is the sole dependency source per user request.
- The earlier script-side path bootstrap was removed; editable install is now the
  supported way to expose `mathalign_dpo`.
- The first smoke run needed non-sandbox network access, and the first formal run
  needed non-sandbox access to the Hugging Face cache lock.
- The formal Stage 1 outputs were republished with `--overwrite` after adding source
  row hashing to the manifest/statistics.

## Recommended Next Stage

Proceed to Stage 2 after review: reasoning step splitting, final answer extraction,
SFT sample construction, rule-based mutation, DPO preference construction, manual
review samples, and schema validation.
