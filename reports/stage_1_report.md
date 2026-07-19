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
- `tests/test_split_normalized.py`
- `tests/test_write_outputs.py`
- `reports/stage_1_report.md`

## Files Modified

- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`

## Commands Executed

- `conda create -n mathalign-dpo python=3.11 -y`
- `conda run -n mathalign-dpo python -m pip install -r requirements.txt`
- `conda run -n mathalign-dpo python -m pytest`
- `conda run -n mathalign-dpo python -m scripts.prepare_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml --smoke-test --output-dir /tmp/mathalign_stage1_smoke --overwrite`
- `conda run -n mathalign-dpo python -m scripts.prepare_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml`
- `wc -l data/processed/normalized_train.jsonl data/processed/normalized_validation.jsonl data/processed/normalized_eval.jsonl`

## Test Results

- Unit tests: `15 passed`.
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
- Normalized rows: 859493
- Rejected rows: 1
- Rejection reasons:
  - `problem_equals_solution`: 1

## Output Hashes

- `data/processed/normalized_train.jsonl`: `3e30cd1a5e48e7fae47be28b511df4797ae8230e4c461ebadc6b86bb8d2409c4`
- `data/processed/normalized_validation.jsonl`: `05978c026409b10ea3d31f59a3e83cbdc8914774673b872086ed9102e64ff660`
- `data/processed/normalized_eval.jsonl`: `006b6e991ed1d63fc7b4608f16ae19060ff7b251df45ac7bfd7148d44a81f21a`
- `data/processed/data_statistics.json`: `2acd9eadceb265c3920db760c53944466462704ecba33badda3a8b038a9582e3`
- `data/processed/split_manifest.json`: `631a61e720e6e9a624eb1b32b9b09063bf8e74f8be7ed18d6507d273e8747006`

## Known Limitations

- The local directory is not a git repository, so no git commit could be recorded.
- Hugging Face access was unauthenticated, so runs may be subject to public rate limits.
- Stage 1 does not validate step, SFT, DPO, tokenizer length, model, or training behavior.

## Deviations From Plan

- `requirements.txt` was added as the dependency source per user request.
- A small `src` path bootstrap was added to `scripts/prepare_data.py` so the planned
  `python -m scripts.prepare_data` command works directly from the source tree.
- The first smoke run needed non-sandbox network access, and the first formal run
  needed non-sandbox access to the Hugging Face cache lock.

## Recommended Next Stage

Proceed to Stage 2 after review: reasoning step splitting, final answer extraction,
SFT sample construction, rule-based mutation, DPO preference construction, manual
review samples, and schema validation.
