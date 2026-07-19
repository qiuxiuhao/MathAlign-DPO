# Stage 2 Report

## Implemented

- Created `plans/stage_2.md`.
- Added Stage 2 parsing, answer extraction, SFT construction, deterministic mutation,
  DPO preference construction, manual review sampling, and manifest view selection.
- Added `scripts/build_stage2_data.py` as the Stage 2 CLI.
- Reused the Stage 1 transactional writer with manifest-last publication and rollback.
- Updated `docs/data_contract.md`, `docs/design.md`, and `README.md` for Stage 2
  schemas, commands, manifest behavior, and the no-tokenizer length policy.
- Generated Stage 2 smoke outputs in `/tmp/mathalign_stage2_smoke`.
- Generated formal Stage 2 outputs in `data/processed`.

## Files Added

- `plans/stage_2.md`
- `reports/stage_2_report.md`
- `scripts/build_stage2_data.py`
- `src/mathalign_dpo/data/build_preferences.py`
- `src/mathalign_dpo/data/build_sft.py`
- `src/mathalign_dpo/data/mutate_steps.py`
- `src/mathalign_dpo/data/parse_steps.py`
- `src/mathalign_dpo/data/prompts.py`
- `src/mathalign_dpo/data/select_views.py`
- `tests/test_parse_steps.py`
- `tests/test_select_views.py`
- `tests/test_stage2_builders.py`

## Files Modified

- `README.md`
- `docs/data_contract.md`
- `docs/design.md`
- `src/mathalign_dpo/data/write_outputs.py`
- `tests/test_packaging.py`
- `tests/test_write_outputs.py`

## Commands Executed

- `conda run -n mathalign-dpo python -m pytest`
- `conda run -n mathalign-dpo python -m scripts.build_stage2_data --help`
- `conda run -n mathalign-dpo python -m scripts.build_stage2_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml --smoke-test --output-dir /tmp/mathalign_stage2_smoke --overwrite`
- `conda run -n mathalign-dpo python -m scripts.build_stage2_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml --overwrite`
- `conda run -n mathalign-dpo python -m pip install -e .`
- `conda run -n mathalign-dpo python -c "import mathalign_dpo; import scripts.build_stage2_data; print('ok')"`
- `conda run -n mathalign-dpo python -c "... manifest/hash validation ..."`
- `git rev-parse --short HEAD`

## Test Results

- Unit tests: `42 passed`.
- Stage 2 CLI help: succeeded.
- Editable install: succeeded after allowing pip network access for isolated build dependencies.
- Package and Stage 2 script imports: succeeded.
- Stage 2 manifest/hash validation:
  - `stage2.completed`: true
  - Stage 2 file entries: 8
  - statistics sha256: `98d79183dbb561558a0f9b11cffc874ebdfec1daa33c02c8ee8de79a9b68d5c9`

## Data Results

- Git commit at report time: `02c1d46`
- Dataset: `AI-MO/NuminaMath-CoT`
- Revision: `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`
- Source split: `train`
- Token length status: `not_checked_no_tokenizer`
- Step rows:
  - train: 5000
  - validation: 200
  - evaluation: 200
- Parse status:
  - train: success 4930, partial 2, failed 68
  - validation: success 197, partial 1, failed 2
  - evaluation: success 199, partial 0, failed 1
- SFT rows:
  - train: 4932
  - validation: 198
- DPO rows:
  - train: 5000
  - validation: 200
- Manual review rows: 100
- DPO applied mutation strategies:
  - train: number_mutation 2602, operator_mutation 2398
  - validation: number_mutation 94, operator_mutation 106

## Output Hashes

- `data/processed/step_train.jsonl`: `16de24798377fa03dbd6a65005feab453aebdc82fd799d37806f728a2e49c08b`
- `data/processed/step_validation.jsonl`: `200ebd7575fcefe2416811d918e1c43382748b6cd7ecf1beb9dfcb15d3e9a62f`
- `data/processed/step_eval.jsonl`: `30f8a050b95d6a5a5299c60ae6579a044881c05b845354b6dd7fa011eb9629d7`
- `data/processed/sft_train.jsonl`: `4d5d0c63951aab5544db6e9e827a0794ce48a4f14b20cdc6fd553a8d073e53e2`
- `data/processed/sft_validation.jsonl`: `ce5f353b7f447309382d6f5531c88b97307d147ed91df0068ab51293b673ea0a`
- `data/processed/dpo_train.jsonl`: `43cd70db7587c54c8d04556782f782a12c13d7f565c2e7986bea9a0f6f169f13`
- `data/processed/dpo_validation.jsonl`: `269fe38c95c8c007f30588d38038ab24e8c98d886f3a75e189c8f24c59ad7551`
- `data/processed/manual_review_preferences.jsonl`: `0b811c7d9e1447a8b171b6854e2aa81e1aa9fd814a8064220ba6f87f1f1d6c87`
- `data/processed/data_statistics.json`: `98d79183dbb561558a0f9b11cffc874ebdfec1daa33c02c8ee8de79a9b68d5c9`
- `data/processed/split_manifest.json`: `55b052c83483b8ee707cefe8a3c3d136ac37c1697b33dc0c86f81554bd45a0ad`

## Known Limitations

- Stage 2 uses deterministic text heuristics only; it does not prove mathematical
  equivalence or evaluate final answers.
- Token counts are intentionally `null`; no tokenizer is downloaded or used.
- SFT rows are fewer than the formal train/validation caps because failed step
  parses remain in `step_*.jsonl` but are excluded from SFT.
- Some mutation candidates are skipped when no safe local edit exists or when the
  rejected step would appear in prompt history.
- Smoke mode scans a slightly larger deterministic Stage 1 prefix so it can still
  produce the configured smoke SFT/DPO counts when early examples fail parsing.

## Deviations From Plan

- The strict string check that rejected both chosen and rejected text in prompt
  history was relaxed for chosen text. The builder already constructs prompts from
  `steps[:step_index]`; exact chosen text can legitimately appear in earlier
  repeated correct steps.
- The first formal run exposed rejected text duplicated in prompt history; those
  pairs are now skipped and counted as `rejected_in_prompt_history`.
- The first `pip install -e .` attempt failed under restricted network while pip
  resolved isolated build dependencies; the same command succeeded after approval.

## Recommended Next Stage

Proceed to Stage 3 after review: Mini SFT with Qwen2.5-0.5B-Instruct on MPS FP16
LoRA. Do not begin Stage 3 until Stage 2 outputs and schemas are approved.
