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
- Completed Stage 2 closeout repairs:
  - Mini step/SFT/DPO views now derive only from Stage 1 manifest Mini IDs.
  - DPO candidates are ranked by stable hash before truncation.
  - Single source contribution is capped at 2 DPO pairs.
  - Final answer extraction records confidence and keeps low-confidence answers
    out of formal DPO.
  - Stage 2 orchestration moved into `src/mathalign_dpo/data/stage2_pipeline.py`.
  - Stage 2 manifest/statistics are separate from Stage 1 outputs.

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
- `src/mathalign_dpo/data/stage2_pipeline.py`
- `tests/test_parse_steps.py`
- `tests/test_select_views.py`
- `tests/test_stage2_builders.py`

## Files Modified

- `README.md`
- `docs/data_contract.md`
- `docs/design.md`
- `src/mathalign_dpo/data/write_outputs.py`
- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/data/build_preferences.py`
- `src/mathalign_dpo/data/parse_steps.py`
- `scripts/prepare_data.py`
- `tests/test_packaging.py`
- `tests/test_config.py`
- `tests/test_parse_steps.py`
- `tests/test_stage2_builders.py`
- `tests/test_stage2_pipeline.py`
- `tests/test_write_outputs.py`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`

## Commands Executed

- `conda run -n mathalign-dpo python -m pytest`
- `conda run -n mathalign-dpo python -m scripts.build_stage2_data --help`
- `conda run -n mathalign-dpo python -m scripts.build_stage2_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml --smoke-test --output-dir /tmp/mathalign_stage2_smoke --overwrite`
- `conda run -n mathalign-dpo python -m scripts.build_stage2_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml --overwrite`
- `conda run -n mathalign-dpo python -m scripts.prepare_data --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml --formal-config configs/qwen25_3b_4090.yaml --overwrite`
- `conda run -n mathalign-dpo python -m pip install -e .`
- `conda run -n mathalign-dpo python -c "import mathalign_dpo; import scripts.build_stage2_data; print('ok')"`
- `conda run -n mathalign-dpo python -c "... manifest/hash validation ..."`
- `conda run -n mathalign-dpo python -c "... Stage 1 clean / Stage 2 Mini source validation ..."`
- `git rev-parse --short HEAD`

## Test Results

- Unit tests after closeout repairs: `47 passed`.
- Stage 2 CLI help: succeeded.
- Editable install: succeeded after allowing pip network access for isolated build dependencies.
- Package and Stage 2 script imports: succeeded.
- Stage 1 manifest preservation:
  - `data/processed/split_manifest.json` is Stage 1 only.
  - no `stage2` section remains in Stage 1 manifest.
  - restored Stage 1 manifest sha256: `fa4ab9a44d3bd887f3117a3b05d9ea6ffa18416545618f6734f95e515b2162af`
- Stage 2 manifest/hash validation:
  - `data/processed/stage2_manifest.json` completed: true
  - Stage 2 file entries: 8
  - Stage 2 statistics sha256: `633a3ab4a6ff378a8237e81567239b1a08d3f35191fda1f6feeb253f2811eceb`
- Mini source validation:
  - Mini step/SFT/DPO source IDs are subsets of Stage 1 Mini IDs.
  - Mini source IDs are also subsets of the formal Stage 1 view.
  - Mini DPO train max pairs per source: 2.

## Data Results

- Git commit at report time: `cf74c87`
- Dataset: `AI-MO/NuminaMath-CoT`
- Revision: `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`
- Source split: `train`
- Token length status: `not_checked_no_tokenizer`
- Step rows:
  - train: 5000
  - validation: 200
  - evaluation: 200
- Parse status:
  - train: success 4772, partial 160, failed 68
  - validation: success 191, partial 7, failed 2
  - evaluation: success 185, partial 14, failed 1
- Answer confidence:
  - train: high 4840, medium 0, low 158, none 2
  - validation: high 193, medium 0, low 6, none 1
  - evaluation: high 186, medium 0, low 14, none 0
- SFT rows:
  - formal train: 4932
  - formal validation: 198
  - mini train: 253
  - mini validation: 32
- DPO rows:
  - formal train: 5000
  - formal validation: 200
  - mini train: 256
  - mini validation: 32
- Manual review rows: 100
- DPO applied mutation strategies:
  - train: number_mutation 2702, operator_mutation 2298
  - validation: number_mutation 104, operator_mutation 96

## Output Hashes

- `data/processed/step_train.jsonl`: `ff7f99c574117511a07d4c178c93b0483d1d3e6e248f93ea0fb2b40907617df1`
- `data/processed/step_validation.jsonl`: `80086d04b7b784f8096568866486e51066b527cd3a3745e83438d902ad4629c0`
- `data/processed/step_eval.jsonl`: `bf60b3a5a103c5983c6db08c1e85e25b0fdcbce8483ae5428b94d7522cc1aba5`
- `data/processed/sft_train.jsonl`: `dc610584988d6c201b013be72c1302bfe989a480cbd86140be89ff0902e2c578`
- `data/processed/sft_validation.jsonl`: `770ae5ffa5fa421aa7d938144185f35262a77578a38520fa7be4db4676608513`
- `data/processed/dpo_train.jsonl`: `f6d0a61c9b6b900f273e35e2016ccc9e22544f79161d24b66bf57cd984e68da9`
- `data/processed/dpo_validation.jsonl`: `f778c9c5746f8525dc27daa24dfdc6c5b25433ba7e54fbf91d40dc4e32612362`
- `data/processed/manual_review_preferences.jsonl`: `f78da1873ec544144f5685dc2a78e43dc74b09614bc14b0d0e9e3e91cf722463`
- `data/processed/stage2_statistics.json`: `633a3ab4a6ff378a8237e81567239b1a08d3f35191fda1f6feeb253f2811eceb`
- `data/processed/stage2_manifest.json`: `c81dd5da4df9ae094335240976d65650d7f1296694f3791f93a7424199faccbf`

## Known Limitations

- Stage 2 uses deterministic text heuristics only; it does not prove mathematical
  equivalence or evaluate final answers.
- Token counts are intentionally `null`; no tokenizer is downloaded or used.
- SFT rows are fewer than the formal train/validation caps because failed step
  parses remain in `step_*.jsonl` but are excluded from SFT.
- Mini SFT train has 253 rows because Stage 1 Mini train IDs contain 3 failed
  step parses. No formal-only source was borrowed to fill Mini.
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
- Stage 2 closeout changed Mini DPO selection from formal prefix slicing to
  Stage 1 Mini-ID candidate generation plus stable rank truncation.
- Stage 2 closeout separated `stage2_manifest.json` and `stage2_statistics.json`
  from Stage 1 `split_manifest.json` and `data_statistics.json`.
- Low-confidence numeric fallback answers now become partial parse rows and are
  excluded from formal DPO.

## Recommended Next Stage

Proceed to Stage 3 after review: Mini SFT with Qwen2.5-0.5B-Instruct on MPS FP16
LoRA. Do not begin Stage 3 until Stage 2 outputs and schemas are approved.
