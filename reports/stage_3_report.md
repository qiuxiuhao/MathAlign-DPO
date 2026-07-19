# Stage 3 Report

## Implemented

- Created `plans/stage_3.md`.
- Added Stage 3 SFT data validation for completed Stage 2 manifest, Stage 1
  manifest hash reference, SFT JSONL row counts, sha256 values, Mini SFT views,
  unique IDs, message roles, non-empty content, and `token_count: null`.
- Added tokenizer chat-template validation and true token length filtering with
  no truncation.
- Added shared model/tokenizer loader with separate MPS FP16 LoRA and CUDA NF4
  QLoRA branches.
- Added MPS backend preflight that rejects unavailable MPS and
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Added TRL `SFTTrainer` orchestration, output directory collision protection,
  smoke overrides, adapter saving, tokenizer saving, trainer state, metrics,
  loss history, runtime metadata, and adapter reload sanity generation.
- Added Stage 3 CLI: `python -m scripts.train_sft`.
- Added Stage 3 tests for SFT data validation, token filtering, model loader
  branching, runtime metadata, CLI behavior, and config compatibility.
- Updated configs with adapter reload settings.
- Updated docs and README for Stage 3 SFT behavior and commands.
- Added training dependencies to `requirements.txt`, excluding `bitsandbytes`.

## Files Added

- `plans/stage_3.md`
- `reports/stage_3_report.md`
- `scripts/train_sft.py`
- `src/mathalign_dpo/training/__init__.py`
- `src/mathalign_dpo/training/model_loader.py`
- `src/mathalign_dpo/training/runtime_metadata.py`
- `src/mathalign_dpo/training/sft_data.py`
- `src/mathalign_dpo/training/train_sft.py`
- `tests/test_model_loader.py`
- `tests/test_runtime_metadata.py`
- `tests/test_sft_data.py`
- `tests/test_train_sft_cli.py`

## Files Modified

- `README.md`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `docs/data_contract.md`
- `docs/design.md`
- `requirements.txt`
- `src/mathalign_dpo/config/load_config.py`
- `tests/test_config.py`
- `tests/test_packaging.py`

## Commands Executed

- `conda run -n mathalign-dpo python -m pip install -r requirements.txt`
- `conda run -n mathalign-dpo python -m pip install -e .`
- `conda run -n mathalign-dpo python -m pytest -q`
- `conda run -n mathalign-dpo python -m scripts.train_sft --help`
- `conda run -n mathalign-dpo python -m pytest tests/test_config.py tests/test_model_loader.py tests/test_train_sft_cli.py -q`
- `conda run -n mathalign-dpo python -m pip check`
- `conda run -n mathalign-dpo python -c "... package version check ..."`
- `conda run -n mathalign-dpo python -c "... MPS availability check ..."`
- `conda run -n mathalign-dpo python -c "... Stage 2 Mini SFT validation ..."`
- `conda run -n mathalign-dpo python -c "... Qwen tokenizer token length validation ..."`
- `conda run -n mathalign-dpo python -m scripts.train_sft --config configs/qwen25_0_5b_m5_24gb_mini.yaml --smoke-test --output-dir outputs/checkpoints/mini/sft_smoke --overwrite`
- `python -m scripts.train_sft --config configs/qwen25_0_5b_m5_24gb_mini.yaml --smoke-test --output-dir outputs/checkpoints/mini/sft_smoke --overwrite`
- `python -m scripts.train_sft --config configs/qwen25_0_5b_m5_24gb_mini.yaml`

## Test Results

- Unit tests before training dependency install: `67 passed`.
- Unit tests after training dependency install and MPS metadata fix:
  `67 passed`.
- Unit tests after TrainerState compatibility fix: `68 passed`.
- Focused Stage 3/formal compatibility tests: `13 passed`.
- CLI help: succeeded.
- Editable install: succeeded.
- Dependency check: no broken requirements found.
- Installed versions:
  - `torch`: `2.13.0`
  - `transformers`: `4.57.6`
  - `trl`: `0.29.1`
  - `peft`: `0.17.1`
  - `accelerate`: `1.14.0`
  - `datasets`: `5.0.0`
  - `safetensors`: `0.8.0`
  - `psutil`: `7.2.2`
- TRL API check:
  - `SFTConfig.completion_only_loss`: present
  - `SFTConfig.eval_strategy`: present
  - `SFTTrainer.processing_class`: present

## Data And Tokenizer Results

- Stage 2 Mini SFT selected rows before tokenizer filtering:
  - train: 253
  - validation: 32
- Qwen tokenizer chat template: present.
- Qwen tokenizer pad token before/after validation: `<|endoftext|>`.
- Full Mini SFT token filtering at `model.max_length = 512`:
  - train input: 253
  - train kept: 135
  - train filtered: 118
  - validation input: 32
  - validation kept: 16
  - validation filtered: 16
- Smoke SFT token filtering at `model.max_length = 512`:
  - train input: 64
  - train kept: 38
  - train filtered: 26
  - validation input: 16
  - validation kept: 7
  - validation filtered: 9

## MPS SFT Results

- Smoke SFT completed on MPS:
  - run ID: `20260719T155230Z_stage3_sft_smoke`
  - output dir: `outputs/checkpoints/mini/sft_smoke`
  - selected rows: train 64, validation 16
  - after token filtering: train 38, validation 7
  - max steps: 10
  - train loss: `0.6184526562690735`
  - train runtime: `15.8653` seconds
  - elapsed seconds: `39.056`
  - peak process memory: `2379.922 MB`
  - adapter reload samples: saved
- Mini SFT completed on MPS:
  - run ID: `20260719T155319Z_stage3_sft_mini`
  - output dir: `outputs/checkpoints/mini/sft/20260719T155319Z_stage3_sft_mini`
  - selected rows: train 253, validation 32
  - after token filtering: train 135, validation 16
  - max steps: 30
  - train loss: `0.5606042861938476`
  - train runtime: `50.1009` seconds
  - train samples/sec: `2.395`
  - train steps/sec: `0.599`
  - elapsed seconds: `73.128`
  - peak process memory: `2364.25 MB`
  - final adapter: `outputs/checkpoints/mini/sft/20260719T155319Z_stage3_sft_mini/final_adapter`
  - adapter reload samples: `outputs/checkpoints/mini/sft/20260719T155319Z_stage3_sft_mini/adapter_reload_samples.jsonl`
- MPS preflight was measured as:
  - `torch.backends.mps.is_built() = True`
  - `torch.backends.mps.is_available() = True`
- A Transformers warning was printed during reload generation:
  `temperature`, `top_p`, and `top_k` generation flags were not valid and may
  be ignored. The run still completed and deterministic adapter reload samples
  were saved.

## Known Limitations

- Stage 3 is a Mini training run only; it is not a formal performance result.
- Token filtering at `max_length = 512` removed 118 of 253 Mini train rows and
  16 of 32 Mini validation rows.
- The first post-install full test run crashed when calling
  `torch.mps.current_allocated_memory()`. Stage 3 now avoids `torch.mps` memory
  counters and records process peak memory instead.

## Deviations From Plan

- TRL resolved to `0.29.1`, which still satisfies the planned `trl>=0.21,<1.0`
  range. The planned API fields were verified against the installed version.
- PyTorch resolved to `2.13.0`, which satisfies `torch>=2.6,<3.0`.
- MPS-specific memory counters were skipped after a real segmentation fault in
  `torch.mps.current_allocated_memory()`.
- The first smoke SFT run completed training but failed while writing
  `trainer_state.json` because the installed Transformers `TrainerState` exposes
  `save_to_json()` rather than `to_json_string()`. The code now supports both
  APIs, and the rerun completed.

## Recommended Next Stage

Proceed to Stage 4 after review: Mini DPO using the Stage 2 preference data and
the completed Stage 3 SFT adapter. Do not run RTX 4090 formal DPO until the Mini
DPO path has completed and reported successfully.
