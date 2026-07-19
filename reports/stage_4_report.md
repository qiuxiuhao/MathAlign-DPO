# Stage 4 Report

## Implemented

- Added Mini DPO data validation, tokenizer filtering, SFT adapter validation,
  TRL `DPOTrainer` orchestration, transactional staging, and DPO CLI.
- Added deterministic formal-pool expansion after tokenizer filtering so Mini
  DPO can obtain 256 train rows and 32 validation rows when the Mini DPO view is
  too short under `max_length=512` and `max_prompt_length=384`.
- Added base-model loading without fresh LoRA and Stage 3 SFT adapter policy
  initialization for DPO.
- Added metadata support for Stage 4 run IDs and DPO effective config fields.
- Added JSON-safe metric writing so MPS/TRL non-finite values such as NaN are
  recorded as JSON `null` instead of crashing artifact publication.
- Reduced DPO adapter reload validation to DPO-specific lightweight defaults
  and release MPS memory before reload/generation to avoid post-training Metal
  command-buffer failures.
- Updated README, design docs, and data contract for Stage 4 Mini DPO.

## Files Added

- `plans/stage_4.md`
- `scripts/train_dpo.py`
- `src/mathalign_dpo/training/dpo_data.py`
- `src/mathalign_dpo/training/train_dpo.py`
- `tests/test_dpo_data.py`
- `tests/test_train_dpo_cli.py`
- `reports/stage_4_report.md`

## Files Modified

- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/training/model_loader.py`
- `src/mathalign_dpo/training/runtime_metadata.py`
- `tests/test_config.py`
- `tests/test_model_loader.py`
- `tests/test_packaging.py`
- `tests/test_runtime_metadata.py`

## Commands Executed

- `conda run -n mathalign-dpo python -m pytest -q`
- `conda run -n mathalign-dpo python -m scripts.train_dpo --help`
- `conda run -n mathalign-dpo python -m scripts.train_dpo --config configs/qwen25_0_5b_m5_24gb_mini.yaml --sft-run-dir outputs/checkpoints/mini/sft/20260719T155319Z_stage3_sft_mini --max-steps 1 --train-samples 1 --validation-samples 1`
- User MPS smoke attempt: `python -m scripts.train_dpo --config configs/qwen25_0_5b_m5_24gb_mini.yaml --sft-run-dir outputs/checkpoints/mini/sft/20260719T170915550766Z_stage3_sft_mini_d4226f86 --smoke-test --output-dir outputs/checkpoints/mini/dpo_smoke --overwrite`
- User MPS Mini DPO: `python -m scripts.train_dpo --config configs/qwen25_0_5b_m5_24gb_mini.yaml --sft-run-dir outputs/checkpoints/mini/sft/20260719T170915550766Z_stage3_sft_mini_d4226f86`

## Test Results

- Unit tests after Stage 4 fixes: `100 passed`.
- Stage 4 CLI help: succeeded.
- Stage 4 preflight with the old Stage 3 Mini SFT run failed before model
  loading, as expected, because the old metadata does not contain the repaired
  fixed model revision required by Stage 4.
- User MPS smoke attempt completed TRL train/eval but hit NaN eval metrics and
  failed while writing strict JSON. The implementation now sanitizes non-finite
  float values to JSON `null`; tests cover this behavior.
- User MPS Mini DPO completed:
  - run ID: `20260719T173512557701Z_stage4_dpo_mini_53bfedfa`
  - output dir: `outputs/checkpoints/mini/dpo/20260719T173512557701Z_stage4_dpo_mini_53bfedfa`
  - source SFT run: `20260719T170915550766Z_stage3_sft_mini_d4226f86`
  - status: `completed`
  - elapsed seconds: `485.5`
  - peak process memory: `3276.953 MB`
  - train runtime: `337.6991`
  - eval runtime: `124.5176`
  - train loss: `0.6924437075853348`
  - eval loss: `0.6906373500823975`
  - candidate pools: train initial 256 / expanded 5000; validation initial 32
    / expanded 200
  - after tokenizer filtering: train 3338, validation 127
  - final actual rows: train 256, validation 32
  - adapter reload samples: 1
  - preference validation rows: 3
  - loss history rows: 23
  - artifacts saved: final adapter, tokenizer, trainer state, train/eval
    metrics, loss history, preference validation, adapter reload samples, and
    run metadata.
- Added tests cover:
  - DPO manifest completion, file sha256, duplicate IDs, missing view IDs, row
    schema, `chosen != rejected`, rejected prompt leaks, and null Stage 2
    `token_count`;
  - tokenizer prompt/total/completion limits and deterministic expansion from
    Mini to formal DPO pools;
  - DPO config validation for beta, loss type, length boundaries, and MPS
    optimizer;
  - required `--sft-run-dir`, smoke overrides, output collision protection,
    Stage 3 SFT adapter metadata validation, and trainer tokenized length
    assertions;
  - base-model loading without fresh LoRA and trainable policy initialization
    from an SFT adapter.

## Known Limitations

- The observed MPS DPO smoke run produced NaN metrics after several steps.
  Stage 4 now records those values safely as `null`, but training stability
  should be reviewed before treating Mini DPO metrics as meaningful.
- Mini DPO has completed on MPS and validates the Stage 4 training chain, but
  Mini metrics are not formal model-quality conclusions.
- Stage 4 does not run RTX 4090 formal DPO, Base/SFT/DPO full evaluation,
  ablations, model merge, or custom trainer logic.

## Deviations From Plan

- Chosen/rejected sanity output records sampled validation pairs and latest TRL
  eval reward metrics when available, rather than running a separate custom
  per-row DPO log-prob loop. This keeps Stage 4 on the official TRL path and
  avoids introducing duplicate loss/math code.

## Recommended Next Stage

Proceed to Stage 5 after review: implement unified Base/SFT/DPO evaluation using
the completed Stage 3 SFT adapter and Stage 4 DPO adapter. Do not start formal
RTX 4090 DPO, ablations, or Stage 6 experiments until Stage 5 is reviewed.
