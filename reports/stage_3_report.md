# Stage 3 Report

## Implemented

- Implemented Mini SFT with TRL `SFTTrainer`, MPS FP16 LoRA, tokenizer length
  filtering, adapter save/reload validation, and runtime metadata.
- Added Stage 3 closeout repairs:
  - effective config is built before metadata collection;
  - metadata records `effective_config`, `original_config_path`, run mode,
    smoke state, and runtime overrides;
  - smoke metadata now records effective `sft.max_steps = 10`, not the original
    Mini `30`;
  - output writes now go to a hidden staging directory and publish only after
    training, explicit evaluation, adapter save, tokenizer save, metadata, and
    adapter reload validation all succeed;
  - failed staging runs preserve old published output directories;
  - both Qwen model configs pin real Hugging Face commit revisions;
  - model, tokenizer, and adapter reload model loading all pass the configured
    revision;
  - Mini SFT selection now filters by true tokenizer length first, expands from
    the Stage 2 Mini view to the deterministic formal Stage 2 candidate pool
    when needed, ranks by stable hash, and selects the exact configured target;
  - Trainer tokenized input lengths are asserted to be no greater than
    `model.max_length`;
  - train metrics and explicit post-train eval metrics are saved separately;
  - adapter reload uses the tokenizer saved in the current output directory;
  - run IDs include microseconds and a short UUID to avoid same-second
    collisions;
  - CUDA QLoRA loader order is quantized load, k-bit preparation, then LoRA.

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
- `.gitignore`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `docs/data_contract.md`
- `docs/design.md`
- `requirements.txt`
- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/training/model_loader.py`
- `src/mathalign_dpo/training/runtime_metadata.py`
- `src/mathalign_dpo/training/sft_data.py`
- `src/mathalign_dpo/training/train_sft.py`
- `tests/test_config.py`
- `tests/test_model_loader.py`
- `tests/test_packaging.py`
- `tests/test_runtime_metadata.py`
- `tests/test_sft_data.py`
- `tests/test_train_sft_cli.py`

## Commands Executed

- `conda run -n mathalign-dpo python -m pip install -r requirements.txt`
- `conda run -n mathalign-dpo python -m pip install -e .`
- `conda run -n mathalign-dpo python -m pytest -q`
- `conda run -n mathalign-dpo python -m scripts.train_sft --help`
- `conda run -n mathalign-dpo python -m pytest tests/test_config.py tests/test_model_loader.py tests/test_train_sft_cli.py -q`
- `conda run -n mathalign-dpo python -m pip check`
- `conda run -n mathalign-dpo python -c "... package version check ..."`
- `conda run -n mathalign-dpo python -c "... Hugging Face model revision lookup ..."`
- `conda run -n mathalign-dpo python -c "... Stage 2 Mini SFT candidate validation and Qwen tokenizer length selection ..."`
- `conda run -n mathalign-dpo python -m scripts.train_sft --config configs/qwen25_0_5b_m5_24gb_mini.yaml --smoke-test --output-dir outputs/checkpoints/mini/sft_smoke --overwrite`

## Test Results

- Unit tests after closeout repairs: `73 passed`.
- CLI help: succeeded.
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
- Added tests cover:
  - effective smoke overrides before metadata;
  - staging publish preserving old output until success;
  - failed staging leaving old output unchanged;
  - 511/512/513 token length boundaries;
  - exact target selection after expanded candidate pool;
  - TrainerState `save_to_json()` compatibility;
  - separate train/eval metric files;
  - model revision presence in both configs.

## Data And Tokenizer Results

- Model revisions:
  - `Qwen/Qwen2.5-0.5B-Instruct`:
    `7ae557604adf67be50417f59c2c2f167def9a775`
  - `Qwen/Qwen2.5-3B-Instruct`:
    `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Qwen tokenizer chat template: present.
- Qwen tokenizer pad token before/after validation: `<|endoftext|>`.
- Revised Mini SFT selection at `max_length = 512`:
  - train initial Mini candidates: 253
  - train expanded candidates: 4932
  - train kept after length filtering in expanded pool: 2653
  - train length filtered in expanded pool: 2279
  - final actual train rows: 256
  - train selected pool: expanded
  - train selection hash:
    `4e8df248e02ce5da407ef15ff9cae735583ff288b7c8dd88a36180565759298b`
  - validation initial Mini candidates: 32
  - validation expanded candidates: 198
  - validation kept after length filtering in expanded pool: 103
  - validation length filtered in expanded pool: 95
  - final actual validation rows: 32
  - validation selected pool: expanded
  - validation selection hash:
    `43ec6b9a43e0ae99889ab0bbe9046da860bfea4b15dbdb45f4368bb04eb18f17`
- `max_length = 512` is sufficient for exactly 256 Mini SFT training rows; no
  change to 768 is required.

## MPS Run Results

- Previous Stage 3 Mini SFT run completed before closeout repairs:
  - run ID: `20260719T155319Z_stage3_sft_mini`
  - selected rows: train 253, validation 32
  - after token filtering: train 135, validation 16
  - train loss: `0.5606042861938476`
  - elapsed seconds: `73.128`
  - peak process memory: `2364.25 MB`
- That previous run is not accepted as final for the revised closeout because it
  trained on 135 rows after token filtering, not the required 256 rows.
- The Codex tool execution context for the closeout repair reports:
  - `torch.backends.mps.is_built() = True`
  - `torch.backends.mps.is_available() = False`
- Therefore the repaired smoke and Mini SFT commands must be rerun from the
  user's MPS-available terminal session before Stage 3 can be marked fully
  runtime-accepted under the revised requirements.
- A repaired smoke attempt from Codex failed before model loading and preserved
  the old `outputs/checkpoints/mini/sft_smoke` result, confirming the new staging
  overwrite behavior does not delete old outputs before success.

## Known Limitations

- Revised MPS smoke and revised 256-row Mini SFT have not yet been measured in
  this tool context because MPS is unavailable here.
- Stage 3 remains Mini SFT only; no DPO training, Base/SFT/DPO evaluation, RTX
  4090 formal SFT, model merge, vLLM, FlashAttention, DeepSpeed, or FSDP was
  implemented.
- MPS-specific memory counters are intentionally skipped because
  `torch.mps.current_allocated_memory()` caused a real segmentation fault in the
  local test environment; Stage 3 records process peak memory instead.

## Deviations From Plan

- The revised exact-256 Mini SFT requirement required expanding beyond the
  original Stage 2 Mini SFT view after tokenizer filtering. Expansion is
  deterministic from the Stage 2 formal SFT view, never random.
- The output overwrite implementation was changed from direct directory deletion
  to staging plus publish.
- Adapter reload now uses the just-saved tokenizer directory instead of loading
  tokenizer files from the Hub.

## Recommended Next Stage

Do not proceed to Stage 4 yet. First rerun repaired Stage 3 smoke and repaired
Stage 3 Mini SFT from an MPS-available terminal, verify status `completed`,
confirm final actual train rows are 256, and update this report with measured
train/eval loss, elapsed time, peak process memory, effective config, and reload
results.
