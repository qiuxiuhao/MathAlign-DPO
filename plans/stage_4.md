# Stage 4 Plan: Mini DPO On MPS With Qwen2.5 SFT Adapter

## Goals

Stage 4 implements Mini DPO training on Mac M5 24GB using PyTorch MPS,
`Qwen/Qwen2.5-0.5B-Instruct`, the completed Stage 3 SFT LoRA adapter, Stage 2
Mini DPO preference data, PEFT, and official TRL `DPOTrainer`.

The stage validates Stage 2 DPO manifest integrity, performs real tokenizer
length checks, trains a small DPO adapter, reloads it, runs chosen/rejected and
generation sanity checks, and records truthful run metadata.

## Scope

- Validate `stage2_manifest.json`, Stage 1 reference hash, DPO JSONL hashes,
  row counts, Mini/formal DPO views, schema, unique IDs, source IDs, and
  `token_count: null`.
- Load the tokenizer saved by the Stage 3 SFT run and require its built-in chat
  template.
- Compute TRL-compatible conversational token lengths for prompt, chosen, and
  rejected.
- Enforce `dpo.max_prompt_length` before creating `DPOTrainer`; do not rely on
  TRL truncation.
- Select legal Mini DPO train and validation rows only from the Stage 2 Mini DPO view.
  The current Mini config uses 179 train rows and 21 validation rows because the
  true tokenizer keeps only those Mini rows under the 512/384 limits.
- Initialize policy from a completed non-smoke Stage 3 Mini SFT adapter.
- Use official TRL `DPOTrainer` with a PEFT model, `ref_model=None`, and no
  custom DPO trainer.
- Save adapter, tokenizer, trainer state, train/eval metrics, loss history,
  chosen/rejected sanity rows, adapter reload generations, and run metadata.
- Keep the same DPO entrypoint compatible with the RTX 4090 formal config,
  without running formal DPO in this stage.

## Non-Goals

- No RTX 4090 formal DPO execution.
- No Base/SFT/DPO full evaluation.
- No ablations.
- No model merge.
- No custom DPO trainer.
- No Reward Model, PPO, GRPO, online rollout, or model-generated negatives.
- No vLLM, FlashAttention, DeepSpeed, FSDP, Registry, Factory, or plugin system.

## File Changes

Add:

- `plans/stage_4.md`
- `scripts/train_dpo.py`
- `src/mathalign_dpo/training/dpo_data.py`
- `src/mathalign_dpo/training/run_artifacts.py`
- `src/mathalign_dpo/training/train_dpo.py`
- `tests/test_dpo_data.py`
- `tests/test_train_dpo_cli.py`
- `reports/stage_4_report.md`

Modify:

- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/training/model_loader.py`
- `src/mathalign_dpo/training/runtime_metadata.py`
- `src/mathalign_dpo/training/train_sft.py`
- `requirements.txt`
- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `tests/test_config.py`
- `tests/test_model_loader.py`
- `tests/test_packaging.py`
- `tests/test_runtime_metadata.py`

## DPO Data Flow

1. Read a single YAML config through `--config`.
2. Refuse actual Stage 4 training unless `project.run_mode == "mini"`.
3. Load and validate the completed Stage 2 manifest.
4. Load `dpo_train.jsonl` and `dpo_validation.jsonl`, verify manifest row
   counts and sha256, and validate every row against the Stage 2 DPO schema.
5. Select initial candidates from `views[run_mode]["dpo"]`.
6. Tokenize initial candidates with the Stage 3 saved tokenizer:
   - `prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)`
   - `chosen_total = tokenizer.apply_chat_template(prompt + chosen)`
   - `rejected_total = tokenizer.apply_chat_template(prompt + rejected)`
7. Keep only rows with:
   - `prompt_len <= dpo.max_prompt_length`
   - `chosen_total <= dpo.max_length`
   - `rejected_total <= dpo.max_length`
   - positive chosen and rejected completion lengths
8. If the Mini pool is too short for `dpo.train_samples` or
   `dpo.validation_samples`, fail clearly; do not borrow from the formal pool.
9. Stable-rank kept rows by `seed|dpo|split|row_id` and select the exact target
   count.
10. Write token statistics and selected IDs only in run metadata; no new
    `data/processed` files are published in Stage 4.

Tokenizer probe before implementation found:

- Mini train: 179 legal rows from 256.
- Mini validation: 21 legal rows from 32.

Therefore the Mini config uses `dpo.train_samples = 179` and
`dpo.validation_samples = 21` for normal Stage 4 DPO.

## Policy And Reference Design

The DPO CLI requires:

```text
--sft-run-dir outputs/checkpoints/mini/sft/<completed_stage3_run>
```

The directory must contain:

```text
run_metadata.json
final_adapter/adapter_model.safetensors
final_adapter/adapter_config.json
tokenizer/
```

Normal Mini DPO rejects smoke SFT adapters. The SFT metadata must be completed,
Stage 3, Mini, `training_stage=sft`, non-smoke, trained on exactly 256 final
rows, and must match the DPO config model name, revision, dtype, backend, LoRA
target modules, and seed.

For MPS:

- Load the base model in FP16 without BitsAndBytes.
- Load the SFT adapter with `PeftModel.from_pretrained(..., is_trainable=True)`.
- Pass the PEFT policy model to `DPOTrainer` with `ref_model=None` and
  `peft_config=None`.
- Rely on TRL 0.29.1 PEFT reference behavior instead of loading a second full
  reference model.
- After trainer initialization, assert trainable parameters are on MPS and that
  both `default` and `ref` adapters exist, `default` is trainable, `ref` is
  frozen, and their initial weights match.

## MPS Memory Strategy

- Batch size remains 1.
- Gradient accumulation remains 4.
- Gradient checkpointing remains enabled.
- `use_cache=false` during training.
- `precompute_ref_log_probs=false` for v1.
- `dataloader_num_workers=0` and `pin_memory=false`.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is rejected.
- Process peak memory is recorded; direct MPS memory counters remain skipped
  because they have previously crashed in this environment.

## Training Flow

Smoke:

```bash
conda run -n mathalign-dpo python -m scripts.train_dpo \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-run-dir outputs/checkpoints/mini/sft/<completed_stage3_run> \
  --smoke-test \
  --output-dir outputs/checkpoints/mini/dpo_smoke \
  --overwrite
```

Mini:

```bash
conda run -n mathalign-dpo python -m scripts.train_dpo \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-run-dir outputs/checkpoints/mini/sft/<completed_stage3_run>
```

The run writes to a hidden staging directory first. The final output directory
is published only after training, explicit evaluation, final adapter save,
tokenizer save, trainer artifacts, chosen/rejected sanity checks, adapter reload
generation, and metadata all succeed. Existing outputs are preserved if staging
fails.

## Tests

- DPO data tests cover manifest completion, Stage 1 hash, DPO JSONL hashes,
  duplicate IDs, missing view IDs, malformed roles, identical chosen/rejected,
  rejected prompt leaks, non-null `token_count`, boundary token lengths, no
  truncation, Mini-only source boundaries, and failure when Mini targets cannot
  be met.
- Config tests cover positive DPO beta, supported loss type, positive DPO
  lengths, `max_prompt_length < max_length`, MPS `adamw_torch`, and CUDA NF4
  compatibility.
- CLI tests cover required `--sft-run-dir`, smoke overrides, output collision
  protection, Mini-only execution, SFT adapter metadata validation, and compact
  CLI JSON.
- Model loader tests cover loading a base model without applying a fresh LoRA
  adapter and loading an SFT adapter as a trainable PEFT policy.
- Runtime metadata tests cover Stage 4 run IDs and `dpo` metadata presence.

## Acceptance Criteria

- Stage 2 DPO integrity is checked before model loading.
- Real Qwen chat-template token lengths are computed for prompt, chosen, and
  rejected.
- Over-length rows are filtered and counted, not truncated.
- Normal Mini DPO uses exactly `dpo.train_samples` train rows and
  `dpo.validation_samples` validation rows from the Mini DPO view.
- Smoke DPO uses `smoke_test.dpo_samples` train rows.
- Policy initializes from a validated Stage 3 SFT adapter.
- TRL `DPOTrainer` is configured without `max_prompt_length`, `peft_config`, or
  a separate full reference model on MPS.
- Saved DPO adapter reloads and produces sanity generations.
- Metadata records date, git commit, config path, SFT source run, seed, device,
  system info, package versions, elapsed time, process peak memory, dataset
  counts, token filtering stats, DPO hyperparameters, output paths, train/eval
  metrics, and validation artifacts.
- `reports/stage_4_report.md` records commands, test results, limitations,
  deviations, and the recommended next stage.
