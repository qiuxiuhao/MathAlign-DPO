# Stage 3 Plan: Mini SFT On MPS With Qwen2.5 LoRA

## Goals

Stage 3 implements the first real training stage: Mini SFT on Mac M5 24GB using
`Qwen/Qwen2.5-0.5B-Instruct`, PyTorch MPS, FP16 LoRA, and TRL `SFTTrainer`.

The stage consumes completed Stage 2 SFT JSONL outputs, validates their
manifest, performs real tokenizer chat-template length checks, trains a small
LoRA adapter, reloads that adapter, runs a few deterministic inference sanity
checks, and records truthful run metadata.

## Scope

- Validate `stage2_manifest.json`, Stage 1 reference hash, SFT JSONL hashes, row
  counts, Mini SFT views, schema, unique IDs, and `token_count: null`.
- Load the configured Qwen tokenizer and require its built-in chat template.
- Compute true rendered token lengths with `tokenizer.apply_chat_template`.
- Filter over-length SFT rows; do not truncate reference solutions.
- Train Mini SFT with MPS FP16 LoRA through official TRL `SFTTrainer`.
- Run a smoke SFT before the normal Mini SFT run.
- Save adapter, tokenizer, trainer state, loss history, metrics, metadata, and
  adapter reload samples.
- Keep the same SFT entrypoint compatible with the RTX 4090 formal config,
  without running formal SFT in this stage.

## Non-Goals

- No DPO training.
- No Base/SFT/DPO formal evaluation.
- No RTX 4090 formal SFT execution.
- No BitsAndBytes execution on Mac.
- No model merge.
- No vLLM, FlashAttention, DeepSpeed, FSDP, Registry, Factory, plugin system, or
  custom Trainer.

## File Changes

Add:

- `src/mathalign_dpo/training/__init__.py`
- `src/mathalign_dpo/training/model_loader.py`
- `src/mathalign_dpo/training/sft_data.py`
- `src/mathalign_dpo/training/train_sft.py`
- `src/mathalign_dpo/training/runtime_metadata.py`
- `scripts/train_sft.py`
- `tests/test_model_loader.py`
- `tests/test_sft_data.py`
- `tests/test_runtime_metadata.py`
- `tests/test_train_sft_cli.py`
- `reports/stage_3_report.md`

Modify:

- `requirements.txt`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `src/mathalign_dpo/config/load_config.py`
- `README.md`
- `docs/design.md`
- `docs/data_contract.md`

## Model Loading Design

The shared model entrypoint is:

```python
load_model_and_tokenizer(config, training_stage="sft")
```

For MPS:

- Require `torch.backends.mps.is_built()` and
  `torch.backends.mps.is_available()`.
- Reject `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Load with `torch_dtype=torch.float16`, no `BitsAndBytesConfig`, and no CUDA
  memory APIs.
- Apply PEFT `LoraConfig` from YAML.
- Enable gradient checkpointing only when configured.
- Keep `model.config.use_cache = False`.
- Move to `mps`.
- Fail if there are zero trainable parameters or trainable parameters are not on
  MPS.

For CUDA:

- Reuse the same loader shape.
- Create BitsAndBytes NF4 config only inside the CUDA branch.
- Keep formal config compatibility tests in Stage 3, but do not run formal SFT.

## Training Flow

1. Load a single YAML config through `--config`.
2. Refuse actual Stage 3 training unless `project.run_mode == "mini"`.
3. Validate completed Stage 2 manifest and selected Mini SFT rows.
4. In smoke mode, cap inputs with `smoke_test.train_samples`,
   `smoke_test.validation_samples`, and `smoke_test.max_steps` before tokenizer
   filtering.
5. Load tokenizer and require chat template.
6. Convert Stage 2 messages to TRL prompt/completion rows:
   `prompt = [system, user]`, `completion = [assistant]`.
7. Compute full chat token length for prompt plus completion.
8. Drop rows with `token_count > model.max_length`; record filtered IDs and
   token statistics.
9. Load model/tokenizer with LoRA.
10. Create `SFTConfig` directly from YAML SFT settings.
11. Train with official TRL `SFTTrainer`.
12. Save final adapter, tokenizer, trainer state, metrics, loss history, and run
    metadata.
13. Reload the saved adapter and generate deterministic outputs for a few
    validation prompts.

Default output root is `config.sft.output_dir/<run_id>`. Explicit non-empty
output directories are rejected unless `--overwrite` is provided.

## Tests

- Manifest validation rejects incomplete manifests, missing files, hash
  mismatches, row count mismatches, missing Mini IDs, duplicate IDs, invalid
  roles, non-empty missing text, and pre-populated token counts.
- Token filtering uses a fake tokenizer to verify chat template rendering, no
  truncation, stable row order, and kept/filtered statistics.
- Config tests cover MPS FP16 LoRA, disabled quantization, `adamw_torch`, CPU
  fallback disabled, CUDA NF4 formal compatibility, and SFT metadata fields.
- Model loader tests monkeypatch imports to prove MPS does not create
  BitsAndBytes config and CUDA creates it only in the CUDA branch.
- Runtime metadata tests cover package versions, git commit handling, elapsed
  time, process peak memory, output paths, and device fields.
- CLI tests cover smoke overrides, output directory collision protection, and
  Mini-only actual training.

## Commands

```bash
conda run -n mathalign-dpo python -m pip install -r requirements.txt
conda run -n mathalign-dpo python -m pip install -e .
conda run -n mathalign-dpo python -m pytest
```

```bash
conda run -n mathalign-dpo python -m scripts.train_sft \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --smoke-test \
  --output-dir outputs/checkpoints/mini/sft_smoke
```

```bash
conda run -n mathalign-dpo python -m scripts.train_sft \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml
```

## Risks

- Initial model/tokenizer download may need network access.
- MPS operator coverage may fail for the installed PyTorch/Transformers/TRL
  combination.
- Token length filtering can reduce the current 253 Mini train rows.
- TRL APIs can change; Stage 3 pins pre-v1 compatible dependency ranges.
- MPS memory accounting is less direct than CUDA; Stage 3 records process peak
  memory and available `torch.mps` counters.

## Acceptance Criteria

- Stage 2 SFT integrity is checked before model loading.
- Real Qwen chat-template token lengths are computed.
- Over-length samples are filtered and counted, not truncated.
- Smoke SFT completes on MPS or fails with a clear non-CPU-fallback reason.
- Mini SFT completes with configured `sft.max_steps` when MPS/model dependencies
  are available.
- Saved LoRA adapter reloads and produces sanity generations.
- Metadata records date, git commit, config path, seed, device, macOS/system
  info, package versions, elapsed time, process peak memory, dataset counts,
  token filtering stats, output paths, and final training metrics.
- `reports/stage_3_report.md` records commands, test results, limitations,
  deviations, and the recommended next stage.
