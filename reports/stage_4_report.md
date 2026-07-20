# Stage 4 Report

## Implemented

- Closed Stage 4 Mini DPO with Mini-only data selection: Stage 4 now uses only
  `views.mini.dpo` and validates each row against `views.mini.dpo_source_ids`
  through `metadata.normalized_id`.
- Added independent DPO sample controls: `dpo.train_samples` and
  `dpo.validation_samples`. Current Mini config uses 179 train rows and 21
  validation rows, matching the legal Mini rows under Qwen 512/384 token limits.
- Added numerical stability gating for loss, reward, margin, log-prob, and
  grad norm. NaN/Inf core metrics now fail the run before completed publication.
- Pinned Stage 4 training-critical dependencies to the measured stack:
  `torch==2.13.0`, `transformers==4.57.6`, `trl==0.29.1`,
  `peft==0.17.1`, and `accelerate==1.14.0`.
- Stage 4 now loads the tokenizer directly from the Stage 3 SFT run
  `tokenizer/` directory and validates tokenizer metadata when Stage 3 recorded
  it.
- Stage 4 validates actual SFT adapter configuration from
  `final_adapter/adapter_config.json`: rank, alpha, dropout, bias, target
  modules, base model name, and base model revision.
- Added strict reference adapter validation after `DPOTrainer` construction:
  `default` and `ref` adapters must exist, `default` must be trainable, `ref`
  must be frozen, and initial adapter weights must match.
- Strengthened Trainer dataset length checks so missing TRL token fields fail
  immediately instead of being skipped.
- Added Stage 2 manifest lineage metadata for future Stage 3 SFT runs and
  Stage 4 lineage validation against the SFT source run.
- Moved shared staging/publish run directory logic into
  `src/mathalign_dpo/training/run_artifacts.py`; DPO no longer imports shared
  artifact helpers from `train_sft.py`.

## Files Added

- `src/mathalign_dpo/training/run_artifacts.py`

## Files Modified

- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `plans/stage_4.md`
- `requirements.txt`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `src/mathalign_dpo/config/load_config.py`
- `src/mathalign_dpo/training/dpo_data.py`
- `src/mathalign_dpo/training/model_loader.py`
- `src/mathalign_dpo/training/sft_data.py`
- `src/mathalign_dpo/training/train_dpo.py`
- `src/mathalign_dpo/training/train_sft.py`
- `tests/test_config.py`
- `tests/test_dpo_data.py`
- `tests/test_packaging.py`
- `tests/test_train_dpo_cli.py`

## Commands Executed

- `conda run -n mathalign-dpo python -m pytest tests/test_dpo_data.py tests/test_train_dpo_cli.py tests/test_config.py tests/test_packaging.py -q`
- `conda run -n mathalign-dpo python -m pytest -q`
- `conda run -n mathalign-dpo python -m scripts.train_dpo --help`
- Mini-only DPO data check with Stage 3 saved tokenizer:
  - candidate pools: train 256, validation 32
  - after tokenizer filtering: train 179, validation 21
  - final selected rows: train 179, validation 21
  - selected pool: `run_mode`
- Stage 3 SFT adapter validation against
  `outputs/checkpoints/mini/sft/20260719T170915550766Z_stage3_sft_mini_d4226f86`:
  rank 8, alpha 16, dropout 0.05, bias `none`, target modules
  `k_proj/o_proj/q_proj/v_proj`, base revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Stage 4 tokenizer/lineage validation with the same SFT run:
  tokenizer loaded from Stage 3 `tokenizer/`, legacy Stage 3 tokenizer metadata
  matches `pad_token_after`, and Stage 2 manifest path matches with current
  hash `c81dd5da4df9ae094335240976d65650d7f1296694f3791f93a7424199faccbf`.
- Attempted MPS smoke rerun:
  `conda run -n mathalign-dpo python -m scripts.train_dpo --config configs/qwen25_0_5b_m5_24gb_mini.yaml --sft-run-dir outputs/checkpoints/mini/sft/20260719T170915550766Z_stage3_sft_mini_d4226f86 --smoke-test --output-dir outputs/checkpoints/mini/dpo_smoke --overwrite`
- Attempted MPS normal Mini DPO rerun:
  `conda run -n mathalign-dpo python -m scripts.train_dpo --config configs/qwen25_0_5b_m5_24gb_mini.yaml --sft-run-dir outputs/checkpoints/mini/sft/20260719T170915550766Z_stage3_sft_mini_d4226f86`

## Test Results

- Targeted Stage 4 closeout tests: `40 passed`.
- Full local test suite: `105 passed`.
- CLI help: succeeded.
- Dependency versions in `mathalign-dpo` environment:
  - `torch`: `2.13.0`
  - `transformers`: `4.57.6`
  - `trl`: `0.29.1`
  - `peft`: `0.17.1`
  - `accelerate`: `1.14.0`
- MPS smoke and normal Mini DPO reruns could not execute in the Codex tool
  environment because `torch.backends.mps.is_built()=True` but
  `torch.backends.mps.is_available()=False`. Both runs failed at backend
  preflight and did not publish completed output directories.
- User-side MPS rerun with Stage 3 SFT
  `20260720T034618167107Z_stage3_sft_mini_9c1f2b1f` passed data filtering
  (`179/21`) and reference adapter validation after a strict allclose tolerance,
  but then emitted repeated Metal command buffer errors during DPO training.
  Logged DPO core metrics were all `0.0`, and the run was manually interrupted
  at step 10/20 while TRL was executing intermediate evaluation.
- Follow-up fix applied on 2026-07-20: Mini MPS DPO now sets
  `dpo.eval_strategy = "no"` for the Mini config, precomputes reference log-probs
  after the project reloads/freezes the `ref` adapter, disables
  `logging_nan_inf_filter`, and rejects all-zero DPO core metric signals.
- User-side MPS smoke after reference precompute completed 10 train steps and
  explicit eval with finite loss/log-prob metrics, but failed the numerical gate
  because Transformers/Accelerate logged NaN `grad_norm` for most optimizer
  steps. Example smoke metrics: `train_loss=0.6927456021308899`,
  `eval_loss=0.6931471824645996`, finite chosen/rejected log-probs, and NaN
  `grad_norm` entries converted to JSON `null` in `loss_history.jsonl`.
- Follow-up fix applied on 2026-07-20: MPS Mini DPO disables Trainer's built-in
  full-model grad clipping (`trainer_max_grad_norm=0.0`) and uses a local
  LoRA-only CPU-norm callback to compute, log, validate, and clip trainable
  adapter gradients with the configured `dpo.max_grad_norm`.
- User-side normal Mini DPO then completed 20 train steps with finite
  `train_loss=0.7087996780872345`, finite `eval_loss=0.6913611888885498`, finite
  DPO log-probs/rewards/grad norm, and one non-finite diagnostic `entropy` value
  at step 20. The previous gate incorrectly matched `loss_history[19].entropy`
  as a core loss metric because the parent path contained `loss_history`.
- Follow-up fix applied on 2026-07-20: numerical stability checks now match only
  the terminal metric name, so non-finite diagnostic entropy is recorded but does
  not fail Stage 4. Core loss/reward/margin/log-prob/grad-norm NaN/Inf still
  fails the run.
- Latest local tests after this fix: `135 passed`.

## Known Limitations

- The previous completed Stage 4 run
  `20260719T173512557701Z_stage4_dpo_mini_53bfedfa` used formal-pool expansion
  and 256/32 DPO rows. It is now superseded by this closeout policy and should
  not be treated as the accepted Stage 4 result.
- Existing Stage 3 SFT run metadata predates the new Stage 2 manifest hash and
  tokenizer hash fields. Stage 4 can validate the saved tokenizer artifact and
  match the legacy Stage 2 manifest path, while future Stage 3 reruns will record
  full hashes for strict lineage validation.
- The current execution environment cannot run MPS training from Codex; final
  smoke/normal DPO must be rerun from an MPS-available terminal.
- MPS DPO still needs a fresh user-side smoke and normal rerun after the
  reference-log-prob precompute and LoRA-only grad-norm callback fixes. A
  successful run must show non-zero finite DPO loss/log-prob metrics, finite
  callback-recorded `grad_norm`, and no published completed directory on
  numerical failure.
- Mini DPO remains a chain validation run, not a formal model-quality result.

## Deviations From Plan

- Normal Mini DPO no longer targets 256/32 by borrowing from formal data. It
  targets the real Mini-only legal counts configured as 179/21.
- NaN/Inf metrics are no longer sanitized into successful artifacts. They may be
  JSON-sanitized in failed metadata, but any core non-finite metric fails the run.
- Stage 4 records legacy lineage compatibility for the existing Stage 3 SFT run;
  future Stage 3 runs will include full Stage 2 manifest and tokenizer hashes.
- The original plan kept `precompute_ref_log_probs=false`. User-side MPS testing
  showed TRL's per-step PEFT reference adapter path produced repeated Metal
  command buffer errors and all-zero DPO metrics, so Mini MPS DPO now precomputes
  reference log-probs after explicit `ref` adapter validation.
- The original plan used Trainer's default gradient clipping. User-side MPS
  smoke showed finite DPO loss/log-prob values but NaN `grad_norm` from the
  default Accelerate/MPS full-model norm path, so Stage 4 now computes/clips
  LoRA gradients through a small local callback.

## Recommended Next Stage

Do not start Stage 5 until an MPS-available terminal reruns Stage 4 smoke and
normal Mini DPO with the Mini-only data policy and produces a completed adapter.
After that review, Stage 5 should implement unified Base/SFT/DPO evaluation only.
