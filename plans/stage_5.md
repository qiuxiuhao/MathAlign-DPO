# Stage 5 Plan: Unified Mini Evaluation

## Goals

Stage 5 adds a deterministic local evaluation entrypoint for comparing Base,
SFT, and DPO on the fixed Mini evaluation view. It uses the Stage 3 tokenizer,
serial model loading on MPS, exact-match answer scoring, preference diagnostics,
transactional outputs, and full runtime metadata.

## Scope

- Evaluate `base`, `sft`, and `dpo` with identical prompts and generation
  parameters.
- Use `step_eval.jsonl` selected by `views.mini.step.evaluation`.
- Use Stage 3 saved tokenizer for all model stages.
- Validate completed Stage 3 SFT and Mini-only Stage 4 DPO inputs.
- Save predictions, summaries, preference diagnostics, comparison examples,
  error cases, report, and metadata.

## Non-Goals

- No new SFT/DPO training.
- No RTX 4090 formal experiment.
- No LLM-as-a-Judge, manual answer edits, model merge, ablations, frontend, or
  visualization system.

## Files

Add:

- `scripts/evaluate_math.py`
- `src/mathalign_dpo/evaluation/evaluate_math.py`
- `src/mathalign_dpo/evaluation/eval_data.py`
- `src/mathalign_dpo/evaluation/answer_normalization.py`
- `src/mathalign_dpo/evaluation/preference_eval.py`
- Stage 5 unit tests and report.

Modify:

- Config validation, README, design doc, data contract, and packaging tests as
  needed for Stage 5.

## Data And Metrics

- Full Mini evaluation uses `config.evaluation.samples = 32`; smoke uses
  `smoke_test.evaluation_samples = 16`.
- Rows must have `parse_status=success` and non-empty `final_answer`.
- Evaluation sources must not overlap selected SFT/DPO train or validation
  sources.
- Exact match is deterministic normalized string equality; failed extraction
  counts as incorrect.
- Preference accuracy compares chosen/rejected average log-prob and is recorded
  as a diagnostic metric only.

## Commands

```bash
conda run -n mathalign-dpo python -m scripts.evaluate_math \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-run-dir outputs/checkpoints/mini/sft/<completed_stage3_run> \
  --dpo-run-dir outputs/checkpoints/mini/dpo/<completed_stage4_run> \
  --smoke-test \
  --output-dir outputs/results/mini/eval_smoke \
  --overwrite
```

```bash
conda run -n mathalign-dpo python -m scripts.evaluate_math \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-run-dir outputs/checkpoints/mini/sft/<completed_stage3_run> \
  --dpo-run-dir outputs/checkpoints/mini/dpo/<completed_stage4_run>
```

## Acceptance

- Tests pass.
- CLI help runs.
- Stage 5 refuses superseded formal-pool Stage 4 DPO runs.
- MPS smoke and full Mini evaluation publish completed outputs only after all
  artifacts are written.
- Report records measured commands, limitations, and stops before Stage 6.
