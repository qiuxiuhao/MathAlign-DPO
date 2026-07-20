# Stage 5 Report

## Implemented

- Added Stage 5 unified Mini evaluation code and CLI.
- Added deterministic answer extraction/normalization and preference diagnostics.
- Added Mini evaluation data loading from Stage 2 `step_eval.jsonl`.
- Added SFT/DPO source validation and leakage checks.

## Files Added

- `plans/stage_5.md`
- `scripts/evaluate_math.py`
- `src/mathalign_dpo/evaluation/__init__.py`
- `src/mathalign_dpo/evaluation/answer_normalization.py`
- `src/mathalign_dpo/evaluation/eval_data.py`
- `src/mathalign_dpo/evaluation/evaluate_math.py`
- `src/mathalign_dpo/evaluation/preference_eval.py`
- Stage 5 tests.

## Files Modified

- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `src/mathalign_dpo/config/load_config.py`

## Commands Executed

- `conda run -n mathalign-dpo python -m pytest tests/test_answer_normalization.py tests/test_evaluation_data.py tests/test_preference_eval.py tests/test_evaluate_math_cli.py -q`
- `conda run -n mathalign-dpo python -m pytest tests/test_answer_normalization.py tests/test_evaluation_data.py tests/test_preference_eval.py tests/test_evaluate_math_cli.py tests/test_config.py tests/test_packaging.py -q`
- `conda run -n mathalign-dpo python -m scripts.evaluate_math --help`
- `conda run -n mathalign-dpo python -m scripts.evaluate_math --config configs/qwen25_0_5b_m5_24gb_mini.yaml --sft-run-dir outputs/checkpoints/mini/sft/20260719T170915550766Z_stage3_sft_mini_d4226f86 --dpo-run-dir outputs/checkpoints/mini/dpo/20260719T173512557701Z_stage4_dpo_mini_53bfedfa --smoke-test --output-dir outputs/results/mini/eval_smoke --overwrite`
- `conda run -n mathalign-dpo python -m pytest -q`

## Test Results

- Stage 5 focused tests: `20 passed`.
- Stage 5 plus config/packaging tests: `37 passed`.
- Full local suite: `127 passed`.
- Stage 5 CLI help succeeded and exposes `--config`, `--sft-run-dir`,
  `--dpo-run-dir`, `--smoke-test`, `--output-dir`, `--samples`, and
  `--overwrite`.
- Runtime smoke attempt with the existing old DPO run failed before MPS/model
  loading, as intended:
  - failure type: `ValueError`
  - message: `Stage 5 rejects DPO runs outside the Mini-only sample policy:
    expected {'train': 179, 'validation': 21}, got {'train': 256,
    'validation': 32}`
  - failed metadata written under hidden staging:
    `outputs/results/mini/.eval_smoke.20260720T033638481951Z_stage5_eval_smoke_be019ed4.staging/run_metadata.json`
  - no completed Stage 5 output was published.

## Known Limitations

- The current repository does not yet contain an accepted Mini-only completed
  Stage 4 DPO run, so runtime evaluation should fail clearly until that adapter
  exists.
- The Codex execution environment was not used for real MPS generation because
  Stage 5 correctly stops at the missing accepted DPO adapter gate first.
- Stage 5 exact match uses deterministic string normalization, not symbolic
  equivalence.

## Deviations From Plan

- Runtime smoke/full MPS evaluation was not run because the only completed DPO
  adapter in the repository is the superseded formal-pool expansion run. This is
  an intentional gate from the Stage 5 plan.

## Recommended Next Stage

Do not start Stage 6 until Stage 5 smoke and full Mini evaluation complete on
an MPS-available terminal and this report is updated with measured results.
