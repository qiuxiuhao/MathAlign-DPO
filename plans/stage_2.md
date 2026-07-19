# Stage 2 Plan: Step Parsing, SFT Samples, And DPO Preferences

## Goals

Stage 2 consumes the completed Stage 1 normalized data and manifest. It selects
Mini/formal views from `split_manifest.json`, parses mathematical solutions into
steps, extracts final answers, builds SFT chat examples, creates deterministic
rule-based negative steps, builds step-level DPO preference examples, writes
schema-valid JSONL/statistics/manual-review files, and publishes all outputs
transactionally.

## Scope

- Validate the completed Stage 1 manifest before reading normalized data.
- Select examples in manifest order for formal and Mini views.
- Parse `solution` into ordered reasoning steps with `success`, `partial`, or
  `failed` status.
- Extract final answers without symbolic simplification or equivalence checks.
- Build SFT chat samples from parsed `success` and `partial` rows.
- Build deterministic number/operator/mixed negative steps.
- Build DPO prompt/chosen/rejected records from previous correct steps only.
- Leave every token count field as `null`; tokenizer length checks are deferred.
- Write Stage 2 files through the same staging, validation, and rollback model
  used by Stage 1.
- Add tests for parsing, answer extraction, SFT/DPO schema, mutation, manifest
  view selection, and transactional failure behavior.

## Non-Goals

- No tokenizer downloads or real token length calculation.
- No model loading, model inference, SFT training, DPO training, or evaluation.
- No MPS/CUDA training adaptation.
- No model-generated negatives.
- No Registry, Factory, multi-parser framework, or training abstraction.

## Files

Add:

- `src/mathalign_dpo/data/parse_steps.py`
- `src/mathalign_dpo/data/prompts.py`
- `src/mathalign_dpo/data/build_sft.py`
- `src/mathalign_dpo/data/mutate_steps.py`
- `src/mathalign_dpo/data/build_preferences.py`
- `src/mathalign_dpo/data/select_views.py`
- `scripts/build_stage2_data.py`
- `tests/test_select_views.py`
- `tests/test_parse_steps.py`
- `tests/test_build_sft.py`
- `tests/test_mutate_steps.py`
- `tests/test_build_preferences.py`
- `tests/test_stage2_pipeline.py`
- `reports/stage_2_report.md`

Modify:

- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `src/mathalign_dpo/data/write_outputs.py`
- Existing tests when needed for reusable transaction coverage.

## Data Flow

1. Load Mini and formal configs.
2. Validate `split_manifest.json` and Stage 1 file hashes/counts.
3. Read normalized JSONL files and select formal/Mini IDs from manifest views.
4. Parse every formal selected normalized row into step examples.
5. Extract final answers using deterministic priority rules.
6. Build SFT examples from `success` and `partial` parsed rows.
7. Build DPO examples from `success` parsed rows only when final answer is
   required.
8. Select canonical formal output counts and record Mini as prefix ID subsets.
9. Write step/SFT/DPO/manual-review outputs, Stage 2 statistics, and manifest
   extension to a staging directory.
10. Validate schema, counts, and sha256, then publish with rollback.

## Key Algorithms

- Step splitting tries numbered or markdown step headings first, then paragraph
  boundaries, then conservative sentence/equation boundaries if needed.
- `success` requires at least `preprocessing.minimum_steps` steps and a final
  answer; `partial` has enough steps but no final answer; `failed` has no
  reliable step sequence.
- Answer extraction priority: last balanced `\boxed{}` or `\fbox{}`, last
  `####`, explicit answer phrases, multiple-choice marker near the end, then
  final numeric/fraction expression.
- Number mutation deterministically chooses a non-label numeric literal and
  applies a configured non-zero offset.
- Operator mutation deterministically changes a supported binary operator while
  avoiding unary minus and step numbering.
- Mixed mutation deterministically chooses a first strategy and falls back to
  the other strategy if needed.
- DPO prompt contains system message, user problem message, and only previous
  correct assistant steps; current `chosen` and `rejected` are never in prompt.

## Test Plan

- Manifest selection rejects incomplete or hash-mismatched Stage 1 inputs and
  preserves manifest order.
- Step parser covers numbered markdown, paragraphs, one-line failures, and
  parse status transitions.
- Answer extraction covers `\boxed{}`, `####`, labels, multiple choice, numbers,
  decimals, fractions, and `\frac`.
- SFT schema tests require exactly system/user/assistant and `token_count: null`.
- Mutation tests require determinism, strategy metadata, fallback behavior, and
  `chosen != rejected`.
- DPO tests require correct prompt prefix steps, deterministic IDs, no rejected
  history, and schema validity.
- Transaction tests require failed staged writes/validation/publish to preserve
  old Stage 2 outputs and not publish `stage2.completed = true`.

## Commands

```bash
python -m pytest
```

```bash
python -m scripts.build_stage2_data \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --smoke-test \
  --output-dir /tmp/mathalign_stage2_smoke \
  --overwrite
```

```bash
python -m scripts.build_stage2_data \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --overwrite
```

## Acceptance Criteria

- Stage 2 code and docs are implemented without Stage 3+ functionality.
- All tests pass.
- Smoke and formal Stage 2 runs generate valid step, SFT, DPO, manual-review,
  statistics, and manifest outputs.
- Token count fields remain `null`, and reports clearly state token length was
  not checked because no tokenizer is used in Stage 2.
- Mini views are recorded as deterministic prefix subsets of formal Stage 2
  views.
- `reports/stage_2_report.md` records commands, counts, parse/mutation stats,
  output hashes, limitations, and deviations.
- Work stops after Stage 2 and waits for review.
