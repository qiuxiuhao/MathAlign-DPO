# MathAlign-DPO Design

## Current Stage

Current stage: Stage 1 data preprocessing refactor.

This stage intentionally rebuilds only the data preprocessing path. SFT, DPO,
and evaluation code remains unchanged and will be migrated in later stages.

## Stage 1 Responsibility

Stage 1 now owns all dataset construction work:

- Download `AI-MO/NuminaMath-CoT` once and save it to `data/raw/numina_math/`.
- Reuse the local raw Dataset with `datasets.load_from_disk()` by default.
- Clean and standardize raw fields.
- Assign deterministic train / validation / evaluation splits.
- Parse reasoning steps.
- Extract final answers.
- Build final SFT, DPO, and evaluation datasets.
- Apply real tokenizer length filtering for both Mini and formal modes.
- Save final Hugging Face Dataset directories under `data/processed/`.

## Removed Data Architecture

The previous Stage 1/2 split is removed from the active design. The project no
longer designs or publishes:

- normalized JSONL files;
- step JSONL files;
- SFT/DPO JSONL files;
- split manifests;
- Stage 2 manifests;
- file-level hash lineage gates;
- candidate pools;
- expanded pools;
- training-time token filtering;
- Mini replenishment from formal pools;
- delayed `token_count = null` rows.

## Determinism

Splits are assigned with a SHA-256 bucket using:

```text
dataset_name | dataset_revision | source_split | source_id | seed
```

Rows within each split are ordered by a second SHA-256 rank. Formal datasets are
selected first. Mini datasets are deterministic prefix subsets of formal datasets
after Mini tokenizer constraints are applied.

## Runtime Boundaries

This refactor accepts a temporary boundary:

- `src/mathalign_dpo/data/` is deleted.
- Existing Stage 3-5 code still imports old data helpers.
- Stage 3-5 execution is therefore not part of this stage's acceptance criteria.

The next implementation stage should update training and evaluation loaders to
read from:

```text
data/processed/<mini|formal>/<sft|dpo|evaluation>
```

with `datasets.load_from_disk()`.
