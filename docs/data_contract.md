# MathAlign-DPO Data Contract

## Current Stage

Current stage: Stage 1 data preprocessing refactor.

Stage 1 is responsible for producing the final datasets consumed by later SFT,
DPO, and evaluation stages. Stage 3-5 code has not been migrated yet, so those
entrypoints are temporarily expected to fail until the next stage updates their
loaders.

## Storage Layout

Raw data:

```text
data/raw/numina_math/
```

Final processed data:

```text
data/processed/
├── metadata.json
├── mini/
│   ├── sft/
│   ├── dpo/
│   └── evaluation/
└── formal/
    ├── sft/
    ├── dpo/
    └── evaluation/
```

All final sample collections are Hugging Face Dataset or DatasetDict directories
written with `save_to_disk()` and loaded with `datasets.load_from_disk()`.

Stage 1 no longer publishes JSONL training data, split manifests, Stage 2
manifests, file-level lineage hashes, candidate pools, expanded pools, or manual
review JSONL files.

## Common Rules

- Every row has `schema_version = "1.0"`.
- Every row has a stable non-empty `id`.
- Every row preserves its source through `source_id` and `metadata.raw_source_id`.
- Fixed split and order are derived from SHA-256 hashes using dataset name,
  revision, source split, source ID, and seed.
- Mini datasets are deterministic prefix subsets of the corresponding formal
  datasets.
- Final datasets must not contain null token counts.
- Token length filtering is completed in Stage 1 with the configured tokenizer
  for each mode.
- Over-length rows are filtered, never truncated.

## SFT Dataset

Location:

```text
data/processed/<mini|formal>/sft/
```

This is a DatasetDict with `train` and `validation` splits.

Columns:

```text
schema_version: string
id: string
source_id: string
prompt: list[{role, content}]
completion: list[{role, content}]
messages: list[{role, content}]
token_count: int
split: string
metadata: object
```

Rules:

- `messages` is exactly system/user/assistant.
- `prompt` is system/user.
- `completion` is one assistant message.
- `token_count` is computed with the mode tokenizer chat template.

## DPO Dataset

Location:

```text
data/processed/<mini|formal>/dpo/
```

This is a DatasetDict with `train` and `validation` splits.

Columns:

```text
schema_version: string
id: string
source_id: string
step_index: int
prompt: list[{role, content}]
chosen: list[{role, content}]
rejected: list[{role, content}]
token_count: object
split: string
metadata: object
```

Rules:

- `chosen` and `rejected` are each exactly one assistant message.
- `chosen` and `rejected` must not be identical.
- `rejected` must not appear in the prompt history.
- `token_count` includes prompt, chosen total, rejected total, chosen
  completion, and rejected completion counts.

## Evaluation Dataset

Location:

```text
data/processed/<mini|formal>/evaluation/
```

This is a single Dataset.

Columns:

```text
schema_version: string
id: string
source_id: string
problem: string
reference_answer: string
prompt_messages: list[{role, content}]
prompt_token_count: int
split: string
metadata: object
```

Rules:

- Every evaluation row has a non-empty `reference_answer`.
- `prompt_token_count + evaluation.max_new_tokens` must fit within the mode
  model length.
- Evaluation rows come only from the deterministic evaluation split.

## Metadata

`data/processed/metadata.json` records only:

```text
schema_version
stage
completed
created_at
smoke_test
dataset_name
dataset_revision
source_split
seed
raw_dataset_path
raw_source_rows
processed_dataset_paths
config_paths
tokenizers
target_counts
actual_counts
filter_counts_by_reason
split_method
selection_method
```

It is descriptive metadata, not a multi-stage manifest or file hash gate.
