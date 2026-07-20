# Stage 1 Data Preprocessing Refactor Plan

## Summary

Stage 1 is now the only data preprocessing stage. It downloads or reuses the raw
NuminaMath Hugging Face Dataset, builds final SFT/DPO/evaluation datasets for
Mini and formal modes, applies tokenizer length filtering before persistence,
and saves only Hugging Face Dataset directories under `data/processed/`.

Stage 3-5 training and evaluation code is intentionally unchanged in this
stage. Because the old `src/mathalign_dpo/data/` package is removed, those
later stages are expected to be temporarily broken until their own migration.

## Data Flow

1. If `data/raw/numina_math/` exists, load it with `datasets.load_from_disk()`.
2. If it does not exist, download `AI-MO/NuminaMath-CoT` at the pinned revision
   and save it to `data/raw/numina_math/`.
3. Normalize fields, assign deterministic split/rank values, parse reasoning
   steps, extract final answers, build SFT/DPO/evaluation rows, and apply real
   tokenizer length filtering in `scripts/prepare_data.py`.
4. Build formal datasets first, then select Mini as deterministic formal-prefix
   subsets after recomputing Mini token counts.
5. Save final outputs:

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

## Implementation Constraints

- The only modified Python file is `scripts/prepare_data.py`.
- `src/mathalign_dpo/data/` and `scripts/build_stage2_data.py` are deleted.
- Stage 1 no longer writes JSONL data, split manifests, Stage 2 manifests, file
  hashes, candidate pools, expanded pools, or delayed null token counts.
- The script avoids `list(dataset)` on the full raw dataset and uses Hugging Face
  Dataset `map`, `filter`, `sort`, and `select` for large data operations.

## Verification

Run smoke preprocessing:

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --smoke-test \
  --overwrite
```

Run full preprocessing:

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --overwrite
```
