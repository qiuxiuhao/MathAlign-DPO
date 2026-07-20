# MathAlign-DPO

Current stage: **Stage 1 data preprocessing refactor**.

This repository is being refactored so Stage 1 produces the final Hugging Face
Datasets for later SFT, DPO, and evaluation stages. Training and evaluation code
is intentionally not migrated in this stage.

## Stage 1 Outputs

Raw Dataset:

```text
data/raw/numina_math/
```

Processed Datasets:

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

All processed datasets are saved with Hugging Face `save_to_disk()` and are
intended to be loaded later with `datasets.load_from_disk()`.

## Stage 1 Command

Smoke preprocessing:

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --smoke-test \
  --overwrite
```

Full preprocessing:

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --overwrite
```

Force raw dataset refresh:

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --refresh-raw \
  --overwrite
```

## Important Status

- Stage 1 data preprocessing is the only active target of this refactor.
- `src/mathalign_dpo/data/` has been removed.
- `scripts/build_stage2_data.py` has been removed.
- Stage 3-5 training and evaluation entrypoints still contain old imports and
  are expected to be migrated in a later stage.
- No training results are claimed by this stage.
