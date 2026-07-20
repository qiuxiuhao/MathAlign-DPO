# MathAlign-DPO

当前阶段：**Stage 3 Mini DPO 已完成**。

MathAlign-DPO 是一个小而完整的数学推理后训练项目。当前有效链路已经迁移到顶层模块，不再使用旧 `src/` 包：

```text
configs/
scripts/prepare_data.py
sft/
dpo/
```

## 数据

Stage 1 会生成所有本地 Hugging Face Dataset：

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

后续训练阶段必须通过 `datasets.load_from_disk()` 直接加载这些数据，不再重新清洗、过滤、排序或选择样本。

## Stage 1 数据预处理

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --overwrite
```

## Stage 2 SFT

Mini smoke：

```bash
python -m sft.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --smoke-test \
  --train-samples 8 \
  --validation-samples 4 \
  --eval-samples 4 \
  --max-steps 1 \
  --output-dir outputs/mini/sft_smoke \
  --overwrite
```

Mini 正式训练：

```bash
python -m sft.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --overwrite
```

## Stage 3 DPO

Mini smoke：

```bash
python -m dpo.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-dir outputs/mini/sft \
  --smoke-test \
  --train-samples 8 \
  --validation-samples 4 \
  --eval-samples 4 \
  --max-steps 1 \
  --output-dir outputs/mini/dpo_smoke \
  --overwrite
```

Mini 正式训练：

```bash
python -m dpo.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-dir outputs/mini/sft \
  --overwrite
```

RTX 4090 smoke：

```bash
python -m dpo.train \
  --config configs/qwen25_3b_4090.yaml \
  --sft-dir outputs/formal/sft \
  --smoke-test \
  --train-samples 32 \
  --validation-samples 8 \
  --eval-samples 8 \
  --max-steps 1 \
  --output-dir outputs/formal/dpo_smoke \
  --overwrite
```

RTX 4090 正式训练：

```bash
python -m dpo.train \
  --config configs/qwen25_3b_4090.yaml \
  --sft-dir outputs/formal/sft \
  --overwrite
```

## 当前状态

- Mini SFT 已运行完成，产物位于 `outputs/mini/sft`。
- Mini DPO 已运行完成，产物位于 `outputs/mini/dpo`。
- Formal DPO 需要先有 `outputs/formal/sft`。
- 旧 `src/` 包、旧 `tests/`、旧 `plans/` 和旧 `scripts/train_dpo.py` 已删除。
- 当前只保留三份报告：
  - `reports/stage_1_refactor_report.md`
  - `reports/stage_2_refactor_report.md`
  - `reports/stage_3_refactor_report.md`
