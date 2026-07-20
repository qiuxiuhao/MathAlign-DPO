# MathAlign-DPO

当前阶段：**Stage 3 Mini DPO 已完成，formal 训练入口已具备**。

MathAlign-DPO 是一个数学推理后训练项目，当前有效链路已经迁移到顶层模块：

```text
configs/
scripts/prepare_data.py
sft/
dpo/
```

旧 `src/` 包、旧 `tests/`、旧 `plans/` 和旧 `scripts/train_dpo.py` 已删除。后续代码不应再 import `mathalign_dpo.*`。

## 项目链路

```text
Stage 1 数据预处理
  -> Stage 2 SFT 训练与 Base/SFT 评价
  -> Stage 3 DPO 训练与 Base/SFT/DPO 评价
```

Stage 1 负责全部数据处理。Stage 2 和 Stage 3 只能通过 `datasets.load_from_disk()` 读取本地 Hugging Face Dataset，不再做数据清洗、长度过滤、样本补齐、重新排序或候选池构造。

## 数据目录

Stage 1 输出如下：

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

原始数据默认保存到：

```text
data/raw/numina_math/
```

本地 raw 数据存在时，Stage 1 默认使用 `datasets.load_from_disk()` 加载，不重复访问 Hugging Face。

## 环境

创建并激活 conda 环境：

```bash
conda create -n mathalign-dpo python=3.11 -y
conda activate mathalign-dpo
```

安装依赖：

```bash
pip install -r requirements.txt
```

Mac Mini 模式使用：

```text
配置：configs/qwen25_0_5b_m5_24gb_mini.yaml
模型：Qwen/Qwen2.5-0.5B-Instruct
本地目录：model/Qwen2.5-0.5B-Instruct
后端：MPS
训练方式：FP16 LoRA
```

RTX 4090 formal 模式使用：

```text
配置：configs/qwen25_3b_4090.yaml
模型：Qwen/Qwen2.5-3B-Instruct
本地目录：model/Qwen2.5-3B-Instruct
后端：CUDA
训练方式：4-bit NF4 QLoRA
```

模型缺失时，训练代码会优先通过 ModelScope 下载到 YAML 指定的 `model/...` 本地目录。

## Stage 1 数据预处理

Stage 1 会自动处理数据下载。首次运行时，它会从 Hugging Face 下载：

```text
AI-MO/NuminaMath-CoT
```

并保存到本地 raw 目录：

```text
data/raw/numina_math/
```

后续再次运行时，只要 `data/raw/numina_math/` 已存在，就会直接使用本地 raw Dataset，不再重复访问远程数据集。

下载 raw 数据并生成 mini/formal 全部处理后数据：

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --overwrite
```

如果需要丢弃本地 raw 数据并强制重新下载：

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --refresh-raw \
  --overwrite
```

## Mac Mini 完整训练与评价

先运行 SFT。训练完成后会自动保存 SFT adapter，并使用 `data/processed/mini/evaluation` 对 Base 和 SFT 做评价：

```bash
python -m sft.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --overwrite
```

SFT 主要产物：

```text
outputs/mini/sft/
├── adapter/
├── best_adapter/
├── tokenizer/
├── train_metrics.json
├── eval_metrics.json
├── best_adapter_metrics.json
├── base_sft_predictions.jsonl
├── base_sft_summary.json
└── run_config.json
```

再运行 DPO。DPO 会从 `outputs/mini/sft` 加载 SFT adapter，训练完成后自动评价 Base、SFT 和 DPO：

```bash
python -m dpo.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-dir outputs/mini/sft \
  --overwrite
```

DPO 主要产物：

```text
outputs/mini/dpo/
├── adapter/
├── best_adapter/
├── tokenizer/
├── train_metrics.json
├── eval_metrics.json
├── best_adapter_metrics.json
├── base_sft_dpo_predictions.jsonl
├── base_sft_dpo_summary.json
└── run_config.json
```

查看 Mini 评价汇总：

```bash
python -m json.tool outputs/mini/sft/base_sft_summary.json
python -m json.tool outputs/mini/dpo/base_sft_dpo_summary.json
```

## RTX 4090 完整训练与评价

先运行 formal SFT。训练完成后会自动使用 `data/processed/formal/evaluation` 评价 Base 和 SFT：

```bash
python -m sft.train \
  --config configs/qwen25_3b_4090.yaml \
  --overwrite
```

formal SFT 主要产物：

```text
outputs/formal/sft/
├── adapter/
├── best_adapter/
├── tokenizer/
├── train_metrics.json
├── eval_metrics.json
├── best_adapter_metrics.json
├── base_sft_predictions.jsonl
├── base_sft_summary.json
└── run_config.json
```

再运行 formal DPO。formal DPO 必须使用 formal SFT adapter，不能复用 Mini adapter：

```bash
python -m dpo.train \
  --config configs/qwen25_3b_4090.yaml \
  --sft-dir outputs/formal/sft \
  --overwrite
```

formal DPO 主要产物：

```text
outputs/formal/dpo/
├── adapter/
├── best_adapter/
├── tokenizer/
├── train_metrics.json
├── eval_metrics.json
├── best_adapter_metrics.json
├── base_sft_dpo_predictions.jsonl
├── base_sft_dpo_summary.json
└── run_config.json
```

查看 formal 评价汇总：

```bash
python -m json.tool outputs/formal/sft/base_sft_summary.json
python -m json.tool outputs/formal/dpo/base_sft_dpo_summary.json
```

## 断点继续训练

训练过程会按配置中的 `save_steps` 保存 `checkpoint-*`。如果训练中断，可以从某个 checkpoint 继续训练。

注意：

- 断点继续时不能加 `--overwrite`，代码会直接拒绝这种组合，避免误删原输出目录。
- `--resume-from-checkpoint` 必须指向包含 `trainer_state.json` 的 checkpoint 目录。
- `max_steps` 表示总训练步数。如果 checkpoint 已经达到原来的 `max_steps`，继续训练前需要在 YAML 中增大 `max_steps`，或通过 CLI 传入更大的 `--max-steps`。

Mac Mini SFT 断点继续：

```bash
python -m sft.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --resume-from-checkpoint outputs/mini/sft/checkpoint-30
```

Mac Mini DPO 断点继续：

```bash
python -m dpo.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-dir outputs/mini/sft \
  --resume-from-checkpoint outputs/mini/dpo/checkpoint-20
```

RTX 4090 SFT 断点继续：

```bash
python -m sft.train \
  --config configs/qwen25_3b_4090.yaml \
  --resume-from-checkpoint outputs/formal/sft/checkpoint-300
```

RTX 4090 DPO 断点继续：

```bash
python -m dpo.train \
  --config configs/qwen25_3b_4090.yaml \
  --sft-dir outputs/formal/sft \
  --resume-from-checkpoint outputs/formal/dpo/checkpoint-200
```

## 当前实测状态

- Stage 1 数据预处理已完成。
- Mini SFT 已运行完成，产物位于 `outputs/mini/sft`。
- Mini DPO 已运行完成，产物位于 `outputs/mini/dpo`。
- formal SFT / formal DPO 入口已准备好，但需要在 RTX 4090 环境中实际运行。

当前只保留三份阶段报告：

```text
reports/stage_1_refactor_report.md
reports/stage_2_refactor_report.md
reports/stage_3_refactor_report.md
```

## 注意事项

- `data/raw/`、`data/processed/`、`model/` 和 `outputs/` 不应提交到 Git。
- Stage 2 和 Stage 3 不负责重新构造数据。
- `adapter/` 保存训练结束时的最新 adapter，`best_adapter/` 保存验证集 `eval_loss` 最低的 adapter。
- `checkpoint-*` 用于断点继续训练，`adapter/` 和 `best_adapter/` 只保存 adapter 权重，不包含 optimizer 和 scheduler 状态。
- DPO 会校验 SFT adapter 的运行模式和模型身份，避免 formal DPO 误用 Mini SFT adapter。
- formal 配置要求 CUDA 设备名包含 `RTX 4090`。
