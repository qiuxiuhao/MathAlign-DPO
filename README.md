# MathAlign-DPO

一套面向数学推理的、可在 Mac M5 24GB 上运行 Mini 全链路，并在 RTX 4090 24GB 上完成正式实验的轻量级后训练项目。

> **当前状态：Stage 4 — Mini DPO 已实现，等待 MPS 运行验收。**  
> Stage 4 已增加 Mini DPO 入口、Stage 2 DPO manifest/hash 校验、真实
> tokenizer 长度过滤、Stage 3 SFT adapter 初始化、事务式输出发布和 adapter
> reload 验证。Base/SFT/DPO 统一评测和消融实验仍为 Planned。

---

## 1. 项目目标

MathAlign-DPO 研究的问题是：

> 一个小型语言模型在先学习正确数学推理过程，再学习偏好正确推理步骤而不是局部合理但错误的推理步骤后，数学推理能力能否获得提升？

项目完整链路：

```text
NuminaMath-CoT
        ↓
数据标准化
        ↓
数学推理步骤拆分
        ↓
SFT 数据构造
        ↓
规则型错误步骤生成
        ↓
逐步骤偏好对构造
        ↓
SFT
        ↓
DPO
        ↓
Base / SFT / DPO 统一评测
```

---

## 2. 双模式定位

项目使用同一套代码，支持两个真实使用场景。

### 2.1 Mac M5 24GB Mini 学习模式

```text
模型：Qwen2.5-0.5B-Instruct
后端：PyTorch MPS
精度：FP16
训练方式：LoRA
序列长度：512
SFT 数据：256 条
DPO 数据：64～256 条
训练步数：几十步
```

用途：

- 在本地完成数据处理；
- 验证 SFT 全链路；
- 验证 DPO 全链路；
- 检查 adapter 保存与加载；
- 运行小规模 Base/SFT/DPO 评测；
- 帮助理解整个后训练项目。

Mini 模式只用于验证链路和学习，不承担正式性能结论。

### 2.2 RTX 4090 24GB 正式实验模式

```text
模型：Qwen2.5-3B-Instruct
后端：PyTorch CUDA
量化：4-bit NF4
计算精度：BF16
训练方式：QLoRA
初始序列长度：1024
```

用途：

- 正式 SFT；
- 正式 DPO；
- Base/SFT/DPO 主实验；
- 负样本策略消融；
- DPO beta 消融；
- 序列长度实验；
- 记录峰值显存和耗时；
- 形成简历中的实验结果。

---

## 3. 同一套代码链路

项目不会维护两套训练工程。

```text
同一份 NuminaMath 数据处理代码
同一份步骤解析代码
同一份偏好数据构造代码
同一份 SFT 入口
同一份 DPO 入口
同一份评测入口
        │
        ├── Mac 配置：0.5B + MPS + FP16 LoRA
        │
        └── 4090 配置：3B + CUDA + NF4 QLoRA
```

设备差异必须主要由配置表达。

---

## 4. 逐步骤偏好数据

对于正确推理：

```text
步骤 1
步骤 2
步骤 3
```

构造：

```text
样本 1
prompt   = 题目
chosen   = 正确步骤 1
rejected = 错误步骤 1
```

```text
样本 2
prompt   = 题目 + 正确步骤 1
chosen   = 正确步骤 2
rejected = 错误步骤 2
```

```text
样本 3
prompt   = 题目 + 正确步骤 1 + 正确步骤 2
chosen   = 正确步骤 3
rejected = 错误步骤 3
```

第一版错误步骤只使用：

- 数字扰动；
- 运算符扰动；
- 两者混合。

---

## 5. 第一版范围

第一版计划支持：

- `Qwen/Qwen2.5-0.5B-Instruct`；
- `Qwen/Qwen2.5-3B-Instruct`；
- `AI-MO/NuminaMath-CoT`；
- PyTorch MPS；
- PyTorch CUDA；
- FP16 LoRA；
- NF4 QLoRA；
- TRL SFTTrainer；
- TRL DPOTrainer；
- JSONL 中间数据；
- 规则型偏好数据；
- Base/SFT/DPO 统一评测；
- 实验元数据记录。

---

## 6. 第一版明确不做

```text
7B 或更大模型
全参数训练
多卡训练
DeepSpeed
FSDP
DoRA
在线 rollout
模型生成负样本
Reward Model
GRPO
PPO
必须依赖 FlashAttention
必须依赖 vLLM
Web 界面
```

这些功能只有在基线完整且确有实验需求时才能考虑。

---

## 7. 技术栈

| 模块 | 技术 |
|---|---|
| 模型与 tokenizer | Transformers |
| 数据处理 | Datasets |
| LoRA/QLoRA | PEFT |
| SFT/DPO | TRL |
| RTX 4090 量化 | BitsAndBytes |
| 运行准备 | Accelerate |
| 配置 | PyYAML |
| 测试 | Pytest |
| Mac 加速 | PyTorch MPS |
| NVIDIA 加速 | PyTorch CUDA |

Mac Mini 模式不使用 BitsAndBytes。

### 7.1 Stage 1 环境安装

```bash
conda create -n mathalign-dpo python=3.11 -y
conda activate mathalign-dpo

python -m pip install -r requirements.txt
python -m pip install -e .
```

### 7.2 Stage 3 Mini SFT 命令

Stage 3 先运行 smoke test，再运行 Mini SFT。首次运行会下载
`Qwen/Qwen2.5-0.5B-Instruct` tokenizer 和模型权重。

```bash
conda run -n mathalign-dpo python -m scripts.train_sft \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --smoke-test \
  --output-dir outputs/checkpoints/mini/sft_smoke
```

```bash
conda run -n mathalign-dpo python -m scripts.train_sft \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml
```

Stage 3 不允许 MPS 静默回退 CPU。如果 `torch.backends.mps` 不可用，或
`PYTORCH_ENABLE_MPS_FALLBACK=1`，训练会直接失败并写入 `run_metadata.json`。

### 7.3 Stage 4 Mini DPO 命令

Stage 4 必须显式指定一个已完成的 Stage 3 Mini SFT 输出目录。正常 Mini DPO
要求该 SFT run 不是 smoke run，并且 tokenizer 过滤后实际训练样本数为 256。

```bash
conda run -n mathalign-dpo python -m scripts.train_dpo \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-run-dir outputs/checkpoints/mini/sft/<completed_stage3_run> \
  --smoke-test \
  --output-dir outputs/checkpoints/mini/dpo_smoke \
  --overwrite
```

```bash
conda run -n mathalign-dpo python -m scripts.train_dpo \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-run-dir outputs/checkpoints/mini/sft/<completed_stage3_run>
```

Stage 4 会在送入 TRL `DPOTrainer` 前检查 `prompt/chosen/rejected` 真实 token
长度；超长样本会被过滤并记录，不会交给 Trainer 静默截断。

---

## 8. Stage 0 文件

```text
MathAlign-DPO/
├── AGENTS.md
├── README.md
├── configs/
│   ├── qwen25_0_5b_m5_24gb_mini.yaml
│   └── qwen25_3b_4090.yaml
└── docs/
    ├── design.md
    └── data_contract.md
```

后续目标目录：

```text
MathAlign-DPO/
├── plans/
├── reports/
├── src/
│   └── mathalign_dpo/
│       ├── config/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       └── learning/
├── scripts/
├── tests/
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── results/
    ├── reports/
    └── figures/
```

---

## 9. 开发阶段

| 阶段 | 主要内容 | 主要设备 |
|---|---|---|
| Stage 0 | 规范、架构、数据契约、双配置 | 不训练 |
| Stage 1 | NuminaMath 标准化与划分 | Mac CPU |
| Stage 2 | 步骤拆分、SFT/DPO 数据 | Mac CPU |
| Stage 3 | 0.5B Mini SFT，验证正式入口 | Mac MPS / 4090 smoke |
| Stage 4 | 0.5B Mini DPO | Mac MPS |
| Stage 5 | 统一评测 | Mac Mini / 4090 正式 |
| Stage 6 | 正式实验、消融、简历化 | RTX 4090 |

每个阶段完成后必须停止并接受审查。

---

## 10. 配置文件

### Mac M5 Mini

```text
configs/qwen25_0_5b_m5_24gb_mini.yaml
```

### RTX 4090 正式

```text
configs/qwen25_3b_4090.yaml
```

二者的数据 Schema 和程序入口完全相同。

---

## 11. 命令接口

Stage 1 已提供 normalized 数据准备入口：

```bash
python -m scripts.prepare_data \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml
```

Stage 2 提供步骤解析、SFT 样本和 DPO 偏好数据构造入口：

```bash
python -m scripts.build_stage2_data \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml
```

Stage 2 不加载 tokenizer，不计算真实 token 长度，也不训练模型。
Stage 2 会生成独立的 `stage2_manifest.json` 和 `stage2_statistics.json`，
不会覆盖 Stage 1 的 `split_manifest.json` 或 `data_statistics.json`。

以下训练和评测接口仍为 Planned：

```bash
python -m scripts.train_sft \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml
```

```bash
python -m scripts.train_dpo \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml
```

```bash
python -m scripts.evaluate \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --model-stage dpo
```

换成 RTX 4090 配置后，同一入口执行正式实验。

---

## 12. 数据产物

计划生成：

```text
data/processed/normalized_train.jsonl
data/processed/normalized_validation.jsonl
data/processed/normalized_eval.jsonl
data/processed/step_train.jsonl
data/processed/step_validation.jsonl
data/processed/step_eval.jsonl
data/processed/sft_train.jsonl
data/processed/sft_validation.jsonl
data/processed/dpo_train.jsonl
data/processed/dpo_validation.jsonl
data/processed/eval.jsonl
data/processed/data_statistics.json
data/processed/split_manifest.json
data/processed/stage2_statistics.json
data/processed/stage2_manifest.json
```

字段定义见：

```text
docs/data_contract.md
```

---

## 13. 计划评测

所有模型必须使用相同：

- Prompt；
- 测试样本；
- 最大生成长度；
- 解码参数；
- 答案提取逻辑。

对比：

```text
Base
SFT
DPO
```

初始指标：

- answer extraction rate；
- exact match accuracy；
- invalid output rate；
- average output tokens；
- elapsed time；
- peak memory。

Mac Mini 只做小样本链路验证。

RTX 4090 执行正式评测和消融实验。

---

## 14. 实验真实性

README 中不得提前填写：

- 未实测准确率；
- 未实测显存；
- 未实测耗时；
- 未运行的功能状态。

统一标记：

```text
Planned
Smoke-tested
Measured
Estimated
```

---

## 15. 项目归属

MathAlign-DPO 是一个独立的干净重构项目。

项目参考“逐步骤数学推理偏好优化”的公开研究思路，但不复制参考仓库源代码。训练使用的模型、数据集和开源库应分别遵守其许可证。
