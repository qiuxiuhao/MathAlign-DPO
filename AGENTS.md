# AGENTS.md

## 1. 项目身份

项目名称：**MathAlign-DPO**

项目目标：实现一套小而完整、代码清晰、适合学习和简历展示的数学推理后训练系统。

项目使用同一套代码支持两种运行模式：

- **Mini 模式**：Apple Silicon MPS，Qwen2.5-0.5B-Instruct，FP16 LoRA。
- **Formal 模式**：RTX 4090 CUDA，Qwen2.5-3B-Instruct，BF16，4-bit NF4 QLoRA。

Mini 不是独立玩具项目，而是正式链路的小规模运行配置。

## 2. 当前架构

当前有效代码采用顶层小模块组织：

```text
configs/                 YAML 配置加载
scripts/prepare_data.py  Stage 1 数据预处理
sft/                     Stage 2 SFT 训练与 Base/SFT 评价
dpo/                     Stage 3 DPO 训练与 Base/SFT/DPO 评价
```

旧 `src/` 包已经删除。不要重新引入 `mathalign_dpo.*` import。

旧 `tests/` 和 `plans/` 目录已经删除。后续如果补测试，只针对当前顶层架构重建。

## 3. 已完成阶段

Stage 1 已完成：

- 下载或复用 `AI-MO/NuminaMath-CoT`。
- 原始数据保存到 `data/raw/numina_math/`。
- 最终 Hugging Face Dataset 保存到 `data/processed/`。
- 完成字段标准化、固定拆分、步骤解析、最终答案提取、SFT/DPO/Evaluation 构造、真实 tokenizer 长度过滤、Mini/formal 数据选择。

Stage 2 已完成：

- 从 `data/processed/mini/sft` 训练 Mini SFT。
- SFT adapter 保存到 `outputs/mini/sft/adapter`。
- 使用 `data/processed/mini/evaluation` 完成 Base/SFT 初步评价。

Stage 3 已完成：

- 从 `data/processed/mini/dpo` 训练 Mini DPO。
- DPO 基于 `outputs/mini/sft` 初始化。
- DPO adapter 保存到 `outputs/mini/dpo/adapter`。
- 使用 `data/processed/mini/evaluation` 完成 Base/SFT/DPO 初步评价。

Formal CUDA SFT/DPO 代码路径已经存在，但需要在 RTX 4090 机器上运行，并且 DPO 需要先有 formal SFT adapter。

## 4. 数据规则

训练阶段必须使用 `datasets.load_from_disk()` 加载 Stage 1 本地产物。

SFT 数据路径：

```text
data/processed/<mode>/sft
```

DPO 数据路径：

```text
data/processed/<mode>/dpo
```

Evaluation 数据路径：

```text
data/processed/<mode>/evaluation
```

SFT 和 DPO 不允许重新执行：

- 数据清洗；
- prompt/completion 重构；
- tokenizer 长度过滤；
- candidate pool 或 expanded pool 构造；
- 样本补齐；
- 稳定重排；
- manifest/hash/lineage 门禁。

## 5. 运行规则

运行参数来自两份 YAML：

```text
configs/qwen25_0_5b_m5_24gb_mini.yaml
configs/qwen25_3b_4090.yaml
```

CLI 覆盖项只允许保持少量调试用途：

- 配置路径；
- smoke 模式；
- 输出目录；
- 少量样本数；
- 最大训练步数。

不要引入 Registry、Factory、插件系统、工作流引擎、依赖注入、多模型 Provider 或兼容层。

正式训练优先使用官方库：

- `datasets`
- `transformers`
- `trl`
- `peft`
- `accelerate`
- `bitsandbytes`，仅用于 CUDA formal 模式

当前版本不实现自定义 SFTTrainer、自定义 DPOTrainer、Reward Model、PPO、GRPO、DeepSpeed、FSDP 或多卡训练。

## 6. 当前命令

Stage 1 数据预处理：

```bash
python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --overwrite
```

Mini SFT：

```bash
python -m sft.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --overwrite
```

Mini DPO：

```bash
python -m dpo.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-dir outputs/mini/sft \
  --overwrite
```

Formal DPO smoke，需先有 formal SFT：

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

## 7. 报告与产物

当前只保留三份阶段报告：

```text
reports/stage_1_refactor_report.md
reports/stage_2_refactor_report.md
reports/stage_3_refactor_report.md
```

不要提交：

- `data/`
- `outputs/`
- `model/`
- 缓存目录；
- 虚拟环境；
- 模型权重；
- 密钥。

这些内容由 `.gitignore` 忽略。
