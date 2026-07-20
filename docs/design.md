# MathAlign-DPO 设计说明

## 当前架构

当前项目采用小型顶层链路：

```text
scripts/prepare_data.py  -> Stage 1 数据预处理
sft/                     -> Stage 2 SFT 训练
dpo/                     -> Stage 3 DPO 训练
evaluation/              -> Stage 4 Base/SFT/DPO 统一评价
configs/                 -> YAML 配置加载
```

旧 `src/` 包已经删除。新代码不应 import `mathalign_dpo.*`。

## 数据流

Stage 1 负责全部数据构造：

- 下载或复用 `AI-MO/NuminaMath-CoT`；
- 清洗并标准化字段；
- 分配确定性的 train / validation / evaluation split；
- 解析推理步骤和最终答案；
- 构造 SFT、DPO、Evaluation Hugging Face Dataset；
- 分别使用 Mini 和 formal tokenizer 完成真实长度过滤；
- 将最终数据保存到 `data/processed/`。

Stage 2、Stage 3 和 Stage 4 只通过 `datasets.load_from_disk()` 加载本地数据。它们不再清洗、过滤、重排、补齐或重新构造样本。

## 训练流

SFT：

- 读取 `data/processed/<mode>/sft`；
- 从本地 `model/` 目录加载配置指定的 Qwen 模型，缺失时从 ModelScope 下载；
- 使用 TRL `SFTTrainer` 训练 LoRA/QLoRA adapter；
- 保存 `adapter/`、`tokenizer/`、metrics 和 reload 样本。

DPO：

- 读取 `data/processed/<mode>/dpo`；
- 从 Base + 已完成的 SFT adapter 初始化 policy；
- 使用 TRL `DPOTrainer`，并设置 `ref_model=None`，由 TRL 创建 PEFT reference adapter；
- 保存 DPO adapter、tokenizer、metrics 和 reload 样本。

Stage 4：

- 读取 `data/processed/<mode>/evaluation`；
- 校验 SFT 和 DPO run directory 的运行模式、模型身份和 adapter/tokenizer 产物；
- 依次加载 Base、SFT adapter 和 DPO adapter；
- 使用相同 prompt 和生成参数输出 `base_sft_dpo_predictions.jsonl`、`base_sft_dpo_summary.json`、正确样例和错误样例。

## 运行模式

Mini 模式：

- 后端：MPS；
- 模型：`model/Qwen2.5-0.5B-Instruct`；
- 精度：FP16；
- 训练方式：LoRA；
- 数据：`data/processed/mini/*`。

Formal 模式：

- 后端：CUDA；
- 模型：`model/Qwen2.5-3B-Instruct`；
- 精度：BF16；
- 量化：4-bit NF4；
- 训练方式：QLoRA；
- 数据：`data/processed/formal/*`。

两种模式共用同一套 SFT、DPO 和 Stage 4 评价代码路径。设备差异应只来自 YAML 配置。

## 已删除设计

当前设计不再使用：

- Stage 1/2 多层 manifest 链；
- 文件级 lineage hash 门禁；
- JSONL 训练输入；
- candidate pool 或 expanded pool；
- 训练阶段数据过滤；
- Mini 从 formal 池补齐；
- 自定义 Trainer；
- Reward Model、PPO 或 GRPO。
