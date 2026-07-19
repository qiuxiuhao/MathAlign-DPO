# MathAlign-DPO 系统设计

## 1. 设计目标

MathAlign-DPO 是一套具有两层运行能力的数学推理后训练系统：

```text
Mac M5 24GB
    运行 0.5B Mini 完整链路
    用于开发、调试和学习

RTX 4090 24GB
    运行 3B 正式实验
    用于完整训练、消融和简历结果
```

设计必须同时满足：

1. 同一套代码；
2. 同一套数据契约；
3. 同一套训练入口；
4. 同一套评测入口；
5. 设备差异主要由配置控制；
6. 初学者可以按数据流逐文件阅读。

---

## 2. 为什么采用 Mini + 正式双模式

仅在 RTX 4090 上开发存在以下问题：

- 每次调试需要租用或占用 GPU；
- 数据错误可能直到训练时才暴露；
- 理解代码必须依赖远程环境；
- 修改和验证反馈较慢。

只在 Mac 上完成项目也存在问题：

- 0.5B 和极小数据难以形成有说服力的正式结果；
- MPS 训练速度和生态不适合大量消融；
- 无法使用成熟的 CUDA NF4 QLoRA 方案。

因此采用：

```text
Mac Mini：完成正确性验证
RTX 4090：完成规模化验证
```

Mini 模式不是单独的 Demo，而是正式链路的小配置。

---

## 3. 高层流程

```text
┌──────────────────────────────────────────┐
│              YAML Configuration          │
│                                          │
│  Mac: 0.5B / MPS / FP16 / LoRA          │
│  CUDA: 3B / NF4 / BF16 / QLoRA          │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│                Data Pipeline             │
│                                          │
│  Load NuminaMath                         │
│      ↓                                   │
│  Normalize fields                        │
│      ↓                                   │
│  Split reasoning steps                   │
│      ↓                                   │
│  Build SFT samples                       │
│      ↓                                   │
│  Mutate incorrect steps                  │
│      ↓                                   │
│  Build DPO preferences                   │
└────────────────────┬─────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────────┐   ┌───────────────────┐
│    SFT Training   │   │   DPO Training    │
│                   │   │                   │
│ Mac: MPS LoRA     │   │ Mac: MPS LoRA     │
│ CUDA: NF4 QLoRA   │   │ CUDA: NF4 QLoRA   │
└─────────┬─────────┘   └─────────┬─────────┘
          └────────────┬───────────┘
                       ▼
┌──────────────────────────────────────────┐
│              Unified Evaluation          │
│                                          │
│      Base / SFT / DPO                    │
│      same prompt and decoding            │
└──────────────────────────────────────────┘
```

---

## 4. 架构原则

### 4.1 共享业务逻辑

以下模块完全共享：

```text
数据加载
字段标准化
数据划分
步骤拆分
最终答案提取
SFT 消息构造
错误步骤生成
DPO 偏好构造
答案评测
指标汇总
```

### 4.2 设备差异集中在模型加载和训练参数

设备差异只能集中在：

```text
src/mathalign_dpo/training/model_loader.py
配置文件
少量 runtime 检查
```

禁止在数据模块中判断 MPS 或 CUDA。

### 4.3 正式 Trainer 使用官方实现

生产训练：

```text
TRL SFTTrainer
TRL DPOTrainer
```

项目核心创新和学习重点：

```text
逐步骤数据解析
规则型 Hard Negative
标准偏好对构造
双环境显存适配
统一实验设计
```

第一版不自定义 Trainer。

---

## 5. 计划模块边界

### 5.1 配置加载

```text
src/mathalign_dpo/config/load_config.py
```

职责：

- 读取 YAML；
- 校验公共字段；
- 校验 MPS/CUDA 专属字段；
- 解析仓库相对路径；
- 输出统一配置对象。

必须校验：

```text
MPS 配置不能启用 4-bit BitsAndBytes
MPS 配置必须使用 adamw_torch
CUDA 正式配置必须启用 NF4
DPO max_prompt_length < max_length
数据比例之和为 1
样本数为正数
```

### 5.2 数据标准化

```text
src/mathalign_dpo/data/load_numina.py
```

职责：

- 加载 NuminaMath；
- 识别源字段；
- 转换标准 Schema；
- 生成稳定 ID；
- 确定性采样和划分；
- 写 JSONL 和统计。

不得负责步骤拆分。

Stage 1 入口必须同时读取 Mini 与 formal 两份配置。二者必须共享
`dataset_name`、`dataset_revision`、`source_split`、seed、split ratio
和 canonical 输出路径。canonical normalized 文件按 formal 配置生成，
Mini 数据由 split manifest 中的 ID 视图表达，且必须是 formal 视图的
确定性前缀子集。

`train_ratio`、`validation_ratio` 和 `evaluation_ratio` 只决定
source-level split 归属；`train_samples`、`validation_samples` 和
`evaluation_samples` 只决定从每个 split 的稳定排序列表中取多少条。

稳定 ID 在字段审计后确定：原始数据存在唯一且非空的原生 ID 字段时优先
使用；否则使用固定 dataset revision 下的原始 split 行号作为 fallback。

所有 Stage 1 输出先写入 `data/processed/.stage_<run_id>/`，完成 schema、
计数和 sha256 校验后再发布最终文件，禁止静默覆盖或留下完成标记不真实的
manifest。

### 5.3 步骤拆分

```text
src/mathalign_dpo/data/parse_steps.py
```

职责：

- 保留原始顺序拆分推理步骤；
- 提取最终答案；
- 记录 parse_status；
- 输出步骤化数据。

### 5.4 SFT 构造

```text
src/mathalign_dpo/data/build_sft.py
```

职责：

- 构造 system/user/assistant 消息；
- assistant 保存完整正确解答；
- 使用 tokenizer chat template 检查长度；
- 超长样本过滤而非静默截断。

### 5.5 错误步骤生成

```text
src/mathalign_dpo/data/mutate_steps.py
```

第一版策略：

```text
number_mutation
operator_mutation
mixed
```

职责：

- 局部修改正确步骤；
- 保持文本形式接近原步骤；
- 确保错误步骤不等于正确步骤；
- 输出修改位置、替换值、失败原因。

### 5.6 DPO 偏好构造

```text
src/mathalign_dpo/data/build_preferences.py
```

对第 i 步：

```text
prompt = 数学题 + 第 0 到 i-1 个正确步骤
chosen = 第 i 个正确步骤
rejected = 第 i 个错误步骤
```

职责：

- 构造 TRL 可读取的偏好格式；
- 保留 source_id 和 step_index；
- 检查长度；
- 输出统计和人工检查样本。

### 5.7 模型加载

```text
src/mathalign_dpo/training/model_loader.py
```

统一入口：

```python
load_model_and_tokenizer(config, training_stage)
```

MPS 路径：

```text
不创建 BitsAndBytesConfig
torch_dtype = float16
device = mps
LoRA
optimizer = adamw_torch
```

CUDA 路径：

```text
创建 BitsAndBytesConfig
4-bit NF4
compute dtype = bfloat16
QLoRA
optimizer = paged_adamw_8bit
```

模型加载模块不得包含数据读取。

### 5.8 SFT 训练

```text
src/mathalign_dpo/training/train_sft.py
```

职责：

- 读取统一 SFT 数据；
- 调用统一模型加载；
- 配置 LoRA；
- 创建 SFTTrainer；
- 保存 adapter；
- 保存实验元数据。

### 5.9 DPO 训练

```text
src/mathalign_dpo/training/train_dpo.py
```

职责：

- 从 SFT adapter 初始化 policy；
- 读取统一偏好数据；
- 创建 DPOTrainer；
- 保存 DPO adapter；
- 保存训练指标和元数据。

TRL 版本兼容方式必须在 Stage 4 基于实际安装版本确定，不得提前编写大量兼容分支。

### 5.10 统一评测

```text
src/mathalign_dpo/evaluation/evaluate_math.py
```

职责：

- 加载 Base/SFT/DPO；
- 使用相同 Prompt；
- 使用相同生成参数；
- 提取答案；
- exact match；
- 保存逐样本和汇总结果。

---

## 6. Mini 模式设计

### 6.1 目标

Mini 模式只证明：

```text
数据能处理
SFT 能训练
SFT adapter 能保存和加载
DPO 能训练
DPO adapter 能保存和加载
Base/SFT/DPO 能统一评测
```

不要求显著提升准确率。

### 6.2 配置

```text
Qwen2.5-0.5B-Instruct
MPS
FP16
LoRA rank 8
max_length 512
SFT 256 条
DPO 64～256 条
SFT 30 steps
DPO 20 steps
batch size 1
gradient accumulation 4
```

### 6.3 MPS 注意事项

- 不使用 BitsAndBytes；
- 不使用 paged AdamW；
- 不要求 BF16；
- 不硬编码 `.cuda()`；
- 不使用 CUDA autocast；
- 不读取 CUDA 显存 API；
- 不把 MPS Tensor 转交 CUDA 专属库；
- 必须检测 `torch.backends.mps.is_available()`；
- 评测 batch size 固定为 1；
- 进程内存与耗时必须实测记录。

如某个 TRL 功能在 MPS 上存在兼容问题，应在报告中记录并做最小修复，不得为此复制整套 Trainer。

---

## 7. RTX 4090 正式模式设计

### 7.1 配置

```text
Qwen2.5-3B-Instruct
CUDA
NF4 4-bit
BF16 compute
QLoRA rank 16
max_length 1024
batch size 1
gradient accumulation 16
gradient checkpointing
```

### 7.2 显存策略

1. 3B 模型；
2. 4-bit NF4；
3. LoRA；
4. batch size 1；
5. gradient accumulation；
6. gradient checkpointing；
7. 1024 长度起步；
8. 训练中不在线生成负样本；
9. 评测 batch size 1；
10. 不强制使用 vLLM。

OOM 排查顺序：

1. 确认 4-bit 生效；
2. 确认 batch size 为 1；
3. 确认未加载额外完整模型副本；
4. 缩短 max_length；
5. 缩短 max_prompt_length；
6. 关闭训练中生成式评测；
7. 必要时降低 LoRA rank。

---

## 8. 数据流

### 8.1 原始到标准化

```text
source row
    ↓
NormalizedMathExample
```

每个源样本最多生成一个标准化样本。

### 8.2 标准化到步骤

```text
NormalizedMathExample
    ↓
StepMathExample
```

必须保留：

```text
problem
solution
steps
final_answer
parse_status
```

### 8.3 步骤到 SFT

```text
StepMathExample
    ↓
SFTExample
```

assistant 内容是完整正确解答。

### 8.4 步骤到 DPO

正确步骤：

```text
s0, s1, ..., sn
```

第 i 个偏好样本：

```text
prompt = problem + s0 ... s(i-1)
chosen = si
rejected = mutate(si)
```

所有前缀步骤必须是正确步骤。

---

## 9. Prompt 设计

第一版统一系统提示词：

```text
You are a careful mathematical reasoning assistant.
```

用户指令：

```text
Solve the following mathematics problem. Show a clear step-by-step derivation and put the final answer in \boxed{}.
```

Prompt 文本必须集中定义，不得分散复制到多个模块。

---

## 10. 错误步骤设计

### 数字扰动

```text
chosen: 2 × 7 = 14
rejected: 2 × 7 = 16
```

### 运算符扰动

```text
chosen: x = 10 - 3
rejected: x = 10 + 3
```

### Mixed

根据 seed 确定性选择一种可用策略。

过滤：

```text
无法修改
修改后相同
空步骤
仅空白变化
长度超限
步骤过短
```

必须输出：

- 成功数；
- 失败数；
- 各策略数量；
- 长度分布；
- 过滤原因；
- 100 条人工检查样本。

---

## 11. 配置结构

两份配置必须拥有相同顶层结构：

```text
schema_version
project
model
quantization
lora
data
preprocessing
negative_sampling
sft
dpo
evaluation
runtime
output
smoke_test
```

这样同一份代码可以加载两者。

---

## 12. 测试设计

### 公共单元测试

- 稳定 ID；
- 标准化；
- 划分；
- 步骤拆分；
- 最终答案；
- SFT Schema；
- 数字扰动；
- 运算符扰动；
- DPO Schema；
- chosen != rejected；
- Token 长度。

### 配置测试

Mac：

```text
backend = mps
quantization.enabled = false
optimizer = adamw_torch
dtype = float16
```

CUDA：

```text
backend = cuda
quantization.enabled = true
quant_type = nf4
optimizer = paged_adamw_8bit
dtype = bfloat16
```

### Smoke Test

Mac：

```text
64 条 SFT
32～64 条 DPO
10 steps
16 条评测
```

CUDA：

```text
小样本 3B SFT
小样本 3B DPO
确认 4-bit 和 adapter
```

---

## 13. 实验记录

统一元数据：

```json
{
  "run_id": "...",
  "mode": "mini",
  "stage": "dpo",
  "git_commit": "...",
  "config_path": "...",
  "seed": 42,
  "device_backend": "mps",
  "device_name": "...",
  "system_memory_gb": 24,
  "software_versions": {},
  "dataset_counts": {},
  "start_time": "...",
  "end_time": "...",
  "elapsed_seconds": 0,
  "peak_memory_mb": 0,
  "output_path": "..."
}
```

RTX 4090 需要额外记录：

```text
CUDA 版本
GPU 峰值显存
```

Mac 需要额外记录：

```text
macOS 版本
MPS 可用状态
进程峰值内存
```

所有值必须来自真实运行。

---

## 14. 正式实验矩阵

只在 RTX 4090 执行：

### 主实验

```text
Base
SFT
DPO
```

### 负样本消融

```text
number_mutation
operator_mutation
mixed
```

### Beta 消融

```text
0.05
0.10
0.20
```

### 长度实验

```text
1024
1536
```

1536 只有在 24GB 显存稳定时执行。

---

## 15. 扩展条件

增加新功能必须同时满足：

1. Mini 和正式基线已经完整；
2. 当前测试全部通过；
3. 有明确实验目的；
4. 先更新 design.md；
5. 如涉及数据，先更新 data_contract.md；
6. 不保留被替代的冗余实现。

可选后续：

```text
模型生成 Hard Negative
符号验证
MATH Benchmark
Alternative DPO Loss
7B 大硬件实验
vLLM 评测加速
```

这些都不属于第一版。
