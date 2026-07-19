# AGENTS.md

## 1. 项目身份

项目名称：**MathAlign-DPO**

项目目标：从零实现一套小而完整、代码清晰、适合学习并能用于简历展示的数学推理后训练系统。

项目采用同一套代码支持两种真实运行模式：

### Mac Mini 学习模式

```text
设备：Apple Silicon Mac M5，24GB 统一内存
后端：PyTorch MPS
模型：Qwen/Qwen2.5-0.5B-Instruct
精度：FP16
训练方式：LoRA
序列长度：512
SFT 数据：256 条
DPO 数据：64～256 条
训练步数：几十步
用途：验证完整链路、调试代码、帮助理解 SFT 与 DPO
```

### RTX 4090 正式实验模式

```text
设备：NVIDIA RTX 4090 24GB
后端：PyTorch CUDA
模型：Qwen/Qwen2.5-3B-Instruct
精度：BF16 计算
量化：4-bit NF4
训练方式：QLoRA
初始序列长度：1024
用途：正式训练、消融实验、简历结果
```

两个模式必须共享：

- 数据处理代码；
- 数据契约；
- SFT 训练入口；
- DPO 训练入口；
- 评测入口；
- 实验记录格式。

不得为 Mac 和 RTX 4090 维护两套独立项目。

---

## 2. 当前阶段

当前阶段：**Stage 0 — 项目规范与架构设计**

Stage 0 只允许创建和修改：

```text
AGENTS.md
README.md
docs/design.md
docs/data_contract.md
configs/qwen25_0_5b_m5_24gb_mini.yaml
configs/qwen25_3b_4090.yaml
```

Stage 0 不允许：

- 编写正式 Python 训练代码；
- 下载完整数据集；
- 下载模型权重；
- 启动训练；
- 伪造训练结果；
- 提前创建 Stage 1 之后的实现；
- 把计划功能写成已完成功能。

只有 Stage 0 通过审查后，才能进入 Stage 1。

---

## 3. 第一版完整链路

```text
AI-MO/NuminaMath-CoT
        ↓
原始字段标准化
        ↓
数学推理步骤拆分
        ↓
SFT 样本构造
        ↓
规则型错误步骤生成
        ↓
逐步骤 chosen/rejected 构造
        ↓
LoRA/QLoRA SFT
        ↓
LoRA/QLoRA DPO
        ↓
Base / SFT / DPO 统一评测
```

Mini 模式和正式模式必须执行相同的逻辑链路。

Mini 模式不是另一个简化项目，而是正式项目的小规模运行配置。

---

## 4. 不可违背的开发原则

### 4.1 第一版保持最小完整

第一版只支持：

- 两个经过批准的运行配置；
- 一个基础模型家族：Qwen2.5-Instruct；
- 一个主训练数据集：NuminaMath-CoT；
- 一条 SFT 路径；
- 一条 DPO 路径；
- 规则型数字扰动和运算符扰动；
- 单设备训练；
- JSONL 中间数据；
- Base/SFT/DPO 统一评测。

不要因为“未来可能有用”而提前增加功能。

### 4.2 优先使用官方库

正式训练必须优先使用：

```text
transformers
datasets
peft
trl
accelerate
```

RTX 4090 正式模式额外使用：

```text
bitsandbytes
```

Mac M5 Mini 模式禁止依赖 CUDA 专属组件，不得要求：

```text
bitsandbytes 4-bit
paged_adamw_8bit
flash-attn
xformers
vllm
CUDA
```

第一版不得自定义 SFTTrainer 或 DPOTrainer。

后续可以在 `src/learning/` 增加教学版 DPO Loss，但教学代码不得被正式训练入口调用。

### 4.3 禁止过度设计

除非经过审查的需求明确要求，否则禁止引入：

- Registry；
- Factory；
- 插件系统；
- 工作流引擎；
- 依赖注入框架；
- 只有一个实现的抽象基类；
- 多模型 Provider；
- 多数据集适配框架；
- 自定义回调框架；
- 向后兼容层；
- 历史版本兼容分支；
- Hydra；
- DeepSpeed；
- FSDP。

只有当至少两个真实实现需要共享逻辑时，才允许提取抽象。

### 4.4 单一配置来源

运行参数的唯一来源是 YAML 配置文件：

```text
configs/qwen25_0_5b_m5_24gb_mini.yaml
configs/qwen25_3b_4090.yaml
```

CLI 第一版只允许：

- 选择配置文件；
- 启用 smoke test；
- 覆盖输出目录；
- 覆盖少量样本数或训练步数用于调试。

禁止在 YAML、CLI 和 Python 默认值中维护三套互相覆盖的训练参数。

### 4.5 同一代码链路，按能力切换

允许存在的真实设备差异只有：

| 项目 | Mac M5 Mini | RTX 4090 正式 |
|---|---|---|
| 设备 | MPS | CUDA |
| 模型 | 0.5B | 3B |
| 量化 | 关闭 | 4-bit NF4 |
| 精度 | FP16 | BF16 |
| 优化器 | adamw_torch | paged_adamw_8bit |
| 最大长度 | 512 | 1024 |
| LoRA Rank | 8 | 16 |
| 数据规模 | Mini | 正式 |
| 目标 | 跑通与学习 | 正式实验 |

除这些配置差异外，禁止复制训练脚本。

推荐逻辑：

```python
if config.runtime.backend == "cuda":
    # 创建 BitsAndBytes 量化配置
elif config.runtime.backend == "mps":
    # 不使用 BitsAndBytes，使用 FP16 LoRA
else:
    raise ValueError(...)
```

不得用大量设备专属分支污染业务逻辑。

### 4.6 每个文件只有一个职责

推荐目录职责：

```text
src/config/load_config.py
src/data/load_numina.py
src/data/parse_steps.py
src/data/build_sft.py
src/data/mutate_steps.py
src/data/build_preferences.py
src/training/model_loader.py
src/training/train_sft.py
src/training/train_dpo.py
src/evaluation/evaluate_math.py
```

避免创建含义不清的文件：

```text
utils.py
helpers.py
common.py
misc.py
manager.py
pipeline.py
```

除非其职责被明确限定和记录。

### 4.7 函数必须小而明确

- 公共函数必须写类型注解；
- 优先控制在 50 行以内；
- 不使用隐藏全局状态；
- 不使用可变默认参数；
- 不静默吞掉异常；
- 不使用宽泛的 `except Exception` 隐藏错误；
- 错误消息必须指出样本、字段或配置项；
- 不通过 fallback 掩盖配置错误。

---

## 5. 数据规则

所有落盘数据必须遵守：

```text
docs/data_contract.md
```

要求：

- 默认使用 UTF-8 JSONL；
- 每行必须有稳定唯一 ID；
- 所有随机行为使用配置中的 seed；
- 每次转换都保留 source_id；
- chosen 和 rejected 不能相同；
- 数据过滤必须记录原因和数量；
- 必须输出数据统计；
- 不得创建未在数据契约中定义的字段；
- 修改 Schema 前必须先更新 data_contract.md。

---

## 6. 设备和显存约束

### 6.1 Mac M5 24GB Mini

必须使用：

```text
Qwen2.5-0.5B-Instruct
PyTorch MPS
FP16
LoRA
max_length = 512
batch_size = 1
gradient accumulation
256 条 SFT 数据
64～256 条 DPO 数据
几十个训练 step
```

Mini 模式的验收目标是：

- 数据链路完整；
- SFT 能训练并保存 adapter；
- DPO 能训练并保存 adapter；
- Base/SFT/DPO 都能推理；
- 指标、日志、结果文件能生成；
- 用户能通过小规模代码理解完整流程。

Mini 结果不能作为正式简历性能结论。

### 6.2 RTX 4090 24GB 正式模式

必须使用：

```text
Qwen2.5-3B-Instruct
CUDA
4-bit NF4
BF16 compute
QLoRA
batch_size = 1
gradient accumulation
gradient checkpointing
max_length = 1024
```

正式模式用于：

- 完整训练；
- 正式评测；
- 负样本消融；
- beta 消融；
- 序列长度实验；
- 峰值显存和耗时记录；
- 简历结果。

### 6.3 第一版明确不支持

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
PPO
GRPO
必须依赖 FlashAttention
必须依赖 vLLM
```

---

## 7. 分阶段开发流程

每个阶段必须按照以下顺序执行：

1. 创建 `plans/stage_<n>.md`；
2. 明确本阶段范围；
3. 列出新增和修改文件；
4. 明确本阶段不做什么；
5. 定义验证命令；
6. 只实现当前阶段；
7. 运行测试；
8. 创建 `reports/stage_<n>_report.md`；
9. 停止并等待审查。

不得自动进入下一阶段。

阶段报告必须包含：

```text
# Stage N Report
## Implemented
## Files Added
## Files Modified
## Commands Executed
## Test Results
## Known Limitations
## Deviations From Plan
## Recommended Next Stage
```

---

## 8. 计划阶段

### Stage 0：项目规范

交付：

- AGENTS.md；
- README.md；
- docs/design.md；
- docs/data_contract.md；
- Mac Mini 配置；
- RTX 4090 正式配置。

不写训练代码。

### Stage 1：数据标准化

交付：

- NuminaMath 加载；
- 字段统一；
- 稳定 ID；
- 确定性划分；
- JSONL 输出；
- 数据统计；
- 单元测试。

阶段 1 必须能在 Mac CPU 上完成。

### Stage 2：步骤和偏好数据

交付：

- 推理步骤拆分；
- 最终答案提取；
- SFT 样本构造；
- 数字扰动；
- 运算符扰动；
- DPO 偏好数据构造；
- 人工抽样文件；
- Schema 验证。

阶段 2 必须能在 Mac CPU 上完成。

### Stage 3：Mini SFT

先在 Mac M5 上完成：

- 0.5B MPS FP16 LoRA；
- 256 条 SFT 数据；
- 几十个 step；
- adapter 保存；
- 推理验证。

随后在 RTX 4090 上验证同一入口可加载正式配置。

### Stage 4：Mini DPO

先在 Mac M5 上完成：

- 64～256 条偏好数据；
- 0.5B MPS LoRA DPO；
- 几十个 step；
- adapter 保存；
- 训练日志；
- 推理验证。

随后在 RTX 4090 上完成 3B QLoRA DPO smoke test。

### Stage 5：统一评测

交付：

- Base/SFT/DPO 同一 Prompt；
- 同一生成参数；
- 最终答案提取；
- exact match；
- 结果 JSON；
- 汇总 CSV；
- 失败案例。

Mac 用小样本验证链路，4090 运行正式评测。

### Stage 6：正式实验与简历化

只在 RTX 4090 正式执行：

- Base/SFT/DPO 主实验；
- 负样本策略消融；
- beta 消融；
- 序列长度比较；
- 峰值显存；
- 耗时；
- 图表；
- 最终 README；
- 简历项目描述。

---

## 9. 测试要求

每个新模块必须有最小测试。

至少包括：

- 配置校验；
- 两份配置都能解析；
- MPS 配置禁止启用 BitsAndBytes；
- CUDA 配置要求 4-bit；
- 稳定 ID；
- 数据划分确定性；
- 步骤解析；
- 答案提取；
- 数字扰动；
- 运算符扰动；
- chosen != rejected；
- Token 长度过滤；
- SFT smoke test；
- DPO smoke test。

普通单元测试不得下载完整模型或完整数据集。

---

## 10. 实验真实性

禁止把估算值写成实测值。

统一使用：

- **Planned**：尚未实现或运行；
- **Smoke-tested**：只在小样本上验证；
- **Measured**：有完整运行记录；
- **Estimated**：明确标记的估算。

每次实测必须记录：

```text
日期
Git commit
设备型号
系统版本
PyTorch 版本
Transformers/TRL/PEFT 版本
配置文件
数据规模
随机种子
峰值内存
训练耗时
输出路径
最终指标
```

Mac 记录：

```text
芯片型号
统一内存
macOS 版本
MPS 是否可用
进程峰值内存
```

RTX 4090 记录：

```text
GPU 型号
CUDA 版本
峰值显存
```

---

## 11. 文档规则

- README 必须标明当前阶段；
- 未实现功能必须标记为 Planned；
- Mini 实验和正式实验必须明确区分；
- 数据 Schema 改动必须先更新 data_contract.md；
- 代码行为变化必须同步修改 design.md；
- 旧功能移除后同步删除旧文档；
- 不得保留失效命令；
- 不得声明尚未验证的设备支持。

---

## 12. 仓库卫生

禁止提交：

```text
模型权重
原始数据集
处理后大数据
Token
API Key
W&B 凭证
缓存
大型日志
本机绝对路径
```

应忽略：

```text
data/raw/
data/processed/
outputs/
checkpoints/
logs/
wandb/
.cache/
__pycache__/
.DS_Store
```

---

## 13. 完成定义

一个阶段只有同时满足以下条件才算完成：

- 计划文件存在；
- 约定文件已完成；
- 测试通过；
- 命令已记录；
- 阶段报告已完成；
- 没有实现阶段外功能；
- 没有伪造结果；
- Mini 与正式模式仍共享同一代码链路；
- 初学者可以按数据流顺序阅读项目。
