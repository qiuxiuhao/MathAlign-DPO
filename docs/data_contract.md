# MathAlign-DPO 数据契约

## 1. 目的

本文件定义 MathAlign-DPO 所有落盘数据格式。

Mac Mini 和 RTX 4090 必须使用完全相同的数据 Schema。

数据生产者和消费者都必须遵守本契约。

修改 Schema 前必须先修改本文件。

---

## 2. 通用约定

### 2.1 文件格式

样本集合：

```text
UTF-8 JSON Lines
一行一个 JSON 对象
扩展名 .jsonl
```

统计：

```text
UTF-8 JSON
扩展名 .json
```

实验汇总可以使用 CSV。

### 2.2 每行公共要求

每行必须：

- 是 JSON Object；
- 包含非空 `id`；
- 包含 `schema_version`；
- 保留源样本 ID；
- 不包含 NaN 和 Infinity；
- `step_index` 从 0 开始；
- 对同一 seed 可重复生成；
- 与设备无关。

MPS 和 CUDA 运行生成的数据不得因设备不同而改变 Schema。

### 2.3 Schema 版本

```json
"schema_version": "1.0"
```

### 2.4 ID

标准化 ID：

```text
numina_train_00000123
```

DPO ID：

```text
numina_train_00000123_step_002_number_mutation
```

ID 不得依赖：

- 运行设备；
- 进程顺序；
- GPU 数量；
- Python 对象地址。

### 2.5 文本标准化

允许：

- CRLF 转 LF；
- 去除首尾空白；
- 合并过多连续空行；
- 保留 LaTeX；
- 保留数学符号；
- 保留有意义换行。

禁止：

- 全部转小写；
- 删除 LaTeX 命令；
- 改写数学表达式；
- 因设备模式改变文本。

---

## 3. 标准化数学样本

### 文件

```text
data/processed/normalized_<split>.jsonl
```

### Schema

```json
{
  "schema_version": "1.0",
  "id": "numina_train_00000123",
  "source": "AI-MO/NuminaMath-CoT",
  "source_split": "train",
  "source_id": "00000123",
  "problem": "A non-empty mathematics problem.",
  "solution": "A complete non-empty reference solution.",
  "metadata": {
    "source_subset": null,
    "original_fields": []
  }
}
```

### 字段

| 字段 | 类型 | 必须 | 含义 |
|---|---|---:|---|
| schema_version | string | 是 | Schema 版本 |
| id | string | 是 | 项目稳定 ID |
| source | string | 是 | 数据集名称 |
| source_split | string | 是 | 原始 split |
| source_id | string | 是 | 原始行 ID |
| problem | string | 是 | 数学问题 |
| solution | string | 是 | 完整正确解答 |
| metadata | object | 是 | 可追溯信息 |

### 拒绝条件

- problem 为空；
- solution 为空；
- problem 与 solution 完全相同；
- 字段不是字符串；
- ID 重复。

最终答案暂时提取失败，不影响标准化样本保留。

---

## 4. 步骤化数学样本

### 文件

```text
data/processed/step_<split>.jsonl
```

### Schema

```json
{
  "schema_version": "1.0",
  "id": "numina_train_00000123",
  "source_id": "numina_train_00000123",
  "problem": "A mathematics problem.",
  "solution": "The original complete solution.",
  "steps": [
    "First correct step.",
    "Second correct step.",
    "Final correct step."
  ],
  "final_answer": "14",
  "parse_status": "success",
  "metadata": {
    "step_count": 3,
    "answer_extraction_method": "boxed",
    "answer_confidence": "high",
    "answer_candidate": "14",
    "parse_failure_reason": null
  }
}
```

### parse_status

```text
success
partial
failed
```

定义：

- success：至少两个可用步骤且提取到最终答案；
- partial：步骤可用，但最终答案未提取；
- failed：没有可靠步骤序列。

`partial` 和 `failed` 的 `final_answer` 必须为 `null`。`failed` 的
`steps` 必须为空数组，并在 `metadata.parse_failure_reason` 中记录原因。
低置信度答案提取结果保留在 `metadata.answer_candidate`，但不写入
`final_answer`。

### answer_confidence

```text
high
medium
low
none
```

规则：

- `boxed`、`fbox`、`####` 和明确 answer 标签：high；
- 选择题标记、最后一行独立答案：medium；
- 末尾数字或分数 fallback：low；
- 未提取答案：none。

正式 DPO 默认只使用 high/medium。low 置信结果保留为 partial 或人工审查
信息，不进入正式 DPO 偏好数据。

### 步骤规则

每个步骤：

- 非空；
- 顺序与原解答一致；
- 不添加“正确步骤”等标签；
- 不与前一步完全重复；
- 不复制题目文本，除非原解答本身如此。

正式 DPO 默认只使用 `success`。

---

## 5. SFT 样本

### 文件

```text
data/processed/sft_<split>.jsonl
```

### Schema

```json
{
  "schema_version": "1.0",
  "id": "numina_train_00000123_sft",
  "source_id": "numina_train_00000123",
  "messages": [
    {
      "role": "system",
      "content": "You are a careful mathematical reasoning assistant."
    },
    {
      "role": "user",
      "content": "Solve the following mathematics problem. Show a clear step-by-step derivation and put the final answer in \\boxed{}.\n\nProblem:\n..."
    },
    {
      "role": "assistant",
      "content": "Complete correct reference solution."
    }
  ],
  "metadata": {
    "final_answer": "14",
    "token_count": null
  }
}
```

### 消息要求

顺序：

```text
system → user → assistant
```

规则：

- 第一版恰好一个 assistant 消息；
- assistant 是完整正确解答；
- 不允许 rejected 内容进入 SFT；
- user 必须包含完整题目；
- Prompt 由统一函数生成。

### 长度

按照当前配置的 tokenizer chat template 渲染。

Stage 2 不加载 tokenizer，不计算真实 token 长度，也不做 token 长度过滤。
因此 Stage 2 生成的 SFT 样本中 `metadata.token_count` 必须为 `null`，并在
统计中记录 `token_length_status = "not_checked_no_tokenizer"`。真实长度过滤
推迟到训练前的 tokenizer 阶段。

Mac Mini：

```text
token_count <= 512
```

RTX 4090 正式：

```text
token_count <= 1024
```

同一源样本可能在 Mini 中因长度被过滤、在正式配置中保留。

过滤必须记录配置和数量，不能静默截断正确解答。

### Stage 3 tokenizer 过滤结果

Stage 3 训练前会读取 Stage 2 SFT JSONL，并用当前模型 tokenizer 的
chat template 计算真实 token 长度。训练使用的内存数据格式为 TRL
prompt/completion 对话格式：

```json
{
  "id": "numina_train_00000123_sft",
  "source_id": "00000123",
  "prompt": [
    {
      "role": "system",
      "content": "You are a careful mathematical reasoning assistant."
    },
    {
      "role": "user",
      "content": "Solve the following mathematics problem..."
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "Complete correct reference solution."
    }
  ],
  "token_count": 384,
  "metadata": {}
}
```

## 6. DPO 样本

### 文件

```text
data/processed/dpo_<split>.jsonl
```

### Schema

```json
{
  "schema_version": "1.0",
  "id": "numina_train_00000123_step_002_number_mutation",
  "source_id": "00000123",
  "step_index": 2,
  "prompt": [
    {"role": "system", "content": "You are a careful mathematical reasoning assistant."},
    {"role": "user", "content": "Solve the following mathematics problem..."},
    {"role": "assistant", "content": "Previous correct step."}
  ],
  "chosen": [
    {"role": "assistant", "content": "Current correct step."}
  ],
  "rejected": [
    {"role": "assistant", "content": "Current mutated incorrect step."}
  ],
  "token_count": null,
  "metadata": {
    "negative_strategy": "mixed",
    "token_length_status": "not_checked_no_tokenizer"
  }
}
```

### Stage 4 tokenizer 过滤结果

Stage 4 训练前读取 Stage 2 DPO JSONL，并用当前模型 tokenizer 的 chat template
计算真实 token 长度。训练使用的内存数据格式仍为 TRL conversational preference
格式：

```json
{
  "id": "numina_train_00000123_step_002_number_mutation",
  "source_id": "00000123",
  "prompt": [],
  "chosen": [],
  "rejected": [],
  "token_count": {
    "prompt": 320,
    "chosen_total": 384,
    "rejected_total": 385,
    "chosen_completion": 64,
    "rejected_completion": 65
  },
  "metadata": {}
}
```

该格式不作为新的 `data/processed` 中间数据发布；它属于单次 DPO 训练运行的
内存视图。Stage 4 必须在送入 TRL `DPOTrainer` 前强制：

```text
prompt <= dpo.max_prompt_length
chosen_total <= dpo.max_length
rejected_total <= dpo.max_length
chosen_completion > 0
rejected_completion > 0
```

TRL `DPOConfig.max_length` 仍会传入 Trainer，但项目不得依赖 Trainer 静默截断。
过滤统计写入 `run_metadata.json`。Mini DPO 只能使用 Stage 2 manifest 中
`views.mini.dpo` 定义的偏好样本，并且每条样本的 `metadata.normalized_id`
必须属于 `views.mini.dpo_source_ids`。如果真实 tokenizer 过滤后不足配置的
`dpo.train_samples` 或 `dpo.validation_samples`，训练必须失败，或先回到
Stage 2 调整 Mini 候选池设计；禁止从 formal DPO view 静默补样。

该格式不作为新的 `data/processed` 中间数据发布；它属于单次训练运行的内存
视图。过滤统计写入 `run_metadata.json`：

```json
{
  "token_statistics": {
    "max_length": 512,
    "train": {
      "candidate_count": 256,
      "kept_count": 179,
      "filtered_count": 77,
      "final_count": 179,
      "selected_pool": "run_mode",
      "selection_hash": "..."
    }
  }
}
```

禁止对超长样本做静默截断。

Mini SFT 正式训练必须在 tokenizer 过滤后选择实际 256 条训练样本。Mini DPO
正式训练样本数独立由 `dpo.train_samples` / `dpo.validation_samples` 控制。
选择过程
必须记录候选数量、长度过滤数量、最终数量、`max_length` 和样本选择 hash。

---

## 6. 错误步骤结果

### Schema

```json
{
  "strategy": "number_mutation",
  "original_step": "2 × 7 = 14.",
  "mutated_step": "2 × 7 = 16.",
  "changed_span": "14",
  "replacement": "16",
  "success": true,
  "reason": null
}
```

### 策略

```text
number_mutation
operator_mutation
mixed
```

### 成功要求

```text
mutated_step.strip() != original_step.strip()
```

### 失败原因

```text
no_numeric_literal
no_supported_operator
empty_step
step_too_short
mutation_unchanged
invalid_output
```

---

## 7. DPO 偏好样本

### 文件

```text
data/processed/dpo_<split>.jsonl
```

### 推荐 Chat Schema

```json
{
  "schema_version": "1.0",
  "id": "numina_train_00000123_step_001_number_mutation",
  "source_id": "numina_train_00000123",
  "step_index": 1,
  "prompt": [
    {
      "role": "system",
      "content": "You are a careful mathematical reasoning assistant."
    },
    {
      "role": "user",
      "content": "Solve the following mathematics problem step by step.\n\nProblem:\n..."
    },
    {
      "role": "assistant",
      "content": "First correct reasoning step."
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "Second correct reasoning step."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "Locally plausible but incorrect second step."
    }
  ],
  "metadata": {
    "negative_strategy": "number_mutation",
    "mutation": {
      "strategy": "number_mutation",
      "changed_span": "14",
      "replacement": "16",
      "success": true,
      "reason": null
    },
    "final_answer": "14",
    "prompt_step_count": 1,
    "prompt_token_count": null,
    "chosen_token_count": null,
    "rejected_token_count": null
  }
}
```

Stage 4 必须根据实际 TRL 版本确认对话式 DPO Schema。若需要等价调整，必须先更新本文件。

### 语义

正确步骤：

```text
s0, s1, ..., sn
```

对 step_index = i：

```text
prompt = problem + s0 ... s(i-1)
chosen = si
rejected = mutate(si)
```

i = 0 时，prompt 中没有历史 assistant 步骤。

### 拒绝条件

- chosen 为空；
- rejected 为空；
- chosen 与 rejected 相同；
- step_index 越界；
- prompt 已包含 chosen；
- prompt 包含错误历史步骤；
- 缺少 mutation metadata；
- 长度超过配置限制。

Stage 2 不加载 tokenizer，不计算真实 token 长度，也不按 token 长度过滤
DPO 样本。`prompt_token_count`、`chosen_token_count` 和
`rejected_token_count` 必须为 `null`，真实长度过滤推迟到训练前。

DPO 候选偏好对必须先全部生成，再使用
`seed|source_id|step_index|actual_strategy` 计算稳定 rank，按 rank 排序后
截取。禁止按遍历顺序达到上限后提前停止。单个 `source_id` 的贡献数量必须
受固定上限控制。

Mini DPO 必须只从 Stage 1/2 manifest 中 Mini ID 集合对应的源题目产生候选。
Mini DPO train 可在这些候选内按稳定 rank 截取 64 到 256 条，validation
由 `dpo.validation_samples` 独立控制。Stage 2 DPO row 的 `source_id` 保留
原始短 ID；训练前使用 `metadata.normalized_id` 对齐 manifest 中的
`dpo_source_ids`。

### 长度规则

```text
prompt_tokens <= dpo.max_prompt_length
prompt_tokens + max(chosen_tokens, rejected_tokens) <= dpo.max_length
```

Mac Mini：

```text
max_prompt_length = 384
max_length = 512
```

RTX 4090：

```text
max_prompt_length = 768
max_length = 1024
```

第一版过滤超长样本，不截断推理步骤。

---

## 8. Stage 5 评测样本

Stage 5 v1 不要求提前发布新的 `data/processed/eval.jsonl`。统一评测直接读取
Stage 2 manifest 中的固定 evaluation 视图：

```text
data/processed/stage2_manifest.json
data/processed/step_eval.jsonl
```

### Schema

```json
{
  "schema_version": "1.0",
  "id": "numina_train_00000123",
  "source_id": "00000123",
  "problem": "A mathematics problem.",
  "final_answer": "14",
  "parse_status": "success",
  "metadata": {
    "answer_extraction_method": "boxed"
  }
}
```

要求：

- problem 非空；
- `parse_status=success`；
- `final_answer` 非空；
- ID 唯一；
- 样本 ID 必须来自 `views.mini.step.evaluation`；
- 与 Stage 3 SFT 和 Stage 4 DPO selected train/validation source 不重叠。

---

## 9. 单样本评测结果

### 文件

```text
outputs/results/<run_id>/predictions.jsonl
```

### Schema

```json
{
  "schema_version": "1.0",
  "id": "eval_000001",
  "run_mode": "mini",
  "model_stage": "dpo",
  "prompt_messages": [],
  "generated_text": "Model response.",
  "predicted_answer": "14",
  "normalized_predicted_answer": "14",
  "reference_answer": "14",
  "normalized_reference_answer": "14",
  "answer_extracted": true,
  "exact_match": true,
  "generation_seconds": 0.0,
  "output_tokens": 0
}
```

run_mode：

```text
mini
formal
```

model_stage：

```text
base
sft
dpo
```

---

## 10. 评测汇总

### 文件

```text
outputs/results/<run_id>/summary.json
```

### Schema

```json
{
  "schema_version": "1.0",
  "run_id": "2026-07-19_mini_dpo_eval_seed42",
  "run_mode": "mini",
  "model_stage": "dpo",
  "device_backend": "mps",
  "num_examples": 32,
  "answer_extraction_rate": 0.0,
  "exact_match_accuracy": 0.0,
  "invalid_output_rate": 0.0,
  "average_output_tokens": 0.0,
  "elapsed_seconds": 0.0,
  "peak_memory_mb": 0,
  "config_path": "configs/qwen25_0_5b_m5_24gb_mini.yaml",
  "git_commit": null
}
```

数值必须实测。

---

## 11. 数据统计

### 文件

```text
data/processed/data_statistics.json
```

### Stage 1 最小字段

Stage 1 只记录标准化和划分统计，不提前写入步骤、SFT 或 DPO
计数。后续阶段生成对应数据时再扩展本文件。

```json
{
  "schema_version": "1.0",
  "stage": 1,
  "seed": 42,
  "dataset_name": "AI-MO/NuminaMath-CoT",
  "dataset_revision": "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
  "source_split": "train",
  "smoke_test": false,
  "source_rows": 0,
  "normalized_rows": 0,
  "normalization_rejected": 0,
  "normalization_rejected_by_reason": {},
  "id_strategy": "row_index_fallback",
  "id_field": null,
  "split_counts_formal": {
    "train": 0,
    "validation": 0,
    "evaluation": 0
  },
  "split_counts_mini": {
    "train": 0,
    "validation": 0,
    "evaluation": 0
  },
  "field_audit": {
    "fields": [],
    "field_types": {},
    "empty_counts": {},
    "problem_field": "problem",
    "solution_field": "solution",
    "source_rows_sha256": "..."
  }
}
```

同一基础数据可以根据两份配置分别输出视图统计。长度过滤统计
在 Stage 2 之后由对应数据生产阶段添加。

### Stage 2 文件

Stage 2 使用独立统计文件，避免覆盖 Stage 1 历史统计：

```text
data/processed/stage2_statistics.json
```

Stage 2 最小字段：

```json
{
  "schema_version": "1.0",
  "stage": 2,
  "completed": true,
  "stage1_manifest_file": {
    "path": "data/processed/split_manifest.json",
    "sha256": "..."
  },
  "token_length_status": "not_checked_no_tokenizer",
  "step_counts_formal": {},
  "step_counts_mini": {},
  "parse_status_counts": {},
  "parse_failure_reason_counts": {},
  "answer_extraction_method_counts": {},
  "answer_confidence_counts": {},
  "sft_counts_formal": {},
  "sft_counts_mini": {},
  "dpo_counts_formal": {},
  "dpo_counts_mini": {},
  "dpo_max_pairs_per_source": 2,
  "dpo_allowed_answer_confidences": ["high", "medium"],
  "dpo_applied_strategy_counts": {},
  "dpo_mutation_failures": {},
  "manual_review_count": 0
}
```

---

## 12. 划分策略

初始计划：

```text
train: 95%
validation: 2.5%
evaluation: 2.5%
```

规则：

- 先按 source_id 划分，再构造步骤级样本；
- 同一道题的所有步骤只能出现在一个 split；
- split 不依赖设备；
- Mini 从 train split 中确定性选择较小子集；
- 正式模式使用更大的确定性子集；
- 保存 split manifest。

Mini 数据必须是正式训练数据的可追踪子集，而不是重新随机划分。

`train_ratio`、`validation_ratio` 和 `evaluation_ratio` 只决定源样本的
split 归属。`train_samples`、`validation_samples` 和
`evaluation_samples` 只决定从对应 split 的稳定排序结果中取多少条。

Stage 1 入口必须同时读取 Mini 和 formal 配置。共享的 normalized
JSONL 文件按 formal 配置需要的最大样本数生成，Mini 视图记录在
`split_manifest.json` 中，并且必须是 formal ID 列表的确定性前缀子集。

### split manifest

文件：

```text
data/processed/split_manifest.json
```

Stage 1 最小 Schema：

```json
{
  "schema_version": "1.0",
  "stage": 1,
  "completed": true,
  "run_id": "20260719T120000Z_stage1",
  "dataset_name": "AI-MO/NuminaMath-CoT",
  "dataset_revision": "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
  "source_split": "train",
  "seed": 42,
  "smoke_test": false,
  "split_method": "sha256_source_id_bucket_v1",
  "source_rows_sha256": "...",
  "split_ratios": {
    "train": 0.95,
    "validation": 0.025,
    "evaluation": 0.025
  },
  "id_strategy": "row_index_fallback",
  "id_field": null,
  "configs": {
    "mini": "configs/qwen25_0_5b_m5_24gb_mini.yaml",
    "formal": "configs/qwen25_3b_4090.yaml"
  },
  "views": {
    "formal": {
      "train": [],
      "validation": [],
      "evaluation": []
    },
    "mini": {
      "train": [],
      "validation": [],
      "evaluation": []
    }
  },
  "files": {},
  "statistics_file": {}
}
```

Stage 2 使用独立 manifest，避免覆盖 Stage 1 历史 manifest：

```text
data/processed/stage2_manifest.json
```

Stage 2 manifest 必须记录 Stage 1 manifest 的路径和 sha256：

```json
{
  "schema_version": "1.0",
  "stage": 2,
  "completed": true,
  "stage1_manifest_file": {
    "path": "data/processed/split_manifest.json",
    "sha256": "..."
  },
  "token_length_status": "not_checked_no_tokenizer",
  "views": {
    "formal": {
      "step": {"train": [], "validation": [], "evaluation": []},
      "sft": {"train": [], "validation": []},
      "dpo": {"train": [], "validation": []}
    },
    "mini": {
      "step": {"train": [], "validation": [], "evaluation": []},
      "sft": {"train": [], "validation": []},
      "sft_source_ids": {"train": [], "validation": []},
      "dpo": {"train": [], "validation": []},
      "dpo_source_ids": {"train": [], "validation": []}
    }
  },
  "files": {},
  "statistics_file": {},
  "manual_review_file": {}
}
```

`completed: true` 只能在所有 staged 文件通过校验并发布后写入最终
manifest。

后续消费者只有在以下条件全部满足时才能读取 Stage 1 数据：

- `split_manifest.json` 存在；
- `completed == true`；
- manifest 中每个文件的 `rows` 与实际 JSONL 行数一致；
- manifest 中每个文件的 `sha256` 与实际文件一致；
- statistics 文件存在且 hash 与 manifest 记录一致。

---

## 13. 文件命名

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
data/processed/manual_review_preferences.jsonl
```

不为 Mac 和 CUDA 分别复制基础数据文件。

Stage 1 `split_manifest.json` 只描述 normalized 数据和基础 split。Stage 2
`stage2_manifest.json` 描述 step/SFT/DPO/manual-review 数据，并引用 Stage 1
manifest 的路径与 sha256。

长度过滤后的训练视图可以在加载时按配置生成，或使用带明确配置后缀的缓存文件；
真实 token 长度过滤推迟到训练阶段。

Stage 1 使用 staging directory 实现完整输出事务：

```text
data/processed/.stage_<run_id>/
```

只有当 JSONL schema 校验、行数校验和 sha256 记录全部完成后，才将
staging 文件发布到最终文件名。默认不得静默覆盖已有输出。覆盖发布必须
先保留旧输出备份；如果新运行在 staging、校验或最终发布阶段失败，旧的
完整输出必须保持可读且 hash 不变。

---

## 14. 运行元数据

### Schema

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "run_mode": "mini",
  "stage": "sft",
  "git_commit": "...",
  "config_path": "...",
  "seed": 42,
  "device_backend": "mps",
  "device_name": "Apple M5",
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

CUDA 运行 metadata 可增加：

```text
cuda_version
gpu_memory_total_mb
peak_gpu_memory_mb
```

MPS 运行 metadata 可增加：

```text
macos_version
mps_available
process_peak_memory_mb
```

---

## 15. 契约执行

后续必须提供 Schema 校验。

校验时机：

1. 标准化后；
2. 步骤拆分后；
3. SFT 构造后；
4. DPO 构造后；
5. 训练前；
6. 评测汇总前。

数据消费者不得静默修复坏数据。

数据生产阶段负责输出合法数据。
