# MathAlign-DPO 数据契约

## 当前状态

当前项目已经完成 Stage 1 数据预处理、Stage 2 Mini SFT 和 Stage 3 Mini DPO。

Stage 1 负责生成后续训练和评价直接消费的最终 Hugging Face Dataset。Stage 2 和 Stage 3 只能通过 `datasets.load_from_disk()` 加载本地数据，不再修改数据内容。

## 存储结构

原始数据：

```text
data/raw/numina_math/
```

最终处理数据：

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

所有最终样本集合都是 Hugging Face Dataset 或 DatasetDict 目录，由 `save_to_disk()` 写入，并由 `datasets.load_from_disk()` 读取。

Stage 1 不再发布 JSONL 训练数据、split manifest、Stage 2 manifest、文件级 hash 血缘记录、candidate pool、expanded pool 或人工复查 JSONL。

## 通用规则

- 每行都有 `schema_version = "1.0"`。
- 每行都有稳定且非空的 `id`。
- 每行通过 `source_id` 和 `metadata.raw_source_id` 保留来源。
- split 和行顺序由 dataset name、revision、source split、source ID 和 seed 的 SHA-256 hash 确定。
- Mini 数据是对应 formal 数据的确定性前缀子集。
- 最终 Dataset 不允许出现空的 token count。
- tokenizer 长度过滤必须在 Stage 1 使用对应模式 tokenizer 完成。
- 超长样本只过滤，不截断。

## SFT 数据集

路径：

```text
data/processed/<mini|formal>/sft/
```

类型：包含 `train` 和 `validation` 的 DatasetDict。

字段：

```text
schema_version: string
id: string
source_id: string
prompt: list[{role, content}]
completion: list[{role, content}]
messages: list[{role, content}]
token_count: int
split: string
metadata: object
```

规则：

- `messages` 必须是 system/user/assistant。
- `prompt` 必须是 system/user。
- `completion` 必须是一条 assistant message。
- `token_count` 使用对应模式 tokenizer 的 chat template 计算。

## DPO 数据集

路径：

```text
data/processed/<mini|formal>/dpo/
```

类型：包含 `train` 和 `validation` 的 DatasetDict。

字段：

```text
schema_version: string
id: string
source_id: string
step_index: int
prompt: list[{role, content}]
chosen: list[{role, content}]
rejected: list[{role, content}]
token_count: object
split: string
metadata: object
```

规则：

- `chosen` 和 `rejected` 各自必须是一条 assistant message。
- `chosen` 和 `rejected` 不得相同。
- `rejected` 不得出现在 prompt history 中。
- `token_count` 包含 prompt、chosen total、rejected total、chosen completion 和 rejected completion 的 token 数。
- DPO 训练阶段只检查这些字段，不重新筛选或重排数据。

## Evaluation 数据集

路径：

```text
data/processed/<mini|formal>/evaluation/
```

类型：单个 Dataset。

字段：

```text
schema_version: string
id: string
source_id: string
problem: string
reference_answer: string
prompt_messages: list[{role, content}]
prompt_token_count: int
split: string
metadata: object
```

规则：

- 每行必须有非空 `reference_answer`。
- `prompt_token_count + evaluation.max_new_tokens` 必须适配对应模式的模型长度上限。
- Evaluation 样本只来自确定性的 evaluation split。

## 元数据

`data/processed/metadata.json` 只记录描述性信息：

```text
schema_version
stage
completed
created_at
smoke_test
dataset_name
dataset_revision
source_split
seed
raw_dataset_path
raw_source_rows
processed_dataset_paths
config_paths
tokenizers
target_counts
actual_counts
filter_counts_by_reason
split_method
selection_method
```

它不是多阶段 manifest，也不是文件 hash 门禁。
