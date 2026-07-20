# Stage 1 重构报告

## 已完成

本阶段已完成 Stage 1 数据链路重构。新的 Stage 1 只负责数据预处理，不执行
SFT、DPO 或 Evaluation 训练/评测。

已实现内容：

- 将 Stage 1 重构为单入口：`scripts/prepare_data.py`。
- 首次从 Hugging Face 下载 `AI-MO/NuminaMath-CoT` 指定 revision，并保存到
  `data/raw/numina_math/`。
- 后续默认使用 `datasets.load_from_disk("data/raw/numina_math")` 读取本地原始
  Dataset，不重复访问远程数据集。
- 在 Stage 1 一次性完成原始字段清洗、固定拆分、步骤拆分、最终答案提取、
  SFT/DPO/Evaluation 数据构造，以及 Mini/formal 真实 tokenizer 长度过滤。
- formal 数据优先选择，Mini 数据作为 formal 的确定性前缀子集。
- 最终数据改为 Hugging Face Dataset / DatasetDict，通过 `save_to_disk()` 保存。
- 删除旧 JSONL、manifest、Stage 2 数据构造、candidate pool、expanded pool、
  训练阶段补筛选等设计。

新的数据目录：

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

## 新增文件

- `plans/refactor_stage1_data.md`
- `reports/stage_1_refactor_report.md`

## 修改文件

- `scripts/prepare_data.py`
- `README.md`
- `docs/design.md`
- `docs/data_contract.md`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`

## 删除文件

- `scripts/build_stage2_data.py`
- `src/mathalign_dpo/data/`
- `data/processed/` 下旧 JSONL、manifest、statistics、manual review 等旧产物

## 执行命令

清理旧数据链路：

```bash
rm -rf data/processed/* src/mathalign_dpo/data scripts/build_stage2_data.py
```

基础静态检查：

```bash
python -m py_compile scripts/prepare_data.py
python scripts/prepare_data.py --help
conda run -n mathalign-dpo python scripts/prepare_data.py --help
```

Stage 1 smoke 验证：

```bash
conda run -n mathalign-dpo python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --smoke-test \
  --overwrite
```

删除 smoke 输出：

```bash
rm -rf data/processed/*
```

Stage 1 正式数据预处理：

```bash
conda run -n mathalign-dpo python scripts/prepare_data.py \
  --mini-config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --formal-config configs/qwen25_3b_4090.yaml \
  --overwrite
```

正式结果检查：

```bash
conda run -n mathalign-dpo python -c "... load_from_disk validation ..."
conda run -n mathalign-dpo python -c "... schema and column validation ..."
```

## 测试结果

基础检查：

- `python -m py_compile scripts/prepare_data.py`：通过。
- `python scripts/prepare_data.py --help`：通过。
- `conda run -n mathalign-dpo python scripts/prepare_data.py --help`：通过。
- Stage 1 smoke preprocessing：通过。
- Stage 1 formal preprocessing：通过。
- `datasets.load_from_disk()` 加载 Mini/formal 全部 Dataset：通过。

正式运行 metadata：

- `completed`: `true`
- `smoke_test`: `false`
- `dataset_name`: `AI-MO/NuminaMath-CoT`
- `dataset_revision`: `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`
- `source_split`: `train`
- `seed`: `42`
- `raw_source_rows`: `859494`
- `created_at`: `2026-07-20T05:32:11Z`
- `raw_dataset_path`: `data/raw/numina_math`

正式数据量：

| 模式 | 数据 | train | validation | evaluation |
|---|---:|---:|---:|---:|
| Mini | SFT | 256 | 32 | - |
| Mini | DPO | 179 | 21 | - |
| Mini | Evaluation | - | - | 32 |
| Formal | SFT | 5000 | 200 | - |
| Formal | DPO | 5000 | 200 | - |
| Formal | Evaluation | - | - | 200 |

目标数量与实际数量一致。

长度过滤结果：

| 模式 | 检查项 | 最大值 | 上限 |
|---|---|---:|---:|
| Mini | SFT token_count | 508 | 512 |
| Mini | DPO prompt token | 384 | 384 |
| Mini | DPO total token | 490 | 512 |
| Mini | Evaluation prompt token | 208 | 512 |
| Formal | SFT token_count | 1024 | 1024 |
| Formal | DPO prompt token | 768 | 768 |
| Formal | DPO total token | 1004 | 1024 |
| Formal | Evaluation prompt token | 311 | 1024 |

Schema 检查：

- Mini/formal SFT columns：
  `schema_version, id, source_id, prompt, completion, messages, token_count, split, metadata`
- Mini/formal DPO columns：
  `schema_version, id, source_id, step_index, prompt, chosen, rejected, split, metadata, token_count`
- Mini/formal Evaluation columns：
  `schema_version, id, source_id, problem, reference_answer, prompt_messages, prompt_token_count, split, metadata`

额外验证：

- SFT `token_count` 无空值。
- DPO `token_count` 无空值。
- Evaluation `prompt_token_count` 无空值。
- DPO `chosen != rejected`。
- Mini SFT train/validation 是 formal SFT 对应 split 的前缀子集。
- Mini DPO train/validation 是 formal DPO 对应 split 的前缀子集。
- Mini Evaluation 是 formal Evaluation 的前缀子集。

过滤统计：

```json
{
  "sft_formal": {
    "mini_prefix_token_too_long": 189,
    "parse_failed": 81,
    "token_too_long": 499
  },
  "dpo_formal": {
    "chosen_too_long": 14,
    "mini_prefix_chosen_too_long": 2,
    "mini_prefix_prompt_too_long": 66,
    "number_mutation:no_number_target;operator_mutation:no_operator_target": 341,
    "operator_mutation:no_operator_target;number_mutation:no_number_target": 341,
    "parse_failed": 16,
    "parse_partial": 32,
    "prompt_too_long": 426,
    "rejected_in_prompt_history": 4
  },
  "evaluation_formal": {
    "mini_prefix_prompt_plus_generation_too_long": 1,
    "parse_failed": 1,
    "parse_partial": 15,
    "prompt_plus_generation_too_long": 1
  }
}
```

## 已知限制

- Stage 3-5 训练和评测入口本阶段未迁移。
- 由于 `src/mathalign_dpo/data/` 已删除，当前旧的 SFT/DPO/Evaluation 代码仍会因
  import 旧数据模块而暂时不可运行。
- 本阶段只确认数据预处理产物正确，不声明任何训练或评测指标。
- 当前 `scripts/prepare_data.py` 为了遵守“只修改一个 Python 文件”的约束，承担了较多
  Stage 1 逻辑；后续可在训练/评测迁移完成后再按职责拆分。

## 计划偏离

- 无重大偏离。
- 按计划接受 Stage 3-5 旧入口暂时不可运行这一迁移边界。
- 为保证 `python scripts/prepare_data.py --help` 在裸 Python 环境也可运行，将
  `PyYAML` import 延迟到读取配置时执行。

## 建议下一阶段

建议下一阶段迁移训练与评测加载逻辑：

- SFT 入口直接读取 `data/processed/<mode>/sft`。
- DPO 入口直接读取 `data/processed/<mode>/dpo`。
- Evaluation 入口直接读取 `data/processed/<mode>/evaluation`。
- 删除训练阶段的 candidate pool、expanded pool、tokenizer 过滤、样本重新选择和
  Stage 2 manifest lineage 校验。
- 运行新的 Stage 3 smoke，验证训练代码可以消费 Stage 1 产物。
