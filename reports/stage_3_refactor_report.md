# Stage 3 重构报告

## 已完成

本阶段已完成 DPO 训练与评价链路重构，并完成 Mini DPO 正式训练验证。

已实现内容：

- 新增顶层 `dpo/` 包，替代旧 `src/mathalign_dpo/training` DPO 代码。
- DPO 训练直接读取 Stage 1 产出的 `data/processed/<mode>/dpo` Hugging Face Dataset。
- DPO 不再进行数据清洗、样本补齐、重新排序、候选池构造、manifest 校验或训练期长度过滤。
- DPO policy 从 Base + Stage 2 SFT adapter 初始化。
- 使用 TRL `DPOTrainer` + `DPOConfig`，`ref_model=None`，由 TRL 为 PEFT policy 创建 reference adapter。
- 训练后保存 DPO adapter 和 tokenizer，并重新加载 DPO adapter 完成生成验证。
- 使用同一份 `data/processed/mini/evaluation` 完成 Base/SFT/DPO 三方评价。
- 删除旧 `src/` 目录和旧 `scripts/train_dpo.py`。

当前活跃代码结构：

```text
configs/
scripts/prepare_data.py
sft/
dpo/
```

## 新增文件

- `plans/stage_3_dpo.md`
- `dpo/__init__.py`
- `dpo/data.py`
- `dpo/modeling.py`
- `dpo/evaluate.py`
- `dpo/train.py`
- `reports/stage_3_refactor_report.md`

## 修改文件

- `README.md`
- `docs/design.md`
- `configs/load_config.py`
- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `pyproject.toml`
- `sft/evaluate.py`

## 删除文件

- `scripts/train_dpo.py`
- `src/`
- 旧 `tests/`
- 旧 reports，仅保留：
  - `reports/stage_1_refactor_report.md`
  - `reports/stage_2_refactor_report.md`
  - `reports/stage_3_refactor_report.md`

## 执行命令

静态检查：

```bash
python -m py_compile \
  configs/load_config.py \
  sft/modeling.py \
  sft/evaluate.py \
  dpo/data.py \
  dpo/modeling.py \
  dpo/evaluate.py \
  dpo/train.py

conda run -n mathalign-dpo python -m dpo.train --help
```

DPO Dataset 与 SFT 依赖检查：

```bash
conda run -n mathalign-dpo python -c "... load_dpo_datasets mini/formal ..."
conda run -n mathalign-dpo python -c "... validate_sft_dir outputs/mini/sft ..."
```

用户执行的 Mini DPO smoke：

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

用户执行的 Mini DPO 正式训练：

```bash
python -m dpo.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --sft-dir outputs/mini/sft \
  --overwrite
```

正式结果核验：

```bash
python -m json.tool outputs/mini/dpo/run_config.json
python -m json.tool outputs/mini/dpo/train_metrics.json
python -m json.tool outputs/mini/dpo/eval_metrics.json
python -m json.tool outputs/mini/dpo/base_sft_dpo_summary.json
wc -l outputs/mini/dpo/loss_history.jsonl \
  outputs/mini/dpo/base_sft_dpo_predictions.jsonl \
  outputs/mini/dpo/correct_cases.jsonl \
  outputs/mini/dpo/error_cases.jsonl \
  outputs/mini/dpo/adapter_reload_samples.jsonl
du -sh outputs/mini/dpo outputs/mini/dpo/adapter outputs/mini/dpo/tokenizer
```

## 测试结果

基础检查：

- `python -m py_compile ...`：通过。
- `conda run -n mathalign-dpo python -m dpo.train --help`：通过。
- `src/` 已删除。
- `scripts/train_dpo.py` 已删除。
- `pyproject.toml` 不再包含 `mathalign_dpo*`。

本地 Dataset 检查：

- Mini DPO：179 train / 21 validation。
- Formal DPO：5000 train / 200 validation。
- Mini DPO smoke override 正确写入 `dpo.train_samples`、`dpo.validation_samples`、`dpo.max_steps`。
- `outputs/mini/sft` 校验通过，包含 SFT adapter、tokenizer 和 completed `run_config.json`。

Smoke 修复记录：

- 首次 DPO smoke 训练完成后，写入 `loss_history.jsonl` 时因 `grad_norm: NaN` 失败。
- 已在共享 JSON writer 中将非有限浮点数转为 JSON `null`。
- `loss/eval_loss` 的 NaN/Inf 拦截仍保留。
- 后续 Mini DPO 正式训练已成功完成。

Mini DPO 正式运行结果：

- 状态：`completed`
- `smoke_test`: `false`
- 配置文件：`configs/qwen25_0_5b_m5_24gb_mini.yaml`
- 输出目录：`outputs/mini/dpo`
- 创建时间：`2026-07-20T07:15:17Z`
- Git commit：`ba5bc04`
- 工作区状态：dirty，包含本阶段重构文件变更，尚未提交。

运行时：

- 后端：MPS。
- 设备：`mps`。
- `mps_is_built`: `true`
- `mps_is_available`: `true`
- `mps_fallback_env`: 空字符串。
- 模型目录：`model/Qwen2.5-0.5B-Instruct`
- SFT adapter：`outputs/mini/sft/adapter`
- tokenizer：`outputs/mini/sft/tokenizer`
- reference policy：`trl_peft_ref_adapter`

DPO 配置：

| 指标 | 数值 |
|---|---:|
| beta | 0.1 |
| loss_type | sigmoid |
| max_length | 512 |
| max_prompt_length_checked | 384 |
| train rows | 179 |
| validation rows | 21 |
| max_steps | 20 |

训练指标：

| 指标 | 数值 |
|---|---:|
| train_loss | 0.6930708885192871 |
| train_runtime | 92.9013 秒 |
| train_samples_per_second | 0.861 |
| train_steps_per_second | 0.215 |
| total_flos | 92230294041600.0 |

验证指标：

| 指标 | 数值 |
|---|---:|
| eval_loss | 0.6958194375038147 |
| eval_runtime | 56.6548 秒 |
| eval_samples_per_second | 0.371 |
| eval_steps_per_second | 0.371 |

loss 轨迹检查：

- `loss_history.jsonl` 共 21 行。
- 训练 loss 记录 20 条。
- 最小训练 loss：0.6766。
- 最大训练 loss：0.7181。
- 最后一条训练 loss：0.6801。
- `eval_loss`: 0.6958194375038147。
- 未发现 NaN/Inf loss。
- `grad_norm` 共 20 条，其中 13 条为 `null`，这是由 MPS/TRL 日志中的非有限 `grad_norm` 转换而来，不作为训练失败条件。

Base/SFT/DPO 评价结果：

| 模型 | 样本数 | Exact Match | 正确数 | 答案提取成功率 | 平均输出 token | 平均生成耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 32 | 0.0625 | 2/32 | 1.0 | 251.78125 | 4.21713884375 秒 |
| SFT | 32 | 0.125 | 4/32 | 1.0 | 242.375 | 5.15002825 秒 |
| DPO | 32 | 0.15625 | 5/32 | 1.0 | 241.5625 | 5.3939896875 秒 |

正确样本 ID：

- Base：
  `numina_train_00628871`, `numina_train_00612904`
- SFT：
  `numina_train_00098078`, `numina_train_00789466`,
  `numina_train_00717283`, `numina_train_00612904`
- DPO：
  `numina_train_00098078`, `numina_train_00789466`,
  `numina_train_00717283`, `numina_train_00626335`,
  `numina_train_00612904`

输出文件：

- `base_sft_dpo_predictions.jsonl`：96 行，Base/SFT/DPO 各 32 条。
- `correct_cases.jsonl`：5 行。
- `error_cases.jsonl`：5 行。
- `adapter_reload_samples.jsonl`：1 行。
- DPO adapter reload 验证完成。

产物大小：

- `outputs/mini/dpo`：79 MB。
- `outputs/mini/dpo/adapter`：4.2 MB。
- `outputs/mini/dpo/tokenizer`：15 MB。

## 已知限制

- 当前结果来自 Mini 32 条 evaluation，只能说明端到端链路可运行，并观察到小样本改善；不能作为正式性能结论。
- DPO `train_loss` 接近 0.693，符合 very small Mini DPO / reference adapter 初期训练的预期，但不代表偏好学习已充分收敛。
- 生成平均长度仍接近 `max_new_tokens=256`，后续统一评测阶段需要继续关注停止条件和答案格式。
- Formal RTX 4090 DPO 代码路径已实现，但尚未在本机运行；需要先有 `outputs/formal/sft`。
- 旧 tests 已按本阶段要求删除，后续阶段应重新建立面向新顶层链路的测试集。

## 计划偏离

- 按用户追加要求，本阶段删除整个 `src/`，不再保留任何旧 `mathalign_dpo` 包代码。
- 按用户追加要求，本阶段删除旧 `tests/`，后续重新建立新测试。
- 按用户追加要求，reports 目录只保留 Stage 1/2/3 refactor reports。
- 为修复 DPO smoke 日志写入失败，额外修改 `sft/evaluate.py` 的共享 JSON writer，将非有限浮点数写为 `null`。

## 建议下一阶段

建议下一阶段处理：

- 在 RTX 4090 上完成 formal SFT，然后运行 formal DPO smoke。
- 重建新测试目录，只覆盖当前顶层链路：`configs/`、`scripts/prepare_data.py`、`sft/`、`dpo/`。
- 独立规划统一评测阶段，将 Base/SFT/DPO 的正式评测、结果汇总和案例分析整理为稳定入口。
