# Stage 2 重构报告

## 已完成

本阶段已完成新的 Stage 2 SFT 链路重构，并完成 Mini 正式 SFT 运行验证。

Stage 2 当前职责限定为：

- 直接使用 Stage 1 产出的本地 Hugging Face Dataset。
- 使用 `datasets.load_from_disk()` 读取 `data/processed/mini/sft`。
- 使用 `datasets.load_from_disk()` 读取 `data/processed/mini/evaluation`。
- 使用 `Qwen/Qwen2.5-0.5B-Instruct` 的本地 ModelScope 下载副本进行 Mini SFT。
- 使用 MPS + FP16 + LoRA + TRL `SFTTrainer` 训练。
- 保存 SFT LoRA adapter 和 tokenizer。
- 训练后重新加载 Base + SFT adapter，完成少量生成验证。
- 使用同一份 Evaluation Dataset 对 Base 与 SFT 做对比评价。

Stage 2 已删除旧 SFT 入口中的数据预处理、样本筛选、旧 JSONL/manifest 依赖、
旧数据模块 import、旧 evaluation 入口。当前 SFT 代码不再负责清洗、重排、补齐、
长度过滤或候选池处理。

本阶段没有运行或修改 DPO。

## 新增文件

- `configs/__init__.py`
- `configs/load_config.py`
- `sft/__init__.py`
- `sft/data.py`
- `sft/modeling.py`
- `sft/evaluate.py`
- `sft/train.py`
- `reports/stage_2_refactor_report.md`

## 修改文件

- `configs/qwen25_0_5b_m5_24gb_mini.yaml`
- `configs/qwen25_3b_4090.yaml`
- `pyproject.toml`
- `requirements.txt`

相关文档在 Stage 1/Stage 2 重构过程中也已同步调整：

- `README.md`
- `docs/design.md`
- `docs/data_contract.md`

## 删除文件

旧 SFT 训练入口已删除：

- `scripts/train_sft.py`
- `src/mathalign_dpo/training/train_sft.py`
- `src/mathalign_dpo/training/sft_data.py`

旧配置加载入口已删除：

- `src/mathalign_dpo/config/__init__.py`
- `src/mathalign_dpo/config/load_config.py`

旧统一评价入口已删除：

- `scripts/evaluate_math.py`
- `src/mathalign_dpo/evaluation/__init__.py`
- `src/mathalign_dpo/evaluation/answer_normalization.py`
- `src/mathalign_dpo/evaluation/eval_data.py`
- `src/mathalign_dpo/evaluation/evaluate_math.py`
- `src/mathalign_dpo/evaluation/preference_eval.py`

Stage 1 阶段已删除的旧数据处理包继续保持删除状态：

- `src/mathalign_dpo/data/`
- `scripts/build_stage2_data.py`

## 执行命令

用户执行的 Stage 2 smoke：

```bash
python -m sft.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --smoke-test \
  --train-samples 8 \
  --validation-samples 4 \
  --eval-samples 4 \
  --max-steps 1 \
  --output-dir outputs/mini/sft_smoke \
  --overwrite
```

用户执行的 Stage 2 Mini 正式训练：

```bash
python -m sft.train \
  --config configs/qwen25_0_5b_m5_24gb_mini.yaml \
  --overwrite
```

本轮报告核验命令：

```bash
find outputs/mini/sft -maxdepth 2 -type f | sort
find outputs/mini/sft -maxdepth 2 -type d | sort
python -m json.tool outputs/mini/sft/run_config.json
python -m json.tool outputs/mini/sft/train_metrics.json
python -m json.tool outputs/mini/sft/eval_metrics.json
python -m json.tool outputs/mini/sft/base_sft_summary.json
wc -l outputs/mini/sft/loss_history.jsonl \
  outputs/mini/sft/base_sft_predictions.jsonl \
  outputs/mini/sft/correct_cases.jsonl \
  outputs/mini/sft/error_cases.jsonl \
  outputs/mini/sft/adapter_reload_samples.jsonl
du -sh outputs/mini/sft outputs/mini/sft/adapter outputs/mini/sft/tokenizer model/Qwen2.5-0.5B-Instruct
python -m py_compile sft/modeling.py sft/evaluate.py sft/train.py
conda run -n mathalign-dpo python -m sft.train --help
```

## 测试结果

基础静态检查：

- `python -m py_compile sft/modeling.py sft/evaluate.py sft/train.py`：通过。
- `conda run -n mathalign-dpo python -m sft.train --help`：通过。

Smoke 运行结果：

- 状态：完成。
- 训练样本：8。
- 验证样本：4。
- 评测样本：4。
- 最大训练步数：1。
- 输出目录：`outputs/mini/sft_smoke`。
- 训练、adapter 保存、adapter 重新加载、Base/SFT 对比评价均完成。

Mini 正式 SFT 运行结果：

- 状态：`completed`
- `smoke_test`: `false`
- 配置文件：`configs/qwen25_0_5b_m5_24gb_mini.yaml`
- 输出目录：`outputs/mini/sft`
- 创建时间：`2026-07-20T06:24:39Z`
- Git commit：`d06cdb1`
- 工作区状态：dirty，包含本阶段重构文件变更，尚未提交。

运行时：

- 后端：MPS。
- 设备：`mps`。
- `mps_is_built`: `true`
- `mps_is_available`: `true`
- `mps_fallback_env`: 空字符串。
- 模型目录：`model/Qwen2.5-0.5B-Instruct`
- ModelScope 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 模型 revision：`7ae557604adf67be50417f59c2c2f167def9a775`
- 精度：`float16`

训练配置：

- SFT train：256 条。
- SFT validation：32 条。
- Evaluation：32 条。
- LoRA rank：8。
- LoRA alpha：16。
- LoRA dropout：0.05。
- LoRA target modules：`q_proj, k_proj, v_proj, o_proj`。
- `max_steps`: 30。
- `per_device_train_batch_size`: 1。
- `gradient_accumulation_steps`: 4。

训练指标：

| 指标 | 数值 |
|---|---:|
| train_loss | 0.5177607357501983 |
| train_runtime | 52.2766 秒 |
| train_samples_per_second | 2.295 |
| train_steps_per_second | 0.574 |
| total_flos | 86340302839296.0 |

验证指标：

| 指标 | 数值 |
|---|---:|
| eval_loss | 0.5158189535140991 |
| eval_runtime | 3.1571 秒 |
| eval_samples_per_second | 10.136 |
| eval_steps_per_second | 10.136 |

loss 轨迹检查：

- `loss_history.jsonl` 共 34 行。
- 训练 loss 记录 30 条。
- 最小训练 loss：0.3116。
- 最大训练 loss：0.892。
- 最后一条训练 loss：0.4554。
- `grad_norm` 范围：0.4524818956851959 到 1.0026227235794067。
- 未发现 NaN/Inf loss。

Base 与 SFT 评价结果：

| 模型 | 样本数 | Exact Match | 正确数 | 答案提取成功率 | 平均输出 token | 平均生成耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 32 | 0.0625 | 2/32 | 1.0 | 251.78125 | 4.494285125 秒 |
| SFT | 32 | 0.125 | 4/32 | 1.0 | 242.375 | 5.14746678125 秒 |

逐样本评价文件：

- `base_sft_predictions.jsonl`：64 行，包含 Base 和 SFT 各 32 条预测。
- `correct_cases.jsonl`：5 行。
- `error_cases.jsonl`：5 行。
- Base 正确样本 ID：
  `numina_train_00628871`, `numina_train_00612904`
- SFT 正确样本 ID：
  `numina_train_00098078`, `numina_train_00789466`,
  `numina_train_00717283`, `numina_train_00612904`

adapter 重新加载验证：

- `adapter_reload_samples.jsonl`：3 行。
- 保存后重新加载 Base + SFT adapter 并完成生成。

输出产物：

```text
outputs/mini/sft/
├── adapter/
├── tokenizer/
├── train_metrics.json
├── eval_metrics.json
├── loss_history.jsonl
├── adapter_reload_samples.jsonl
├── base_sft_predictions.jsonl
├── base_sft_summary.json
├── correct_cases.jsonl
├── error_cases.jsonl
├── run_config.json
├── trainer_state.json
├── checkpoint-15/
└── checkpoint-30/
```

产物大小：

- `outputs/mini/sft`：75 MB。
- `outputs/mini/sft/adapter`：4.2 MB。
- `outputs/mini/sft/tokenizer`：15 MB。
- `model/Qwen2.5-0.5B-Instruct`：956 MB。

## 已知限制

- 本阶段只完成 Mini SFT，不处理 DPO。
- 当前 Exact Match 来自 32 条 Mini evaluation，只能说明链路可运行和小样本上有观测改善，
  不能作为正式性能结论。
- Base/SFT 生成平均 token 数接近 `max_new_tokens=256`，说明部分样本可能没有自然提前停止；
  后续统一评测阶段需要继续观察生成长度、停止条件和答案抽取质量。
- RTX 4090 formal SFT 尚未在本阶段实测。
- 旧 DPO 代码仍未迁移，且会受旧配置/旧 SFT/旧数据模块删除影响，后续 Stage 3 需要处理。
- 当前未记录峰值 MPS 内存、macOS 版本、PyTorch/Transformers/TRL/PEFT 版本等完整实验环境信息。

## 计划偏离

- `plans/stage_2_sft.md` 当前未在仓库中找到。本报告按已实现的新 Stage 2 SFT 行为与正式运行结果记录。
- 为满足用户“旧 SFT 相关代码全部删除，新的 SFT 代码放在仓库根目录 `sft/`”的要求，
  Stage 2 采用新的 `sft/` 顶层包，而不是继续使用 `src/mathalign_dpo/training/`。
- 为满足用户“读取 YAML 配置的代码放在 `configs/`，原有的也删除”的要求，
  新配置加载代码位于 `configs/load_config.py`，旧 `src/mathalign_dpo/config/` 已删除。
- 本阶段已删除旧 evaluation 相关代码，并在 `sft/evaluate.py` 内实现 Base/SFT 对比评价；
  DPO 评价仍不在本阶段处理。
- 用户执行 smoke 后发现 reload 阶段曾被 `PYTORCH_ENABLE_MPS_FALLBACK=1` 硬拦截。
  已调整为记录该环境变量但不因此中断已可用的 MPS 运行。
- 用户确认后，为减少日志噪声，已将模型加载参数从 deprecated `torch_dtype` 改为 `dtype`，
  并在推理加载后清理 sampling generation defaults。

## 建议下一阶段

建议下一阶段进入 DPO 重构：

- 删除旧 DPO 训练入口对旧数据模块、旧配置模块和旧 SFT 代码的依赖。
- 新建独立 DPO 代码目录，直接读取 `data/processed/<mode>/dpo`。
- DPO 阶段只加载 Stage 1 已构造好的 `prompt/chosen/rejected`，不再做样本筛选、候选池补齐或长度过滤。
- DPO smoke 先在 Mini MPS 上验证 Base + SFT adapter -> DPO adapter 链路。
- DPO 正式评价仍留到后续统一评测阶段。
