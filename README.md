# Fast-dLLM v1 × BFCL 多工具提前预测预实验

本目录实现 LLaDA-8B-Instruct 在 BFCL 非 Live `parallel_multiple` 上的无缓存全局去噪实验。所有新增/修改代码均在本目录；原始 `Fast-dLLM` 工作树（包括其中已有的未提交改动）未被修改。

## 已核对的本机状态

- Fast-dLLM：`/data/home/yzx/Fast-dLLM`，审计时 commit 为 `aadcf060e12b6413b494bf9aebb172cfe3cf911c`。
- 权重：`/data/labshare/Param/llada`，配置为 32 层、8B、BF16、`mask_token_id=126336`、上下文 4096。
- 模型实现：运行时从 `Fast-dLLM/v1/llada/model` 导入；无缓存生成/transfer 代码复制在 `fastdllm_generate.py` 中。
- 实验节点有 8 张 RTX 3090；OMTP smoke 使用 GPU 0 实际完成。若受限沙箱内 `nvidia-smi` 报驱动不可访问，需要在具备 GPU device 权限的终端运行命令。
- 本机 `llada` conda 环境审计时为 Python 3.10、Torch 2.11.0+cu130、Transformers 4.38.2；Fast-dLLM v1 的仓库要求是 Transformers 4.49.0，正式复现前建议对齐依赖。

## 数据与 Prompt

BFCL 官方文件虽以 `.json` 结尾，实际是 JSONL。当前 V4 文件中的非 Live `parallel_multiple` 是 BFCL 初始版本延续下来的 200 条 AST 类别。输入结构是：

```text
id
question: [[{"role": "user", "content": "..."}]]
function: [{"name": ..., "description": ..., "parameters": ...}, ...]
```

答案文件按 `id` 对齐，结构是：

```text
ground_truth: [{"function_name": {"argument": [allowed values ...]}}, ...]
```

`prompting.py` 只读取 question 和候选 function definitions。ground truth 只在生成结束后进入离线评测，绝不会进入 prompt、observer 或在线 stable trigger。

下载脚本默认固定 Gorilla 的 BFCL V4 release commit，并记录每个文件 SHA-256：

```bash
cd /data/home/yzx/dLLM_Toolcall
python download_bfcl.py --output-dir data/bfcl
```

若上游移除了该历史 revision，可显式选择另一个 commit；不要用浮动 `main` 做最终可复现实验：

```bash
python download_bfcl.py --revision <GORILLA_COMMIT> --output-dir data/bfcl_<commit>
```

## 手动运行命令

先在用于 Fast-dLLM v1 的环境中对齐依赖。不要在正在服务其他任务的环境中直接升级：

```bash
conda activate llada
cd /data/home/yzx/dLLM_Toolcall
python -m pip install -r requirements.txt
```

20 条 smoke test（全局去噪、32 次 forward、observer 稳定阈值 N=2）。`--parity-samples 1` 会对第 1 条先做一次不计时的 observer-off warmup，再测 observer-off/on 的 token 级一致性和墙钟开销，避免把 CUDA 冷启动误算成 observer 开销：

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/smoke20_global128_s32_n2 \
  --limit 20 \
  --gen-length 128 \
  --block-length 128 \
  --steps 32 \
  --temperature 0 \
  --stable-n 2 \
  --seed 20261009 \
  --parity-samples 1
```

注意：question 和 answer 在上游不同子目录中同名。为避免 basename 冲突，下载脚本实际落盘名称分别是：

```text
data/bfcl/BFCL_v4_parallel_multiple.json
data/bfcl/BFCL_v4_parallel_multiple_answers.json
```

完整 200 条：

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --output-dir results/full200_global128_s32_n2 \
  --limit 200 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --stable-n 2 --seed 20261009
```

如果 128 token 组有大量 `invalid_json`，按原要求新增 256 token 组，不要截断或修复原输出：

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --output-dir results/full200_global256_s32_n2 \
  --limit 200 --gen-length 256 --block-length 256 --steps 32 \
  --temperature 0 --stable-n 2 --seed 20261009
```

分 Block 对照组仍使用无缓存 `generate()`，只是 transfer mask 按 32-token block 推进：

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --output-dir results/full200_block32_s32_n2 \
  --limit 200 --gen-length 128 --block-length 32 --steps 32 \
  --temperature 0 --stable-n 2 --seed 20261009
```

N=1 或 N=3 只需修改 `--stable-n`。完整 logits 默认不保存；仅对极少量调试样本使用 `--save-full-logits`，因为每步 `[gen_length, vocab_size]` 张量会非常大。

## Observer 与计时语义

`observation_callback=None` 时，生成函数不增加同步、计时、解码或 tensor copy，采样和 token transfer 与复制来源保持一致。启用时每个 forward 保存：

- forward/block/step 编号；
- forward CUDA 同步完成时间和 transfer CUDA 同步完成时间；
- transfer 前 generation token IDs、MASK 相对位置；
- 每个 MASK 的 logits top-1 token 和 logit（可选 top-2/margin）；
- transfer 后 generation token IDs；
- 保留 MASK gap 的 committed text；
- argmax 填满当前所有 generation MASK 后的 temporary text；
- committed/predicted 调用列表、工具类型集合、连续稳定次数与在线 trigger。

三个时间概念不能混用：

- `forward_complete_time_s`：从 generation 开始到当前 `model(x).logits` 真正完成；CUDA 已同步，不含当前 observer 处理。
- `commit_complete_time_s`：同一步原始 token transfer 完成时间，不含当前 observer 处理。
- `prediction_available_time_s`：加入当前 observer top-k copy、decode 和 parser 后的实际可用时间。

后续 forward 的 wall-clock timeline 会自然包含此前 observer 开销；`model_forward_time_s` 是同步 forward 区间之和，`observer_processing_time_s` 是 observer CPU/GPU 处理之和，`generation_time_s` 是实际开启 observer 的端到端时间。

Committed 文本不会简单删除 MASK 再拼接，因为那会把原本不连续的 token 拼成伪造的 `"name"` 字段。实现会在连续 committed runs 之间插入不可匹配的 gap marker。工具提取只接受完整 JSON 中的 `name`，或局部完整的 `"name": "candidate"` 字段；arguments 中出现候选名不会被识别。

## 在线指标与离线 oracle 指标

- Natural：用 transfer 后的 committed tokens 检测。
- Argmax：保留 transfer 前 committed token，用本次 logits top-1 临时填满仍为 MASK 的生成位置。
- Stable：预测工具类型集合连续 N 次相同后触发；空集合不触发。集合改变后可再次触发，从而支持追加新工具。

`predict_all_detect_time_s` 是第一次在线 N-stable trigger 时间，决策本身不知道 ground truth。`predict_trigger_set_exact/precision/recall` 是实验结束后的离线标注。`oracle_correct_detect_time_s` 是未经 stable 限制的 argmax 集合首次恰好等于真值的时间；`offline_stable_correct_trigger_time_s` 是 stable 事件首次恰好正确的时间。二者都明确是 oracle，不能作为线上策略宣称。

`prediction_lead_time_s` 和 `available_warmup_window_s` 使用 forward-complete 的算法时间；`actual_prediction_lead_time_s`、`predict_all_available_time_s` 与 `actual_available_warmup_window_s` 包含当前 observer 处理。它们只在第一次在线 trigger 的集合经离线确认完全正确时填写，否则为 null。启动模拟使用实际 prediction-available 时间。失败样本不会用 0 代替。

## 输出

每个 run 目录包含：

- `samples.jsonl`：逐样本完整结果和 `steps`；
- `steps.jsonl`：展开后的逐 forward 记录；
- `summary.csv`：总体均值、有效样本数和启动模拟；
- `failures.jsonl`：JSON、集合、AST、预测、observer parity 等失败类型；
- `metadata.json`：参数、命令、数据 hash、Fast-dLLM commit、软件和硬件；
- `figures/`：F1-forward、检测时间分布、oracle 累计曲线、启动收益图。

`raw_output` 保存完整 generation region 的 `skip_special_tokens=True` 解码，`raw_output_with_special_tokens` 保留特殊 token；严格 JSON 评测使用正常生成语义下首个 EOS/PAD 之前的 `evaluated_output_before_eos`。三者都保留，评测不会用修复后的文本替代原输出。

最终格式指标分为三层，避免混淆：`final_json_valid` 只表示 JSON 语法可解析，`tool_call_format_valid` 要求顶层数组且每项恰含 `name/arguments`，`candidate_names_valid` 再要求所有 name 来自候选列表。语法正确但调用未知工具的条目仍保留在 `generated_tool_calls` 中，因而会正确计入名称误报和调用次数。

如评测适配器更新，可只重算既有输出而不重复模型推理，也不会覆盖原 run：

```bash
python reanalyze_run.py \
  --source-run results/smoke20_global128_s32_n2 \
  --output-dir results/smoke20_global128_s32_n2_reanalyzed_v2 \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json
```

本地 AST 指标名为 `bfcl_ast_compatible_match`：它实现 BFCL parallel 的无序、重复调用匹配、schema 类型、可选参数 `""` 和字符串归一化，但没有伪装成已调用外部 `bfcl_eval` 包。最终官方榜单分数仍应把原始输出适配到与数据 revision 相同的 BFCL 官方 evaluator 复核。

启动模拟假设每个工具独立并行启动，不考虑设备争用或 Docker 调度开销。错误预测的类型计入 `wrong_preload_cost_s`；漏掉的必要类型回退到它的 natural 首次发现时间再启动。

## OMTP：Commit 策略迁移实验

`run_omtp_experiment.py` 在同一次无缓存生成的每个已有 Forward 上同时计算六种方法，不新增模型 Forward：

- A `natural_decoding`：committed token 中完整的 tool name；
- B `intermediate_argmax`：第一次非空的临时 Argmax 工具集合；
- C `whole_set_argmax_stable_n2`：原有整集合 Stable N=2；
- D `per_tool_argmax_stable_n2`：每个候选工具独立做 Argmax Stable N=2；
- E `omtp_latent_region_n2`：`name` anchor 轨迹、半径 4 的 candidate-aware region top-2-mean score、每轨迹 Stable N=2；
- F `omtp_with_fallback`：每个 occurrence 轨迹按真实可用时间采用首次 `latent_region`、唯一 `prefix` 或 `natural` 事件。

E/F 只使用当前请求的 function definitions 建立候选表。Ground Truth 和最终输出仅在生成结束后标注历史事件，不参与 anchor、评分、稳定判定或触发。Anchor 只扫描 generation region，并以 JSON 对象深度排除 `arguments` 嵌套对象中的 `name` 字段。所有事件原样保留；错误事件不会被最终答案回删。

20 条 smoke（其中第一条执行 warm observer-off、计时 observer-off、observer-on parity）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python run_omtp_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/omtp_smoke20_global128_s32_n2 \
  --limit 20 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --stable-n 2 --region-radius 4 --region-top-k 2 \
  --min-seen 2 --min-support 0.5 --seed 20261009 --parity-samples 1
```

Smoke 通过后运行完整 200 条（使用新目录，程序拒绝覆盖已有结果）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python run_omtp_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/omtp_full200_global128_s32_n2 \
  --limit 200 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --stable-n 2 --region-radius 4 --region-top-k 2 \
  --min-seen 2 --min-support 0.5 --seed 20261009 --parity-samples 1
```

新增输出为 `omtp_events.jsonl`、`omtp_trajectory.jsonl`、`omtp_samples.jsonl`、`omtp_summary.csv`、`omtp_comparison.csv` 及五张 `figures/` 对比图。`online_first_trigger_time_s` 是因果在线触发时间；`oracle_*` 字段是生成结束后的离线正确性标注。所有正式触发时间均在产生该事件的 observer 检测/评分完成后采集，不使用 Forward 完成时刻代替。

### 已完成的 20 条 smoke

最终代码对应结果目录是 `results/omtp_smoke20_global128_s32_n2_v2`。20/20 请求均完成且每条都是 32 NFE；parity 样本的 observer on/off token 输出和 NFE 完全一致。最终生成 JSON 正确率为 85%，Ground Truth 工具集合 EM 为 75%，本地 BFCL-compatible AST 为 60%。

相对 Ground Truth，以“运行期间所有不可撤回在线事件的工具类型并集”计分：

| 方法 | Precision | Recall | F1 | Set EM | Oracle 精确集合时间 mean / P50 / P95 | 有效数 |
|---|---:|---:|---:|---:|---:|---:|
| Whole-set Argmax Stable N=2 | 0.9583 | 1.0000 | 0.9733 | 0.90 | 3.138 / 3.561 / 4.969 s | 20/20 |
| Per-tool Argmax Stable N=2 | 0.9583 | 1.0000 | 0.9733 | 0.90 | 3.138 / 3.561 / 4.969 s | 20/20 |
| OMTP latent-region N=2 | 0.9500 | 0.9750 | 0.9583 | 0.90 | 2.811 / 3.013 / 5.134 s | 18/20 |
| OMTP + fallback | 0.9500 | 0.9750 | 0.9583 | 0.90 | 2.721 / 3.013 / 4.473 s | 18/20 |
| Natural | 0.9333 | 0.9250 | 0.9233 | 0.85 | 5.041 / 5.798 / 6.665 s | 17/20 |

OMTP 与 Whole-set Stable 同时达到过 Ground Truth 精确集合的 18 条配对样本上，OMTP 早 0.399 s（P50 0.071 s，P95 3.485 s）。但 OMTP 的 F1 低 0.015，且精确集合历史覆盖为 18/20、低于 Stable 的 20/20。因此本次小样本结论是 OMTP 有平均时延收益，但尚不能称为准确率和覆盖率上的全面改进。完整数字、两套 reference（最终生成 / Ground Truth）及有效分母以 `omtp_comparison.csv` 为准。

## OMTP 性能优化与可靠性消融

`--observer-mode` 保留三个互不覆盖的实现：

- `original`：原始 OMTP，作为算法和性能基线；
- `optimized`：缓存候选名称 tokenization、融合 top-1/row-max、一次式 JSON 深度扫描、局部 log-partition 复用，并按名称 token 长度批量计算候选/位置分数；
- `selective`：先在 anchor 精确位置执行廉价评分，仅在 top-1 或 margin 不确定时执行完整 region scoring。

可靠性开关为 `--min-top1-score`、`--min-margin`、`--stable-n {1,2,3}` 和 `--disable-prefix-fallback`。默认阈值均关闭，因此不会改变原始策略。每个 latent 轨迹额外记录 `name_uncommitted_ratio`。`--profile-observer` 会在模块边界同步 GPU，仅用于诊断；正式墙钟对照不得开启它。

同一批前 5 条样本的同步 Profile 显示：原始 anchor detection 为 41.433 ms/Forward，优化后为 2.204 ms/Forward；observer 总处理时间从 1.472 s/样本降至 0.225 s/样本。原始实现中 anchor detection 占排除重叠同步计时后的 94.3%，瓶颈来自每个 Forward 重复扫描生成区/词表和重复 token decode。优化版只搬运 token id 与小型分数，不把完整 logits 搬到 CPU；5 条共记录 GPU→CPU 383,944 bytes，CPU→GPU 127,957 bytes。

数据严格分为三段：0–19 只用于 Smoke/实现验证，20–39 只用于 Selective 路由和 confidence gate 校准，40–59 是未参与阈值选择的保留测试。校准集选择 `selective_refine_margin=0.5`、`selective_refine_top1=-4.0`；E 另外使用 `min_margin=0.5` 并关闭 prefix fallback。绝对 top-1 gate 在校准集上损害 recall，因此保持关闭。

保留集 40–59（Ground Truth reference）结果：

| 方法 | P / R / F1 | Set EM | 精确集合历史覆盖 | 精确时间 mean / P50 / P95 | Observer / 样本 | 生成墙钟 / 样本 |
|---|---:|---:|---:|---:|---:|---:|
| A Argmax Stable N=2 | .850 / .883 / .857 | .75 | 16/20 | 2.477 / 2.526 / 3.466 s | .053 s | 5.456 s |
| B 原始 OMTP | .933 / .983 / .950 | .85 | 18/20 | 3.366 / 3.180 / 6.264 s | 1.617 s | 6.919 s |
| C 优化 OMTP | .933 / .983 / .950 | .85 | 18/20 | 2.588 / 2.437 / 5.172 s | .216 s | 5.587 s |
| D Selective OMTP | .933 / .983 / .950 | .85 | 18/20 | 2.572 / 2.441 / 5.191 s | .264 s | 5.629 s |
| E Selective + gates | .925 / .867 / .868 | .65 | 14/20 | 2.462 / 2.457 / 4.512 s | .271 s | 5.645 s |

C 相对 B 的 observer 时间下降 86.6%，生成墙钟下降 19.3%；相对 A 的生成墙钟开销仅 2.4%。B/C 的模型输出和 OMTP 事件签名均为 20/20 相同，所有 A–E 模型输出均相同、每条均为 32 NFE。C 相对 A 在 16 条共同成功样本上平均提前 0.134 s、P50 0.040 s、P95 1.898 s，胜/负为 8/8，另有 2 条只有 C 成功。D 没有降低成本：校准后的路由仍有 821 次 refine、345 次 cheap，且 observer 比 C 慢 21.9%，因此本数据上 Selective Region Scoring 不比直接优化后的 full-region 实现更有价值。E 的门限没有在保留集上泛化，漏检明显增加，不能作为默认策略。

复现 C（将 `--observer-mode` 换为 `original` 或 `selective` 可运行 B/D；E 再增加 `--min-margin 0.5 --disable-prefix-fallback`）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python run_omtp_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/holdout40_59_C_optimized_omtp_rerun \
  --start 40 --limit 20 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --stable-n 2 --seed 20261009 \
  --observer-mode optimized --parity-samples 1
```

配对总表和校验位于 `results/ablation_ABCDE_holdout40_59/`；各方法的五张图和逐事件日志位于对应的 `results/holdout40_59_*` 目录。

## 无 GPU 验证

```bash
cd /data/home/yzx/dLLM_Toolcall
python -m pytest -q
```

当前实现的无 GPU 测试覆盖 BFCL 数据适配、ground-truth prompt 隔离、严格/局部 parser、重复调用与 AST、observer 开关输出一致性、block 元数据、N-stable 和启动模拟。
