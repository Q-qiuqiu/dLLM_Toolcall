# Llama 3 AR Natural Baseline

`run_ar_experiment.py` 是独立的 AR 入口。复用现有 TaskBench/BFCL 数据适配、
Prompt 内容、输出格式和 `NaturalOnlyObserver`，不修改 LLaDA 或 OMTP 的实现。
每个实验仍只写一个 `samples.jsonl`，已有非空结果目录不能覆盖。

## 推理与公平性

- 本地权重：`/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct`。
  使用 `AutoTokenizer` / `AutoModelForCausalLM`、`local_files_only=True` 和模型自带
  chat template，不启动 HTTP 服务，不使用 GGUF、vLLM、推测模型或约束解码。
- 默认 BF16、单卡、PyTorch 原生 SDPA、标准 KV Cache、greedy、batch size 1。
  标准 KV Cache 是 AR 的正常配置，不能故意关闭以放大 dLLM 优势。
  `--attn-implementation eager --dtype float16` 可显式复现旧 `llama3_server.py`
  的 attention/精度选择，但应作为另一组配置单独报告。
- 使用原生 `model.generate()`，只添加只读 Forward 计时 hook 和永远返回 False
  的 `StoppingCriteria` 检测回调。每个新 token（包括最后一个 EOS）完成后运行
  与 dLLM 相同的 committed-token Natural 识别函数；不做 Prefix/OMTP。
- 只检测生成区，不读取 Prompt 中的候选名称。候选完整目录来自数据适配器。
  GT、最终输出及参考依赖不参与生成、在线识别或提前停止。
- 默认**不设置实验输出 token 预算**，一直生成到 checkpoint 的 EOS / Llama turn-end。
  不因 JSON 闭合或检测到完整工具集合而额外停止，也不会在 128 tokens 停下。
  HF 的默认 `max_length=20` 及 checkpoint 保存的长度限制不会生效：仅以模型本身
  上下文上限（当前 Llama 3 为 Prompt+输出共 8192 tokens）作为硬边界。
  未产生 EOS 就撞到该边界会标记 `context_limit`、`natural_generation_complete=false`，
  不能算自然结束。`--max-new-tokens` 仅保留为显式诊断选项，正式命令不传它。
- 时间零点在原生 `generate()` 调用前，包含 HF generation setup、Prompt prefill、
  后续串行生成及 Natural 检测；排除模型加载、Prompt 分词、离线分析和文件写出。
  Forward/选 token 后正确 CUDA synchronize，事件时间包含 GPU→CPU、解码和解析。
- NFE 使用真实模型 hook 计数：AR 首次完整 Prompt prefill 产生第一个 token，
  之后每个 KV decode 产生一个 token。它不应被强制为 dLLM 的 32 NFE。
  `model_forward_time_s` 是同步后的 forward 墙钟之和，不包含选 token 和 Natural。
- `--parity-samples 1` 对首条做未计入结果的 warmup，然后 Detection Off/On 对照。
  两者均保留同步计时 hook，Off 只关闭名称检测，不是完全无插桩的速度测试。
- AR 使用自然 EOS，dLLM 保持原有固定生成画布，不改变 dLLM 配置或输出行为。
  报告时需说明不同停止策略及实际输出长度，以相同目标工具集合的识别时间做比较。
  不预设 AR 必定更慢。若 AR 少生成工具、漏检或调用错误，不能算作完成相同任务更快。

## 先运行 HuggingFace 20 条 Smoke

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_ar_experiment.py \
  --benchmark taskbench \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct \
  --output-dir results/taskbench_huggingface_llama3_ar_natural_eos_smoke20 \
  --prompt-profile compact_json_dep_v4 --output-schema flat_name_v1 \
  --seed 20261009 --dtype bfloat16 \
  --attn-implementation sdpa --parity-samples 1
```

查看单轮结果：

```bash
/home/yzx/miniconda3/envs/llada/bin/python summarize_experiment.py \
  --input results/taskbench_huggingface_llama3_ar_natural_eos_smoke20/samples.jsonl
```

Smoke 使用独立的 `smoke20.jsonl`，不能与 `formal200` 的前 20 条直接配对。
不同时在同一张 GPU 上运行 LLaDA/其他工具模型与 AR 计时实验。
此前 128-budget 的 smoke 结果保留不动，不能冒充 EOS 完整生成结果。

## TaskBench 三领域正式实验

各领域 200 条，与现有 LLaDA `formal200` 完全相同的列表。命令顺序运行：

```bash
for domain in dailylifeapis huggingface multimedia; do
  CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
    run_ar_experiment.py --benchmark taskbench \
    --manifest "data/taskbench/data_${domain}/splits/formal200.jsonl" \
    --tool-catalog "data/taskbench/data_${domain}/tool_desc.json" \
    --model-path /data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct \
    --output-dir "results/taskbench_${domain}_llama3_ar_natural_eos_formal200" \
    --prompt-profile compact_json_dep_v4 --output-schema flat_name_v1 \
    --seed 20261009 --dtype bfloat16 \
    --attn-implementation sdpa --parity-samples 1 || break
done
```

只跑 HuggingFace 正式实验时，将循环改为 `for domain in huggingface; do`。

## BFCL 200 条正式实验

BFCL 与现有 dLLM 基线一样使用原始 Prompt、`standard` 输出格式：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_ar_experiment.py --benchmark bfcl \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct \
  --output-dir results/bfcl_llama3_ar_natural_eos_full200 \
  --limit 200 --start 0 --prompt-profile original --output-schema standard \
  --seed 20261009 --dtype bfloat16 \
  --attn-implementation sdpa --parity-samples 1
```

BFCL Smoke 可以改成 `--limit 20` 并使用新的输出目录。

## 与 dLLM 配对比较

HuggingFace 正式实验完成后：

```bash
/home/yzx/miniconda3/envs/llada/bin/python compare_ar_baseline.py \
  --dllm-input results/taskbench_huggingface_llada15_prefixfix_gated_formal200_20261016/samples.jsonl \
  --ar-input results/taskbench_huggingface_llama3_ar_natural_eos_formal200/samples.jsonl
```

其他领域替换双方领域名；BFCL 替换成对应 dLLM/AR 结果。
比较脚本不新增文件，同时输出：

1. `dllm_online_vs_ar_natural`：OMTP+Fallback 的首次事件 vs AR 自然识别。
2. `dllm_natural_vs_ar_natural`：dLLM 自然识别 vs AR 自然识别。

每套比较分别保留三种参考，不把它们混在一起：

- `ground_truth_complete_coverage_time`：**同一 GT 工具类型集合**的完整覆盖时间，
  报告双方成功、仅一方成功、双方失败。只有双方成功的请求形成时间配对，
  失败数量和覆盖率不能省略。允许存在额外错误触发，另看 Precision/Exact Match。
- `identical_nonempty_final_tool_set_complete_time`：两模型最终生成了完全相同的
  非空工具类型集合时，比较识别这一集合的时间；单独报告符合条件的请求数。
- `common_final_tool_identities`：比较两模型都实际生成的工具名的首次识别时间，
  每请求按工具名称去重。它是条件子集分析，不代表全测试集任务成功。

`own_final_tool_set_complete_time_descriptive_only` 是各自最终集合的描述性时间；
双方生成的工具不同或数量不同时，不能用它直接宣称完成同样任务的加速比。
另保留第 1–4 个正确工具（按检测时间排序）的统计，而不是调用位置匹配。

时间优势统一为 `T_AR - T_dLLM`，正值才表示 dLLM 更早；负值原样保留。
输出 Mean/P50/P95、样本数、胜平负、覆盖失败、模型各自正确率、生成时间和开销。
不要求两个模型 Raw output、token 数或 NFE 相同。检查的是相同请求、候选目录、
数据/Prompt 内容、种子和输出格式。历史 dLLM 无 messages hash 时明确提示。

## JSONL 信息

沿用 `tool_first_v1` 元数据及逐样本格式，保留：

- Raw output、GT（仅离线）、候选目录、样本 ID/seed。
- 每种工具首次 Natural 事件及 `natural_first_times`，含真实时间和 AR Forward/token 序号。
- TTFT、prefill forward、forward 总计、生成总时间、Natural callback 耗时。
- 实际输出 token 数、EOS/上下文硬边界停止原因、`natural_generation_complete`、
  JSON/调用结构合法性。显式诊断预算则另标 `length_budget`。
- 各自最终工具集合及 GT 完整覆盖时间，明确为离线注释。
- 模型配置/Tokenizer/权重索引哈希、权重分片尺寸、可用的 revision、dtype、attention、设备和依赖版本。
- 首条 On/Off 输出与 NFE 一致性；失败请求时间为 null，不丢弃。

默认不保存完整 logits 或每 token 文本；需要调试时可用 `--debug-output` 将
Natural 轨迹放在同一个 JSONL 内。正式计时勿加该开关。
