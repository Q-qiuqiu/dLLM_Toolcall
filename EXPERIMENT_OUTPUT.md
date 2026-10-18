# 当前实验入口：单 JSONL 与首次工具识别

`run_taskbench_experiment.py` 和 `run_omtp_experiment.py` 默认都只输出
`--output-dir/samples.jsonl`。第一行是 `record_type=metadata`，之后每条请求一行。
不自动生成重复样本文件、事件文件、CSV、图或 JSON Repair/AST 报告。
历史结果不会删除或覆盖；指定已有非空目录会直接拒绝运行。

## Prefix 修复

- 仅从当前 name 字段的值起点检测，不在邻近位置寻找名称碎片。
- 用完整的已确定文本前缀，对比当前请求完整候选目录中的全部名称。
  `Audio-to-Audio` 与 `Audio-to-Text` 都在目录中时，`Audio-to` 不触发。
- 遇到已确定的错误后缀，拒绝候选，不再只保留前面的匹配部分。
  MASK 后已确定的 token 也必须兼容候选；已确定的名称结束边界不能冲突。
- 完整的 `Object Classification` 不会通过截取 `Object` 触发 `Object Detection`。
- 同名工具由 OMTP、Prefix、独立 Natural 检测中最早的在线事件触发；
  不同轨迹重复预测该工具不重复记录。错误触发不事后删除。
- Natural 首次完成时间独立记录，即使该工具此前已经被预测，也不会漏记基线。
- 没有真实工具启动或状态检查；此处只是等价的首次事件去重记录。

## 模型生成非法名称的归因

若当时的 committed 前缀确实只对应一个合法工具、没有后缀冲突，随后模型却在
**同一字段**生成目录外名称，则 `offline_prefix_audit.status=model_invalid_name`。
按当前实验定义，它不计入 Prefix 检测错误。但仍保留实际触发，且仍属于相对
最终合法工具集合的额外触发，不能从真实启动成本或原始 Precision 中删除。

这与已经看见冲突后缀还触发不同：后者属于检测实现问题，此次直接拒绝。
无法确认同一字段、字段不完整或缺少在线验证证据，标记 `unresolved`，不自动豁免。
所有归因只在生成结束后进行，不修改预测名称、触发时刻或在线决策。

## 运行 20 条 HuggingFace Smoke

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=1 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_huggingface_prefixfix_singlefile_smoke20 \
  --cache-mode none --response-mode native \
  --output-schema flat_name_v1 --prompt-profile compact_json_dep_v4 \
  --generation-profile fixed128 --parity-samples 1 \
  --omtp-strict-anchor --omtp-min-top1-score -5 --omtp-min-margin 0.1 \
  --omtp-append-natural
```

正式运行把 `smoke20.jsonl` 改成 `formal200.jsonl`，使用新的输出目录。
Multimedia 同时替换 manifest 和 tool catalog 的领域目录。GPU 编号按空闲情况选择。
其余 Prompt、模型生成、评分门控、Stable 参数保持不变，不增加模型 Forward。

## BFCL 门控入口与效果对照

`run_omtp_experiment.py` 已暴露与 TaskBench 相同的 `--omtp-strict-anchor`、
`--omtp-min-top1-score`、`--omtp-min-margin` 等可靠性开关。
旧的 `--min-top1-score` / `--min-margin` 仍兼容；同时传入不同值会明确报错。
未加门控开关时，保持无门控的默认配置，不会静默更换 Prompt 或模型。

`run_bfcl_gate_ablation.py` 用于准确率配对诊断：四组观察器读取同一次 Forward
的 logits 和 committed tokens，分别为无门控、仅可信 Anchor、仅 -5/0.1
分数门控、以及两者同时开启。每组都保留相同的 Prefix 修复和自然兜底。
它不增加模型 Forward；首条可另做已有的 Observer Off/On Parity 检查。
结果仍只有一个 `samples.jsonl`，每条样本的 `ablation_variants` 保留四组结果。
因为四组在同一回调里顺序执行，**不能把其中的墙钟时间作为四组的公平速度对照**，
只比较识别准确率、错误触发、真实提前识别比例和提前 Forward 数。

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_bfcl_gate_ablation.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/bfcl_gate_ablation_llada15_smoke20_new \
  --limit 20 --start 0 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --seed 20261009 --stable-n 2 \
  --cache-mode none --response-mode native --output-schema standard \
  --parity-samples 1
```

显存不足时可以显式使用 `--device-map auto --max-memory-json '{"0":"3GiB","cpu":"48GiB"}'`
进行分层加载/CPU 权重卸载；不量化、不改权重，但布局与额外传输必须在报告中说明。
只有显式选择该选项才改变模型放置，常规单卡路径保持不变。

```bash
/home/yzx/miniconda3/envs/llada/bin/python summarize_experiment.py \
  --input results/taskbench_huggingface_prefixfix_singlefile_smoke20/samples.jsonl
```

汇总只打印到终端，不新建文件。已移除只支持旧多文件格式的独立分析和可靠性
逐轮开发脚本。当前结果使用 `summarize_experiment.py`；正式 AST/参数指标使用
`evaluate_official_results.py`，AR 对照使用 `compare_ar_baseline.py`。

## 每条请求保存的信息

- raw output；若 EOS 前的文本不同，同时保留 `output_before_eos`。
- 每种工具的 `first_tool_events`：工具名、来源、真实可用时间、Forward。
  Prefix 事件额外保存前缀、在线验证证据、离线归因；不保存重复轨迹事件。
- 独立 `natural_first_times`，便于按工具首次出现计算配对提前量。
- 生成墙钟耗时、Observer 耗时、NFE、Prompt/输出 token 数、EOS、剩余 MASK 数。
- 样本 ID、实际随机种子、候选名称、GT 调用与依赖边（只用于离线分析）。
- 已执行的 On/Off Parity 结果。失败请求也保留一行，时间为 null 而不是 0。

metadata 保存数据与源码哈希、模型版本/运行配置、硬件和计时语义。
时间零点在生成画布构建前：包含模型的 Prompt 计算，不包含 Prompt 分词、模型加载、
生成后的归因和文件写出。默认 OMTP 模式的自然时间来自同一个 observer-on 请求，
不能冒充独立 observer-off 请求的自然延迟。独立基线使用下面的 `--observer-mode natural`。
没有 EOS 仅表示可能用尽预算，不能直接认定截断。

只有 `--debug-output` 才在同一 JSONL 的 `debug` 字段中保留完整 Forward 轨迹及
历史评测细节；仍不额外写文件。默认不累计完整历史，只保留当前步给评分使用。

## 独立 Natural-only 基线

新增 `natural_observer.py`，两个现有入口都可指定 `--observer-mode natural`。
默认仍是 `omtp`，不改变已有方法及默认运行行为。Natural-only：

- 与 OMTP 共用 `observer.detect_committed_tools`，只处理生成区的 committed tokens。
  保留 MASK 间隔，检测完整候选名称，不把分离片段拼接成名称。
- 不实例化 OMTP 观察器，不读取 logits，不执行 Argmax、Anchor、Track、Region、
  Stable 或 Prefix。候选名称仍只来自完整目录，不使用 GT/最终输出作在线判断。
- 复用原生成函数、Prompt、随机数种子、Token Transfer 及同步回调；不增加 Forward。
- 时间包含当前 token 的 GPU→CPU 拷贝、解码和解析耗时，不使用 Forward 完成时间
  冒充检测完成时间。同一工具只记录第一次 Natural 事件。
- 仍有少量 committed-token 检测和原同步/快照回调开销，**不是 Observer Off**。
  `observer_time_s` 是回调处理时间，不包含生成器内的同步/快照开销；后者包含在
  `generation_time_s` 中。`--parity-samples 1` 对首条另做原有的 Off/On 检查。
- Natural 模式若传入 OMTP 门控或 profile 开关会报错，请从基线命令中移除这些开关。
  不建议加 `--debug-output` 进行正式计时。

### HuggingFace 200 条：与当前完整 OMTP 运行配对

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --observer-mode natural \
  --manifest data/taskbench/data_huggingface/splits/formal200.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_huggingface_llada15_natural_only_formal200 \
  --cache-mode none --response-mode native --output-schema flat_name_v1 \
  --prompt-profile compact_json_dep_v4 --generation-profile fixed128 \
  --dtype bfloat16 --parity-samples 1
```

随机种子仍为固定的 `20261009 + 样本序号`。先测 20 条可同时将
`formal200.jsonl` 换为 `smoke20.jsonl`，输出目录换为一个新的 smoke 目录。
但 smoke 结果只能与同样 smoke 样本及种子的 OMTP 运行配对。

跑完后，比较当前已完成的 HuggingFace OMTP 正式实验：

```bash
/home/yzx/miniconda3/envs/llada/bin/python summarize_experiment.py \
  --input results/taskbench_huggingface_llada15_prefixfix_gated_formal200_20261016/samples.jsonl \
  --natural-baseline results/taskbench_huggingface_llada15_natural_only_formal200/samples.jsonl
```

报告只打印终端，不新增文件。重点看
`independent_natural_baseline.paired_comparison.net_paired_lead_s` 的
`mean/p50/p95/count`，定义为：

```text
净提前量 = 独立 Natural-only 运行的该工具首次检测时间
         - OMTP-on 运行的该工具首次事件时间（Latent/Prefix/Natural 谁先用谁）
```

这里**不再减一次 Observer 总耗时**，因为 OMTP 事件时间已经包含真实观察开销。
负值也保留，代表在该工具上反而晚于 Natural-only。按“请求 × 去重工具类型”配对，
重复调用不重复计收益，不只看第一个工具；同时给出每请求平均提前量、完整集合提前量、
双方检测/单方检测/双方缺失、胜/平/负及生成总时间差。
两套准确率（相对最终 Raw Text、相对 GT）仍分别报告；错误预测不被删除。

配对检查模型、数据哈希、Prompt 配置、生成参数、种子和候选目录。输出不同、NFE
不同、运行失败的请求单独列出，不混入时间收益。新运行另存实际 Prompt 哈希；
历史 OMTP 没有该字段时明确提示，仅能检查已有配置与源码/目录哈希。
不同轮次的共享 GPU 负载可能影响墙钟差，正式论文计时建议相同 GPU、稳定负载下
交替重复运行；比值只代表识别时间比，**不是实际工具执行端到端加速比**。

### TaskBench 三领域顺序运行

以下命令只运行基线，各领域各写一个独立 JSONL。不与 OMTP 同时抢同一张 GPU：

```bash
for domain in dailylifeapis huggingface multimedia; do
  CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
    run_taskbench_experiment.py --observer-mode natural \
    --manifest "data/taskbench/data_${domain}/splits/formal200.jsonl" \
    --tool-catalog "data/taskbench/data_${domain}/tool_desc.json" \
    --model-path /data/labshare/Param/llada1.5 \
    --fastdllm-root /data/home/yzx/Fast-dLLM \
    --output-dir "results/taskbench_${domain}_llada15_natural_only_formal200" \
    --cache-mode none --response-mode native --output-schema flat_name_v1 \
    --prompt-profile compact_json_dep_v4 --generation-profile fixed128 \
    --dtype bfloat16 --parity-samples 1 || break
done
```

上面的 HuggingFace 单独命令和循环是二选一，重复使用同一结果目录会被拒绝。

### BFCL 200 条基线

BFCL 继续使用其原始 Prompt 和 `standard` 输出，不套用 TaskBench Prompt。

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_omtp_experiment.py --observer-mode natural \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/bfcl_llada15_natural_only_full200 \
  --limit 200 --start 0 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --seed 20261009 --cache-mode none --response-mode native \
  --output-schema standard --dtype bfloat16 --parity-samples 1
```

同样使用 `summarize_experiment.py --input <OMTP samples.jsonl>
--natural-baseline <Natural-only samples.jsonl>` 配对。

## 已完成的验证

回放旧 HuggingFace formal200 的 806 次 Prefix 事件：拒绝旧事件 16 次，其中
14 次与最终合法工具不符；保留 790 次，其中 789 次对应最终同位置名称，1 次
是合法唯一前缀后模型生成 `Audio-to-Audio Enhancement`，单独归因为模型非法名称。
这是旧轨迹离线回放，不是修改后模型实跑，也不能据此宣称新的端到端时间指标。
