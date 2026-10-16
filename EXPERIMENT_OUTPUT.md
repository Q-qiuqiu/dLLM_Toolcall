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

汇总只打印到终端，不新建文件。旧版 `report_omtp_reliability_ablation.py`、
TaskBench 多文件报告器仍用于历史结果，不要对新格式执行旧版分析命令。

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
生成后的归因和文件写出。自然时间来自同一个 observer-on 请求，不能冒充独立
observer-off 请求的自然延迟。没有 EOS 仅表示可能用尽预算，不能直接认定截断。

只有 `--debug-output` 才在同一 JSONL 的 `debug` 字段中保留完整 Forward 轨迹及
历史评测细节；仍不额外写文件。默认不累计完整历史，只保留当前步给评分使用。

## 已完成的验证

回放旧 HuggingFace formal200 的 806 次 Prefix 事件：拒绝旧事件 16 次，其中
14 次与最终合法工具不符；保留 790 次，其中 789 次对应最终同位置名称，1 次
是合法唯一前缀后模型生成 `Audio-to-Audio Enhancement`，单独归因为模型非法名称。
这是旧轨迹离线回放，不是修改后模型实跑，也不能据此宣称新的端到端时间指标。
