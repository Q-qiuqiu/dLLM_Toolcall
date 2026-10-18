# 历史说明：Fast-dLLM v1 × BFCL 多工具提前预测预实验

> 本文仅保留早期实验设计与命令记录。多文件格式、逐轨迹策略、工具启动模拟和
> 部分独立分析脚本已退役；下文历史命令不保证仍可执行，不是当前运行入口。
> 当前使用方法见 [项目 README](../../README.md) 与
> [单 JSONL / Natural 基线说明](../../EXPERIMENT_OUTPUT.md)。
> 下文文件和命令中的相对路径均按项目根目录理解。

本目录实现 LLaDA-8B-Instruct 在 BFCL 非 Live `parallel_multiple` 上的无缓存全局去噪实验。所有新增/修改代码均在本目录；原始 `Fast-dLLM` 工作树（包括其中已有的未提交改动）未被修改。

当前入口与输出格式以 [EXPERIMENT_OUTPUT.md](../../EXPERIMENT_OUTPUT.md) 为准：
默认每次只保存一个 `samples.jsonl`，按工具名称记录 OMTP / Prefix / Natural
中最早的事件，并独立保留自然识别时间。Prefix 已修复共享前缀歧义和已确定后缀冲突。
两个入口均支持 `--observer-mode natural`：独立运行只含 committed-token 检测的
Natural 基线，不执行 OMTP、Argmax、Anchor、Prefix 或评分。检测函数与 OMTP 共用。
独立运行命令及净提前量统计见 [EXPERIMENT_OUTPUT.md](../../EXPERIMENT_OUTPUT.md#独立-natural-only-基线)。
原生 Hugging Face Llama 3 AR 基线见 [AR_BASELINE.md](../../AR_BASELINE.md)：默认自然 EOS 结束、
不设实验输出 token 预算（仅保留模型上下文硬边界），标准 KV Cache、
逐 token 共用 Natural 检测、单 JSONL，并分开报告共同工具和同一 GT 集合的时间对照。
下文多文件报告及旧版消融命令属于历史流程；新结果使用 `summarize_experiment.py`，
不再自动输出 CSV、轨迹文件或图片。已有结果及其离线分析工具保留。

BFCL 已补齐可靠性开关；同 Forward 的四组真实 20 条门控对照结果见
[BFCL_GATE_ABLATION.md](../../BFCL_GATE_ABLATION.md)。这批样本上完整门控没有提高
准确率，平均提前步数由 11.37 降至 7.39，因此没有将其设为 BFCL 默认配置。

OMTP 可靠性逐项开发实验见 [OMTP_RELIABILITY.md](OMTP_RELIABILITY.md)。
Multimedia 开发集实跑了原版及五轮各 20 条的独立测试，推荐保留可信 Anchor、
分数门控和新 Natural 工具追加。运行时显式添加
`--omtp-strict-anchor --omtp-min-top1-score -5 --omtp-min-margin 0.1 --omtp-append-natural`；
不添加这些开关仍复现原版。开发集 GT Precision/F1 为 85.25%/86.87%，
相对最终 Raw Text 的 Precision 为 93.58%；二者不能混为任务或集合准确率。

## 已核对的本机状态

- Fast-dLLM：`/data/home/yzx/Fast-dLLM`，审计时 commit 为 `aadcf060e12b6413b494bf9aebb172cfe3cf911c`。
- 权重：`/data/labshare/Param/llada`，配置为 32 层、8B、BF16、`mask_token_id=126336`、上下文 4096。
- 模型实现：运行时从 `Fast-dLLM/v1/llada/model` 导入；无缓存生成/transfer 代码复制在 `fastdllm_generate.py` 中。
- 实验节点有 8 张 RTX 3090；OMTP smoke 使用 GPU 0 实际完成。若受限沙箱内 `nvidia-smi` 报驱动不可访问，需要在具备 GPU device 权限的终端运行命令。
- 本机 `llada` conda 环境当前为 Python 3.10、Torch 2.11.0+cu130、Transformers 4.49.0，已与 Fast-dLLM v1 的仓库要求对齐。

## 可替换的模型后端

`--model-path` 现在会根据 checkpoint 的 `config.json` 自动选择模型实现，
并从模型自己的 config/tokenizer 解析 MASK、EOS、PAD、词表大小和上下文上限。
OMTP 候选名称也始终用当前模型的 tokenizer 重新编码，不再依赖 LLaDA-8B
的固定 token ID。

| 模型 | 本地路径 | 加载方式 | MASK / 上下文 | Cache 支持 |
|---|---|---|---|---|
| LLaDA-8B-Instruct | `/data/labshare/Param/llada` | Fast-dLLM v1 `LLaDAModelLM` | config 自动读取（当前为 126336 / 4096） | No Cache、Dual Cache |
| LLaDA 1.5 8B | `/data/labshare/Param/llada1.5` | Fast-dLLM v1 `LLaDAModelLM` | config 自动读取（当前为 126336 / 4096） | No Cache、Dual Cache |
| iLLaDA-8B-Instruct | `/data/labshare/Param/iLLADA-8B-Instruct` | checkpoint 自带 `ILLaDAForCausalLM` | tokenizer/config 自动读取（当前为 5 / 8192） | No Cache |

本机的 LLaDA 1.5 来自官方 `GSAI-ML/LLaDA-1.5`，下载时固定 revision
`84346fd91ba60252d260022201ad6fc5a3468fb2`，而不是浮动的 `main`。
现有 iLLaDA checkpoint 的本地 Hugging Face metadata 记录 revision 为
`5769f049224b2cff61106ebce0429ede5deeb434`。

iLLaDA 的现有目录外面多包了一层，程序会自动解析到唯一包含
`config.json` 的子目录。它的原生 chat template 不支持 system role；仅对这一已知
错误，`prompting.py` 会把 system 指令和原 user 内容合并成一个 user turn，再调用
原生 template。LLaDA/LLaDA 1.5 的消息结构保持不变。

iLLaDA 的网络结构和 KV 维度不同，不能安全套用当前只为
`LLaDAModelLM` 实现的 Dual Cache；选择 `--cache-mode dual` 会在生成前明确报错，
不会静默退化或产生不可比较结果。三种模型都可以使用相同的 No Cache 命令，
只需替换 `--model-path`，并为结果使用独立目录。例如 TaskBench：

```bash
# LLaDA 1.5
CUDA_VISIBLE_DEVICES=0 python run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/formal200.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_huggingface_llada15_nocache \
  --cache-mode none --prompt-profile compact_json_v3 \
  --generation-profile fixed128 --parity-samples 1

# iLLaDA（外层路径和内层实际 checkpoint 路径均可）
CUDA_VISIBLE_DEVICES=0 python run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/formal200.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/iLLADA-8B-Instruct \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_huggingface_illada_nocache \
  --cache-mode none --prompt-profile compact_json_v3 \
  --generation-profile fixed128 --parity-samples 1
```

### iLLaDA 直接 JSON 响应消融

iLLaDA checkpoint 没有等价于 `enable_thinking=False` 的原生开关。为测试不先生成
`<think>` 的效果，程序提供显式的 `--response-mode json_prefill`：在 128-token
生成区开头提交纯结构前缀 `[{"name":"`，余下位置仍按原 No Cache 算法去噪。
它不修改 Prompt、不包含候选或正确工具名、不增加 Forward；前缀会保留在
`raw_output` 中并由 JSON 评测正常检查。该模式改变了解码初始条件，所以结果会在
逐样本记录和 `metadata.json` 中标记，必须作为独立消融报告。默认的
`--response-mode native` 完全保持原生行为。

20 条 HuggingFace smoke：

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=0 python run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/iLLADA-8B-Instruct \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_huggingface_illada_json_prefill_smoke20 \
  --cache-mode none --response-mode json_prefill \
  --prompt-profile compact_json_v3 --generation-profile fixed128 \
  --parity-samples 1
```

重点检查 `taskbench_generation_summary.csv` 中的 raw/repair JSON 合法率，并在
`taskbench_samples.jsonl` 中确认 `response_mode=json_prefill`、
`generation_prefix_is_model_generated=false`、`forward_count=32` 和 parity 字段。

### 扁平 Tool Call 输出格式

`--output-schema flat_name_v1` 保留 OMTP 使用的外层 `"name"` 字段，但把参数
直接放在同一个调用对象中。候选定义会渲染为无 `parameters` 包装的函数签名，
Prompt 同时提供签名到扁平 JSON 的正向示例。例如模型输出：

```json
[{"name":"get_weather","location":"Beijing","date":"tomorrow"}]
```

程序在生成结束后确定性转换为：

```json
[{"name":"get_weather","arguments":{"location":"Beijing","date":"tomorrow"}}]
```

转换不进入 Prompt、Forward 或 OMTP 事件计时。原始模型对象保存在
`model_output_tool_calls`，标准对象保存在 `converted_standard_tool_calls`。
若工具的参数本身叫 `name`，Prompt 签名会生成无冲突别名（当前为
`arg_name`），转换时自动还原。兼容转换器还会递归识别模型误加的
`parameters`、`arguments`、`args`、`input` 或其他可由 Schema 确认的嵌套层；
最终只有候选工具名、必需参数、字段名和 JSON 类型通过验证的标准调用才标记为
转换成功。未知多余字段会从标准调用中丢弃，但原始模型输出始终保留。
默认 `--output-schema standard` 保持旧格式不变。

使用 LLaDA 1.5 运行同一批 HuggingFace 20 条配对 Smoke：

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=0 python run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_huggingface_llada15_flat_name_smoke20 \
  --cache-mode none --response-mode native \
  --output-schema flat_name_v1 --prompt-profile compact_json_v3 \
  --generation-profile fixed128 --parity-samples 1
```

也可将单独保存的扁平 JSON 转为标准 Tool Call：

```bash
python convert_flat_tool_calls.py \
  --input flat_calls.json \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --output standard_calls.json
```

TaskBench 的依赖感知 Prompt 作为独立的 `compact_json_dep_v4` 保留，不覆盖
`compact_json_v3` 基线。它仍输出相同的扁平 `{name, ...inputs}` 数组，只用
`<node-N>` 表示对数组中更早一次调用输出的引用，不增加 `depends_on` 字段，也不
改变 OMTP 的 `"name"` Anchor。例如：

```json
[{"name":"Image-to-Text","input_1":"example.jpg"},{"name":"Summarization","input_1":"<node-0>"}]
```

运行依赖感知的 LLaDA 1.5 Smoke Test：

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=0 python run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_huggingface_llada15_flat_name_dep_v4_smoke20 \
  --cache-mode none --response-mode native \
  --output-schema flat_name_v1 --prompt-profile compact_json_dep_v4 \
  --generation-profile fixed128 --parity-samples 1
```

Prompt 只说明引用语法并提供正向示例，不包含样本 Ground Truth、实际依赖边或
参考执行计划。当前转换器会把 `<node-N>` 原样保存在标准 `arguments` 中。

依赖正确性使用独立的离线语义评测：按模型生成数组的 0-based 顺序把
`<node-N>` 恢复为“第 N 个调用的工具 → 当前工具”，再与 TaskBench 的
`dependency_edges` 比较。它不要求模型引用字符串与官方 `tool_nodes` 中的字面
编号一致，也不修改 Prompt、OMTP、模型输出或计时。每次新实验会额外生成：

- `taskbench_dependency_summary.csv`：Raw/Repair 的依赖边 Precision、Recall、
  F1、Exact Match，以及工具集合/顺序与依赖边联合正确率。
- `taskbench_dependency_edges.jsonl`：逐样本有效、越界、自引用、格式错误引用和
  TP/FP/FN 边。

`Literal AST` 指标继续保留。对旧实验目录可直接离线补算，不必重新推理：

```bash
python evaluate_taskbench_dependencies.py \
  --result-dir results/taskbench_huggingface_llada15_flat_name_dep_v4_smoke20 \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json
```

每轮的 `metadata.json` 会保存 requested/resolved model path、模型 family、
architecture、MASK/EOS/PAD、词表大小、上下文上限以及 Dual Cache 能力，便于确认
实际运行的不是错误 checkpoint。更换模型会改变生成质量、时间和中间 logits，
因此不同模型必须分别报告结果，不能和原 LLaDA run 混写到同一目录。

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

## 固定方法：OMTP + Fallback

`run_omtp_experiment.py` 现在只运行优化后的 `omtp_fallback`。每个 occurrence 轨迹不可撤回地采用最早到达的事件：OMTP latent region Stable N、唯一 committed prefix，最后是 natural completion。`latent_region`、`prefix` 和 `natural` 原始事件仍保留作审计，但不再作为独立在线方法计分。自然解码作为显式分析基线，与 OMTP 在同一轮 Forward、同一时间零点下记录。

算法只使用当前请求 function definitions 中的候选名称。Ground Truth 和最终输出仅在生成结束后标注历史事件，不参与 anchor、评分、稳定判定或触发。Anchor 只扫描 generation region，并以 JSON 对象深度排除 `arguments` 嵌套对象中的 `name` 字段。错误在线事件不会被最终答案回删。

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

新增输出为 `omtp_events.jsonl`（所有原始事件）、`omtp_fallback_events.jsonl`（真正采用的不可撤回事件）、`natural_events.jsonl`（committed token 自然识别事件）、`position_records.jsonl`（逐调用位置的 OMTP/Natural 配对记录）、`omtp_trajectory.jsonl`、`omtp_samples.jsonl`、`omtp_summary.csv`、`omtp_comparison.csv`、`omtp_position_summary.csv`、`omtp_diagnostics.csv` 和 `figures/`。其中 diagnostics 汇总去重后的错误工具触发、Final Raw Text 与 Strict JSON 工具集合差异，以及完整的 OMTP/Natural 成功组合；这些都是生成结束后的离线统计。每个样本的 `natural_baseline` 保存自然基线分别相对最终输出和 Ground Truth 的 P/R/F1、Set EM 与检测时间；`per_tool_predictions` 同时保存 `T_online`、`T_natural` 和配对提前量。`online_first_trigger_time_s` 是因果在线触发时间；`oracle_*` 是生成结束后的离线正确性标注。触发时间包含 observer 检测和评分开销。

评测按宽松到严格分层：`all_tool_types_prefetched` 只检查去重后的必需工具类型是否全部被预取，重复次数和额外误报不影响通过；`all_tool_occurrences_prefetched` 按 multiset 检查所有调用次数是否覆盖；`tool_occurrence_multiset_exact_match` 检查无序名称及次数完全一致；`position_accuracy`/`position_exact_match` 检查按生成位置排序后的第 i 个调用是否对应。`position_records` 保存每个位置的目标名称、OMTP/Natural 名称、时间、Forward、正确性和提前量。这部分匹配与 CSV 汇总在 `_generate_once` 返回后离线执行，不计入生成墙钟或在线触发时间，也不改变 NFE。

固定实现缓存候选名称 tokenization、融合 top-1/row-max、一次式 JSON 深度扫描、复用局部 log-partition，并按名称 token 长度批量评分。`--stable-n`、`--min-top1-score` 和 `--min-margin` 是该方法自身参数；默认关闭置信度门限。Fallback 始终启用，命令行不再提供关闭开关。`--profile-observer` 仅用于同步诊断，不应用于正式墙钟结果。

保留集 40–59 的历史 C 结果仍位于原结果目录，仅用于追溯：Ground Truth P/R/F1 为 .933/.983/.950，Set EM .85，observer 约 .216 秒/样本。由于当前输出语义已固定为 OMTP+fallback，旧的多方法 CSV 不应与新 run 混用。

固定方法运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python run_omtp_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/holdout40_59_omtp_fallback_fixed \
  --start 40 --limit 20 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --stable-n 2 --seed 20261009 --parity-samples 1
```

## 无 GPU 验证

```bash
cd /data/home/yzx/dLLM_Toolcall
python -m pytest -q
```

当前实现的无 GPU 测试覆盖 BFCL 数据适配、ground-truth prompt 隔离、严格/局部 parser、重复调用与 AST、observer 开关输出一致性、block 元数据、N-stable 和启动模拟。

## TaskBench DailyLife 跨 Benchmark 实验

TaskBench 实验只增加数据适配、固定样本清单、离线评测和报表；生成仍复用
`run_omtp_experiment._run_sample` 与现有 Optimized OMTP + Prefix/Natural
Fallback。命令行不开放生成或 OMTP 参数，固定为 generation/block length 128、
32 steps、temperature 0、Stable N=2、region radius 4、region top-k 2、
min-seen 2、min-support .5、无 confidence gate。

官方 DailyLife 数据保存在 `data/taskbench/data_dailylifeapis/`。使用系统 Python
可从 Parquet 重新生成固定且互斥的清单：

```bash
cd /data/home/yzx/dLLM_Toolcall
python prepare_taskbench.py \
  --parquet data/taskbench/data_dailylifeapis/improved.parquet \
  --tool-catalog data/taskbench/data_dailylifeapis/tool_desc.json \
  --output-dir data/taskbench/data_dailylifeapis/splits \
  --seed 20261009 --smoke-per-type 10 --formal-per-type 100
```

Smoke Test（10 Chain + 10 DAG）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_dailylifeapis/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_dailylifeapis/tool_desc.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_dailylife_smoke20_fixed \
  --parity-samples 1
```

Smoke 的 JSON、事件、NFE=32 和 parity 均正常后，再运行互斥的正式 200 条：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_dailylifeapis/splits/formal200.jsonl \
  --tool-catalog data/taskbench/data_dailylifeapis/tool_desc.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_dailylife_formal200_fixed \
  --parity-samples 1
```

同一固定128/32配置可通过 `--cache-mode` 选择 Fast-dLLM v1 No Cache
或 Dual Cache。Dual Cache 的 observer 直接复用缓存解码每次已有的 logits，
不增加模型 Forward；全局单 Block 的128个生成位置在每步仍全部可见。
Dual Cache Smoke 使用独立结果目录：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_dailylifeapis/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_dailylifeapis/tool_desc.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --cache-mode dual \
  --output-dir results/taskbench_dailylife_smoke20_dual_cache_fixed \
  --parity-samples 1
```

省略 `--cache-mode` 或显式传入 `--cache-mode none` 即为原始 No Cache
策略。Parity 比较的是同一种 cache mode 下 Observer Off/On 的输出和NFE；
No Cache与Dual Cache之间的生成一致性使用两个独立run按 sample ID离线配对。

每条 Prompt 都包含完整的 40 个 DailyLife 候选工具，只读取 instruction 与工具
目录；Ground Truth 调用、依赖边和参考步骤仅随结果保存并用于生成结束后的离线评测。
逐位置指标按 TaskBench `tool_nodes`/参考步骤顺序定义；依赖边只用于 Chain/DAG
分组。本次固定清单不存在重复工具调用样本，因此 occurrence/multiset 指标会保留，
但不能据此评估重复调用泛化能力。

独立结果目录包含 `taskbench_samples.jsonl`、三种方法各自事件流、逐 Forward
轨迹、逐位置记录、失败记录、metadata、四张汇总 CSV 与 `figures/`。三种方法为
Natural、纯 latent-region OMTP、正式 OMTP+Fallback；所有正确性与配对提前量均为
生成结束后的 Ground Truth 离线标注，不反馈到在线策略。

TaskBench 还会在生成彻底结束后使用 `json-repair` 做独立的离线语法修复。
原始文本、原始严格 JSON 指标和修复后指标会同时保留；修复结果不进入
OMTP、Natural 检测或模型输入。对已有结果可运行：

```bash
/home/yzx/miniconda3/envs/llada/bin/python repair_taskbench_results.py \
  --result-dir results/taskbench_dailylife_smoke20_dual_cache_fixed \
  --manifest data/taskbench/data_dailylifeapis/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_dailylifeapis/tool_desc.json
```

该命令不会覆盖原始 `taskbench_samples.jsonl`，而是新增
`taskbench_json_repair_samples.jsonl`、`taskbench_json_repair_summary.csv` 和
`taskbench_json_repair_failures.jsonl`。

## TaskBench HuggingFace / Multimedia 固定跨领域实验

两个资源型领域使用与 DailyLife 完全相同的 OMTP + Fallback 核心及
No Cache 128/128/32 推理配置。官方数据固定为 `microsoft/Taskbench`
revision `d12764a8fa650f49f50507a02c3e707c71c67621`；预先生成的 Smoke 20
与 Formal 200 清单互不重叠，选择种子为 `20261009`。HuggingFace 使用完整
23 工具目录，Multimedia 使用完整 40 工具目录。Ground Truth、依赖边和参考
执行步骤都不会进入 Prompt。

`--prompt-profile original` 精确复现最初结果，`strict_json_v2` 保留此前严格
Prompt 消融；当前默认 `compact_json_v3`。紧凑版本继续将 Schema 标准化为
`{"type":"object"}`，只正向说明 JSON 数组及 `name`/`arguments` 结构，让模型
根据函数参数 Schema 生成输入；不加入候选名 enum、参数长度规定或负向禁止条款。

校验已经下载的官方原始文件及其 SHA（首次获取时去掉 `--verify-only`）：

```bash
cd /data/home/yzx/dLLM_Toolcall
python download_taskbench_domains.py --verify-only
```

先审计候选名的 tokenizer/Prefix 兼容性（该命令不运行模型）：

```bash
cd /data/home/yzx/dLLM_Toolcall
/home/yzx/miniconda3/envs/llada/bin/python audit_taskbench_tool_names.py \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --tokenizer /data/labshare/Param/llada \
  --output data/taskbench/data_huggingface/splits/tool_name_audit.json

/home/yzx/miniconda3/envs/llada/bin/python audit_taskbench_tool_names.py \
  --tool-catalog data/taskbench/data_multimedia/tool_desc.json \
  --tokenizer /data/labshare/Param/llada \
  --output data/taskbench/data_multimedia/splits/tool_name_audit.json
```

两个 20 条 No Cache Smoke Test：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM --cache-mode none \
  --prompt-profile original \
  --output-dir results/taskbench_huggingface_smoke20_nocache_fixed \
  --parity-samples 1

CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_multimedia/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_multimedia/tool_desc.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM --cache-mode none \
  --prompt-profile original \
  --output-dir results/taskbench_multimedia_smoke20_nocache_fixed \
  --parity-samples 1
```

确认 Smoke 的 NFE、Parity 和失败记录后，再分别运行互斥的正式 200 条：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/formal200.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM --cache-mode none \
  --prompt-profile original \
  --output-dir results/taskbench_huggingface_formal200_nocache_fixed \
  --parity-samples 1

CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_multimedia/splits/formal200.jsonl \
  --tool-catalog data/taskbench/data_multimedia/tool_desc.json \
  --model-path /data/labshare/Param/llada \
  --fastdllm-root /data/home/yzx/Fast-dLLM --cache-mode none \
  --prompt-profile original \
  --output-dir results/taskbench_multimedia_formal200_nocache_fixed \
  --parity-samples 1
```

正式结果完成后生成三领域描述性对照（不会重新运行推理）：

```bash
/home/yzx/miniconda3/envs/llada/bin/python compare_taskbench_domains.py \
  --dailylife results/taskbench_dailylife_formal200_nocache_fixed \
  --huggingface results/taskbench_huggingface_formal200_nocache_fixed \
  --multimedia results/taskbench_multimedia_formal200_nocache_fixed \
  --output-dir results/taskbench_cross_domain_fixed
```

每个领域目录会额外生成 `taskbench_tool_name_summary.csv` 和
`taskbench_special_name_failures.jsonl`。后者只标注含空格、连字符或括号的
漏检，不能单凭一条漏检断言是 Anchor 错误。Multimedia 目录里存在已知 Prefix
关系 `Image Search` / `Image Search (by Image)`，保留原始名字且不为它改算法。

### Strict JSON Prompt V2 消融

在相同 128/128/32 Smoke 样本上测试新 Prompt，并使用独立目录：

```bash
for domain in huggingface multimedia; do
  CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
    run_taskbench_experiment.py \
    --manifest data/taskbench/data_${domain}/splits/smoke20.jsonl \
    --tool-catalog data/taskbench/data_${domain}/tool_desc.json \
    --model-path /data/labshare/Param/llada \
    --fastdllm-root /data/home/yzx/Fast-dLLM \
    --cache-mode none --generation-profile fixed128 \
    --prompt-profile strict_json_v2 \
    --output-dir results/taskbench_${domain}_smoke20_nocache_prompt_v2 \
    --parity-samples 1
done
```

新 Prompt 后的最大上下文长度（含 128 生成区）分别为 HuggingFace 2620、
Multimedia 3268，仍低于模型 4096 上限。

### Compact JSON Prompt V3

只考察最终 JSON 合法性时，使用默认的紧凑 Prompt。V2 仍可显式选择，不会被
覆盖。两个领域的配对 Smoke Test 命令如下：

```bash
for domain in huggingface multimedia; do
  CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
    run_taskbench_experiment.py \
    --manifest data/taskbench/data_${domain}/splits/smoke20.jsonl \
    --tool-catalog data/taskbench/data_${domain}/tool_desc.json \
    --model-path /data/labshare/Param/llada \
    --fastdllm-root /data/home/yzx/Fast-dLLM \
    --cache-mode none --generation-profile fixed128 \
    --prompt-profile compact_json_v3 \
    --output-dir results/taskbench_${domain}_smoke20_nocache_prompt_v3 \
    --parity-samples 1
done
```

### 256-token 独立诊断组

主实验默认仍是 `fixed128`。若要判断输出预算对 JSON 和工具集合正确率的影响，
可在同一 Smoke 20 上运行 `diagnostic256`；它只把 generation/block length 改为
256、steps 改为 64，不改变 Prompt、样本、种子或 OMTP 参数，并必须使用独立
结果目录：

```bash
for domain in huggingface multimedia; do
  CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
    run_taskbench_experiment.py \
    --manifest data/taskbench/data_${domain}/splits/smoke20.jsonl \
    --tool-catalog data/taskbench/data_${domain}/tool_desc.json \
    --model-path /data/labshare/Param/llada \
    --fastdllm-root /data/home/yzx/Fast-dLLM \
    --cache-mode none --generation-profile diagnostic256 \
    --prompt-profile original \
    --output-dir results/taskbench_${domain}_smoke20_nocache_256diag \
    --parity-samples 1
done
```

## 独立真实工具模型冷启动实验

模型权重、固定输入、版本与 SHA 清单位于 `models/`，服务和 benchmark 源码位于
`models/src/`。这一实验与 OMTP 完全隔离；详细命令和时间字段定义见
[`models/README.md`](../../models/README.md)。BLIP 只代表图像描述能力，不被标记为
Multimedia OCR 工具的完整实现。
