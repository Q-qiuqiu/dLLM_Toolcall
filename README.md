# OMTP + Fallback 工具名称在线预测实验

基于 Fast-dLLM v1，比较 LLaDA 的 OMTP + Prefix/Natural Fallback、独立
Natural-only，以及原生 Hugging Face Llama 3 AR Natural 基线。
支持 BFCL v4 `parallel_multiple` 和 TaskBench 的 DailyLife、HuggingFace、
Multimedia 三领域。在线预测不使用 Ground Truth，不启动任何真实工具。

## 当前保留的配置

- 主实验：`gen_length=128`、`block_length=128`、`steps=32`、temperature=0、batch_size=1。
- OMTP：Stable N=2；每条请求、每种工具只记录 Latent/Prefix/Natural 中最早的事件。
- 同轮独立记录 Natural 首次识别时间；公平净收益使用另跑的 `--observer-mode natural`。
- `--cache-mode none` 和 `dual` 都保留；Dual Observer 使用全局 Block，要求
  `block_length == gen_length`，不会为观测增加模型 Forward。
- TaskBench 使用 `compact_json_dep_v4` + `flat_name_v1`，例如
  `[{"name":"tool_a","input_1":"example.jpg"},{"name":"tool_b","input_1":"<node-0>"}]`。
  离线转换成通用 `name/arguments` 结构。
- BFCL 使用原始 Prompt 和 `standard` 输出格式；正式对照沿用无门控策略。
- 每次推理运行只保存一个 `samples.jsonl`，已有非空目录不能覆盖。
- 已退役额外选择理由输出与分 Block Dual 消融，原有无额外理由 Prompt 不变。

模型通过 `config.json` 自动适配 LLaDA、LLaDA 1.5 与 iLLaDA；下面显式使用已完成
正式实验的 `/data/labshare/Param/llada1.5`。模型权重不纳入 Git。

## 目录与入口

```text
dLLM_Toolcall/
├── run_taskbench_experiment.py    # TaskBench OMTP / Natural
├── run_omtp_experiment.py         # BFCL OMTP / Natural
├── run_ar_experiment.py           # Hugging Face AR Natural
├── fastdllm_generate.py           # No Cache / Dual Cache，不改变 transfer
├── *_observer.py                 # OMTP、Prefix、Natural 和可靠性门控
├── summarize_experiment.py       # 单 JSONL 指标及独立 Natural 净提前量
├── compare_ar_baseline.py         # AR 与 dLLM 离线时间/正确率对照
├── evaluate_official_results.py   # 已保存输出的官方离线评测
├── export_paper_workbook.py       # 已支持正式结果的 Excel 汇总
├── data/                         # 原始数据、完整工具目录、固定样本清单
├── results/                      # 正式推理日志与最终官方评测报告
├── models/                       # 独立工具模型；models/src 为服务/冷启动代码
├── tests/                        # 保留功能的 CPU 回归测试
├── third_party/official_eval/    # 固定版本的官方评测源码
└── docs/legacy/                  # 历史流程与开发结论，不是当前运行入口
```

早期多文件结果的独立比较、重分析和逐轮可靠性开发脚本已移除。
仍被当前生成器、debug 评测或回归测试调用的通用模块没有删除。
数据下载/适配、工具格式转换、语义依赖检查和工具模型冷启动程序也保留。

## TaskBench：20 条 Smoke

从项目根目录执行；GPU 编号按实际情况选择：

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_huggingface/splits/smoke20.jsonl \
  --tool-catalog data/taskbench/data_huggingface/tool_desc.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --cache-mode none --response-mode native \
  --output-schema flat_name_v1 --prompt-profile compact_json_dep_v4 \
  --generation-profile fixed128 --parity-samples 1 \
  --omtp-strict-anchor --omtp-min-top1-score -5 --omtp-min-margin 0.1 \
  --omtp-append-natural \
  --output-dir results/taskbench_huggingface_omtp_smoke20_NEW
```

三领域目录分别为 `data_dailylifeapis`、`data_huggingface`、`data_multimedia`。
正式实验将清单换为同领域的 `formal200.jsonl`，并换新的输出目录；它与 Smoke 不重叠。

运行匹配的独立 Natural-only 时，保持模型、数据、Prompt、Cache、生成配置和种子一致，
添加 `--observer-mode natural`，移除全部 `--omtp-*` 门控选项，并另设输出目录。
完整基线命令见 [EXPERIMENT_OUTPUT.md](EXPERIMENT_OUTPUT.md#独立-natural-only-基线)。
若比较 Dual Cache，则两条命令都改为 `--cache-mode dual` 并使用独立目录。
不能拿另一种 Cache / Prompt 的运行冒充匹配基线。

## BFCL：20 条 Smoke

```bash
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python \
  run_omtp_experiment.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --limit 20 --start 0 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --seed 20261009 --stable-n 2 \
  --cache-mode none --response-mode native --output-schema standard \
  --parity-samples 1 --output-dir results/bfcl_omtp_smoke20_NEW
```

全量复现使用 `--limit 200`；Natural-only 添加 `--observer-mode natural` 并更换目录。
若先用 20 条选择新策略，则保留测试应使用其他样本，而不是将同一批开发样本视作独立测试。

## 结果分析

```bash
/home/yzx/miniconda3/envs/llada/bin/python summarize_experiment.py \
  --input results/taskbench_huggingface_omtp_smoke20_NEW/samples.jsonl \
  --natural-baseline results/taskbench_huggingface_natural_smoke20_NEW/samples.jsonl
```

汇总默认只打印终端，不额外创建文件。

- 时间按请求 × 去重工具类型配对，不只看第一个工具，不重复计算同名调用。
- 净提前量是独立 Natural-only 的检测时间减 OMTP-on 的首次在线事件时间；已经
  包含 Observer 成本，不要再减一次全部 Observer 耗时。保留负值、缺失和失败。
- 识别时间比不是工具执行端到端加速比；本系统没有连接工具预取或执行。
- 对 Final Raw Text 的忠实度与对 GT 的正确率分开。工具集合正确不代表 JSON、
  调用次数、参数或完整任务正确；单次类型首次事件不能恢复 occurrence / 位置预测轨迹。
- 官方 AST / TaskBench 指标由保存结果离线适配计算；JSON Repair 不改变原始日志、
  在线事件或计时。官方指标不能用本地兼容 AST 或工具集合指标替代。

更多说明：

- [当前 JSONL、计时和 Natural 基线](EXPERIMENT_OUTPUT.md)
- [AR 运行与对照](AR_BASELINE.md)
- [官方评测、版本与评估分母](OFFICIAL_EVALUATION.md)
- [BFCL 门控开发结论](BFCL_GATE_ABLATION.md)
- [独立真实工具模型服务与冷启动](models/README.md)
- [历史配置与旧版流程](docs/legacy/README_HISTORY.md)
- [历史可靠性开发记录](docs/legacy/OMTP_RELIABILITY.md)

## CPU 回归测试

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yzx/miniconda3/envs/llada/bin/python -B -m pytest -q -p no:cacheprovider
```

GPU 推理由用户在具备 CUDA 权限的终端手动执行。CPU 测试不代表真实模型 Smoke 已重新运行。
