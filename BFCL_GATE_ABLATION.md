# BFCL 门控效果验证：实际 20 条结果

结论：这批样本上，门控没有改善 OMTP + Fallback 的工具集合准确率，
但减少了正确预测的提前步数；完整门控对提前识别能力属于负优化。
只适用于本轮 LLaDA 1.5 / BFCL 前 20 条 Smoke，不能外推为全量或其他模型结论。

## 实验设置

- `GSAI-ML/LLaDA-1.5` 本地 `/data/labshare/Param/llada1.5`，BF16、不量化。
- BFCL v4 `parallel_multiple_0` 至 `parallel_multiple_19`，候选数 2–3。
- No Cache、gen_length=128、block_length=128、steps=32、temperature=0。
- 原 BFCL Prompt，标准 `{name, arguments}` 数组，Stable N=2。
- 随机种子 `20261009 + sample_index`。候选工具只来自请求 function definitions。
- 所有组都启用已修复的 Prefix 校验、按工具类型首次事件去重和独立 Natural 记录。
- 四组观察器读取**同一轮 Forward** 的 logits / committed tokens，
  不为每组增加 Forward，也不更改 token transfer。
- GPU 0–3 共享资源，使用 Accelerate 分层加载与 CPU 权重卸载。
  这次只评价准确率和 Forward 提前量，不比较四组墙钟时间。

| 组别 | 可信 Anchor | Top-1 阈值 | Margin 阈值 |
|---|---|---:|---:|
| A 无门控 | 关闭 | 关闭 | 关闭 |
| B 仅 Anchor | 开启 | 关闭 | 关闭 |
| C 仅 Confidence | 关闭 | -5 | 0.1 |
| D 完整门控 | 开启 | -5 | 0.1 |

阈值沿用既有开发配置，本轮未依据 BFCL 结果调整。

## 主结果：OMTP + Fallback

参照模型最终 Raw Text 的合法候选工具名称，每种工具在每条请求内只计首次。
20 条共 38 个最终工具类型实例；全部纳入，兜底时的提前量为 0。

| 组别 | Precision | Recall | 工具集合完全一致 | 正确工具提前识别比例 | 平均提前 Forward | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 无门控 | 95.00% | 100% | 90% | 34/38 = 89.47% | 11.368 | 11 | 26.3 |
| B 仅 Anchor | 95.00% | 100% | 90% | 34/38 = 89.47% | 7.632 | 8 | 17.0 |
| C 仅 Confidence | 95.00% | 100% | 90% | 31/38 = 81.58% | 11.184 | 11 | 26.3 |
| D 完整门控 | 95.00% | 100% | 90% | 31/38 = 81.58% | 7.395 | 8 | 17.0 |

四组每条请求的最终触发工具集合完全相同。相对 BFCL GT 的指标也完全相同：
Precision=92.50%、Recall=100%、F1=96.10%、Set EM=85%。
这些 GT 指标与最终生成忠实度指标分开计算。

完整门控相对无门控：38 个工具中 17 个识别更晚、21 个不变、0 个更早；
平均推迟 3.974 次 Forward，平均提前窗口减少 34.95%。
逐请求先平均其正确工具的识别变化：14 条更晚、6 条不变、0 条更早。

Recall=100% 来自 Natural 兜底，不意味着纯 Latent 全部识别成功。

## 错误归因

各组都是 40 个去重首次触发、38 个最终出现、2 个额外触发：

1. `parallel_multiple_2`：Latent 错误预测 `rectangle.calculate_perimeter`，
   最终只有两个 circle 工具。这是方法自身错误，四组都没有拦住。
2. `parallel_multiple_3`：唯一 committed 前缀 `integ` 合法触发 `integral`，
   模型随后在同一字段生成 `integral.integolution`。
   按用户定义，标记 `model_invalid_name`，不计 Prefix 检测错误；
   原始触发仍保留，不能从物理额外启动成本中删除。

因此原始额外触发率为 2/40=5%；排除模型非法名称事件后的检测错误率
为 1/39=2.56%，四组仍然相同。

## 纯 Latent 的附加观察

| 组别 | TP / FP / FN | Precision | Recall |
|---|---|---:|---:|
| A 无门控 | 37 / 1 / 1 | 97.37% | 97.37% |
| B 仅 Anchor | 37 / 1 / 1 | 97.37% | 97.37% |
| C 仅 Confidence | 30 / 1 / 8 | 96.77% | 78.95% |
| D 完整门控 | 28 / 1 / 10 | 96.55% | 73.68% |

这些是纯 Latent 的去重类型识别指标，不将 Prefix / Natural 混入。

## 延后的原因与实例

- 可信 Anchor 必须等待临时 argmax 中出现可信字段上下文，可能错过更早的正确
  logits 预测。例如 `parallel_multiple_1` 中，`area_circle.calculate`
  从无门控的 Forward 4 延到 Forward 19；矩形工具从 6 延到 16。
- 当前 Margin 门控将 Top-2 不存在（只有一个有效可评分候选）当作不通过。
  `parallel_multiple_0` 中 `math_toolkit.product_of_primes` 原本在 Forward 12
  正确触发，margin=None；开启门控后改为 Forward 14 的 Natural 兜底。
  类似情况还出现在 `train_ticket.buy` 和 `animal_population.get_history`。
- 剩下的 Latent 错误事件 Margin 约 0.444，大于 0.1，因此没有被拦截。
  比较分数不是校准正确概率，不能保证错误预测分数一定更低。

建议：BFCL 暂不默认套用这套完整门控；保留可配置开关。
若要修改 Margin 缺失规则，需另做明确消融，不能悄悄并入本轮结果。
后续可在未参与本轮的 180 条上确认泛化，不将本轮 Smoke 作为全量结论。

## 完整性检查与产物

- 实际 20 条全部完成，无运行失败；每条 32 NFE。
- 四组的独立 Natural 工具名称及 Forward 完全一致。
- 共享回调检查 committed / before token tensors 未被观察器修改。
- 首条 Observer Off/On 输出 token 与 NFE 一致。
- 73 项 CPU 测试通过，包含新 BFCL 参数入口、旧参数兼容和四组不增加 NFE。
- 有效结果唯一文件：
  `results/bfcl_gate_ablation_llada15_smoke20_r3/samples.jsonl`。
  `ablation_variants` 保存全部四组信息；metadata 保存参数、源码哈希、模型及设备布局。
- r1 在首条 Parity 阶段主动中止、r2 在模型加载阶段遇到共享显存 OOM，
  都没有完成任何样本，均明确标记且不纳入结果。

## 实际成功运行命令

```bash
cd /data/home/yzx/dLLM_Toolcall
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
TOKENIZERS_PARALLELISM=false /home/yzx/miniconda3/envs/llada/bin/python \
  run_bfcl_gate_ablation.py \
  --questions data/bfcl/BFCL_v4_parallel_multiple.json \
  --answers data/bfcl/BFCL_v4_parallel_multiple_answers.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/bfcl_gate_ablation_llada15_smoke20_r3 \
  --limit 20 --start 0 --gen-length 128 --block-length 128 --steps 32 \
  --temperature 0 --seed 20261009 --stable-n 2 \
  --cache-mode none --response-mode native --output-schema standard \
  --parity-samples 1 --device-map auto \
  --max-memory-json '{"0":"2600MiB","1":"3000MiB","2":"2800MiB","3":"2800MiB","cpu":"48GiB"}'
```

复跑必须换新输出目录；不能覆盖已有结果。
