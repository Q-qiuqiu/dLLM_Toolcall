# OMTP reliability development experiment

> 历史策略开发记录：逐轮开发运行器和旧多文件分析脚本已经退役。
> 本文保留开发结论，不是当前可执行命令清单。正式入口与独立 Natural 基线见
> [项目 README](../../README.md) 和 [EXPERIMENT_OUTPUT.md](../../EXPERIMENT_OUTPUT.md)。

This document describes the historical ablation and multi-file output. Current
runner output is a single `samples.jsonl`; use [EXPERIMENT_OUTPUT.md](../../EXPERIMENT_OUTPUT.md).
The current prefix validation and first-per-tool arbitration are bug fixes on
top of the selected policy, not the untouched historical algorithm.

These are explicitly opt-in changes to the original Optimized OMTP + Fallback.
The prompt, model, seeds, token transfer and forward count are held fixed.
All decisions use only current/past tokens and logits, plus the full candidate
catalog. Ground truth is consulted after inference for development selection.

The 20 Multimedia smoke samples are now a development set. Their selected
numbers must not be presented as an independent final evaluation. The existing
Multimedia formal200 split is disjoint from these development examples.

## Sequential candidates

1. `--omtp-name-boundary`: candidate alignments must end immediately before a
   closing quote, which may be committed or predicted by current argmax. This
   supports BPE tokens containing the quote and following comma/brace. It
   prevents a shorter candidate from matching the interior of a longer name.
2. `--omtp-consecutive-stability`: reset temporal stability when a track misses
   an intervening forward. Track association otherwise remains identical.
3. `--omtp-strict-anchor`: speculative scoring requires the current temporary
   tokens to actually form a name-field anchor after an outer-object opening
   or comma, outside argument string contents. This also gates prefix events.
   Original track association and Natural detection remain unchanged.
4. `--omtp-min-top1-score -5 --omtp-min-margin 0.1`: initial development gates.
   These scores are not calibrated probabilities. A missing margin rejects a
   latent trigger under the margin gate; prefix/natural fallback stays active.
5. `--omtp-append-natural`: append newly discovered natural tool identities
   even when a track committed to a different tool earlier. Wrong events remain
   in history. The merge is chronological and does not consult later events
   when deciding an earlier event's inclusion.

The default CLI, with no reliability flags, retains the original method.
The ablation driver saves an independent real 20-example inference run per
candidate and accepts a candidate only if:

- request-macro tool-type precision and F1 do not decrease, with at least one
  strictly increasing;
- final generated token IDs, NFE and Natural event signatures match baseline;
- observer-off/on parity passes;
- paired correctly detected unique tool types retain positive mean lead.

Each next trial inherits only previously accepted flags. Rejected flags are
removed from subsequent configurations; all rejected result artifacts remain
available. No original results are overwritten. Recall, set exact match,
complete coverage, FP/FN and all failure cases are reported alongside selection.

## Run

First run the unmodified-policy 20-example baseline into `00_baseline` using
`run_taskbench_experiment.py` with `compact_json_dep_v4`, `flat_name_v1`,
`fixed128`, no cache and the same manifest as the following command. Then:

```bash
CUDA_VISIBLE_DEVICES=1 /home/yzx/miniconda3/envs/llada/bin/python \
  run_omtp_reliability_ablation.py \
  --output-root results/omtp_reliability_multimedia_dev20_r1 \
  --baseline-dir results/omtp_reliability_multimedia_dev20_r1/00_baseline
```

The driver outputs `ablation_decisions.jsonl`, `ablation_comparison.csv`,
`selected_policy.json`, per-trial logs and full normal TaskBench artifacts.
`selected_policy.json` records all accepted flags, real commands and code hashes.
The precision goal is 0.9; reaching it is a measured outcome, not a guarantee.
This refers to de-duplicated tool-type precision, not request set exact match.
Timing from a shared GPU is reported as observed; publication-quality timing
should be rerun on an otherwise idle device after the policy is frozen.

## Reference and Natural-baseline audit

After the ablations finish:

```bash
python report_omtp_reliability_ablation.py \
  --result-root results/omtp_reliability_multimedia_dev20_r1
```

This adds `reference_comparison.csv`, `paired_unique_tool_timing.csv` and a
cross-trial accuracy figure. Metrics are reported against both GT and final
Raw Text. Final-output prediction fidelity is not the same as task correctness.
The same command also supports a single normal TaskBench result directory.

In addition to the historical track-based Natural baseline, the report audits
`natural_committed_fields`: first complete candidate `name` fields detected by
the existing sparse-committed-token scanner. Its timestamps use the already
recorded `prediction_available_time_s`, after that lightweight detection and
before OMTP scoring. This requires no generation-time work or extra forward.
The paired timing CSV uses this independent baseline, so speculative track
association and region-scoring overhead cannot inflate Natural completion
times. The original baseline and ablation decisions remain available unchanged.

The original stable policy counts valid observations and can retain a count
across a missing anchor. Only the explicit `--omtp-consecutive-stability` flag
requires adjacent forwards; do not describe a selected policy without this
flag as strict consecutive-forward stability.

## Selected policy and independent formal evaluation

Actual development results and all keep/revert decisions are in
`results/omtp_reliability_multimedia_dev20_r1/REPORT.md` and the CSV/JSON files
alongside it. The kept flags are strict Anchor, score -5 / margin 0.1, and new
Natural identity append. Global name boundary and strict consecutive stability
are not part of the selected policy. No-flags CLI continues to mean original.

To evaluate the frozen selected policy on the disjoint Multimedia formal200:

```bash
CUDA_VISIBLE_DEVICES=1 /home/yzx/miniconda3/envs/llada/bin/python \
  run_taskbench_experiment.py \
  --manifest data/taskbench/data_multimedia/splits/formal200.jsonl \
  --tool-catalog data/taskbench/data_multimedia/tool_desc.json \
  --model-path /data/labshare/Param/llada1.5 \
  --fastdllm-root /data/home/yzx/Fast-dLLM \
  --output-dir results/taskbench_multimedia_llada15_reliable_v1_formal200 \
  --cache-mode none --response-mode native \
  --output-schema flat_name_v1 --prompt-profile compact_json_dep_v4 \
  --generation-profile fixed128 --parity-samples 1 \
  --omtp-strict-anchor --omtp-min-top1-score -5 --omtp-min-margin 0.1 \
  --omtp-append-natural
```

Then compute the independent Natural baseline and dual-reference analysis:

```bash
python report_omtp_reliability_ablation.py \
  --result-root results/taskbench_multimedia_llada15_reliable_v1_formal200
```

The same frozen flags apply to HuggingFace: replace the manifest/catalog domain
and use a new `results/taskbench_huggingface_llada15_reliable_v1_formal200`
directory. Formal evaluation has not been run as part of this development task.
