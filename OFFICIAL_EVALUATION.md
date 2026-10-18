# Saved-output official evaluation

This is a CPU-only, offline evaluation of saved LLaDA-1.5 and optional AR outputs. It never
loads model weights, reruns inference, edits generation logs, changes OMTP, or
feeds ground truth into prediction. All code and outputs remain in this project.

## Run

```bash
cd /data/home/yzx/dLLM_Toolcall
python download_official_evaluators.py
/home/yzx/miniconda3/envs/llada/bin/python -m pip install \
  --target .official_eval_deps --no-deps -r requirements-official-eval.txt
/home/yzx/miniconda3/envs/llada/bin/python evaluate_official_results.py \
  --output-dir results/official_eval_REPLACE_WITH_NEW_DIRECTORY
```

Download/setup requires network access once. Scoring is offline, without GPU
or remote APIs. The target dependencies do not replace packages in the model
environment. Existing numpy/scipy/sklearn/pandas/click/openpyxl/json-repair are
reused; the exact dependency versions used are recorded in `metadata.json`.
An existing output directory is rejected, preserving historical results.
`--domains bfcl` or `--domains dailylifeapis huggingface multimedia` selects
part of the suite. Default inputs are the fixed full/formal 200-query runs:
No Cache Natural, No Cache OMTP + Fallback, and Dual OMTP + Fallback.
Add `--include-ar` to also score the four saved Llama 3 8B Natural EOS runs
using the identical official sources, adapters, repair policy and GT subsets:

```bash
/home/yzx/miniconda3/envs/llada/bin/python evaluate_official_results.py \
  --include-ar --output-dir results/official_eval_with_ar_NEW_DIRECTORY
```

The default three-method dLLM suite is unchanged. AR cache is labeled `ar_kv`,
not dLLM Dual Cache. Its recorded `context_limit` cases remain in the applicable
evaluation denominators and are counted separately from natural EOS completion.
No AR token budget is added and no model is rerun.

In the saved AR TaskBench runs, all 200 answers per domain contain introductory
and/or trailing prose (for example, `Here is the JSON array for the request:`).
Under the same strict whole-response JSON policy, their `original` scores are
therefore zero. This is a format-compliance failure, NOT evidence that every
tool choice was wrong. Report it separately and use the common `json_repair`
condition when comparing recovered tool/argument quality across both models.
The raw outputs and strict-zero rows are preserved, not relabeled as valid.

## Official source provenance

- BFCL revision: `58f57e9124ea981403792dd51e00a6577e621fae`, identical to the
  local BFCL V4 data download revision.
- TaskBench evaluator: dataset repository `microsoft/Taskbench`, revision
  `d12764a8fa650f49f50507a02c3e707c71c67621`.
- Original downloaded source files stay unmodified under `third_party/official_eval/`.
  URLs, byte counts, revisions and SHA256 are locked in `sources.json`, and
  checked before execution. All scoring functions are the official functions,
  not the former local `bfcl_compatible_ast_match` implementation.
- BFCL's inference-provider registry is bridged by a single custom-model
  registration with `underscore_to_dot=False`. This preserves dotted names
  without importing API SDKs. The AST checker and type converters are unchanged.

These scores mean **official scoring code on our selected samples and adapted
JSON output**, not an unqualified reproduction of the full official leaderboard
protocol, native prompts, or all test categories.

## Fixed conversion policy

Both modes use `output_before_eos` when present, otherwise `raw_output`.

- `original`: require syntactically valid JSON in the experiment's array format.
- `json_repair`: repair invalid JSON once with the installed `json-repair`
  version, then use the same adapter. Already valid JSON is unchanged.
- Standard BFCL calls become `{function_name: arguments}` dictionaries for
  the official Python `parallel_multiple` AST checker. Call order and duplicates
  are retained; matching is performed only by the official checker.
- TaskBench flat calls restore the reversible reserved `arg_name` alias and
  recognize dictionary wrappers `arguments/args/parameters/params/inputs/input`.
  Conflicting values are conversion failures. Unknown tools, extra parameters,
  bad parameter types and missing required parameters are NOT removed or filled:
  the official evaluator judges them. This is intentionally different from the
  older convenience converter that filtered fields using the schema.
- DailyLife becomes native named arguments. Predicted temporal edges are only
  derived from exact generated `<node-N>` argument references. No GT-based edges
  or implicit sequential edges are added. Low e-F1 here is partly a limitation
  of the frozen output protocol, not proof that all tools were identified wrongly.
- HuggingFace/Multimedia restore `input_1/input_2/...` to positional values.
  Official resource scoring resolves references to tool names and resource types.
  Forward references allowed by the official evaluator are not rejected.
- Invalid JSON/structure is scored as an empty prediction, never silently dropped.
  Original canonical calls, errors and final official inputs remain auditable.

## Upstream script reliability and denominators

The original TaskBench script catches some errors with `except: pass`; an invalid
resource reference may reuse a previous normalized argument, even one from GT.
The upstream file is NOT patched. A preflight check detects unsupported types,
malformed/out-of-range references, and scores that full resource prediction as
empty. Such requests remain in the denominator. Their name-only projection is
still evaluated separately without inventing or correcting the model's names.

We also found original GT annotations unsupported by the official script:

| Domain | GT issues | Native e/t/v-F1 denominator |
| --- | ---: | ---: |
| DailyLife | 1 | 199 |
| HuggingFace | 2 | 198 |
| Multimedia | 3 | 197 |

Examples include named parameters with `value_from_node` but no `value`, missing
`arguments`, a `<node-Image Classification>` reference rather than a numeric
index, non-string resource arguments, and `<node-0-text>`. No GT is filled or
repaired. Exact sample IDs and original annotations are saved in each domain's
`gt_annotation_errors.json` and the experiment metadata.

`n_precision/n_recall/n_f1` use an explicit name-only projection: both prediction
and GT keep their tool nodes but remove arguments, and then the unmodified
official evaluator computes node F1 on **all 200 queries**. This prevents a GT
parameter annotation problem from invalidating an otherwise scorable node metric.
It still requires the generated JSON structure to be parseable. Unknown names
remain in the official input; the official node metric counts only positive
catalog classes and therefore does not penalize unknown-class names as FPs.
Keep reporting the separate invalid-name and custom trigger metrics.

`n_f1_full_adapter` is the additional node score obtained through the native full
parameter path, on the GT-valid subset; it also reflects whole-output preflight
failures. It must not be confused with the all-200 node-only score.

Every evaluation checks that official node/link/argument support equals the
declared eligible denominator. Any undeclared upstream skip aborts the run.
All model failures within that denominator are preserved. Native parameters and
dependencies are not falsely advertised as having 200 valid GT annotations.

## Output

- `official_metrics.xlsx`: overall, Chain/DAG, definitions and output parity.
- `official_summary.csv`: machine-readable scores and explicit denominators.
- `evaluation_samples.jsonl`: original-output paths, canonical predictions,
  parse/preflight errors and BFCL AST explanations per sample/mode.
- `metadata.json`: source/dependency versions, input hashes and GT annotation issues.
- Domain folders: untouched GT snapshots, selected native GT, name projections,
  official-format predictions, exact official score dictionaries and logs.
- `output_parity.json`: No Cache Natural/OMTP output and score equivalence.
- `sanity_*NOT_A_MODEL*`: GT-as-prediction and empty-prediction integration tests,
  explicitly not experiment/model results. Official four F1 scores must be 1/0.

`normalized_edit_distance` is the source script's `1 - Levenshtein.ratio`, so
lower is better. Its order dependence is mainly meaningful for Chain tasks.
Task decomposition ROUGE/BERTScore and real tool execution success are not
evaluated because the frozen protocol does not produce decomposition text and
does not execute tools.

## Tests

```bash
/home/yzx/miniconda3/envs/llada/bin/python -m pytest -q tests/test_official_evaluation.py
```

On/Off inference timing and OMTP trigger accuracy remain separate metrics.
When a Natural run and an OMTP run have identical final output, their official
final-call scores must be identical; the runner asserts this invariant.
