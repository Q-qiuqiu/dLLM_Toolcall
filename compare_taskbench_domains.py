"""Compare completed independent TaskBench domain runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dailylife", required=True)
parser.add_argument("--huggingface", required=True)
parser.add_argument("--multimedia", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
rows = []
for domain in ("dailylife", "huggingface", "multimedia"):
    root = Path(getattr(args, domain))
    with (root / "taskbench_method_comparison.csv").open() as handle:
        methods = {row["method"]: row for row in csv.DictReader(handle)}
    with (root / "taskbench_generation_summary.csv").open() as handle:
        generation = next(row for row in csv.DictReader(handle) if row["group"] == "overall")
    for method in ("natural_decoding", "omtp_fallback"):
        source = methods[method]
        row = {
            "domain": domain, "method": method,
            "sample_count": source["sample_count"],
            "tool_precision": source["tool_precision"],
            "tool_recall": source["tool_recall"],
            "tool_f1": source["tool_f1"],
            "tool_set_exact_match": source["tool_set_exact_match"],
            "all_tool_types_prefetched": source["all_tool_types_prefetched"],
            "occurrence_f1": source["occurrence_f1"],
            "position_accuracy": source["position_accuracy"],
            "first_correct_time_mean_s": source["oracle_first_correct_prediction_time_s_mean"],
            "all_coverage_time_mean_s": source["oracle_all_correct_coverage_time_s_mean"],
            "paired_coverage_lead_mean_s": source.get("paired_coverage_lead_s_mean", ""),
            "paired_coverage_lead_forwards_mean": source.get("paired_coverage_lead_forwards_mean", ""),
            "original_json_valid_rate": generation["final_json_valid_rate"],
            "repaired_json_valid_rate": generation.get("repaired_json_valid_rate", ""),
            "repaired_output_valid_rate": generation.get("repaired_output_valid_rate", ""),
            "generation_time_mean_s": generation["generation_time_s_mean"],
            "observer_processing_time_mean_s": generation["observer_processing_time_s_mean"],
        }
        rows.append(row)
output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
with (output / "taskbench_cross_domain.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(output / "taskbench_cross_domain.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for method in ("natural_decoding", "omtp_fallback"):
    selected = [row for row in rows if row["method"] == method]
    axes[0].plot([r["domain"] for r in selected], [float(r["tool_f1"]) for r in selected], marker="o", label=method)
    axes[1].plot([r["domain"] for r in selected], [float(r["all_tool_types_prefetched"]) for r in selected], marker="o", label=method)
axes[0].set_title("Tool F1"); axes[1].set_title("Complete type coverage")
for axis in axes: axis.set_ylim(0, 1.05); axis.legend(); axis.tick_params(axis="x", rotation=15)
figure.tight_layout(); figure.savefig(output / "taskbench_cross_domain.png", dpi=180)
print(output.resolve())
