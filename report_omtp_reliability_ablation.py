"""Report both GT accuracy and final-generation fidelity for real ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from omtp_evaluation import evaluate_events
from run_omtp_reliability_ablation import load_rows
from taskbench_report import stats


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def committed_field_events(sample):
    """Reuse the already timed lightweight scan, independently of OMTP tracks."""

    seen = set()
    events = []
    for step in sample["steps"]:
        for tool in step["committed_tool_types"]:
            if tool in seen:
                continue
            seen.add(tool)
            events.append({
                "predicted_tool": tool,
                "forward_step": step["forward_index"],
                "predict_time_s": step["prediction_available_time_s"],
                "source": "natural_committed_fields",
            })
    return events


def build_report(root):
    root = Path(root)
    comparison_path = root / "reference_comparison.csv"
    timing_path = root / "paired_unique_tool_timing.csv"
    if comparison_path.exists() or timing_path.exists():
        raise FileExistsError("refusing to overwrite reliability reports")
    ablation = (root / "ablation_decisions.jsonl").is_file()
    decisions = (
        load_rows(root / "ablation_decisions.jsonl") if ablation
        else [{"stage": ".", "accepted": True}]
    )
    comparisons = []
    timings = []
    baseline_samples = {
        row["sample_id"]: row
        for row in load_rows(
            (root / "00_baseline" if ablation else root) / "taskbench_samples.jsonl"
        )
    }
    for decision in decisions:
        stage = decision["stage"]
        samples = load_rows(root / stage / "taskbench_samples.jsonl")
        for reference in ("ground_truth", "final_raw_text"):
            expected = [
                set(sample["ground_truth_tools"] if reference == "ground_truth"
                    else sample["final_raw_tool_types"])
                for sample in samples
            ]
            streams = {
                "natural": [sample["natural_baseline"]["events"] for sample in samples],
                "natural_committed_fields": [committed_field_events(sample) for sample in samples],
                "omtp_fallback": [sample["omtp_fallback_events"] for sample in samples],
            }
            evaluated = {}
            for method, events in streams.items():
                values = [evaluate_events(stream, truth) for stream, truth in zip(events, expected)]
                evaluated[method] = values
                fp = sum(value["false_positive_count"] for value in values)
                fn = sum(value["false_negative_count"] for value in values)
                tp = sum(len(set(value["predicted_tools"]) & truth) for value, truth in zip(values, expected))
                row = {
                    "stage": stage, "accepted": decision["accepted"],
                    "reference": reference, "method": method,
                    "sample_count": len(samples),
                    "precision_macro": mean(v["tool_precision"] for v in values),
                    "recall_macro": mean(v["tool_recall"] for v in values),
                    "f1_macro": mean(v["tool_f1"] for v in values),
                    "precision_micro": tp / (tp + fp) if tp + fp else 0.,
                    "recall_micro": tp / (tp + fn) if tp + fn else 0.,
                    "set_exact_match": mean(v["tool_set_exact_match"] for v in values),
                    "all_required_types_covered": mean(v["all_tool_types_prefetched"] for v in values),
                    "true_positive_types": tp, "false_positive_types": fp,
                    "false_negative_types": fn,
                    "tasks_with_wrong_trigger": sum(v["false_positive_count"] > 0 for v in values),
                }
                if reference == "ground_truth":
                    row["wrong_types_also_in_final_raw_text"] = sum(
                        len(set(value["false_positives"]) & set(sample["final_raw_tool_types"]))
                        for value, sample in zip(values, samples)
                    )
                    row["wrong_types_absent_from_final_raw_text"] = fp - row["wrong_types_also_in_final_raw_text"]
                comparisons.append(row)
            natural_times = []; online_times = []; forward_leads = []
            common_forward_delays = []
            both = natural_only = online_only = neither = 0
            for sample, truth, natural, online in zip(samples, expected, evaluated["natural_committed_fields"], evaluated["omtp_fallback"]):
                baseline = evaluate_events(
                    baseline_samples[sample["sample_id"]]["omtp_fallback_events"], truth
                )
                for tool in sorted(truth):
                    n = natural["first_times_by_tool"].get(tool)
                    o = online["first_times_by_tool"].get(tool)
                    if n is not None and o is not None:
                        both += 1; natural_times.append(n); online_times.append(o)
                        forward_leads.append(natural["first_forwards_by_tool"][tool] - online["first_forwards_by_tool"][tool])
                        if tool in baseline["first_forwards_by_tool"]:
                            common_forward_delays.append(
                                online["first_forwards_by_tool"][tool]
                                - baseline["first_forwards_by_tool"][tool]
                            )
                    elif n is not None: natural_only += 1
                    elif o is not None: online_only += 1
                    else: neither += 1
            leads = [n - o for n, o in zip(natural_times, online_times)]
            row = {
                "stage": stage, "reference": reference,
                "natural_baseline": "natural_committed_fields",
                "total_required_task_tool_pairs": sum(map(len, expected)),
                "both_detected": both, "natural_only_detected": natural_only,
                "omtp_only_detected": online_only, "both_missed": neither,
                "paired_wins": sum(value > 0 for value in leads),
                "paired_ties": sum(abs(value) <= 1e-12 for value in leads),
                "paired_losses": sum(value < 0 for value in leads),
                "paired_latency_ratio_of_means": (
                    mean(natural_times) / mean(online_times) if online_times else None
                ),
            }
            for prefix, values in (
                ("natural_time_s", natural_times), ("omtp_time_s", online_times),
                ("lead_s", leads), ("lead_forwards", forward_leads),
                ("prediction_forward_delay_vs_original_on_common_pairs", common_forward_delays),
            ):
                row.update({f"{prefix}_{key}": value for key, value in stats(values).items()})
            timings.append(row)
    _write_csv(comparison_path, comparisons)
    _write_csv(timing_path, timings)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rows = [r for r in comparisons if r["method"] == "omtp_fallback" and r["reference"] == "ground_truth"]
    for offset, key in enumerate(("precision_macro", "recall_macro", "f1_macro")):
        ax.bar([i + (offset - 1) * .24 for i in range(len(rows))], [r[key] for r in rows], width=.24, label=key)
    ax.set_xticks(range(len(rows)), [r["stage"] for r in rows], rotation=20, ha="right")
    ax.axhline(.9, color="grey", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Tool-type score vs GT (dev20)"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "reliability_accuracy.png", dpi=180); plt.close(fig)
    print(f"Reference and paired unique-tool timing reports written under {root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    build_report(parser.parse_args().result_root)
