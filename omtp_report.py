"""Aggregate OMTP sample records and build the requested comparison figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from omtp_evaluation import METHODS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def stats(values: list[Any]) -> dict[str, Any]:
    clean = [float(x) for x in values if x is not None]
    return {
        "mean": mean(clean) if clean else None,
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
        "valid_count": len(clean),
    }


def build_omtp_report(output_dir: str | Path) -> list[dict[str, Any]]:
    output = Path(output_dir)
    samples = load_jsonl(output / "omtp_samples.jsonl")
    if not samples:
        raise ValueError("no OMTP samples to summarize")
    summary_rows = []
    comparison_rows = []
    for reference, key in (("final", "vs_final"), ("ground_truth", "vs_ground_truth")):
        for method in METHODS:
            results = [sample["method_comparison"][method][key] for sample in samples]
            row = {"reference": reference, "method": method, "sample_count": len(samples)}
            for metric in (
                "tool_precision", "tool_recall", "tool_f1",
                "tool_set_exact_match", "false_positive_count",
                "correct_trigger_coverage",
            ):
                row[metric] = mean(float(result[metric]) for result in results)
            for metric in (
                "online_first_trigger_time_s",
                "oracle_first_correct_prediction_time_s",
                "oracle_all_correct_coverage_time_s",
                "oracle_exact_set_time_s",
                "paired_exact_set_lead_vs_natural_s",
            ):
                values = stats([result[metric] for result in results])
                for stat, value in values.items():
                    row[f"{metric}_{stat}"] = value
            row["untriggered_rate"] = mean(
                float(result["online_first_trigger_time_s"] is None)
                for result in results
            )
            row["exact_set_unreached_rate"] = mean(
                float(result["oracle_exact_set_time_s"] is None)
                for result in results
            )
            comparison_rows.append(row)
            for metric, value in row.items():
                if metric in ("reference", "method", "sample_count"):
                    continue
                summary_rows.append({
                    "section": reference,
                    "method": method,
                    "metric": metric,
                    "value": value,
                    "sample_count": len(samples),
                })

    overhead_fields = (
        "observer_processing_time_s", "observer_mean_processing_time_s",
        "omtp_processing_time_s",
        "generation_time_s", "observer_off_generation_time_s",
        "observer_wall_overhead_ratio", "observer_output_identical",
        "observer_nfe_identical",
    )
    for field in overhead_fields:
        values = stats([sample.get(field) for sample in samples])
        for stat, value in values.items():
            summary_rows.append({
                "section": "observer", "method": "omtp", "metric": f"{field}_{stat}",
                "value": value, "sample_count": len(samples),
            })
    required_tool_rows = [
        row
        for sample in samples
        for row in sample.get("per_tool_predictions", [])
        if row.get("tool") in set(sample.get("ground_truth_tools", []))
    ]
    for field in ("T_latent", "T_prefix", "T_natural", "T_online", "latent_lead_vs_natural_s"):
        values = stats([row.get(field) for row in required_tool_rows])
        for stat, value in values.items():
            summary_rows.append({
                "section": "required_tool_timing", "method": "omtp",
                "metric": f"{field}_{stat}", "value": value,
                "sample_count": len(samples),
            })
        summary_rows.append({
            "section": "required_tool_timing", "method": "omtp",
            "metric": f"{field}_missing_rate",
            "value": (
                mean(float(row.get(field) is None) for row in required_tool_rows)
                if required_tool_rows else None
            ),
            "sample_count": len(samples),
        })
    for reference, key in (("final", "vs_final"), ("ground_truth", "vs_ground_truth")):
        exact_advantages = []
        trigger_advantages = []
        for sample in samples:
            baseline = sample["method_comparison"]["whole_set_argmax_stable_n2"][key]
            omtp = sample["method_comparison"]["omtp_latent_region_n2"][key]
            if baseline["oracle_exact_set_time_s"] is not None and omtp["oracle_exact_set_time_s"] is not None:
                exact_advantages.append(
                    float(baseline["oracle_exact_set_time_s"])
                    - float(omtp["oracle_exact_set_time_s"])
                )
            if baseline["online_first_trigger_time_s"] is not None and omtp["online_first_trigger_time_s"] is not None:
                trigger_advantages.append(
                    float(baseline["online_first_trigger_time_s"])
                    - float(omtp["online_first_trigger_time_s"])
                )
        paired = {
            "reference": reference,
            "method": "paired_omtp_vs_whole_set_argmax_stable_n2",
            "sample_count": len(samples),
            "omtp_tool_f1_delta": mean(
                sample["method_comparison"]["omtp_latent_region_n2"][key]["tool_f1"]
                - sample["method_comparison"]["whole_set_argmax_stable_n2"][key]["tool_f1"]
                for sample in samples
            ),
            "omtp_tool_set_exact_match_delta": mean(
                float(sample["method_comparison"]["omtp_latent_region_n2"][key]["tool_set_exact_match"])
                - float(sample["method_comparison"]["whole_set_argmax_stable_n2"][key]["tool_set_exact_match"])
                for sample in samples
            ),
        }
        for prefix, values in (("paired_exact_time_advantage_s", exact_advantages), ("paired_first_trigger_advantage_s", trigger_advantages)):
            for stat, value in stats(values).items():
                paired[f"{prefix}_{stat}"] = value
        comparison_rows.append(paired)
    with (output / "omtp_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "method", "metric", "value", "sample_count"))
        writer.writeheader()
        writer.writerows(summary_rows)
    fields = sorted({key for row in comparison_rows for key in row})
    with (output / "omtp_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    profile_names = sorted({
        name
        for sample in samples
        for name in sample.get("observer_profile", {}).get("modules", {})
    })
    if profile_names:
        total_forwards = sum(
            int(sample.get("observer_profile", {}).get("forward_count") or 0)
            for sample in samples
        )
        profile_rows = []
        for name in profile_names:
            total = sum(
                float(sample.get("observer_profile", {}).get("modules", {}).get(name, {}).get("total_s") or 0)
                for sample in samples
            )
            calls = sum(
                int(sample.get("observer_profile", {}).get("modules", {}).get(name, {}).get("calls") or 0)
                for sample in samples
            )
            profile_rows.append({
                "module": name, "total_s": total,
                "mean_per_forward_s": total / total_forwards if total_forwards else None,
                "mean_per_call_s": total / calls if calls else None,
                "calls": calls, "forward_count": total_forwards,
                "gpu_to_cpu_bytes": "", "cpu_to_gpu_bytes": "",
            })
        transfer = {
            direction: sum(
                int(sample.get("observer_profile", {}).get("transfer_bytes", {}).get(direction) or 0)
                for sample in samples
            )
            for direction in ("gpu_to_cpu", "cpu_to_gpu")
        }
        profile_rows.append({
            "module": "recorded_tensor_transfers", "total_s": "",
            "mean_per_forward_s": "", "mean_per_call_s": "", "calls": "",
            "forward_count": total_forwards,
            "gpu_to_cpu_bytes": transfer["gpu_to_cpu"],
            "cpu_to_gpu_bytes": transfer["cpu_to_gpu"],
        })
        with (output / "omtp_profile.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=profile_rows[0].keys())
            writer.writeheader(); writer.writerows(profile_rows)
    _figures(output, samples, comparison_rows)
    return comparison_rows


def _figures(output: Path, samples: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (directory / "README.txt").write_text("Install matplotlib to generate OMTP figures.\n", encoding="utf-8")
        return

    ground = [
        row for row in rows
        if row["reference"] == "ground_truth" and row["method"] in METHODS
    ]
    labels = [row["method"] for row in ground]
    x = list(range(len(labels)))
    width = 0.25
    plt.figure(figsize=(11, 5))
    for offset, metric in ((-width, "tool_precision"), (0, "tool_recall"), (width, "tool_f1")):
        plt.bar([v + offset for v in x], [row[metric] for row in ground], width, label=metric.replace("tool_", ""))
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()
    plt.savefig(directory / "method_precision_recall_f1.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 4))
    for method, label in (("whole_set_argmax_stable_n2", "Argmax Stable N=2"), ("omtp_latent_region_n2", "OMTP")):
        values = [sample["method_comparison"][method]["vs_ground_truth"]["paired_exact_set_lead_vs_natural_s"] for sample in samples]
        values = [x for x in values if x is not None]
        if values:
            plt.hist(values, bins="auto", alpha=0.55, label=label)
    plt.xlabel("Lead vs natural (s)"); plt.ylabel("Samples"); plt.legend(); plt.tight_layout()
    plt.savefig(directory / "omtp_vs_argmax_lead_distribution.png", dpi=160); plt.close()

    maximum = max((sample.get("forward_count") or 0) for sample in samples)
    curves = {"per_tool_argmax_stable_n2": [], "omtp_latent_region_n2": []}
    for forward in range(1, maximum + 1):
        for method in curves:
            coverages = []
            for sample in samples:
                expected = set(sample["ground_truth_tools"])
                predicted = set()
                for event in sample["method_comparison"][method]["events"]:
                    if (event.get("forward_step") or 0) <= forward:
                        predicted.update(event.get("predicted_tools") or [event.get("predicted_tool")])
                coverages.append(len(predicted & expected) / len(expected) if expected else 0.0)
            curves[method].append(mean(coverages))
    plt.figure(figsize=(7, 4))
    for method, values in curves.items():
        plt.plot(range(1, maximum + 1), values, label=method)
    plt.xlabel("Forward"); plt.ylabel("Correct-tool coverage"); plt.ylim(0, 1.02); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(directory / "correct_tool_coverage_by_forward.png", dpi=160); plt.close()

    counts = {source: 0 for source in ("latent_region", "prefix", "natural")}
    for sample in samples:
        for event in sample["method_comparison"]["omtp_with_fallback"]["events"]:
            counts[event["source"]] = counts.get(event["source"], 0) + 1
    plt.figure(figsize=(6, 4)); plt.bar(list(counts), list(counts.values())); plt.ylabel("Committed occurrences"); plt.tight_layout()
    plt.savefig(directory / "omtp_event_source_distribution.png", dpi=160); plt.close()

    observed = [sample["generation_time_s"] for sample in samples if sample.get("generation_time_s") is not None]
    baseline = [sample["observer_off_generation_time_s"] for sample in samples if sample.get("observer_off_generation_time_s") is not None]
    processing = [sample["observer_processing_time_s"] for sample in samples if sample.get("observer_processing_time_s") is not None]
    plt.figure(figsize=(6, 4)); plt.bar(("observer off", "observer on", "observer processing"), (mean(baseline) if baseline else 0, mean(observed) if observed else 0, mean(processing) if processing else 0)); plt.ylabel("Seconds"); plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(directory / "observer_overhead.png", dpi=160); plt.close()


__all__ = ["build_omtp_report", "stats"]
