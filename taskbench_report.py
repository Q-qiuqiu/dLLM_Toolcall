"""Aggregate TaskBench Natural/latent/fallback results and draw comparisons."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from taskbench_evaluation import FALLBACK, LATENT, METHODS, NATURAL


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
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] if lo == hi else (
        ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
    )


def stats(values: list[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "mean": mean(clean) if clean else None,
        "p50": percentile(clean, 0.5),
        "p95": percentile(clean, 0.95),
        "valid_count": len(clean),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _method_row(samples: list[dict[str, Any]], method: str) -> dict[str, Any]:
    results = [
        sample["taskbench_methods"][method]["vs_ground_truth"]
        for sample in samples
    ]
    row: dict[str, Any] = {"method": method, "sample_count": len(samples)}
    for metric in (
        "tool_precision", "tool_recall", "tool_f1", "tool_set_exact_match",
        "all_tool_types_prefetched", "false_positive_count",
        "false_negative_count", "occurrence_precision", "occurrence_recall",
        "occurrence_f1", "all_tool_occurrences_prefetched",
        "tool_occurrence_multiset_exact_match", "position_accuracy",
        "position_exact_match",
    ):
        row[metric] = mean(float(result[metric]) for result in results)
    for metric in (
        "oracle_first_correct_prediction_time_s",
        "oracle_all_correct_coverage_time_s", "oracle_exact_set_time_s",
        "oracle_all_occurrences_covered_time_s",
    ):
        values = [result.get(metric) for result in results]
        for name, value in stats(values).items():
            row[f"{metric}_{name}"] = value
        row[f"{metric}_missing_rate"] = mean(
            float(value is None) for value in values
        )

    if method != NATURAL:
        natural = [
            sample["taskbench_methods"][NATURAL]["vs_ground_truth"]
            for sample in samples
        ]
        current_times = [r["oracle_all_correct_coverage_time_s"] for r in results]
        natural_times = [r["oracle_all_correct_coverage_time_s"] for r in natural]
        leads = [
            float(n) - float(c)
            for c, n in zip(current_times, natural_times)
            if c is not None and n is not None
        ]
        for name, value in stats(leads).items():
            row[f"paired_coverage_lead_s_{name}"] = value
        row.update({
            "paired_both_success": sum(
                c is not None and n is not None
                for c, n in zip(current_times, natural_times)
            ),
            "method_only_success": sum(
                c is not None and n is None
                for c, n in zip(current_times, natural_times)
            ),
            "natural_only_success": sum(
                c is None and n is not None
                for c, n in zip(current_times, natural_times)
            ),
            "both_failed": sum(
                c is None and n is None
                for c, n in zip(current_times, natural_times)
            ),
            "paired_wins": sum(value > 0 for value in leads),
            "paired_ties": sum(abs(value) <= 1e-12 for value in leads),
            "paired_losses": sum(value < 0 for value in leads),
        })
    return row


def _groups(samples: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups = [("overall", samples)]
    for graph_type in ("chain", "dag"):
        groups.append((graph_type, [s for s in samples if s["graph_type"] == graph_type]))
    for key, prefix in (
        ("ground_truth_call_count", "call_count"),
        ("ground_truth_tool_type_count", "tool_type_count"),
    ):
        for value in sorted({int(s[key]) for s in samples}):
            groups.append((
                f"{prefix}_{value}", [s for s in samples if int(s[key]) == value]
            ))
    return [(name, rows) for name, rows in groups if rows]


def _position_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group, members in (
        ("overall", samples),
        ("chain", [s for s in samples if s["graph_type"] == "chain"]),
        ("dag", [s for s in samples if s["graph_type"] == "dag"]),
    ):
        for method in METHODS:
            for position in range(1, 5):
                records = []
                for sample in members:
                    records.extend(
                        row for row in sample["taskbench_methods"][method]
                        ["vs_ground_truth"]["position_records"]
                        if row["position"] == position and row["expected_tool"] is not None
                    )
                if not records:
                    continue
                correct_times = [
                    row["predict_time_s"] for row in records if row["correct"]
                ]
                value = {
                    "group": group, "method": method, "position": position,
                    "case_count": len(records),
                    "correct_count": sum(bool(row["correct"]) for row in records),
                    "position_accuracy": mean(float(row["correct"]) for row in records),
                    "correct_detection_rate": len(correct_times) / len(records),
                }
                for name, stat in stats(correct_times).items():
                    value[f"correct_time_s_{name}"] = stat
                rows.append(value)
    return rows


def _generation_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group, members in (
        ("overall", samples),
        ("chain", [s for s in samples if s["graph_type"] == "chain"]),
        ("dag", [s for s in samples if s["graph_type"] == "dag"]),
    ):
        if not members:
            continue
        row: dict[str, Any] = {
            "group": group, "sample_count": len(members),
            "final_json_valid_rate": mean(float(s["final_json_valid"]) for s in members),
            "final_strict_tool_set_exact_rate": mean(float(s["tool_set_exact_match"]) for s in members),
            "final_raw_tool_set_exact_rate": mean(float(s["final_raw_tool_set_exact_match"]) for s in members),
            "repaired_json_valid_rate": mean(float(s.get("repaired_json_valid", False)) for s in members),
            "repaired_tool_call_format_valid_rate": mean(float(s.get("repaired_tool_call_format_valid", False)) for s in members),
            "repaired_output_valid_rate": mean(float(s.get("repaired_output_valid", False)) for s in members),
            "repaired_tool_set_exact_rate": mean(float(s.get("repaired_tool_set_exact_match", False)) for s in members),
            "repaired_tool_type_f1": mean(float(s.get("repaired_tool_type_f1", 0.0)) for s in members),
            "final_call_count_match_rate": mean(float(s["tool_call_count_match"]) for s in members),
            "likely_length_truncated_count": sum(bool(s["likely_length_truncated"]) for s in members),
            "runtime_failure_count": sum(bool(s.get("runtime_error")) for s in members),
            "expected_nfe": (
                members[0].get("expected_forward_count", 32)
                if len({s.get("expected_forward_count", 32) for s in members}) == 1
                else "mixed"
            ),
            "nfe_expected_rate": mean(
                float(s.get("forward_count") == s.get("expected_forward_count", 32))
                for s in members
            ),
            "parity_checked_count": sum(s.get("observer_output_identical") is not None for s in members),
            "parity_output_identical_rate": (
                mean(float(s["observer_output_identical"]) for s in members
                     if s.get("observer_output_identical") is not None)
                if any(s.get("observer_output_identical") is not None for s in members)
                else None
            ),
            "parity_nfe_identical_rate": (
                mean(float(s["observer_nfe_identical"]) for s in members
                     if s.get("observer_nfe_identical") is not None)
                if any(s.get("observer_nfe_identical") is not None for s in members)
                else None
            ),
        }
        for field in (
            "generation_time_s", "observer_processing_time_s",
            "observer_mean_processing_time_s", "observer_off_generation_time_s",
            "observer_wall_overhead_ratio", "prompt_token_count",
        ):
            for name, value in stats([s.get(field) for s in members]).items():
                row[f"{field}_{name}"] = value
        rows.append(row)
    return rows


def _figures(output: Path, methods: list[dict[str, Any]],
             positions: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    labels = [row["method"] for row in methods]
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, metric in enumerate(("tool_precision", "tool_recall", "tool_f1")):
        ax.bar([i + (offset - 1) * width for i in x],
               [row[metric] for row in methods], width, label=metric)
    ax.set_xticks(list(x), labels, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "method_accuracy.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in (LATENT, FALLBACK):
        leads = []
        for sample in samples:
            m = sample["taskbench_methods"][method]["vs_ground_truth"]["oracle_all_correct_coverage_time_s"]
            n = sample["taskbench_methods"][NATURAL]["vs_ground_truth"]["oracle_all_correct_coverage_time_s"]
            if m is not None and n is not None: leads.append(float(n) - float(m))
        if leads: ax.hist(leads, bins=min(20, max(5, len(leads) // 5)), alpha=.55, label=method)
    ax.axvline(0, color="black", linewidth=1); ax.set_xlabel("lead over Natural (s)")
    ax.set_ylabel("paired samples"); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "paired_coverage_lead.png", dpi=180); plt.close(fig)

    overall = [r for r in positions if r["group"] == "overall"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in METHODS:
        rows = sorted((r for r in overall if r["method"] == method), key=lambda r:r["position"])
        ax.plot(
            [r["position"] for r in rows],
            [r["correct_time_s_mean"] if r["correct_time_s_mean"] is not None else float("nan") for r in rows],
            marker="o", label=method,
        )
    ax.set_xticks([1,2,3,4]); ax.set_xlabel("ground-truth call position"); ax.set_ylabel("correct trigger time (s)")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "position_1_4_timing.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in METHODS:
        vals=[]
        for graph in ("chain", "dag"):
            members=[s for s in samples if s["graph_type"]==graph]
            vals.append(
                _method_row(members, method)["tool_f1"]
                if members else float("nan")
            )
        ax.plot(("chain","dag"), vals, marker="o", label=method)
    ax.set_ylim(0,1.05); ax.set_ylabel("Tool F1"); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "chain_dag_f1.png", dpi=180); plt.close(fig)

    generation = _generation_rows(samples)[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    timing_labels = ("generation", "observer")
    timing_values = (
        generation["generation_time_s_mean"] or 0.0,
        generation["observer_processing_time_s_mean"] or 0.0,
    )
    axes[0].bar(timing_labels, timing_values, color=("#4c78a8", "#f58518"))
    axes[0].set_ylabel("mean seconds / request")
    quality_labels = ("JSON valid", "strict set EM", "raw set EM")
    quality_values = (
        generation["final_json_valid_rate"],
        generation["final_strict_tool_set_exact_rate"],
        generation["final_raw_tool_set_exact_rate"],
    )
    axes[1].bar(quality_labels, quality_values, color=("#54a24b", "#e45756", "#72b7b2"))
    axes[1].set_ylim(0, 1.05); axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylabel("rate")
    fig.tight_layout()
    fig.savefig(figures / "observer_and_generation_quality.png", dpi=180)
    plt.close(fig)


def build_taskbench_report(output_dir: str | Path) -> None:
    output = Path(output_dir)
    samples = load_jsonl(output / "taskbench_samples.jsonl")
    if not samples:
        raise ValueError("no TaskBench samples to summarize")
    method_rows = [_method_row(samples, method) for method in METHODS]
    group_rows = []
    for group, members in _groups(samples):
        for method in METHODS:
            group_rows.append({"group": group, **_method_row(members, method)})
    position_rows = _position_rows(samples)
    generation_rows = _generation_rows(samples)
    _write_csv(output / "taskbench_method_comparison.csv", method_rows)
    _write_csv(output / "taskbench_group_summary.csv", group_rows)
    _write_csv(output / "taskbench_position_summary.csv", position_rows)
    _write_csv(output / "taskbench_generation_summary.csv", generation_rows)
    _figures(output, method_rows, position_rows, samples)


__all__ = ["build_taskbench_report", "load_jsonl", "percentile", "stats"]
