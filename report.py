"""Aggregate JSONL results into CSV summaries and the requested figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation import set_quality


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_summary(output_dir: str | Path) -> list[dict[str, Any]]:
    output = Path(output_dir)
    samples = _load_jsonl(output / "samples.jsonl")
    if not samples:
        raise ValueError(f"No samples found in {output / 'samples.jsonl'}")

    metric_names = [
        "final_json_valid",
        "tool_call_format_valid",
        "candidate_names_valid",
        "tool_name_precision",
        "tool_name_recall",
        "tool_name_f1",
        "tool_set_exact_match",
        "tool_call_count_match",
        "bfcl_ast_compatible_match",
        "predict_trigger_set_exact",
        "generation_time_s",
        "model_forward_time_s",
        "observer_processing_time_s",
        "observer_output_identical",
        "observer_wall_overhead_ratio",
        "eos_at_last_position",
        "natural_all_detect_time_s",
        "predict_all_detect_time_s",
        "predict_all_available_time_s",
        "oracle_correct_detect_time_s",
        "offline_stable_correct_trigger_time_s",
        "prediction_lead_time_s",
        "actual_prediction_lead_time_s",
        "offline_stable_correct_lead_time_s",
        "available_warmup_window_s",
        "actual_available_warmup_window_s",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        values = [sample.get(metric) for sample in samples]
        present = [value for value in values if value is not None]
        numeric = [float(value) for value in present]
        rows.append(
            {
                "section": "overall",
                "metric": metric,
                "value": mean(numeric) if numeric else None,
                "valid_count": len(numeric),
                "sample_count": len(samples),
            }
        )
    detection_rate_fields = {
        "natural_all_detection_success_rate": "natural_all_detect_time_s",
        "oracle_correct_prediction_success_rate": "oracle_correct_detect_time_s",
        "stable_correct_prediction_success_rate": "offline_stable_correct_trigger_time_s",
    }
    for metric, source_field in detection_rate_fields.items():
        rows.append(
            {
                "section": "overall",
                "metric": metric,
                "value": mean(
                    float(sample.get(source_field) is not None) for sample in samples
                ),
                "valid_count": len(samples),
                "sample_count": len(samples),
            }
        )
    for duration in (1.0, 1.5, 2.0):
        simulation_rows = [
            row
            for sample in samples
            for row in sample.get("warmup_simulation", [])
            if float(row["startup_duration_s"]) == duration
        ]
        for metric in (
            "ready_time_savings_s",
            "natural_ready_by_generation_end",
            "prediction_ready_by_generation_end",
            "wrong_preload_cost_s",
        ):
            values = [row.get(metric) for row in simulation_rows]
            numeric = [float(value) for value in values if value is not None]
            rows.append(
                {
                    "section": f"startup_{duration:g}s",
                    "metric": metric,
                    "value": mean(numeric) if numeric else None,
                    "valid_count": len(numeric),
                    "sample_count": len(samples),
                }
            )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("section", "metric", "value", "valid_count", "sample_count"),
        )
        writer.writeheader()
        writer.writerows(rows)
    _build_figures(output, samples)
    return rows


def _build_figures(output: Path, samples: list[dict[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (figures / "README.txt").write_text(
            "Install matplotlib and rerun `python report.py OUTPUT_DIR` to generate figures.\n",
            encoding="utf-8",
        )
        return

    maximum_forward = max(len(sample.get("steps", [])) for sample in samples)
    forward_indices, f1_values = [], []
    for index in range(maximum_forward):
        values = []
        for sample in samples:
            steps = sample.get("steps", [])
            if index < len(steps):
                values.append(
                    set_quality(
                        steps[index]["predicted_tool_types"],
                        sample["ground_truth_tool_types"],
                    )["f1"]
                )
        if values:
            forward_indices.append(index + 1)
            f1_values.append(mean(values))
    plt.figure(figsize=(7, 4))
    plt.plot(forward_indices, f1_values, marker="o", markersize=3)
    plt.xlabel("Forward index")
    plt.ylabel("Mean tool-type F1")
    plt.title("Intermediate prediction F1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "prediction_f1_by_forward.png", dpi=160)
    plt.close()

    natural = [
        sample["natural_all_detect_time_s"]
        for sample in samples
        if sample.get("natural_all_detect_time_s") is not None
    ]
    predicted = [
        sample["offline_stable_correct_trigger_time_s"]
        for sample in samples
        if sample.get("offline_stable_correct_trigger_time_s") is not None
    ]
    plt.figure(figsize=(7, 4))
    if natural:
        plt.hist(natural, alpha=0.6, label="Natural all", bins="auto")
    if predicted:
        plt.hist(predicted, alpha=0.6, label="Stable prediction (offline-correct)", bins="auto")
    plt.xlabel("Seconds from generation start")
    plt.ylabel("Samples")
    plt.title("Detection-time distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "detection_time_distribution.png", dpi=160)
    plt.close()

    oracle_times = sorted(
        sample["oracle_correct_detect_time_s"]
        for sample in samples
        if sample.get("oracle_correct_detect_time_s") is not None
    )
    plt.figure(figsize=(7, 4))
    if oracle_times:
        fractions = [(index + 1) / len(samples) for index in range(len(oracle_times))]
        plt.step(oracle_times, fractions, where="post")
    plt.xlabel("Seconds from generation start")
    plt.ylabel("Fraction of all samples")
    plt.ylim(0, 1.02)
    plt.title("Cumulative exact tool-set prediction (offline oracle)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "correct_set_cumulative_over_time.png", dpi=160)
    plt.close()

    durations, savings = [], []
    for duration in (1.0, 1.5, 2.0):
        values = [
            row["ready_time_savings_s"]
            for sample in samples
            for row in sample.get("warmup_simulation", [])
            if float(row["startup_duration_s"]) == duration
            and row.get("ready_time_savings_s") is not None
        ]
        durations.append(str(duration))
        savings.append(mean(values) if values else 0.0)
    plt.figure(figsize=(6, 4))
    plt.bar(durations, savings)
    plt.xlabel("Tool startup duration (s)")
    plt.ylabel("Mean ready-time savings (s)")
    plt.title("Simulated tool warmup benefit")
    plt.tight_layout()
    plt.savefig(figures / "warmup_simulation_savings.png", dpi=160)
    plt.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    build_summary(args.output_dir)


if __name__ == "__main__":
    main()
