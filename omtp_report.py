"""Aggregate fixed OMTP + fallback records and build figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation import set_quality
from omtp_evaluation import METHOD_NAME, NATURAL_BASELINE_NAME
from tool_parser import extract_partial_names


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
    clean = [float(value) for value in values if value is not None]
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
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for reference, key in (
        ("final", "vs_final"),
        ("ground_truth", "vs_ground_truth"),
    ):
        for method, record_key in (
            (METHOD_NAME, "omtp_evaluation"),
            (NATURAL_BASELINE_NAME, "natural_baseline"),
        ):
            results = [sample[record_key][key] for sample in samples]
            row = {
                "reference": reference,
                "method": method,
                "sample_count": len(samples),
            }
            for metric in (
                "tool_precision", "tool_recall", "tool_f1",
                "tool_set_exact_match", "false_positive_count",
                "false_negative_count", "correct_trigger_coverage",
                "all_tool_types_prefetched",
                "occurrence_precision", "occurrence_recall", "occurrence_f1",
                "all_tool_occurrences_prefetched",
                "tool_occurrence_count_match",
                "tool_occurrence_multiset_exact_match",
                "position_accuracy", "position_exact_match",
            ):
                row[metric] = mean(float(result[metric]) for result in results)
            for metric in (
                "online_first_trigger_time_s",
                "oracle_first_correct_prediction_time_s",
                "oracle_all_correct_coverage_time_s",
                "oracle_exact_set_time_s",
                "natural_exact_set_time_s",
                "paired_exact_set_lead_vs_natural_s",
                "oracle_all_occurrences_covered_time_s",
                "oracle_exact_occurrence_multiset_time_s",
                "natural_all_occurrences_covered_time_s",
                "paired_all_occurrences_lead_vs_natural_s",
            ):
                for stat, value in stats(
                    [result.get(metric) for result in results]
                ).items():
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

    for field in (
        "observer_processing_time_s", "observer_mean_processing_time_s",
        "omtp_processing_time_s", "generation_time_s",
        "observer_off_generation_time_s", "observer_wall_overhead_ratio",
        "observer_output_identical", "observer_nfe_identical",
    ):
        for stat, value in stats(
            [sample.get(field) for sample in samples]
        ).items():
            summary_rows.append({
                "section": "observer", "method": METHOD_NAME,
                "metric": f"{field}_{stat}", "value": value,
                "sample_count": len(samples),
            })

    required_tools = [
        row
        for sample in samples
        for row in sample.get("per_tool_predictions", [])
        if row.get("tool") in set(sample.get("ground_truth_tools", []))
    ]
    for field in (
        "T_latent", "T_prefix", "T_natural", "T_online",
        "latent_lead_vs_natural_s", "online_lead_vs_natural_s",
    ):
        for stat, value in stats(
            [row.get(field) for row in required_tools]
        ).items():
            summary_rows.append({
                "section": "required_tool_timing", "method": METHOD_NAME,
                "metric": f"{field}_{stat}", "value": value,
                "sample_count": len(samples),
            })
        summary_rows.append({
            "section": "required_tool_timing", "method": METHOD_NAME,
            "metric": f"{field}_missing_rate",
            "value": (
                mean(float(row.get(field) is None) for row in required_tools)
                if required_tools else None
            ),
            "sample_count": len(samples),
        })

    with (output / "omtp_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("section", "method", "metric", "value", "sample_count"),
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    fields = sorted({key for row in comparison_rows for key in row})
    with (output / "omtp_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    _write_profile(output, samples)
    _write_position_summary(output, samples)
    _write_diagnostics(output, samples)
    _figures(output, samples, comparison_rows)
    return comparison_rows


def _write_profile(output: Path, samples: list[dict[str, Any]]) -> None:
    names = sorted({
        name for sample in samples
        for name in sample.get("observer_profile", {}).get("modules", {})
    })
    if not names:
        return
    forwards = sum(
        int(sample.get("observer_profile", {}).get("forward_count") or 0)
        for sample in samples
    )
    rows = []
    for name in names:
        total = sum(
            float(sample.get("observer_profile", {}).get("modules", {})
                  .get(name, {}).get("total_s") or 0)
            for sample in samples
        )
        calls = sum(
            int(sample.get("observer_profile", {}).get("modules", {})
                .get(name, {}).get("calls") or 0)
            for sample in samples
        )
        rows.append({
            "module": name, "total_s": total,
            "mean_per_forward_s": total / forwards if forwards else None,
            "mean_per_call_s": total / calls if calls else None,
            "calls": calls, "forward_count": forwards,
            "gpu_to_cpu_bytes": "", "cpu_to_gpu_bytes": "",
        })
    transfers = {
        direction: sum(
            int(sample.get("observer_profile", {}).get("transfer_bytes", {})
                .get(direction) or 0)
            for sample in samples
        )
        for direction in ("gpu_to_cpu", "cpu_to_gpu")
    }
    rows.append({
        "module": "recorded_tensor_transfers", "total_s": "",
        "mean_per_forward_s": "", "mean_per_call_s": "", "calls": "",
        "forward_count": forwards,
        "gpu_to_cpu_bytes": transfers["gpu_to_cpu"],
        "cpu_to_gpu_bytes": transfers["cpu_to_gpu"],
    })
    with (output / "omtp_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_position_summary(
    output: Path, samples: list[dict[str, Any]]
) -> None:
    rows = []
    for reference in ("vs_final", "vs_ground_truth"):
        positions = sorted({
            int(row["position"])
            for sample in samples
            for row in sample.get("position_comparison", {}).get(reference, [])
            if row.get("expected_tool") is not None
        })
        for position in positions:
            records = [
                row
                for sample in samples
                for row in sample.get("position_comparison", {}).get(reference, [])
                if int(row["position"]) == position
                and row.get("expected_tool") is not None
            ]
            row: dict[str, Any] = {
                "reference": reference,
                "position": position,
                "case_count": len(records),
                "omtp_prediction_rate": mean(
                    float(item.get("omtp_predicted_tool") is not None)
                    for item in records
                ),
                "omtp_position_accuracy": mean(
                    float(item.get("omtp_correct", False)) for item in records
                ),
                "natural_detection_rate": mean(
                    float(item.get("natural_predicted_tool") is not None)
                    for item in records
                ),
                "natural_position_accuracy": mean(
                    float(item.get("natural_correct", False)) for item in records
                ),
            }
            metric_values = {
                "omtp_time_s": [
                    item.get("omtp_predict_time_s") for item in records
                ],
                "omtp_correct_time_s": [
                    item.get("omtp_predict_time_s")
                    for item in records if item.get("omtp_correct")
                ],
                "natural_time_s": [
                    item.get("natural_detect_time_s") for item in records
                ],
                "natural_correct_time_s": [
                    item.get("natural_detect_time_s")
                    for item in records if item.get("natural_correct")
                ],
                "paired_correct_lead_s": [
                    item.get("lead_vs_natural_s")
                    for item in records
                    if item.get("omtp_correct") and item.get("natural_correct")
                ],
            }
            for metric, values in metric_values.items():
                for stat, value in stats(values).items():
                    row[f"{metric}_{stat}"] = value
            rows.append(row)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (output / "omtp_position_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_diagnostics(
    output: Path, samples: list[dict[str, Any]]
) -> None:
    """Write offline diagnostics without changing generation-time accounting."""

    rows: list[dict[str, Any]] = []
    for method, key in (
        (METHOD_NAME, "omtp_evaluation"),
        (NATURAL_BASELINE_NAME, "natural_baseline"),
    ):
        false_counts = [
            int(sample[key]["vs_ground_truth"]["false_positive_count"])
            for sample in samples
        ]
        rows.append({
            "section": "deduplicated_wrong_tool_triggers",
            "label": method,
            "sample_count": len(samples),
            "mean_wrong_tool_types_per_task": mean(false_counts),
            "tasks_with_wrong_tool_trigger": sum(
                count > 0 for count in false_counts
            ),
            "tasks_with_wrong_tool_trigger_rate": mean(
                float(count > 0) for count in false_counts
            ),
            "total_wrong_tool_types": sum(false_counts),
            "max_wrong_tool_types_per_task": max(false_counts, default=0),
        })

    final_sets = {"raw_text": [], "strict_json": []}
    for sample in samples:
        candidates = sample.get("candidate_tools", [])
        final_sets["raw_text"].append(set(extract_partial_names(
            sample.get("raw_output", ""), candidates
        )))
        final_sets["strict_json"].append({
            str(call["name"])
            for call in sample.get("generated_tool_calls", [])
        })
    for label, predicted_sets in final_sets.items():
        qualities = [
            set_quality(predicted, set(sample["ground_truth_tools"]))
            for predicted, sample in zip(predicted_sets, samples)
        ]
        rows.append({
            "section": "final_tool_set_quality",
            "label": label,
            "sample_count": len(samples),
            "tool_set_exact_match": mean(
                float(quality["exact"]) for quality in qualities
            ),
            "all_tool_types_covered": mean(
                float(set(sample["ground_truth_tools"]) <= predicted)
                for predicted, sample in zip(predicted_sets, samples)
            ),
            "tool_precision": mean(
                float(quality["precision"]) for quality in qualities
            ),
            "tool_recall": mean(
                float(quality["recall"]) for quality in qualities
            ),
            "tool_f1": mean(float(quality["f1"]) for quality in qualities),
        })
    raw_exact = [
        predicted == set(sample["ground_truth_tools"])
        for predicted, sample in zip(final_sets["raw_text"], samples)
    ]
    strict_exact = [
        predicted == set(sample["ground_truth_tools"])
        for predicted, sample in zip(final_sets["strict_json"], samples)
    ]
    rows.append({
        "section": "final_tool_set_quality",
        "label": "raw_exact_but_strict_failed",
        "sample_count": len(samples),
        "case_count": sum(
            raw and not strict for raw, strict in zip(raw_exact, strict_exact)
        ),
    })

    outcomes: dict[str, list[tuple[float | None, float | None]]] = {
        "both_success": [], "omtp_only": [],
        "natural_only": [], "both_failed": [],
    }
    for sample in samples:
        omtp = sample["omtp_evaluation"]["vs_ground_truth"]
        natural = sample["natural_baseline"]["vs_ground_truth"]
        omtp_ok = bool(omtp["all_tool_types_prefetched"])
        natural_ok = bool(natural["all_tool_types_prefetched"])
        category = (
            "both_success" if omtp_ok and natural_ok else
            "omtp_only" if omtp_ok else
            "natural_only" if natural_ok else "both_failed"
        )
        outcomes[category].append((
            omtp["oracle_all_correct_coverage_time_s"],
            natural["oracle_all_correct_coverage_time_s"],
        ))
    for category, pairs in outcomes.items():
        omtp_times = [float(a) for a, _ in pairs if a is not None]
        natural_times = [float(b) for _, b in pairs if b is not None]
        leads = [
            float(b) - float(a)
            for a, b in pairs if a is not None and b is not None
        ]
        row: dict[str, Any] = {
            "section": "broad_type_coverage_outcomes",
            "label": category,
            "sample_count": len(samples),
            "case_count": len(pairs),
            "case_rate": len(pairs) / len(samples),
        }
        for prefix, values in (
            ("omtp_time_s", omtp_times),
            ("natural_time_s", natural_times),
            ("paired_lead_s", leads),
        ):
            for stat, value in stats(values).items():
                row[f"{prefix}_{stat}"] = value
        rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with (output / "omtp_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _figures(
    output: Path,
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (directory / "README.txt").write_text(
            "Install matplotlib to generate OMTP figures.\n", encoding="utf-8"
        )
        return

    ground = {
        row["method"]: row
        for row in rows if row["reference"] == "ground_truth"
    }
    metrics = ("tool_precision", "tool_recall", "tool_f1")
    positions = list(range(len(metrics)))
    width = 0.36
    plt.figure(figsize=(7, 4))
    for offset, method in ((-width / 2, METHOD_NAME),
                           (width / 2, NATURAL_BASELINE_NAME)):
        plt.bar(
            [position + offset for position in positions],
            [ground[method][metric] for metric in metrics],
            width,
            label=method,
        )
    plt.xticks(positions, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "omtp_fallback_quality.png", dpi=160)
    plt.close()

    leads = [
        sample["omtp_evaluation"]["vs_ground_truth"]
        .get("paired_exact_set_lead_vs_natural_s")
        for sample in samples
    ]
    leads = [value for value in leads if value is not None]
    if leads:
        plt.figure(figsize=(7, 4))
        plt.hist(leads, bins="auto")
        plt.xlabel("OMTP + fallback lead vs natural (s)")
        plt.ylabel("Samples")
        plt.tight_layout()
        plt.savefig(directory / "omtp_fallback_lead_distribution.png", dpi=160)
        plt.close()

    maximum = max((sample.get("forward_count") or 0) for sample in samples)
    coverage = []
    for forward in range(1, maximum + 1):
        values = []
        for sample in samples:
            expected = set(sample["ground_truth_tools"])
            predicted = {
                event["predicted_tool"]
                for event in sample.get("omtp_fallback_events", [])
                if (event.get("forward_step") or 0) <= forward
            }
            values.append(
                len(predicted & expected) / len(expected) if expected else 0.0
            )
        coverage.append(mean(values))
    if coverage:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, maximum + 1), coverage, label="OMTP + fallback")
        plt.xlabel("Forward")
        plt.ylabel("Correct-tool coverage")
        plt.ylim(0, 1.02)
        plt.grid(alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(directory / "correct_tool_coverage_by_forward.png", dpi=160)
        plt.close()

    counts = {source: 0 for source in ("latent_region", "prefix", "natural")}
    for sample in samples:
        for event in sample.get("omtp_fallback_events", []):
            source = event["source"]
            counts[source] = counts.get(source, 0) + 1
    plt.figure(figsize=(6, 4))
    plt.bar(list(counts), list(counts.values()))
    plt.ylabel("Committed occurrences")
    plt.tight_layout()
    plt.savefig(directory / "omtp_event_source_distribution.png", dpi=160)
    plt.close()

    observed = [sample["generation_time_s"] for sample in samples
                if sample.get("generation_time_s") is not None]
    baseline = [sample["observer_off_generation_time_s"] for sample in samples
                if sample.get("observer_off_generation_time_s") is not None]
    processing = [sample["observer_processing_time_s"] for sample in samples
                  if sample.get("observer_processing_time_s") is not None]
    plt.figure(figsize=(6, 4))
    plt.bar(
        ("observer off", "observer on", "observer processing"),
        (mean(baseline) if baseline else 0, mean(observed) if observed else 0,
         mean(processing) if processing else 0),
    )
    plt.ylabel("Seconds")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(directory / "observer_overhead.png", dpi=160)
    plt.close()


__all__ = ["build_omtp_report", "stats"]
