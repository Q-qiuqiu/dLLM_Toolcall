"""Build a paired A--E report from independently generated experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from omtp_evaluation import evaluate_events


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def stats(values) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "mean": mean(clean) if clean else None,
        "p50": percentile(clean, .5),
        "p95": percentile(clean, .95),
        "count": len(clean),
    }


def argmax_events(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source": "whole_set_argmax_stable",
            "predicted_tools": event["tool_types"],
            "forward_step": event["forward_index"],
            "predict_time_s": event.get("available_time_s") or event["time_s"],
        }
        for event in sample.get("stable_events", [])
    ]


def natural_events(sample: dict[str, Any], *, baseline: bool) -> list[dict[str, Any]]:
    if baseline:
        first = {}
        for step in sample.get("steps", []):
            for tool in step.get("committed_tool_types", []):
                first.setdefault(tool, step["prediction_available_time_s"])
    else:
        first = sample.get("natural_detection_times", {})
    return [
        {"source": "natural", "predicted_tool": tool, "predict_time_s": when}
        for tool, when in first.items()
    ]


def expand_tools(events: list[dict[str, Any]]) -> list[tuple[str, float]]:
    rows = []
    for event in events:
        tools = event.get("predicted_tools") or [event.get("predicted_tool")]
        for tool in tools:
            if tool is not None:
                rows.append((str(tool), float(event["predict_time_s"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--d", required=True)
    parser.add_argument("--e", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    directories = {
        "A_argmax_stable_n2": (Path(args.a), True),
        "B_original_omtp": (Path(args.b), False),
        "C_optimized_omtp": (Path(args.c), False),
        "D_selective_omtp": (Path(args.d), False),
        "E_selective_gated": (Path(args.e), False),
    }
    loaded = {}
    for method, (directory, baseline) in directories.items():
        filename = "samples.jsonl" if baseline else "omtp_samples.jsonl"
        rows = load(directory / filename)
        loaded[method] = {row["sample_id"]: row for row in rows}
    ids = list(loaded["A_argmax_stable_n2"])
    if any(set(rows) != set(ids) for rows in loaded.values()):
        raise ValueError("A--E sample IDs differ")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sample_rows = []
    method_results: dict[str, list[dict[str, Any]]] = {name: [] for name in directories}
    for sample_id in ids:
        per_method = {}
        gt = set(loaded["A_argmax_stable_n2"][sample_id]["ground_truth_tool_types"])
        for method, (_, baseline) in directories.items():
            sample = loaded[method][sample_id]
            events = (
                argmax_events(sample)
                if baseline
                else sample["method_comparison"]["omtp_latent_region_n2"]["events"]
            )
            final = set(
                sample["final_predicted_tool_types"] if baseline
                else sample["final_tools"]
            )
            result_gt = evaluate_events(events, gt)
            result_final = evaluate_events(events, final)
            natural = evaluate_events(
                natural_events(sample, baseline=baseline), gt
            )
            exact = result_gt["oracle_exact_set_time_s"]
            natural_time = natural["oracle_exact_set_time_s"]
            result_gt["lead_vs_natural_s"] = (
                natural_time - exact
                if natural_time is not None and exact is not None else None
            )
            expanded = expand_tools(events)
            unique = {tool for tool, _ in expanded}
            result_gt["event_wrong_rate"] = (
                sum(tool not in gt for tool, _ in expanded) / len(expanded)
                if expanded else 0.0
            )
            result_gt["unique_tool_wrong_trigger_rate"] = (
                len(unique - gt) / len(unique) if unique else 0.0
            )
            result_gt["required_tool_first_times"] = {
                tool: min(
                    (when for predicted, when in expanded if predicted == tool),
                    default=None,
                )
                for tool in sorted(gt)
            }
            per_method[method] = {
                "events": events,
                "vs_ground_truth": result_gt,
                "vs_final": result_final,
                "natural_exact_time_s": natural_time,
                "observer_processing_time_s": sample.get("observer_processing_time_s"),
                "generation_time_s": sample.get("generation_time_s"),
                "observer_output_identical": sample.get("observer_output_identical"),
                "observer_nfe_identical": (
                    sample.get("observer_nfe_identical")
                    if not baseline else sample.get("observer_output_identical")
                ),
                "forward_count": sample.get("forward_count"),
                "raw_output": sample.get("raw_output"),
            }
            method_results[method].append(per_method[method])
        sample_rows.append({
            "sample_id": sample_id,
            "ground_truth_tools": sorted(gt),
            "methods": per_method,
        })

    summary = []
    for method, results in method_results.items():
        gt_rows = [row["vs_ground_truth"] for row in results]
        row = {
            "method": method,
            "sample_count": len(results),
            "tool_precision": mean(x["tool_precision"] for x in gt_rows),
            "tool_recall": mean(x["tool_recall"] for x in gt_rows),
            "tool_f1": mean(x["tool_f1"] for x in gt_rows),
            "tool_set_exact_match": mean(float(x["tool_set_exact_match"]) for x in gt_rows),
            "exact_coverage_rate": mean(float(x["oracle_exact_set_time_s"] is not None) for x in gt_rows),
            "event_wrong_rate": sum(x["event_wrong_rate"] * x["trigger_count"] for x in gt_rows) / max(1, sum(x["trigger_count"] for x in gt_rows)),
            "unique_tool_wrong_trigger_rate": mean(x["unique_tool_wrong_trigger_rate"] for x in gt_rows),
            "required_tool_trigger_coverage": mean(x["correct_trigger_coverage"] for x in gt_rows),
            "nfe_pass_rate": mean(float(x["forward_count"] == 32 and x["observer_nfe_identical"] is not False) for x in results),
        }
        measured_parity = [
            x["observer_output_identical"] for x in results
            if x["observer_output_identical"] is not None
        ]
        row["parity_measured_count"] = len(measured_parity)
        row["parity_pass_count"] = sum(bool(value) for value in measured_parity)
        for prefix, values in (
            ("exact_time_s", [x["oracle_exact_set_time_s"] for x in gt_rows]),
            ("lead_vs_natural_s", [x["lead_vs_natural_s"] for x in gt_rows]),
            ("observer_processing_s", [x["observer_processing_time_s"] for x in results]),
            ("generation_time_s", [x["generation_time_s"] for x in results]),
            ("required_tool_time_s", [
                value
                for x in gt_rows
                for value in x["required_tool_first_times"].values()
                if value is not None
            ]),
        ):
            for key, value in stats(values).items():
                row[f"{prefix}_{key}"] = value
        summary.append(row)

    baseline_by_id = {
        row["sample_id"]: row["methods"]["A_argmax_stable_n2"]["vs_ground_truth"]
        for row in sample_rows
    }
    paired = []
    for method in list(directories)[1:]:
        differences = []
        wins = ties = losses = method_only = baseline_only = 0
        for row in sample_rows:
            base = baseline_by_id[row["sample_id"]]["oracle_exact_set_time_s"]
            current = row["methods"][method]["vs_ground_truth"]["oracle_exact_set_time_s"]
            if base is None and current is not None:
                method_only += 1
            elif base is not None and current is None:
                baseline_only += 1
            elif base is not None and current is not None:
                delta = float(base) - float(current)
                differences.append(delta)
                if delta > 1e-6: wins += 1
                elif delta < -1e-6: losses += 1
                else: ties += 1
        paired.append({
            "method": method, "reference": "A_argmax_stable_n2",
            "paired_count": len(differences), "wins": wins, "ties": ties,
            "losses": losses, "method_only_success": method_only,
            "baseline_only_success": baseline_only,
            **{f"advantage_s_{key}": value for key, value in stats(differences).items()},
        })

    b = loaded["B_original_omtp"]
    c = loaded["C_optimized_omtp"]
    def signature(sample):
        return [
            (event["track_id"], event["predicted_tool"], event["source"], event["forward_step"])
            for event in sample["omtp_events"]
        ]
    validation = {
        "sample_count": len(ids),
        "model_output_identical_count": sum(b[i]["raw_output"] == c[i]["raw_output"] for i in ids),
        "omtp_event_signature_identical_count": sum(signature(b[i]) == signature(c[i]) for i in ids),
        "all_method_model_output_identical_count": sum(
            len({loaded[method][i]["raw_output"] for method in loaded}) == 1
            for i in ids
        ),
        "all_nfe_32": all(
            loaded[method][sample_id]["forward_count"] == 32
            for method in loaded for sample_id in ids
        ),
    }
    d = loaded["D_selective_omtp"]
    e = loaded["E_selective_gated"]
    validation["optimized_vs_selective_event_signature_identical_count"] = sum(
        signature(c[i]) == signature(d[i]) for i in ids
    )
    validation["selective_vs_gated_event_signature_identical_count"] = sum(
        signature(d[i]) == signature(e[i]) for i in ids
    )
    validation["selective_routing"] = {
        method: {
            key: sum(
                int(loaded[method][i].get("observer_profile", {}).get("selective", {}).get(key) or 0)
                for i in ids
            )
            for key in ("cheap_count", "refine_count")
        }
        for method in ("D_selective_omtp", "E_selective_gated")
    }
    whole_per_tool = []
    for sample_id in ids:
        sample = b[sample_id]
        whole = sample["method_comparison"]["whole_set_argmax_stable_n2"]["vs_ground_truth"]
        per_tool = sample["method_comparison"]["per_tool_argmax_stable_n2"]["vs_ground_truth"]
        whole_per_tool.append({
            "sample_id": sample_id,
            "same_final_predicted_set": whole["predicted_tools"] == per_tool["predicted_tools"],
            "whole_exact_time_s": whole["oracle_exact_set_time_s"],
            "per_tool_exact_time_s": per_tool["oracle_exact_set_time_s"],
        })
    validation["whole_vs_per_tool_same_set_count"] = sum(
        row["same_final_predicted_set"] for row in whole_per_tool
    )
    validation["whole_vs_per_tool_exact_time_delta_s"] = stats([
        row["whole_exact_time_s"] - row["per_tool_exact_time_s"]
        for row in whole_per_tool
        if row["whole_exact_time_s"] is not None
        and row["per_tool_exact_time_s"] is not None
    ])

    with (output / "ablation_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for filename, rows in (("ablation_summary.csv", summary), ("paired_vs_argmax.csv", paired)):
        fields = sorted({key for row in rows for key in row})
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    (output / "validation.json").write_text(
        json.dumps({"bc_equivalence": validation, "whole_vs_per_tool": whole_per_tool}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure_dir = output / "figures"; figure_dir.mkdir()
        plt.figure(figsize=(8, 4))
        for row in paired:
            method = row["method"]
            values = []
            for sample in sample_rows:
                base = sample["methods"]["A_argmax_stable_n2"]["vs_ground_truth"]["oracle_exact_set_time_s"]
                current = sample["methods"][method]["vs_ground_truth"]["oracle_exact_set_time_s"]
                if base is not None and current is not None:
                    values.append(base - current)
            if values:
                values.sort(); plt.plot(values, label=method)
        plt.axhline(0, color="black", linewidth=.8); plt.ylabel("A time - method time (s)"); plt.xlabel("Paired samples, sorted"); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(figure_dir / "paired_advantage_distribution.png", dpi=160); plt.close()
    except ImportError:
        pass
    print(output.resolve())


if __name__ == "__main__":
    main()
