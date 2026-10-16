"""Sequential real-model dev-set ablations with explicit keep/revert decisions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

from taskbench_report import percentile


STAGES = (
    ("01_name_boundary", ["--omtp-name-boundary"]),
    ("02_consecutive", ["--omtp-consecutive-stability"]),
    ("03_strict_anchor", ["--omtp-strict-anchor"]),
    ("04_confidence_gates", ["--omtp-min-top1-score", "-5", "--omtp-min-margin", "0.1"]),
    ("05_append_natural", ["--omtp-append-natural"]),
)


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows):
    metrics = [row["taskbench_methods"]["omtp_fallback"]["vs_ground_truth"] for row in rows]
    natural = [row["taskbench_methods"]["natural_decoding"]["vs_ground_truth"] for row in rows]
    tp = sum(len(set(row["predicted_tools"]) & set(sample["ground_truth_tools"])) for row, sample in zip(metrics, rows))
    fp = sum(row["false_positive_count"] for row in metrics)
    fn = sum(row["false_negative_count"] for row in metrics)
    paired_unique = []
    paired_coverage = []
    both = omtp_only = natural_only = neither = 0
    for sample, predicted, baseline in zip(rows, metrics, natural):
        for name in set(sample["ground_truth_tools"]):
            online_time = predicted["first_times_by_tool"].get(name)
            natural_time = baseline["first_times_by_tool"].get(name)
            if online_time is not None and natural_time is not None:
                paired_unique.append(natural_time - online_time)
        online_time = predicted["oracle_all_correct_coverage_time_s"]
        natural_time = baseline["oracle_all_correct_coverage_time_s"]
        if online_time is not None and natural_time is not None:
            both += 1
            paired_coverage.append(natural_time - online_time)
        elif online_time is not None:
            omtp_only += 1
        elif natural_time is not None:
            natural_only += 1
        else:
            neither += 1
    result = {
        "sample_count": len(rows),
        "precision_macro": mean(row["tool_precision"] for row in metrics),
        "recall_macro": mean(row["tool_recall"] for row in metrics),
        "f1_macro": mean(row["tool_f1"] for row in metrics),
        "precision_micro": tp / (tp + fp) if tp + fp else 0.,
        "recall_micro": tp / (tp + fn) if tp + fn else 0.,
        "set_exact_match": mean(row["tool_set_exact_match"] for row in metrics),
        "all_required_types_covered": mean(row["all_tool_types_prefetched"] for row in metrics),
        "true_positive_types": tp,
        "false_positive_types": fp,
        "false_negative_types": fn,
        "tasks_with_wrong_trigger": sum(row["false_positive_count"] > 0 for row in metrics),
        "generation_time_s_mean": mean(row["generation_time_s"] for row in rows),
        "observer_time_s_mean": mean(row["observer_processing_time_s"] for row in rows),
        "nfe32_count": sum(row["forward_count"] == 32 for row in rows),
        "parity_checked_count": sum(row.get("observer_output_identical") is not None for row in rows),
        "parity_all_passed": all(row.get("observer_output_identical") is not False and row.get("observer_nfe_identical") is not False for row in rows),
        "both_coverage_success": both, "omtp_only_success": omtp_only,
        "natural_only_success": natural_only, "both_failed": neither,
    }
    for prefix, values in (("unique_tool_lead_s", paired_unique), ("coverage_lead_s", paired_coverage)):
        result.update({
            f"{prefix}_count": len(values),
            f"{prefix}_mean": mean(values) if values else None,
            f"{prefix}_p50": percentile(values, .5),
            f"{prefix}_p95": percentile(values, .95),
            f"{prefix}_wins": sum(value > 0 for value in values),
            f"{prefix}_losses": sum(value < 0 for value in values),
        })
    return result


def parity_signature(row):
    return row["sample_id"], row["steps"][-1]["generation_token_ids_after"], row["forward_count"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--manifest", default="data/taskbench/data_multimedia/splits/smoke20.jsonl")
    parser.add_argument("--tool-catalog", default="data/taskbench/data_multimedia/tool_desc.json")
    parser.add_argument("--model-path", default="/data/labshare/Param/llada1.5")
    parser.add_argument("--fastdllm-root", default="/data/home/yzx/Fast-dLLM")
    parser.add_argument("--wait-for-baseline", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "ablation_decisions.jsonl"
    if ledger.exists():
        raise FileExistsError(f"refusing to overwrite {ledger}")
    if args.wait_for_baseline:
        marker = Path(args.baseline_dir) / "taskbench_generation_summary.csv"
        print(f"Waiting for baseline report: {marker}", flush=True)
        while not marker.exists():
            time.sleep(5)
    baseline = load_rows(Path(args.baseline_dir) / "taskbench_samples.jsonl")
    expected_ids = [row["sample_id"] for row in load_rows(args.manifest)]
    if len(baseline) != 20 or [row["sample_id"] for row in baseline] != expected_ids:
        raise ValueError("exactly the same 20 fixed dev samples are required")
    previous_metrics = summarize(baseline)
    accepted_flags = []
    accepted_stages = []
    commands = []
    decisions = [{"stage": "00_baseline", "accepted": True, **previous_metrics}]
    with ledger.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(decisions[0]) + "\n")
        handle.flush()
        for stage, flags in STAGES:
            directory = output / stage
            cmd = [
                sys.executable, "run_taskbench_experiment.py",
                "--manifest", args.manifest, "--tool-catalog", args.tool_catalog,
                "--model-path", args.model_path, "--fastdllm-root", args.fastdllm_root,
                "--output-dir", str(directory), "--cache-mode", "none",
                "--response-mode", "native", "--output-schema", "flat_name_v1",
                "--prompt-profile", "compact_json_dep_v4", "--generation-profile", "fixed128",
                "--parity-samples", "1", *accepted_flags, *flags,
            ]
            commands.append(cmd)
            print(f"START {stage}; inherited={accepted_stages}", flush=True)
            with (output / f"{stage}.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"{stage} failed; inspect {stage}.log")
            rows = load_rows(directory / "taskbench_samples.jsonl")
            same_output = [parity_signature(row) for row in rows] == [parity_signature(row) for row in baseline]
            same_natural = [
                [(event["predicted_tool"], event["forward_step"], event["best_position"]) for event in row["natural_baseline"]["events"]]
                for row in rows
            ] == [
                [(event["predicted_tool"], event["forward_step"], event["best_position"]) for event in row["natural_baseline"]["events"]]
                for row in baseline
            ]
            metrics = summarize(rows)
            precision_delta = metrics["precision_macro"] - previous_metrics["precision_macro"]
            f1_delta = metrics["f1_macro"] - previous_metrics["f1_macro"]
            improved = precision_delta >= -1e-9 and f1_delta >= -1e-9 and (precision_delta > 1e-9 or f1_delta > 1e-9)
            accepted = bool(
                improved and same_output and same_natural and metrics["parity_all_passed"]
                and metrics["nfe32_count"] == 20
                and (metrics["unique_tool_lead_s_mean"] or 0) > 0
            )
            decision = {
                "stage": stage, "accepted": accepted,
                "parent_stages": list(accepted_stages), "trial_flags": flags,
                "output_token_parity_vs_baseline": same_output,
                "natural_event_parity_vs_baseline": same_natural,
                "precision_delta_vs_parent": precision_delta,
                "f1_delta_vs_parent": f1_delta,
                **metrics,
            }
            decisions.append(decision)
            handle.write(json.dumps(decision) + "\n"); handle.flush()
            print(json.dumps(decision), flush=True)
            if accepted:
                accepted_flags.extend(flags)
                accepted_stages.append(stage)
                previous_metrics = metrics
            print(f"{'KEEP' if accepted else 'REVERT'} {stage}", flush=True)
    columns = list(dict.fromkeys(key for decision in decisions for key in decision))
    with (output / "ablation_comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(decisions)
    code_hashes = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ("reliable_omtp_observer.py", "omtp_observer.py", "optimized_omtp_observer.py", "prompting.py")
    }
    result = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "development_set_only": True, "sample_count": 20,
        "selection_rule": "macro precision and F1 never decrease; at least one strictly improves; positive paired unique-tool lead; all output/NFE/Natural parity pass",
        "accepted_stages": accepted_stages, "accepted_flags": accepted_flags,
        "precision_target": .9, "precision_target_reached": previous_metrics["precision_macro"] >= .9,
        "metrics": previous_metrics, "commands": commands, "code_sha256": code_hashes,
    }
    (output / "selected_policy.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DONE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
