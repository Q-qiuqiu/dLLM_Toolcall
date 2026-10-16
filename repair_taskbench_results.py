"""Replay json-repair on an existing TaskBench run without changing it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from json_repair_evaluation import evaluate_json_repair
from taskbench_data import iter_taskbench_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tool-catalog", required=True)
    return parser.parse_args()


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return mean(float(bool(row.get(field))) for row in rows)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return mean(float(row.get(field, 0.0)) for row in rows)


def summary_row(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    metrics = {
        "json_valid_rate": ("final_json_valid", "repaired_json_valid"),
        "tool_call_format_valid_rate": (
            "tool_call_format_valid", "repaired_tool_call_format_valid"
        ),
        "candidate_names_valid_rate": (
            "candidate_names_valid", "repaired_candidate_names_valid"
        ),
        "fully_valid_rate": (None, "repaired_output_valid"),
        "tool_set_exact_rate": (
            "tool_set_exact_match", "repaired_tool_set_exact_match"
        ),
        "tool_call_count_match_rate": (
            "tool_call_count_match", "repaired_tool_call_count_match"
        ),
        "ast_compatible_rate": (
            "bfcl_ast_compatible_match", "repaired_ast_compatible_match"
        ),
    }
    value: dict[str, Any] = {
        "group": group,
        "sample_count": len(rows),
        "repair_attempted_count": sum(bool(r["json_repair_attempted"]) for r in rows),
        "repair_changed_count": sum(bool(r["json_repair_changed"]) for r in rows),
        "repair_exception_count": sum(r["json_repair_exception"] is not None for r in rows),
        "repair_processing_time_s_mean": _mean(rows, "json_repair_processing_time_s"),
    }
    for name, (before_field, after_field) in metrics.items():
        if before_field is None:
            before = mean(float(
                bool(r.get("final_json_valid"))
                and bool(r.get("tool_call_format_valid"))
                and bool(r.get("candidate_names_valid"))
            ) for r in rows)
        else:
            before = _rate(rows, before_field)
        after = _rate(rows, after_field)
        value[f"original_{name}"] = before
        value[f"repaired_{name}"] = after
        value[f"absolute_gain_{name}"] = after - before
    for name, before_field, after_field in (
        ("tool_name_precision", "tool_name_precision", "repaired_tool_name_precision"),
        ("tool_name_recall", "tool_name_recall", "repaired_tool_name_recall"),
        ("tool_name_f1", "tool_name_f1", "repaired_tool_name_f1"),
        ("tool_type_precision", "tool_type_precision", "repaired_tool_type_precision"),
        ("tool_type_recall", "tool_type_recall", "repaired_tool_type_recall"),
        ("tool_type_f1", "tool_type_f1", "repaired_tool_type_f1"),
    ):
        before, after = _mean(rows, before_field), _mean(rows, after_field)
        value[f"original_{name}"] = before
        value[f"repaired_{name}"] = after
        value[f"absolute_gain_{name}"] = after - before
    return value


def main() -> None:
    args = parse_args()
    output = Path(args.result_dir)
    source = output / "taskbench_samples.jsonl"
    destination = output / "taskbench_json_repair_samples.jsonl"
    summary = output / "taskbench_json_repair_summary.csv"
    failures = output / "taskbench_json_repair_failures.jsonl"
    for path in (destination, summary, failures):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    samples = {
        sample.sample_id: sample for sample in iter_taskbench_samples(
            args.manifest, args.tool_catalog
        )
    }
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = samples[row["sample_id"]]
            row.update(evaluate_json_repair(
                row.get("evaluated_output_before_eos", row.get("raw_output", "")),
                row["candidate_tools"], sample,
                output_schema=row.get("output_schema", "standard"),
            ))
            rows.append(row)

    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with failures.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row["repaired_output_valid"]:
                handle.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "error": row["repaired_json_error"],
                    "repaired_json_text": row["repaired_json_text"],
                }, ensure_ascii=False) + "\n")

    groups = [
        summary_row(rows, "overall"),
        summary_row([r for r in rows if r.get("graph_type") == "chain"], "chain"),
        summary_row([r for r in rows if r.get("graph_type") == "dag"], "dag"),
    ]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    print(f"Repaired {len(rows)} samples; results written under {output.resolve()}")


if __name__ == "__main__":
    main()
