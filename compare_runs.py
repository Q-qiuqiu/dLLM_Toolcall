"""Create a paired comparison of two completed BFCL experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation import set_quality


METRICS = (
    "final_json_valid",
    "tool_call_format_valid",
    "candidate_names_valid",
    "tool_name_f1",
    "tool_set_exact_match",
    "tool_call_count_match",
    "bfcl_ast_compatible_match",
    "predict_trigger_set_exact",
    "generation_time_s",
    "model_forward_time_s",
    "observer_processing_time_s",
    "natural_all_detect_time_s",
    "oracle_correct_detect_time_s",
    "offline_stable_correct_trigger_time_s",
    "offline_stable_correct_lead_time_s",
)


def _load(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["sample_id"]: row for row in rows}


def _average(rows: list[dict[str, Any]], metric: str) -> tuple[float | None, int]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return (mean(values) if values else None, len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    left = _load(Path(args.left_run) / "samples.jsonl")
    right = _load(Path(args.right_run) / "samples.jsonl")
    common = sorted(left.keys() & right.keys())
    if not common:
        raise ValueError("Runs have no common sample IDs")
    left_rows = [left[sample_id] for sample_id in common]
    right_rows = [right[sample_id] for sample_id in common]

    comparison_rows = []
    for metric in METRICS:
        left_value, left_count = _average(left_rows, metric)
        right_value, right_count = _average(right_rows, metric)
        comparison_rows.append(
            {
                "metric": metric,
                args.left_label: left_value,
                f"{args.left_label}_valid_count": left_count,
                args.right_label: right_value,
                f"{args.right_label}_valid_count": right_count,
                "right_minus_left": (
                    right_value - left_value
                    if left_value is not None and right_value is not None
                    else None
                ),
            }
        )
    fields = list(comparison_rows[0])
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    with (output / "paired_samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in common:
            item = {"sample_id": sample_id}
            for metric in METRICS:
                item[f"{args.left_label}_{metric}"] = left[sample_id].get(metric)
                item[f"{args.right_label}_{metric}"] = right[sample_id].get(metric)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    maximum_forward = min(
        max(len(row.get("steps", [])) for row in left_rows),
        max(len(row.get("steps", [])) for row in right_rows),
    )
    trajectory = []
    for index in range(maximum_forward):
        row: dict[str, Any] = {"forward_index": index + 1}
        for label, records in ((args.left_label, left_rows), (args.right_label, right_rows)):
            qualities = [
                set_quality(
                    record["steps"][index]["predicted_tool_types"],
                    record["ground_truth_tool_types"],
                )
                for record in records
                if index < len(record.get("steps", []))
            ]
            row[f"{label}_mean_f1"] = mean(q["f1"] for q in qualities)
            row[f"{label}_exact_rate"] = mean(float(q["exact"]) for q in qualities)
        trajectory.append(row)
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)

    metadata = {
        "left_run": str(Path(args.left_run).resolve()),
        "right_run": str(Path(args.right_run).resolve()),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "paired_sample_count": len(common),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Compared {len(common)} paired samples in {output.resolve()}")


if __name__ == "__main__":
    main()
