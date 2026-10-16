"""Add offline semantic dependency metrics to an existing TaskBench run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskbench_data import iter_taskbench_samples
from taskbench_dependency_evaluation import (
    audit_ground_truth_references, evaluate_semantic_dependencies,
)
from taskbench_report import write_dependency_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tool-catalog", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.result_dir)
    source = output / "taskbench_samples.jsonl"
    destination = output / "taskbench_dependency_samples.jsonl"
    generated = (
        destination,
        output / "taskbench_dependency_summary.csv",
        output / "taskbench_dependency_edges.jsonl",
    )
    for path in generated:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    samples = {
        sample.sample_id: sample
        for sample in iter_taskbench_samples(args.manifest, args.tool_catalog)
    }
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = samples[row["sample_id"]]
            raw_valid = bool(
                row.get("final_json_valid")
                and row.get("tool_call_format_valid")
                and row.get("candidate_names_valid")
            )
            row["semantic_dependency_raw"] = evaluate_semantic_dependencies(
                row.get("generated_tool_calls", []),
                sample.dependencies,
                sample.ground_truth_calls,
                output_valid=raw_valid,
            )
            row["semantic_dependency_repaired"] = evaluate_semantic_dependencies(
                row.get("repaired_tool_calls", []),
                sample.dependencies,
                sample.ground_truth_calls,
                output_valid=bool(row.get("repaired_output_valid")),
            )
            row["ground_truth_dependency_reference_audit"] = (
                audit_ground_truth_references(sample.ground_truth_calls)
            )
            rows.append(row)

    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_dependency_report(output, rows)
    print(
        f"Evaluated semantic dependencies for {len(rows)} samples under "
        f"{output.resolve()}"
    )


if __name__ == "__main__":
    main()
