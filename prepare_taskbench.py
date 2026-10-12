"""Audit official TaskBench Parquet and create fixed disjoint manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--tool-catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--smoke-per-type", type=int, default=10)
    parser.add_argument("--formal-per-type", type=int, default=100)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json(value: Any, *, sample_id: str, field: str) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise ValueError(f"{sample_id}: invalid {field}") from error


def _normalize(row: dict[str, Any], domain: str) -> dict[str, Any]:
    source_id = str(row["id"])
    nodes = _parse_json(row["tool_nodes"], sample_id=source_id, field="tool_nodes")
    links = _parse_json(row["tool_links"], sample_id=source_id, field="tool_links")
    steps = _parse_json(row["tool_steps"], sample_id=source_id, field="tool_steps")
    calls = []
    for node in nodes:
        raw_arguments = node.get("arguments", [])
        arguments = {}
        for index, argument in enumerate(raw_arguments, start=1):
            if isinstance(argument, dict) and "name" in argument:
                arguments[str(argument["name"])] = argument.get("value")
            else:
                # Resource-dependency domains store positional values rather
                # than named API parameters.  The adapter exposes stable names
                # without changing the original tool ID or value.
                arguments[f"input_{index}"] = argument
        calls.append({"name": str(node["task"]), "arguments": arguments})
    return {
        "sample_id": f"taskbench_{domain.removeprefix('data_')}_{source_id}",
        "domain": domain,
        "source_id": source_id,
        "source_seed": int(row["seed"]),
        "graph_type": str(row["type"]).lower(),
        "instruction": str(row["instruction"]),
        "ground_truth_calls": calls,
        "dependencies": [
            {"source": str(edge["source"]), "target": str(edge["target"])}
            for edge in links
        ],
        # Retained for offline auditing only. The prompt builder never reads it.
        "reference_steps": [str(step) for step in steps],
    }


def main() -> None:
    args = parse_args()
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "prepare_taskbench.py requires PyArrow; use the system Python in this workspace"
        ) from error
    catalog_value = json.loads(Path(args.tool_catalog).read_text(encoding="utf-8"))
    catalog = {str(node["id"]) for node in catalog_value["nodes"]}
    domain = Path(args.tool_catalog).resolve().parent.name
    table = parquet.read_table(args.parquet)
    rows = table.to_pylist()
    counts = Counter()
    eligible: dict[str, list[dict[str, Any]]] = {"chain": [], "dag": []}
    exclusions = Counter()
    declared_tool_count_mismatches = 0
    for row in rows:
        graph_type = str(row.get("type", "")).lower()
        counts[graph_type] += 1
        if graph_type not in eligible:
            continue
        try:
            normalized = _normalize(row, domain)
        except (KeyError, TypeError, ValueError):
            exclusions["malformed_json_or_fields"] += 1
            continue
        if int(row.get("n_tools", -1)) != len(normalized["ground_truth_calls"]):
            declared_tool_count_mismatches += 1
        calls = normalized["ground_truth_calls"]
        names = [call["name"] for call in calls]
        edges = normalized["dependencies"]
        if len(calls) < 2:
            exclusions["fewer_than_two_calls"] += 1
        elif len(set(names)) < 2:
            exclusions["fewer_than_two_tool_types"] += 1
        elif not set(names) <= catalog:
            exclusions["unknown_ground_truth_tool"] += 1
        elif not edges:
            exclusions["no_dependency_edges"] += 1
        elif not ({edge["source"] for edge in edges} | {edge["target"] for edge in edges}) <= set(names):
            exclusions["edge_endpoint_not_in_ground_truth"] += 1
        else:
            eligible[graph_type].append(normalized)

    rng = random.Random(args.seed)
    for graph_type in eligible:
        eligible[graph_type].sort(key=lambda row: row["source_id"])
        rng.shuffle(eligible[graph_type])
    needed = args.smoke_per_type + args.formal_per_type
    if any(len(rows) < needed for rows in eligible.values()):
        raise ValueError("not enough eligible Chain/DAG samples for requested split")

    smoke_parts = {
        key: value[:args.smoke_per_type] for key, value in eligible.items()
    }
    formal_parts = {
        key: value[args.smoke_per_type:needed] for key, value in eligible.items()
    }

    def interleave(parts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        result = []
        for index in range(max(map(len, parts.values()))):
            for graph_type in ("chain", "dag"):
                if index < len(parts[graph_type]):
                    result.append(parts[graph_type][index])
        return result

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = {
        "smoke20.jsonl": interleave(smoke_parts),
        "formal200.jsonl": interleave(formal_parts),
    }
    for name, selected in splits.items():
        with (output / name).open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    smoke_ids = {row["sample_id"] for row in splits["smoke20.jsonl"]}
    formal_ids = {row["sample_id"] for row in splits["formal200.jsonl"]}
    if smoke_ids & formal_ids:
        raise AssertionError("Smoke and formal TaskBench splits overlap")
    audit = {
        "source_parquet": str(Path(args.parquet).resolve()),
        "source_parquet_sha256": _sha256(args.parquet),
        "tool_catalog": str(Path(args.tool_catalog).resolve()),
        "tool_catalog_sha256": _sha256(args.tool_catalog),
        "selection_seed": args.seed,
        "domain": domain,
        "source_rows": len(rows),
        "source_schema": {
            field.name: str(field.type) for field in table.schema
        },
        "source_type_counts": dict(sorted(counts.items())),
        "declared_n_tools_mismatch_count": declared_tool_count_mismatches,
        "candidate_tool_count": len(catalog),
        "eligible_counts": {key: len(value) for key, value in eligible.items()},
        "eligible_repeated_call_case_counts": {
            key: sum(
                len(row["ground_truth_calls"])
                > len({call["name"] for call in row["ground_truth_calls"]})
                for row in value
            )
            for key, value in eligible.items()
        },
        "eligible_call_count_distributions": {
            key: dict(sorted(Counter(
                len(row["ground_truth_calls"]) for row in value
            ).items()))
            for key, value in eligible.items()
        },
        "exclusion_counts": dict(sorted(exclusions.items())),
        "splits": {
            name: {
                "sample_count": len(selected),
                "chain_count": sum(row["graph_type"] == "chain" for row in selected),
                "dag_count": sum(row["graph_type"] == "dag" for row in selected),
                "repeated_call_case_count": sum(
                    len(row["ground_truth_calls"])
                    > len({call["name"] for call in row["ground_truth_calls"]})
                    for row in selected
                ),
                "ids": [row["sample_id"] for row in selected],
            }
            for name, selected in splits.items()
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
