"""TaskBench DailyLife adapter for the existing BFCL-shaped prompt/evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


TYPE_MAP = {
    "string": "string",
    "date": "string",
    "integer": "integer",
    "float": "float",
    "boolean": "boolean",
    "array": "array",
    "object": "dict",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tool_catalog(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = value.get("nodes") if isinstance(value, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("TaskBench tool_desc.json must contain a non-empty nodes list")
    functions = []
    seen: set[str] = set()
    for node in nodes:
        name = str(node.get("id", ""))
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate TaskBench tool id: {name!r}")
        seen.add(name)
        properties = {}
        required = []
        parameters = node.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, list):
                raise ValueError(f"{name}: parameters must be a list")
            for parameter in parameters:
                parameter_name = str(parameter.get("name", ""))
                if not parameter_name:
                    raise ValueError(f"{name}: empty parameter name")
                schema = {
                    "type": TYPE_MAP.get(str(parameter.get("type", "string")), "string"),
                    "description": str(parameter.get("desc", "")),
                }
                properties[parameter_name] = schema
                required.append(parameter_name)
        else:
            input_types = node.get("input-type", [])
            if not isinstance(input_types, list):
                raise ValueError(f"{name}: input-type must be a list")
            for index, input_type in enumerate(input_types, start=1):
                parameter_name = f"input_{index}"
                properties[parameter_name] = {
                    "type": "string",
                    "description": f"TaskBench positional {input_type} input",
                }
                required.append(parameter_name)
        output_types = node.get("output-type", [])
        output_description = (
            f" Outputs: {', '.join(map(str, output_types))}."
            if isinstance(output_types, list) and output_types else ""
        )
        functions.append({
            "name": name,
            "description": str(node.get("desc", "")) + output_description,
            "parameters": {
                "type": "dict",
                "properties": properties,
                "required": required,
            },
        })
    return functions


@dataclass(frozen=True)
class TaskBenchSample:
    sample_id: str
    source_id: str
    source_seed: int
    domain: str
    graph_type: str
    messages: list[dict[str, str]]
    functions: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    dependencies: list[dict[str, str]]
    reference_steps: list[str]

    @property
    def candidate_names(self) -> list[str]:
        return [str(function["name"]) for function in self.functions]

    @property
    def ground_truth_calls(self) -> list[dict[str, Any]]:
        return [
            {"name": call["name"], "arguments": call["arguments"]}
            for call in self.calls
        ]

    @property
    def ground_truth_tool_types(self) -> set[str]:
        return {str(call["name"]) for call in self.calls}

    @property
    def ground_truth(self) -> list[dict[str, dict[str, list[Any]]]]:
        # The existing AST-compatible checker expects each allowed argument
        # value to be wrapped in a list.
        return [
            {
                str(call["name"]): {
                    str(name): [value]
                    for name, value in call["arguments"].items()
                }
            }
            for call in self.calls
        ]

    @property
    def tool_call_count(self) -> int:
        return len(self.calls)

    @property
    def tool_type_count(self) -> int:
        return len(self.ground_truth_tool_types)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def iter_taskbench_samples(
    manifest_path: str | Path,
    tool_catalog_path: str | Path,
) -> Iterator[TaskBenchSample]:
    functions = load_tool_catalog(tool_catalog_path)
    candidates = {function["name"] for function in functions}
    seen_ids: set[str] = set()
    manifest_path = Path(manifest_path)
    inferred_domain = next(
        (part for part in manifest_path.parts if part.startswith("data_")),
        "data_dailylifeapis",
    )
    for row in _load_jsonl(manifest_path):
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate TaskBench sample id: {sample_id}")
        seen_ids.add(sample_id)
        graph_type = str(row["graph_type"]).lower()
        if graph_type not in ("chain", "dag"):
            raise ValueError(f"{sample_id}: expected Chain/DAG, got {graph_type}")
        calls = row.get("ground_truth_calls")
        dependencies = row.get("dependencies")
        if not isinstance(calls, list) or len(calls) < 2:
            raise ValueError(f"{sample_id}: expected at least two calls")
        names = [str(call.get("name", "")) for call in calls]
        if len(set(names)) < 2 or not set(names) <= candidates:
            raise ValueError(f"{sample_id}: invalid ground-truth tool names")
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError(f"{sample_id}: expected dependency edges")
        yield TaskBenchSample(
            sample_id=sample_id,
            source_id=str(row["source_id"]),
            source_seed=int(row["source_seed"]),
            domain=str(row.get("domain", inferred_domain)),
            graph_type=graph_type,
            messages=[{"role": "user", "content": str(row["instruction"])}],
            functions=functions,
            calls=[
                {
                    "name": str(call["name"]),
                    "arguments": dict(call.get("arguments", {})),
                }
                for call in calls
            ],
            dependencies=[
                {"source": str(edge["source"]), "target": str(edge["target"])}
                for edge in dependencies
            ],
            reference_steps=[str(step) for step in row.get("reference_steps", [])],
        )


def validate_taskbench_manifest(
    manifest_path: str | Path,
    tool_catalog_path: str | Path,
) -> dict[str, Any]:
    samples = list(iter_taskbench_samples(manifest_path, tool_catalog_path))
    return {
        "sample_count": len(samples),
        "graph_types": {
            graph_type: sum(sample.graph_type == graph_type for sample in samples)
            for graph_type in ("chain", "dag")
        },
        "candidate_tool_count": len(samples[0].functions) if samples else 0,
        "min_calls": min((sample.tool_call_count for sample in samples), default=0),
        "max_calls": max((sample.tool_call_count for sample in samples), default=0),
        "min_tool_types": min((sample.tool_type_count for sample in samples), default=0),
        "max_tool_types": max((sample.tool_type_count for sample in samples), default=0),
        "manifest_sha256": sha256_file(manifest_path),
        "tool_catalog_sha256": sha256_file(tool_catalog_path),
    }


__all__ = [
    "TaskBenchSample", "iter_taskbench_samples", "load_tool_catalog",
    "sha256_file", "validate_taskbench_manifest",
]
