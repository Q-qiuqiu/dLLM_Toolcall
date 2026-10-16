"""Offline semantic dependency evaluation for TaskBench tool-call arrays.

The model-facing convention is deliberately simple: ``<node-N>`` refers to
the output of array item N in the model's own generated call array.  This
module resolves those references into tool-name edges and compares the edges
with TaskBench's name-level ``tool_links``.  Ground truth is never consulted
during generation or online OMTP triggering.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


NODE_REFERENCE = re.compile(r"^<node-(\d+)>$")
NODE_LIKE = re.compile(r"node\s*[-_]\s*\d+", re.IGNORECASE)


def _argument_strings(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _argument_strings(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _argument_strings(item, path + (str(index),))


def resolve_node_references(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Resolve exact references using the generated array's zero-based order."""

    calls = list(calls)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for target_index, call in enumerate(calls):
        target_tool = str(call.get("name", ""))
        arguments = call.get("arguments", {})
        for argument_path, value in _argument_strings(arguments):
            match = NODE_REFERENCE.fullmatch(value)
            common = {
                "target_call_index": target_index,
                "target_tool": target_tool,
                "argument_path": list(argument_path),
                "reference": value,
            }
            if match is None:
                if NODE_LIKE.search(value):
                    malformed.append({**common, "reason": "malformed_reference"})
                continue
            source_index = int(match.group(1))
            if source_index >= len(calls):
                invalid.append({
                    **common, "source_call_index": source_index,
                    "reason": "source_index_out_of_range",
                })
            elif source_index >= target_index:
                invalid.append({
                    **common, "source_call_index": source_index,
                    "source_tool": str(calls[source_index].get("name", "")),
                    "reason": "self_or_forward_reference",
                })
            else:
                valid.append({
                    **common, "source_call_index": source_index,
                    "source_tool": str(calls[source_index].get("name", "")),
                })
    return {
        "reference_count": len(valid) + len(invalid),
        "valid_reference_count": len(valid),
        "invalid_reference_count": len(invalid),
        "malformed_reference_count": len(malformed),
        "valid_references": valid,
        "invalid_references": invalid,
        "malformed_references": malformed,
    }


def _edge_dicts(edges: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"source": source, "target": target}
        for source, target in sorted(edges)
    ]


def evaluate_semantic_dependencies(
    calls: Iterable[dict[str, Any]],
    ground_truth_edges: Iterable[dict[str, Any]],
    ground_truth_calls: Iterable[dict[str, Any]],
    *,
    output_valid: bool,
) -> dict[str, Any]:
    """Evaluate name-level dependency edges recovered from generated calls.

    TaskBench dependency annotations identify tools by name, not occurrence
    IDs.  Edge metrics therefore use de-duplicated ``(source, target)`` pairs.
    Raw occurrence-level reference records remain available for auditing.
    """

    calls = list(calls)
    ground_truth_calls = list(ground_truth_calls)
    resolved = resolve_node_references(calls)
    predicted_edges = {
        (row["source_tool"], row["target_tool"])
        for row in resolved["valid_references"]
    }
    expected_edges = {
        (str(edge["source"]), str(edge["target"]))
        for edge in ground_truth_edges
    }
    true_positive = predicted_edges & expected_edges
    false_positive = predicted_edges - expected_edges
    false_negative = expected_edges - predicted_edges
    precision = (
        len(true_positive) / len(predicted_edges) if predicted_edges else 0.0
    )
    recall = len(true_positive) / len(expected_edges) if expected_edges else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    predicted_names = [str(call.get("name", "")) for call in calls]
    expected_names = [str(call.get("name", "")) for call in ground_truth_calls]
    tool_set_exact = set(predicted_names) == set(expected_names)
    occurrence_sequence_exact = predicted_names == expected_names
    edge_exact = predicted_edges == expected_edges
    semantic_plan_exact = tool_set_exact and edge_exact
    ordered_plan_exact = occurrence_sequence_exact and edge_exact
    return {
        "offline_only": True,
        "reference_convention": "generated_call_array_zero_based",
        "output_valid": bool(output_valid),
        **resolved,
        "predicted_dependency_edges": _edge_dicts(predicted_edges),
        "ground_truth_dependency_edges": _edge_dicts(expected_edges),
        "true_positive_edges": _edge_dicts(true_positive),
        "false_positive_edges": _edge_dicts(false_positive),
        "false_negative_edges": _edge_dicts(false_negative),
        "edge_true_positive_count": len(true_positive),
        "edge_false_positive_count": len(false_positive),
        "edge_false_negative_count": len(false_negative),
        "dependency_edge_precision": precision,
        "dependency_edge_recall": recall,
        "dependency_edge_f1": f1,
        "dependency_edge_exact_match": edge_exact,
        "tool_set_exact_match": tool_set_exact,
        "tool_occurrence_sequence_exact_match": occurrence_sequence_exact,
        "semantic_plan_exact_match": semantic_plan_exact,
        "ordered_semantic_plan_exact_match": ordered_plan_exact,
        "valid_semantic_plan_exact_match": bool(output_valid) and semantic_plan_exact,
        "valid_ordered_semantic_plan_exact_match": (
            bool(output_valid) and ordered_plan_exact
        ),
    }


def audit_ground_truth_references(
    ground_truth_calls: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Audit literal TaskBench node strings without using them as edge truth."""

    result = resolve_node_references(ground_truth_calls)
    return {
        "reference_count": result["reference_count"],
        "backward_valid_reference_count": result["valid_reference_count"],
        "non_backward_reference_count": result["invalid_reference_count"],
        "malformed_reference_count": result["malformed_reference_count"],
        "note": (
            "audit only; semantic scoring uses TaskBench dependency_edges, "
            "not literal ground-truth <node-N> equality"
        ),
    }


__all__ = [
    "NODE_REFERENCE", "audit_ground_truth_references",
    "evaluate_semantic_dependencies", "resolve_node_references",
]
