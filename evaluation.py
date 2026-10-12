"""Final-call quality, timing, and offline analysis metrics."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable, Optional

from bfcl_data import BFCLSample
from observer import ToolCallObserver


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def precision_recall_f1(
    predicted: Iterable[str], expected: Iterable[str]
) -> dict[str, float]:
    predicted_counter = Counter(predicted)
    expected_counter = Counter(expected)
    true_positive = sum(
        min(predicted_counter[name], expected_counter[name])
        for name in predicted_counter.keys() | expected_counter.keys()
    )
    predicted_count = sum(predicted_counter.values())
    expected_count = sum(expected_counter.values())
    precision = _safe_div(true_positive, predicted_count)
    recall = _safe_div(true_positive, expected_count)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def set_quality(predicted: Iterable[str], expected: Iterable[str]) -> dict[str, Any]:
    predicted_set, expected_set = set(predicted), set(expected)
    overlap = predicted_set & expected_set
    precision = _safe_div(len(overlap), len(predicted_set))
    recall = _safe_div(len(overlap), len(expected_set))
    return {
        "exact": predicted_set == expected_set,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "false_positives": sorted(predicted_set - expected_set),
        "false_negatives": sorted(expected_set - predicted_set),
    }


def _standardize_string(value: str) -> str:
    # Mirrors the normalization documented by BFCL's AST checker.
    return re.sub(r"[ ,./\-_*^]", "", value).lower().replace("'", '"')


def _value_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return _standardize_string(actual) == _standardize_string(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _value_equal(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _value_equal(actual[key], expected[key]) for key in actual
        )
    # bool is a subclass of int, so require exact Python types here.
    return type(actual) is type(expected) and actual == expected


def _schema_type_ok(value: Any, schema: dict[str, Any]) -> bool:
    expected_type = schema.get("type", "any")
    if expected_type == "float" and type(value) is int:
        # BFCL's official Python checker explicitly applies int -> float.
        return True
    type_map = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "array": list,
        "tuple": list,
        "dict": dict,
        "object": dict,
        "any": object,
    }
    python_type = type_map.get(expected_type, object)
    if python_type is object:
        return True
    if type(value) is not python_type:
        return False
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_schema_type_ok(item, schema["items"]) for item in value)
    return True


def _coerce_schema_value(value: Any, schema: dict[str, Any]) -> Any:
    """Apply the same Python numeric coercion used by BFCL's AST checker."""

    expected_type = schema.get("type", "any")
    if expected_type == "float" and type(value) is int:
        return float(value)
    if expected_type in ("array", "tuple") and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return [_coerce_schema_value(item, item_schema) for item in value]
    return value


def _call_matches(
    predicted: dict[str, Any],
    ground_truth: dict[str, dict[str, list[Any]]],
    functions_by_name: dict[str, dict[str, Any]],
) -> bool:
    if len(ground_truth) != 1:
        return False
    expected_name, expected_arguments = next(iter(ground_truth.items()))
    if predicted.get("name") != expected_name:
        return False
    actual_arguments = predicted.get("arguments")
    if not isinstance(actual_arguments, dict):
        return False
    function = functions_by_name.get(expected_name, {})
    properties = function.get("parameters", {}).get("properties", {})
    for key, value in actual_arguments.items():
        if key not in expected_arguments or key not in properties:
            return False
        if not _schema_type_ok(value, properties[key]):
            return False
        value = _coerce_schema_value(value, properties[key])
        allowed = expected_arguments[key]
        if not any(
            option != "" and _value_equal(value, option) for option in allowed
        ):
            return False
    for key, allowed in expected_arguments.items():
        if key not in actual_arguments and "" not in allowed:
            return False
    return True


def bfcl_compatible_ast_match(
    predicted_calls: list[dict[str, Any]], sample: BFCLSample
) -> bool:
    """Unordered duplicate-aware BFCL AST compatibility check.

    This follows BFCL's parallel-call matching, schema type checks, optional
    ``""`` convention, and string normalization.  The run metadata labels it
    ``compatible`` rather than claiming the external BFCL package was invoked.
    """

    if len(predicted_calls) != len(sample.ground_truth):
        return False
    functions_by_name = {function["name"]: function for function in sample.functions}
    edges = [
        [
            index
            for index, expected in enumerate(sample.ground_truth)
            if _call_matches(predicted, expected, functions_by_name)
        ]
        for predicted in predicted_calls
    ]

    def match(row: int, used: set[int]) -> bool:
        if row == len(edges):
            return True
        for target in edges[row]:
            if target not in used and match(row + 1, used | {target}):
                return True
        return False

    return match(0, set())


def _first_time(
    steps: list[dict[str, Any]], predicate, time_key: str
) -> Optional[float]:
    for step in steps:
        if predicate(step):
            return float(step[time_key])
    return None


def analyze_observer(
    observer: ToolCallObserver, expected_types: set[str]
) -> dict[str, Any]:
    steps = observer.steps
    natural_first_times = {
        name: observer.first_natural_by_tool.get(name) for name in sorted(expected_types)
    }
    present_natural = [time for time in natural_first_times.values() if time is not None]
    natural_all = (
        max(present_natural)
        if len(present_natural) == len(expected_types) and expected_types
        else None
    )
    first_natural = min(present_natural) if present_natural else None
    first_predict = _first_time(
        steps,
        lambda step: bool(set(step["predicted_tool_types"]) & expected_types),
        "forward_complete_time_s",
    )
    oracle_correct = _first_time(
        steps,
        lambda step: set(step["predicted_tool_types"]) == expected_types,
        "forward_complete_time_s",
    )
    first_trigger = observer.stable_events[0] if observer.stable_events else None
    trigger_types = first_trigger.tool_types if first_trigger else []
    trigger_quality = set_quality(trigger_types, expected_types)
    stable_correct = next(
        (
            event.time_s
            for event in observer.stable_events
            if set(event.tool_types) == expected_types
        ),
        None,
    )
    return {
        "natural_first_detect_time_s": first_natural,
        "natural_last_detect_time_s": natural_all,
        "natural_all_detect_time_s": natural_all,
        "predict_first_detect_time_s": first_predict,
        # This is the first online N-stable trigger and does not use ground truth.
        "predict_all_detect_time_s": first_trigger.time_s if first_trigger else None,
        "predict_all_available_time_s": (
            first_trigger.available_time_s if first_trigger else None
        ),
        "predict_trigger_tool_types": trigger_types,
        "predict_trigger_set_exact": trigger_quality["exact"],
        "predict_trigger_precision": trigger_quality["precision"],
        "predict_trigger_recall": trigger_quality["recall"],
        "predict_trigger_false_positives": trigger_quality["false_positives"],
        "predict_trigger_false_negatives": trigger_quality["false_negatives"],
        # Both fields below are explicitly offline/oracle analyses.
        "oracle_correct_detect_time_s": oracle_correct,
        "offline_stable_correct_trigger_time_s": stable_correct,
        "natural_first_times_by_tool": natural_first_times,
        "online_natural_first_times_all_candidates": dict(
            sorted(observer.first_natural_by_tool.items())
        ),
        "predict_first_times_by_tool": {
            name: observer.first_predict_by_tool.get(name)
            for name in sorted(expected_types)
        },
        "online_predict_first_times_all_candidates": dict(
            sorted(observer.first_predict_by_tool.items())
        ),
        "stable_events": [event.__dict__ for event in observer.stable_events],
    }


def evaluate_final_calls(
    sample: BFCLSample, predicted_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    predicted_names = [call["name"] for call in predicted_calls]
    expected_names = [call["name"] for call in sample.ground_truth_calls]
    name_metrics = precision_recall_f1(predicted_names, expected_names)
    type_metrics = set_quality(predicted_names, expected_names)
    return {
        "tool_name_precision": name_metrics["precision"],
        "tool_name_recall": name_metrics["recall"],
        "tool_name_f1": name_metrics["f1"],
        "tool_set_exact_match": type_metrics["exact"],
        "tool_type_precision": type_metrics["precision"],
        "tool_type_recall": type_metrics["recall"],
        "tool_type_f1": type_metrics["f1"],
        "tool_call_count_match": len(predicted_calls) == len(expected_names),
        "bfcl_ast_compatible_match": bfcl_compatible_ast_match(
            predicted_calls, sample
        ),
    }


__all__ = [
    "analyze_observer",
    "bfcl_compatible_ast_match",
    "evaluate_final_calls",
    "precision_recall_f1",
    "set_quality",
]
