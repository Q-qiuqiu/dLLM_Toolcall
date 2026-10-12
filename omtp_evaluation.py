"""Causal and oracle-labelled evaluation for OMTP comparison methods."""

from __future__ import annotations

from typing import Any, Iterable

from evaluation import set_quality
from omtp_observer import OMTPObserver


METHODS = (
    "intermediate_argmax",
    "whole_set_argmax_stable_n2",
    "per_tool_argmax_stable_n2",
    "omtp_latent_region_n2",
    "omtp_with_fallback",
    "natural_decoding",
)


def _tool_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        tools = event.get("predicted_tools")
        if tools is None:
            tools = [event["predicted_tool"]]
        for tool in tools:
            rows.append({**event, "predicted_tool": tool})
    return sorted(rows, key=lambda row: (float(row["predict_time_s"]), str(row["predicted_tool"])))


def evaluate_events(
    events: list[dict[str, Any]], expected: set[str]
) -> dict[str, Any]:
    expanded = _tool_events(events)
    predicted: set[str] = set()
    first_correct = None
    all_coverage = None
    exact_time = None
    first_trigger = expanded[0]["predict_time_s"] if expanded else None
    per_tool: dict[str, float] = {}
    for event in expanded:
        tool = str(event["predicted_tool"])
        when = float(event["predict_time_s"])
        predicted.add(tool)
        per_tool.setdefault(tool, when)
        if tool in expected and first_correct is None:
            first_correct = when
        if expected and expected <= predicted and all_coverage is None:
            all_coverage = when
        if expected and predicted == expected and exact_time is None:
            exact_time = when
    quality = set_quality(predicted, expected)
    return {
        "predicted_tools": sorted(predicted),
        "tool_precision": quality["precision"],
        "tool_recall": quality["recall"],
        "tool_f1": quality["f1"],
        "tool_set_exact_match": quality["exact"],
        "false_positive_count": len(quality["false_positives"]),
        "false_negative_count": len(quality["false_negatives"]),
        "false_positives": quality["false_positives"],
        "false_negatives": quality["false_negatives"],
        "correct_trigger_coverage": quality["recall"],
        "online_first_trigger_time_s": first_trigger,
        "oracle_first_correct_prediction_time_s": first_correct,
        "oracle_all_correct_coverage_time_s": all_coverage,
        "oracle_exact_set_time_s": exact_time,
        "first_times_by_tool": dict(sorted(per_tool.items())),
        "trigger_count": len(expanded),
    }


def evaluate_omtp(
    observer: OMTPObserver,
    *,
    final_tools: Iterable[str],
    ground_truth_tools: Iterable[str],
) -> dict[str, Any]:
    final_set = set(final_tools)
    gt_set = set(ground_truth_tools)
    method_events = observer.method_events()
    steps = observer.steps

    first_argmax = next(
        (step for step in steps if step["predicted_tool_types"]), None
    )
    method_events["intermediate_argmax"] = (
        [{
            "source": "intermediate_argmax",
            "track_id": None,
            "predicted_tools": list(first_argmax["predicted_tool_types"]),
            "forward_step": first_argmax["forward_index"],
            "predict_time_s": first_argmax["prediction_available_time_s"],
        }]
        if first_argmax else []
    )
    method_events["natural_decoding"] = [
        {
            "source": "natural",
            "track_id": None,
            "predicted_tool": name,
            "forward_step": None,
            "predict_time_s": when,
        }
        for name, when in observer.natural_available_by_tool.items()
    ]

    comparisons: dict[str, Any] = {}
    for method in METHODS:
        events = method_events.get(method, [])
        final_eval = evaluate_events(events, final_set)
        gt_eval = evaluate_events(events, gt_set)
        for label, result in (("final", final_eval), ("ground_truth", gt_eval)):
            natural = evaluate_events(method_events["natural_decoding"], final_set if label == "final" else gt_set)
            exact_time = result["oracle_exact_set_time_s"]
            natural_time = natural["oracle_exact_set_time_s"]
            result["paired_exact_set_lead_vs_natural_s"] = (
                float(natural_time) - float(exact_time)
                if natural_time is not None and exact_time is not None else None
            )
        comparisons[method] = {
            "events": events,
            "vs_final": final_eval,
            "vs_ground_truth": gt_eval,
        }

    def first_by_source(source: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for event in observer.events:
            if event["source"] == source:
                result.setdefault(event["predicted_tool"], float(event["predict_time_s"]))
        return dict(sorted(result.items()))

    latent_times = first_by_source("latent_region")
    prefix_times = first_by_source("prefix")
    natural_times = dict(sorted(observer.natural_available_by_tool.items()))
    online_times = comparisons["omtp_with_fallback"]["vs_final"]["first_times_by_tool"]
    per_tool_predictions = []
    for name in observer.candidate_names:
        t_latent = latent_times.get(name)
        t_natural = natural_times.get(name)
        per_tool_predictions.append({
            "tool": name,
            "T_latent": t_latent,
            "T_prefix": prefix_times.get(name),
            "T_natural": t_natural,
            "T_online": online_times.get(name),
            "latent_lead_vs_natural_s": (
                t_natural - t_latent
                if t_natural is not None and t_latent is not None else None
            ),
        })
    return {
        "methods": comparisons,
        "per_tool_predictions": per_tool_predictions,
        "natural_detection_times": natural_times,
    }


__all__ = ["METHODS", "evaluate_events", "evaluate_omtp"]
