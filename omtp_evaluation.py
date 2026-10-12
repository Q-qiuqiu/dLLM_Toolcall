"""Causal evaluation for the fixed OMTP + fallback online policy."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from evaluation import set_quality
from omtp_observer import OMTPObserver


METHOD_NAME = "omtp_fallback"
NATURAL_BASELINE_NAME = "natural_decoding"


def _tool_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        tools = event.get("predicted_tools")
        if tools is None:
            tools = [event["predicted_tool"]]
        for tool in tools:
            rows.append({**event, "predicted_tool": tool})
    return sorted(
        rows,
        key=lambda row: (
            float(row["predict_time_s"]), str(row["predicted_tool"])
        ),
    )


def evaluate_events(
    events: list[dict[str, Any]], expected: set[str]
) -> dict[str, Any]:
    """Evaluate immutable online triggers; correctness is assigned offline."""

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
        "all_tool_types_prefetched": bool(expected) and expected <= predicted,
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


def _position_ordered_events(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order occurrence events by their generated-sequence location."""

    return sorted(
        _tool_events(events),
        key=lambda event: (
            event.get("best_position") is None,
            int(event.get("best_position") or 0),
            int(event.get("track_id") or 0),
            float(event["predict_time_s"]),
        ),
    )


def evaluate_occurrences(
    events: list[dict[str, Any]], expected_occurrences: Iterable[str]
) -> dict[str, Any]:
    """Evaluate call multiplicity and generated-position alignment offline."""

    ordered_events = _position_ordered_events(events)
    expected = [str(tool) for tool in expected_occurrences]
    predicted = [str(event["predicted_tool"]) for event in ordered_events]
    expected_counts = Counter(expected)
    predicted_counts = Counter(predicted)
    matched = sum(
        min(count, predicted_counts[tool])
        for tool, count in expected_counts.items()
    )
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    all_prefetched = bool(expected) and all(
        predicted_counts[tool] >= count
        for tool, count in expected_counts.items()
    )

    position_rows = []
    correct_count = 0
    for index in range(max(len(expected), len(ordered_events))):
        expected_tool = expected[index] if index < len(expected) else None
        event = ordered_events[index] if index < len(ordered_events) else None
        predicted_tool = event["predicted_tool"] if event is not None else None
        correct = expected_tool is not None and predicted_tool == expected_tool
        correct_count += int(correct)
        position_rows.append({
            "position": index + 1,
            "expected_tool": expected_tool,
            "predicted_tool": predicted_tool,
            "correct": correct,
            "predict_time_s": (
                float(event["predict_time_s"]) if event is not None else None
            ),
            "forward_step": event.get("forward_step") if event else None,
            "source": event.get("source") if event else None,
            "track_id": event.get("track_id") if event else None,
            "sequence_position": event.get("best_position") if event else None,
        })

    running = Counter()
    covered_time = None
    exact_time = None
    for event in _tool_events(events):
        running[str(event["predicted_tool"])] += 1
        when = float(event["predict_time_s"])
        if expected and covered_time is None and all(
            running[tool] >= count for tool, count in expected_counts.items()
        ):
            covered_time = when
        if expected and exact_time is None and running == expected_counts:
            exact_time = when

    return {
        "predicted_tool_occurrences": predicted,
        "expected_tool_occurrences": expected,
        "occurrence_precision": precision,
        "occurrence_recall": recall,
        "occurrence_f1": f1,
        "all_tool_occurrences_prefetched": all_prefetched,
        "tool_occurrence_count_match": len(predicted) == len(expected),
        "tool_occurrence_multiset_exact_match": predicted_counts == expected_counts,
        "oracle_all_occurrences_covered_time_s": covered_time,
        "oracle_exact_occurrence_multiset_time_s": exact_time,
        "position_correct_count": correct_count,
        "position_total": len(expected),
        "position_accuracy": correct_count / len(expected) if expected else 0.0,
        "position_exact_match": (
            len(predicted) == len(expected) and correct_count == len(expected)
        ),
        "position_records": position_rows,
    }


def _first_by_source(
    observer: OMTPObserver, source: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    for event in observer.events:
        if event["source"] == source:
            result.setdefault(
                str(event["predicted_tool"]), float(event["predict_time_s"])
            )
    return dict(sorted(result.items()))


def evaluate_omtp_fallback(
    observer: OMTPObserver,
    *,
    final_tools: Iterable[str],
    ground_truth_tools: Iterable[str],
    final_occurrences: Iterable[str] | None = None,
    ground_truth_occurrences: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate only OMTP + fallback against final output and BFCL truth.

    The selected online event is the first causal event for each occurrence
    track. It may come from latent region scoring, a unique committed prefix,
    or natural completion. Ground truth is used only after generation.
    """

    events = observer.fallback_events()
    final_set = set(final_tools)
    gt_set = set(ground_truth_tools)
    vs_final = evaluate_events(events, final_set)
    vs_ground_truth = evaluate_events(events, gt_set)
    final_occurrences = list(final_occurrences or final_tools)
    ground_truth_occurrences = list(
        ground_truth_occurrences or ground_truth_tools
    )
    vs_final.update(evaluate_occurrences(events, final_occurrences))
    vs_ground_truth.update(
        evaluate_occurrences(events, ground_truth_occurrences)
    )

    # Natural occurrence events already exist because natural fallback is part
    # of OMTP. Reusing them avoids another scan inside the timed callback.
    natural_events = [
        {**event, "source": NATURAL_BASELINE_NAME}
        for event in observer.events if event["source"] == "natural"
    ]
    natural_vs_final = evaluate_events(natural_events, final_set)
    natural_vs_ground_truth = evaluate_events(natural_events, gt_set)
    natural_vs_final.update(
        evaluate_occurrences(natural_events, final_occurrences)
    )
    natural_vs_ground_truth.update(
        evaluate_occurrences(natural_events, ground_truth_occurrences)
    )
    for natural, result in (
        (natural_vs_final, vs_final),
        (natural_vs_ground_truth, vs_ground_truth),
    ):
        online_time = result["oracle_exact_set_time_s"]
        natural_time = natural["oracle_exact_set_time_s"]
        result["natural_exact_set_time_s"] = natural_time
        result["paired_exact_set_lead_vs_natural_s"] = (
            float(natural_time) - float(online_time)
            if natural_time is not None and online_time is not None
            else None
        )
        occurrence_time = result["oracle_all_occurrences_covered_time_s"]
        natural_occurrence_time = natural[
            "oracle_all_occurrences_covered_time_s"
        ]
        result["natural_all_occurrences_covered_time_s"] = (
            natural_occurrence_time
        )
        result["paired_all_occurrences_lead_vs_natural_s"] = (
            float(natural_occurrence_time) - float(occurrence_time)
            if natural_occurrence_time is not None and occurrence_time is not None
            else None
        )

    latent_times = _first_by_source(observer, "latent_region")
    prefix_times = _first_by_source(observer, "prefix")
    natural_times = dict(sorted(observer.natural_available_by_tool.items()))
    online_times = vs_final["first_times_by_tool"]
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
            "online_lead_vs_natural_s": (
                t_natural - online_times[name]
                if t_natural is not None and name in online_times else None
            ),
        })

    def position_comparison(
        omtp_result: dict[str, Any], natural_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        natural_by_position = {
            row["position"]: row
            for row in natural_result["position_records"]
        }
        rows = []
        for row in omtp_result["position_records"]:
            natural = natural_by_position.get(row["position"], {})
            omtp_time = row["predict_time_s"]
            natural_time = natural.get("predict_time_s")
            rows.append({
                "position": row["position"],
                "expected_tool": row["expected_tool"],
                "omtp_predicted_tool": row["predicted_tool"],
                "omtp_correct": row["correct"],
                "omtp_predict_time_s": omtp_time,
                "omtp_forward_step": row["forward_step"],
                "omtp_source": row["source"],
                "omtp_track_id": row["track_id"],
                "omtp_sequence_position": row["sequence_position"],
                "natural_predicted_tool": natural.get("predicted_tool"),
                "natural_correct": natural.get("correct", False),
                "natural_detect_time_s": natural_time,
                "natural_forward_step": natural.get("forward_step"),
                "lead_vs_natural_s": (
                    float(natural_time) - float(omtp_time)
                    if natural_time is not None and omtp_time is not None
                    else None
                ),
            })
        return rows

    return {
        "method": METHOD_NAME,
        "events": events,
        "vs_final": vs_final,
        "vs_ground_truth": vs_ground_truth,
        "natural_baseline": {
            "method": NATURAL_BASELINE_NAME,
            "events": natural_events,
            "vs_final": natural_vs_final,
            "vs_ground_truth": natural_vs_ground_truth,
        },
        "position_comparison": {
            "vs_final": position_comparison(vs_final, natural_vs_final),
            "vs_ground_truth": position_comparison(
                vs_ground_truth, natural_vs_ground_truth
            ),
        },
        "per_tool_predictions": per_tool_predictions,
        "natural_detection_times": natural_times,
    }


__all__ = [
    "METHOD_NAME", "NATURAL_BASELINE_NAME", "evaluate_events",
    "evaluate_occurrences",
    "evaluate_omtp_fallback",
]
