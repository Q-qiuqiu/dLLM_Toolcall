"""Offline TaskBench evaluation for the three fixed online event streams.

Nothing in this module participates in generation or online triggering.  The
TaskBench ground truth is consulted only after the immutable event streams have
been produced.
"""

from __future__ import annotations

from typing import Any, Iterable

from omtp_evaluation import evaluate_events, evaluate_occurrences


NATURAL = "natural_decoding"
LATENT = "omtp_latent_region"
FALLBACK = "omtp_fallback"
METHODS = (NATURAL, LATENT, FALLBACK)


def _evaluate(
    events: list[dict[str, Any]],
    expected_types: Iterable[str],
    expected_occurrences: Iterable[str],
) -> dict[str, Any]:
    result = evaluate_events(events, set(expected_types))
    result.update(evaluate_occurrences(events, expected_occurrences))
    return result


def evaluate_taskbench_methods(
    *,
    latent_events: list[dict[str, Any]],
    fallback_events: list[dict[str, Any]],
    natural_events: list[dict[str, Any]],
    ground_truth_occurrences: Iterable[str],
    final_occurrences: Iterable[str],
) -> dict[str, Any]:
    """Evaluate Natural, pure latent OMTP, and OMTP+Fallback offline."""

    ground_truth_occurrences = list(ground_truth_occurrences)
    final_occurrences = list(final_occurrences)
    streams = {
        NATURAL: natural_events,
        LATENT: latent_events,
        FALLBACK: fallback_events,
    }
    result: dict[str, Any] = {}
    for method, events in streams.items():
        result[method] = {
            "event_count": len(events),
            "events": events,
            "vs_ground_truth": _evaluate(
                events, set(ground_truth_occurrences), ground_truth_occurrences
            ),
            "vs_final": _evaluate(
                events, set(final_occurrences), final_occurrences
            ),
        }

    natural = result[NATURAL]["vs_ground_truth"]
    for method in (LATENT, FALLBACK):
        current = result[method]["vs_ground_truth"]
        for prefix, key in (
            ("coverage", "oracle_all_correct_coverage_time_s"),
            ("exact", "oracle_exact_set_time_s"),
            ("occurrence", "oracle_all_occurrences_covered_time_s"),
        ):
            current_time = current[key]
            natural_time = natural[key]
            current[f"natural_{prefix}_time_s"] = natural_time
            current[f"paired_{prefix}_lead_vs_natural_s"] = (
                float(natural_time) - float(current_time)
                if current_time is not None and natural_time is not None
                else None
            )
        current_forward = current["oracle_all_correct_coverage_forward"]
        natural_forward = natural["oracle_all_correct_coverage_forward"]
        current["natural_coverage_forward"] = natural_forward
        current["paired_coverage_lead_forwards"] = (
            int(natural_forward) - int(current_forward)
            if current_forward is not None and natural_forward is not None
            else None
        )
    return result


__all__ = [
    "FALLBACK", "LATENT", "METHODS", "NATURAL",
    "evaluate_taskbench_methods",
]
