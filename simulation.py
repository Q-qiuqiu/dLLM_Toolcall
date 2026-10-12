"""Independent parallel tool-startup timing simulation."""

from __future__ import annotations

from typing import Any, Iterable, Optional


def simulate_warmup(
    *,
    expected_types: set[str],
    natural_first_times: dict[str, Optional[float]],
    stable_events: Iterable[dict[str, Any]],
    generation_time_s: float,
    startup_durations: Iterable[float] = (1.0, 1.5, 2.0),
) -> list[dict[str, Any]]:
    """Simulate independently and concurrently starting each tool type.

    The prediction strategy starts a type on its first online stable event.
    Required types never predicted early fall back to natural discovery.  An
    incorrect type consumes a startup but does not delay other independent
    starts under this model.
    """

    predicted_start: dict[str, float] = {}
    for event in stable_events:
        event_start = event.get("available_time_s")
        if event_start is None:
            event_start = event["time_s"]
        for name in event["tool_types"]:
            predicted_start.setdefault(name, float(event_start))

    rows: list[dict[str, Any]] = []
    for duration in startup_durations:
        natural_ready_values = [
            time + duration
            for name in expected_types
            if (time := natural_first_times.get(name)) is not None
        ]
        natural_complete = len(natural_ready_values) == len(expected_types)
        prediction_ready_values = []
        missing_without_fallback: list[str] = []
        for name in expected_types:
            start = predicted_start.get(name, natural_first_times.get(name))
            if start is None:
                missing_without_fallback.append(name)
            else:
                prediction_ready_values.append(start + duration)
        prediction_complete = not missing_without_fallback
        natural_ready = max(natural_ready_values) if natural_complete else None
        prediction_ready = (
            max(prediction_ready_values) if prediction_complete else None
        )
        wrong_types = sorted(set(predicted_start) - expected_types)
        rows.append(
            {
                "startup_duration_s": duration,
                "natural_all_tools_ready_time_s": natural_ready,
                "prediction_all_tools_ready_time_s": prediction_ready,
                "ready_time_savings_s": (
                    natural_ready - prediction_ready
                    if natural_ready is not None and prediction_ready is not None
                    else None
                ),
                "natural_ready_by_generation_end": (
                    natural_ready <= generation_time_s
                    if natural_ready is not None
                    else False
                ),
                "prediction_ready_by_generation_end": (
                    prediction_ready <= generation_time_s
                    if prediction_ready is not None
                    else False
                ),
                "wrong_preload_types": wrong_types,
                "wrong_preload_cost_s": len(wrong_types) * duration,
                "required_types_without_any_start": missing_without_fallback,
            }
        )
    return rows


__all__ = ["simulate_warmup"]
