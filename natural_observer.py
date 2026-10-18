"""Minimal online Natural baseline, sharing OMTP's committed-token detector."""

from __future__ import annotations

import time
from typing import Any, Iterable

from fastdllm_generate import ForwardObservation
from observer import detect_committed_tools


class NaturalOnlyObserver:
    """Read only committed generation IDs; no logits, argmax, prefix or tracks.

    Reuse the existing synchronized generation callback, without additional
    forwards. This is a low-overhead Natural-only baseline, not observer-off:
    token copying, sparse decoding, parsing and callback synchronization remain.
    """

    def __init__(
        self, *, tokenizer: Any, candidate_names: Iterable[str],
        record_history: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.candidate_names = list(candidate_names)
        self.record_history = record_history
        self.first_tool_events: dict[str, dict[str, Any]] = {}
        self.natural_first_events: dict[str, dict[str, Any]] = {}
        self.total_processing_s = 0.0
        self.forward_count = 0
        self.steps: list[dict[str, Any]] = []

    def __call__(self, observation: ForwardObservation) -> None:
        started = time.perf_counter()
        self.forward_count += 1
        ids = observation.committed_x[0, observation.prompt_length:].detach().cpu().tolist()
        text, names, tools = detect_committed_tools(
            self.tokenizer, ids, observation.mask_id, self.candidate_names
        )
        new_tools = [name for name in tools if name not in self.first_tool_events]
        # commit_elapsed_s is device-complete, NOT the online detection time.
        # Include D2H copying, decoding, parsing and deduplication before trigger.
        detected = time.perf_counter()
        available_s = observation.commit_elapsed_s + detected - started
        for name in new_tools:
            event = {
                "predicted_tool": name, "source": "natural",
                "forward_step": observation.forward_index,
                "predict_time_s": available_s,
            }
            self.first_tool_events[name] = event
            self.natural_first_events[name] = event
        if self.record_history:
            self.steps.append({
                "forward_index": observation.forward_index,
                "block_index": observation.block_index,
                "block_step": observation.block_step,
                "forward_complete_time_s": observation.elapsed_s,
                "commit_complete_time_s": observation.commit_elapsed_s,
                "forward_time_s": observation.forward_time_s,
                "prediction_available_time_s": available_s,
                "observer_detection_s": detected - started,
                "committed_call_names": names,
                "committed_tool_types": tools,
                "committed_text": text,
            })
        self.total_processing_s += time.perf_counter() - started


def add_observer_arguments(parser: Any) -> None:
    parser.add_argument(
        "--observer-mode", choices=("omtp", "natural"), default="omtp",
        help="omtp keeps the current method; natural runs only the shared committed-token detector.",
    )


def validate_observer_mode(args: Any) -> None:
    if getattr(args, "observer_mode", "omtp") != "natural":
        return
    # Fail loudly instead of claiming unused OMTP gates were applied.
    if any(
        value is not None and value is not False
        for key, value in vars(args).items()
        if key.startswith("omtp_") or key in ("min_top1_score", "min_margin", "profile_observer")
    ):
        raise ValueError("--observer-mode natural does not run OMTP; remove OMTP gates/profile flags")
