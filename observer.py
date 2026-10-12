"""Per-forward observation for committed and temporary argmax sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable, Optional

import torch

from fastdllm_generate import ForwardObservation
from tool_parser import extract_partial_names, ordered_unique


MASK_GAP = "<|OBSERVED_MASK_GAP|>"


def decode_sparse_committed(
    tokenizer: Any, token_ids: list[int], mask_id: int
) -> str:
    """Decode committed runs while retaining a separator for every MASK gap.

    Removing MASK ids and decoding the remaining ids as one sequence could join
    distant fragments into a fake JSON field.  Run-wise decoding prevents that
    false positive.
    """

    pieces: list[str] = []
    run: list[int] = []
    in_gap = False
    barriers = set(getattr(tokenizer, "all_special_ids", []) or []) | {mask_id}
    for token_id in token_ids:
        if token_id in barriers:
            if run:
                pieces.append(tokenizer.decode(run, skip_special_tokens=True))
                run = []
            if not in_gap:
                pieces.append(MASK_GAP)
                in_gap = True
        else:
            run.append(token_id)
            in_gap = False
    if run:
        pieces.append(tokenizer.decode(run, skip_special_tokens=True))
    return "".join(pieces)


def decode_temporary(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode the visible completion prefix, stopping at EOS/PAD."""

    stop_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    }
    stop = next(
        (index for index, token_id in enumerate(token_ids) if token_id in stop_ids),
        len(token_ids),
    )
    return tokenizer.decode(token_ids[:stop], skip_special_tokens=True)


@dataclass(frozen=True)
class OnlineStableEvent:
    forward_index: int
    time_s: float
    tool_types: list[str]
    available_time_s: float | None = None


class ToolCallObserver:
    """Synchronous observer used only when explicitly enabled."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        candidate_names: Iterable[str],
        stable_n: int = 2,
        top_k: int = 1,
        full_logits_dir: Optional[str | Path] = None,
        enable_stable_prediction: bool = True,
    ) -> None:
        if stable_n not in (1, 2, 3):
            raise ValueError("stable_n must be 1, 2, or 3")
        if top_k not in (1, 2):
            raise ValueError("top_k must be 1 or 2")
        self.tokenizer = tokenizer
        self.candidate_names = list(candidate_names)
        self.stable_n = stable_n
        self.top_k = top_k
        self.enable_stable_prediction = enable_stable_prediction
        self.steps: list[dict[str, Any]] = []
        self.stable_events: list[OnlineStableEvent] = []
        self.first_natural_by_tool: dict[str, float] = {}
        self.first_predict_by_tool: dict[str, float] = {}
        self._previous_predicted_set: Optional[frozenset[str]] = None
        self._stable_run = 0
        self._last_emitted_set: Optional[frozenset[str]] = None
        self.total_processing_s = 0.0
        self.full_logits_dir = Path(full_logits_dir) if full_logits_dir else None
        if self.full_logits_dir:
            self.full_logits_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, observation: ForwardObservation) -> None:
        processing_start = time.perf_counter()
        prompt_length = observation.prompt_length

        before = observation.x[0, prompt_length:]
        after = observation.committed_x[0, prompt_length:]
        generation_logits = observation.logits[0, prompt_length:, :]
        mask = before == observation.mask_id

        values, ids = torch.topk(generation_logits, k=self.top_k, dim=-1)
        before_ids = before.detach().cpu().tolist()
        after_ids = after.detach().cpu().tolist()
        mask_positions = (
            torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()
        )
        top_ids = ids[:, 0]
        temporary = torch.where(mask, top_ids, before)
        temporary_ids = temporary.detach().cpu().tolist()
        masked_top1_ids = top_ids[mask].detach().cpu().tolist()
        masked_top1_logits = values[:, 0][mask].float().detach().cpu().tolist()

        committed_text_before = decode_sparse_committed(
            self.tokenizer, before_ids, observation.mask_id
        )
        committed_text = decode_sparse_committed(
            self.tokenizer, after_ids, observation.mask_id
        )
        temporary_text = decode_temporary(self.tokenizer, temporary_ids)

        natural_names = extract_partial_names(
            committed_text, self.candidate_names
        )
        predicted_names = extract_partial_names(
            temporary_text, self.candidate_names
        )
        natural_types = ordered_unique(natural_names)
        predicted_types = ordered_unique(predicted_names)
        predicted_set = frozenset(predicted_types)

        for name in natural_types:
            self.first_natural_by_tool.setdefault(name, observation.commit_elapsed_s)
        for name in predicted_types:
            self.first_predict_by_tool.setdefault(name, observation.elapsed_s)

        stable_trigger = False
        if self.enable_stable_prediction:
            if predicted_set == self._previous_predicted_set:
                self._stable_run += 1
            else:
                self._stable_run = 1
                self._previous_predicted_set = predicted_set
            if (
                predicted_set
                and self._stable_run >= self.stable_n
                and predicted_set != self._last_emitted_set
            ):
                stable_trigger = True
                self._last_emitted_set = predicted_set
                self.stable_events.append(
                    OnlineStableEvent(
                        observation.forward_index,
                        observation.elapsed_s,
                        sorted(predicted_set),
                    )
                )

        step: dict[str, Any] = {
            "forward_index": observation.forward_index,
            "block_index": observation.block_index,
            "block_step": observation.block_step,
            "cache_mode": observation.cache_mode,
            "observation_sequence_offset": observation.sequence_offset,
            "forward_complete_time_s": observation.elapsed_s,
            "commit_complete_time_s": observation.commit_elapsed_s,
            "forward_time_s": observation.forward_time_s,
            "mask_count_before": len(mask_positions),
            "mask_positions_before": mask_positions,
            "generation_token_ids_before": before_ids,
            "generation_token_ids_after": after_ids,
            "mask_top1_token_ids": masked_top1_ids,
            "mask_top1_logits": masked_top1_logits,
            "committed_text_before": committed_text_before,
            "committed_text": committed_text,
            "temporary_text": temporary_text,
            "committed_call_names": natural_names,
            "committed_tool_types": natural_types,
            "predicted_call_names": predicted_names,
            "predicted_tool_types": predicted_types,
            "prediction_stable_run": self._stable_run,
            "stable_n": self.stable_n,
            "stable_trigger": stable_trigger,
        }
        if self.top_k == 2:
            step["mask_top2_token_ids"] = ids[:, 1][mask].detach().cpu().tolist()
            step["mask_top2_logits"] = (
                values[:, 1][mask].float().detach().cpu().tolist()
            )
            step["mask_top1_top2_margin"] = (
                (values[:, 0] - values[:, 1])[mask].float().detach().cpu().tolist()
            )
        if self.full_logits_dir:
            path = self.full_logits_dir / f"forward_{observation.forward_index:04d}.pt"
            torch.save(generation_logits.detach().cpu(), path)
            step["full_logits_path"] = str(path)

        processing_end = time.perf_counter()
        processing_s = processing_end - processing_start
        self.total_processing_s += processing_s
        step["observer_processing_s"] = processing_s
        step["prediction_available_time_s"] = (
            observation.commit_elapsed_s + processing_s
        )
        if stable_trigger:
            event = self.stable_events[-1]
            self.stable_events[-1] = OnlineStableEvent(
                forward_index=event.forward_index,
                time_s=event.time_s,
                tool_types=event.tool_types,
                available_time_s=step["prediction_available_time_s"],
            )
        self.steps.append(step)


__all__ = [
    "MASK_GAP",
    "OnlineStableEvent",
    "ToolCallObserver",
    "decode_sparse_committed",
    "decode_temporary",
]
