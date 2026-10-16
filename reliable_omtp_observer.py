"""Opt-in, causally evaluated reliability ablations for the frozen OMTP."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import re
from typing import Any

import torch

from optimized_omtp_observer import OptimizedOMTPObserver
from omtp_observer import _json_object_depth


@dataclass(frozen=True)
class ReliabilityOptions:
    name_boundary: bool = False
    consecutive_stability: bool = False
    strict_anchor: bool = False
    append_natural: bool = False

    @property
    def enabled(self) -> bool:
        return any(asdict(self).values())


def reliability_options(args: Any) -> ReliabilityOptions:
    return ReliabilityOptions(**{
        field: bool(getattr(args, f"omtp_{field}", False))
        for field in ReliabilityOptions.__dataclass_fields__
    })


def add_reliability_arguments(parser: Any) -> None:
    for field, help_text in (
        ("name_boundary", "Require a committed or current-argmax closing name quote."),
        ("consecutive_stability", "Reset latent stability after a missing forward."),
        ("strict_anchor", "Require a current top-1 name field in an outer object."),
        ("append_natural", "Append newly naturally recognized identities after earlier commits."),
    ):
        parser.add_argument(
            "--omtp-" + field.replace("_", "-"), action="store_true", help=help_text
        )
    parser.add_argument("--omtp-min-top1-score", type=float, default=None)
    parser.add_argument("--omtp-min-margin", type=float, default=None)


def _outside_string(text: str) -> bool:
    quoted = escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
    return not quoted


class ReliableOMTPObserver(OptimizedOMTPObserver):
    def __init__(self, *, reliability: ReliabilityOptions, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reliability = reliability
        self._reliable_temporary_ids: list[int] = []
        self._boundary_surface_cache: dict[int, bool] = {}
        self._trusted_anchor_centers: set[int] = set()

    def _detect_anchors(self, observation, temporary_ids):
        self._reliable_temporary_ids = temporary_ids
        anchors = super()._detect_anchors(observation, temporary_ids)
        if not self.reliability.strict_anchor:
            return anchors
        accepted = []
        for anchor in anchors:
            start = anchor["anchor_start"] - observation.prompt_length
            pattern = anchor["anchor_token_ids"]
            if temporary_ids[start:start + len(pattern)] != pattern:
                continue
            prefix = self.tokenizer.decode(temporary_ids[:start], skip_special_tokens=True)
            # Variants without the opening quote start inside an empty key.
            if not anchor["anchor_variant"].startswith('"'):
                if not prefix.endswith('"'):
                    continue
                prefix = prefix[:-1]
            if (
                _outside_string(prefix)
                and _json_object_depth(prefix) == 1
                and re.search(r"[\{,]\s*$", prefix)
            ):
                accepted.append(anchor)
        self._trusted_anchor_centers = {row["value_start"] for row in accepted}
        # Preserve the original track association and Natural detector for a
        # paired baseline. Trust only controls speculative scoring/fallback.
        return anchors

    def _score_region(self, observation, center):
        if self.reliability.strict_anchor and center not in self._trusted_anchor_centers:
            return None
        return super()._score_region(observation, center)

    def _prefix_and_natural(self, observation, center):
        prefix, natural = super()._prefix_and_natural(observation, center)
        if self.reliability.strict_anchor and center not in self._trusted_anchor_centers:
            prefix = None
        return prefix, natural

    def _associate(self, anchors, forward):
        previous_forwards = {track["track_id"]: track["last_forward"] for track in self.tracks}
        associated = super()._associate(anchors, forward)
        if self.reliability.consecutive_stability:
            for track, _, _ in associated:
                previous = previous_forwards.get(track["track_id"])
                if previous is not None and previous != forward - 1:
                    track["stable_count"] = 0
                    track["last_prediction"] = None
        return associated

    def _closes_name(self, token_id: int) -> bool:
        if token_id not in self._boundary_surface_cache:
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self._boundary_surface_cache[token_id] = text.startswith('"')
        return self._boundary_surface_cache[token_id]

    def _score_bounds(self, observation, *, left, right, top_k):
        if not self.reliability.name_boundary:
            return super()._score_bounds(observation, left=left, right=right, top_k=top_k)
        # Identical candidate scoring and top-k aggregation, with one extra
        # eligibility condition at each candidate's end. The boundary uses
        # only current committed / argmax tokens, never the eventual output.
        prompt = observation.prompt_length
        logits = observation.logits[0, prompt:, :].float()
        committed_ids = self.baseline.steps[-1]["generation_token_ids_after"]
        total = len(committed_ids)
        by_length = defaultdict(list)
        for name, ids in self._candidate_ids.items():
            by_length[len(ids)].append((name, ids))
        candidates = {}
        with self._profile("candidate_token_scoring", observation.logits):
            score_groups = []
            for width, group in by_length.items():
                starts = list(range(left, min(right, total - width - 1) + 1))
                if not starts:
                    continue
                boundary_ok = [
                    self._closes_name(self._reliable_temporary_ids[start + width])
                    for start in starts
                ]
                if not any(boundary_ok):
                    continue
                compatible = torch.tensor([
                    [boundary_ok[index] and all(
                        actual in (observation.mask_id, expected)
                        for actual, expected in zip(committed_ids[start:start + width], ids)
                    ) for index, start in enumerate(starts)]
                    for _, ids in group
                ], device=logits.device, dtype=torch.bool)
                self._record_transfer("cpu_to_gpu", compatible.numel())
                start_tensor = torch.tensor(starts, device=logits.device)
                offsets = torch.arange(width, device=logits.device)
                positions = start_tensor[:, None] + offsets[None, :]
                token_tensor = torch.tensor([list(ids) for _, ids in group], device=logits.device)
                self._record_transfer("cpu_to_gpu", (start_tensor.numel() + token_tensor.numel()) * 8)
                selected = logits[positions[None, :, :], token_tensor[:, None, :]]
                partitions = self._ensure_partitions(logits, positions)
                scores = (selected - partitions[positions][None, :, :]).mean(dim=-1)
                scores = torch.where(compatible, scores, torch.full_like(scores, -torch.inf))
                score_groups.append((group, start_tensor, scores))
        with self._profile("region_scoring", observation.logits):
            for group, start_tensor, scores in score_groups:
                for index, (name, _) in enumerate(group):
                    valid = torch.isfinite(scores[index])
                    count = int(valid.sum().item())
                    if not count:
                        continue
                    count_top = min(top_k, count)
                    values, order = torch.topk(scores[index][valid], k=count_top)
                    best_position = int(start_tensor[valid][order[0]].detach().cpu())
                    candidates[name] = {
                        "score": float(values.mean().detach().cpu()),
                        "best_position": prompt + best_position,
                        "alignment_count": count,
                    }
                    self._record_transfer("gpu_to_cpu", 8 + 4 * count_top)
        if not candidates:
            return None
        ranked = sorted(candidates.items(), key=lambda item: item[1]["score"], reverse=True)
        name, top1 = ranked[0]
        top2 = ranked[1][1]["score"] if len(ranked) > 1 else None
        start = top1["best_position"] - prompt
        width = len(self._candidate_ids[name])
        return {
            "predicted_tool": name, "top1_score": top1["score"], "top2_score": top2,
            "margin": None if top2 is None else top1["score"] - top2,
            "best_position": top1["best_position"],
            "region_left": prompt + left, "region_right": prompt + right,
            "candidate_scores": candidates,
            "name_uncommitted_ratio": sum(
                token == observation.mask_id for token in committed_ids[start:start + width]
            ) / float(width),
            "boundary_verified": True,
        }

    def fallback_events(self):
        events = super().fallback_events()
        if not self.reliability.append_natural:
            return events
        stream = [(event, False) for event in events] + [
            (event, True) for event in self.events if event["source"] == "natural"
        ]
        identities = set()
        result = []
        for event, supplemental in sorted(stream, key=lambda item: item[0]["predict_time_s"]):
            if supplemental and event["predicted_tool"] in identities:
                continue
            result.append(event)
            identities.add(event["predicted_tool"])
        return result


__all__ = [
    "ReliabilityOptions", "ReliableOMTPObserver", "add_reliability_arguments",
    "reliability_options",
]
