"""Online Multi-Tool Prediction (OMTP) on existing no-cache forwards.

The implementation is a read-only adaptation of the local
Agent-Oriented-Planning commit observer: speculative JSON ``name`` anchors,
persistent local tracks, candidate-aware region scoring, stable-N commit, and
unique-prefix/natural fallbacks.  It never mutates the decoding canvas and
never requests an extra model forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import re
import time
from typing import Any, Iterable

import torch

from fastdllm_generate import ForwardObservation
from observer import ToolCallObserver
from prefix_detection import CommittedPrefixMatcher


@dataclass(frozen=True)
class OMTPConfig:
    stable_n: int = 2
    region_radius: int = 4
    region_top_k: int = 2
    min_seen: int = 2
    min_support: float = 0.5
    position_tolerance: int = 4
    merge_distance: int = 6
    max_track_misses: int = 3
    anchor_min_logit_margin: float = -6.0
    min_anchor_gap: int = 8
    min_top1_score: float | None = None
    min_margin: float | None = None

    def validate(self) -> None:
        if self.stable_n not in (1, 2, 3):
            raise ValueError("stable_n must be 1, 2, or 3")
        if self.region_radius < 0 or self.region_top_k < 1:
            raise ValueError("invalid region parameters")
        if self.min_seen < 1 or not 0.0 <= self.min_support <= 1.0:
            raise ValueError("invalid track support parameters")


def _json_object_depth(text: str) -> int:
    """Return unmatched object depth while ignoring braces inside strings."""

    depth = 0
    quoted = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth = max(0, depth - 1)
    return depth


class OMTPObserver:
    """OMTP observer with latent, unique-prefix, and natural fallback events."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        candidate_names: Iterable[str],
        stable_n: int = 2,
        config: OMTPConfig | None = None,
        profile_enabled: bool = False,
        record_history: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.candidate_names = list(dict.fromkeys(str(x) for x in candidate_names))
        self.config = config or OMTPConfig(stable_n=stable_n)
        self.config.validate()
        self.profile_enabled = bool(profile_enabled)
        self.record_history = record_history
        self.forward_count = 0
        self.profile_stats: dict[str, dict[str, float | int]] = {}
        self.profile_transfer_bytes = {"gpu_to_cpu": 0, "cpu_to_gpu": 0}
        self.baseline = ToolCallObserver(
            tokenizer=tokenizer,
            candidate_names=self.candidate_names,
            stable_n=stable_n,
            top_k=1,
            enable_stable_prediction=False,
        )
        self._candidate_ids = {
            name: tuple(self._encode(name)) for name in self.candidate_names
        }
        if any(not ids for ids in self._candidate_ids.values()):
            raise ValueError("candidate tool name encoded to an empty token sequence")
        self._candidate_closed_ids = {
            name: tuple(self._encode(name + '"'))
            for name in self.candidate_names
        }
        self._prefix_matcher = CommittedPrefixMatcher(tokenizer, self.candidate_names)
        self._prefix_evidence: dict[int, dict[str, Any]] = {}
        self._anchor_variants = self._build_anchor_variants()
        self.tracks: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.latent_events: list[dict[str, Any]] = []
        self.fallback_commits: dict[int, dict[str, Any]] = {}
        self.natural_available_by_tool: dict[str, float] = {}
        # One causal first detection per tool identity, independent of tracks.
        # This is a record of triggers, not a real tool/service controller.
        self.first_tool_events: dict[str, dict[str, Any]] = {}
        self.natural_first_events: dict[str, dict[str, Any]] = {}
        self._dual_prompt_offset: int | None = None
        self.omtp_processing_s = 0.0
        self.total_processing_s = 0.0
        self._callback_started = 0.0
        self._callback_base_s = 0.0

    def _profile_sync(self, tensor: torch.Tensor | None) -> None:
        if tensor is None or not tensor.is_cuda:
            return
        started = time.perf_counter()
        torch.cuda.synchronize(tensor.device)
        elapsed = time.perf_counter() - started
        row = self.profile_stats.setdefault(
            "cpu_gpu_synchronization", {"total_s": 0.0, "calls": 0}
        )
        row["total_s"] = float(row["total_s"]) + elapsed
        row["calls"] = int(row["calls"]) + 1

    @contextmanager
    def _profile(self, name: str, tensor: torch.Tensor | None = None):
        if not self.profile_enabled:
            yield
            return
        self._profile_sync(tensor)
        started = time.perf_counter()
        try:
            yield
        finally:
            self._profile_sync(tensor)
            elapsed = time.perf_counter() - started
            row = self.profile_stats.setdefault(
                name, {"total_s": 0.0, "calls": 0}
            )
            row["total_s"] = float(row["total_s"]) + elapsed
            row["calls"] = int(row["calls"]) + 1

    def _record_transfer(self, direction: str, bytes_count: int) -> None:
        if self.profile_enabled:
            self.profile_transfer_bytes[direction] += int(bytes_count)

    def profile_summary(self) -> dict[str, Any]:
        forwards = max(1, self.forward_count)
        modules = {}
        for name, values in self.profile_stats.items():
            total = float(values["total_s"])
            modules[name] = {
                "total_s": total,
                "mean_per_forward_s": total / forwards,
                "calls": int(values["calls"]),
            }
        return {
            "enabled": self.profile_enabled,
            "forward_count": self.forward_count,
            "modules": modules,
            "transfer_bytes": dict(self.profile_transfer_bytes),
        }

    def _confidence_gate(self, score: dict[str, Any]) -> bool:
        if (
            self.config.min_top1_score is not None
            and float(score["top1_score"]) < self.config.min_top1_score
        ):
            return False
        if self.config.min_margin is not None:
            margin = score.get("margin")
            if margin is None or float(margin) < self.config.min_margin:
                return False
        return True

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self.baseline.steps

    @property
    def first_natural_by_tool(self):
        return self.baseline.first_natural_by_tool

    def _encode(self, text: str) -> list[int]:
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = self.tokenizer.encode(text)
        return [int(x) for x in ids]

    def _build_anchor_variants(self) -> list[dict[str, Any]]:
        canonical = ('name":"', 'name": "', '"name":"', '"name": "')
        texts = list(canonical)
        for before in (" ", "\t", "\n", "\n  "):
            for after in ("", " ", "\t", "\n", "\n  "):
                texts.extend((f'name"{before}:{after}"', f'"name"{before}:{after}"'))
        unique: dict[tuple[int, ...], dict[str, Any]] = {}
        for text in dict.fromkeys(texts):
            ids = tuple(self._encode(text))
            if not ids:
                continue
            row = {"text": text, "ids": ids, "speculative": text in canonical}
            old = unique.get(ids)
            if old is None or (row["speculative"] and not old["speculative"]):
                unique[ids] = row
        if not unique:
            raise ValueError("tokenizer produced no JSON name anchor variants")
        return list(unique.values())

    def _relative_now(self) -> float:
        return self._callback_base_s + (time.perf_counter() - self._callback_started)

    def __call__(self, observation: ForwardObservation) -> None:
        callback_start = time.perf_counter()
        self.forward_count += 1
        if observation.cache_mode == "dual" and self._dual_prompt_offset is None:
            self._dual_prompt_offset = observation.sequence_offset
        self._callback_started = callback_start
        self._callback_base_s = observation.commit_elapsed_s
        self.baseline(observation)
        baseline_step = self.baseline.steps[-1]
        for name in baseline_step["committed_tool_types"]:
            event = {
                "predicted_tool": name, "source": "natural",
                "forward_step": observation.forward_index,
                "predict_time_s": baseline_step["prediction_available_time_s"],
            }
            self.natural_first_events.setdefault(name, event)
            self.first_tool_events.setdefault(name, event)
        omtp_start = time.perf_counter()
        self._callback_started = callback_start
        self._callback_base_s = observation.commit_elapsed_s
        detection_start_s = self._relative_now()
        self._observe_omtp(observation, detection_start_s)
        omtp_end = time.perf_counter()
        self.omtp_processing_s += omtp_end - omtp_start
        self.total_processing_s += omtp_end - callback_start
        step = self.baseline.steps[-1]
        step["omtp_detection_start_time_s"] = detection_start_s
        step["omtp_detection_complete_time_s"] = self._relative_now()
        step["omtp_processing_s"] = omtp_end - omtp_start
        step["combined_observer_processing_s"] = omtp_end - callback_start
        step["omtp_latent_committed_tools"] = sorted(
            {event["predicted_tool"] for event in self.latent_events}
        )
        step["omtp_fallback_committed_tools"] = sorted(
            {event["predicted_tool"] for event in self.fallback_commits.values()}
        )
        if not self.record_history:
            # Scoring needs only the current committed IDs, never past steps.
            self.baseline.steps[:] = [step]
            self.trajectory.clear()

    def _temporary_ids(self, observation: ForwardObservation) -> torch.Tensor:
        generation = observation.committed_x[0, observation.prompt_length:]
        logits = observation.logits[0, observation.prompt_length:, :]
        top1 = logits.argmax(dim=-1)
        return torch.where(generation == observation.mask_id, top1, generation)

    def _outer_object_anchor(
        self, temporary_ids: list[int], relative_start: int
    ) -> bool:
        prefix = self.tokenizer.decode(
            temporary_ids[:relative_start],
            skip_special_tokens=True,
        )
        # A tool-call name lives in a top-level array element object (depth 1).
        # Depth >=2 is conservatively treated as an arguments/nested-object key.
        return _json_object_depth(prefix) <= 1

    def _detect_anchors(
        self, observation: ForwardObservation, temporary_ids: list[int]
    ) -> list[dict[str, Any]]:
        prompt = observation.prompt_length
        generation = observation.committed_x[0, prompt:]
        logits = observation.logits[0, prompt:, :]
        row_max = logits.float().amax(dim=-1)
        by_start: dict[int, dict[str, Any]] = {}
        length = generation.shape[0]
        for variant in self._anchor_variants:
            pattern = variant["ids"]
            width = len(pattern)
            count = length - width + 1
            if count <= 0:
                continue
            scores = torch.zeros(count, device=logits.device, dtype=torch.float32)
            compatible = torch.ones(count, device=logits.device, dtype=torch.bool)
            observed = torch.zeros(count, device=logits.device, dtype=torch.float32)
            for offset, token_id in enumerate(pattern):
                selected = logits[offset:offset + count, token_id].float()
                scores += selected - row_max[offset:offset + count]
                current = generation[offset:offset + count]
                compatible &= (current == observation.mask_id) | (current == token_id)
                observed += (current == token_id).float()
            scores /= float(width)
            observed /= float(width)
            eligible = compatible & (
                ((scores >= self.config.anchor_min_logit_margin) | (observed == 1.0))
                if variant["speculative"] else (observed == 1.0)
            )
            local_indices = torch.nonzero(eligible, as_tuple=False).flatten()
            local_list = local_indices.detach().cpu().tolist()
            score_list = scores[local_indices].detach().cpu().tolist()
            observed_list = observed[local_indices].detach().cpu().tolist()
            self._record_transfer(
                "gpu_to_cpu", len(local_list) * (8 + 4 + 4)
            )
            for local, score_value, observed_value in zip(local_list, score_list, observed_list):
                if not self._outer_object_anchor(temporary_ids, int(local)):
                    continue
                absolute = prompt + int(local)
                row = {
                    "anchor_start": absolute,
                    "anchor_end": absolute + width,
                    "value_start": absolute + width,
                    "anchor_variant": variant["text"],
                    "anchor_token_ids": list(pattern),
                    "anchor_score": float(score_value),
                    "anchor_observed_ratio": float(observed_value),
                }
                old = by_start.get(absolute)
                if old is None or (row["anchor_observed_ratio"], row["anchor_score"]) > (
                    old["anchor_observed_ratio"], old["anchor_score"]
                ):
                    by_start[absolute] = row
        cluster_radius = max(len(row["ids"]) for row in self._anchor_variants)
        clusters: list[list[dict[str, Any]]] = []
        for row in sorted(by_start.values(), key=lambda x: x["anchor_start"]):
            if not clusters or row["anchor_start"] - clusters[-1][-1]["anchor_start"] >= cluster_radius:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        ranked = [
            max(group, key=lambda x: (x["anchor_observed_ratio"], x["anchor_score"]))
            for group in clusters
        ]
        selected: list[dict[str, Any]] = []
        for row in ranked:
            if all(abs(row["anchor_start"] - old["anchor_start"]) >= self.config.min_anchor_gap for old in selected):
                selected.append(row)
        return selected

    def _new_track(self, anchor: dict[str, Any], forward: int) -> dict[str, Any]:
        track = {
            "track_id": len(self.tracks),
            "first_forward": forward,
            "last_forward": forward,
            "seen_count": 0,
            "center": anchor["value_start"],
            "center_history": [],
            "last_prediction": None,
            "stable_count": 0,
            "latent_triggered": False,
            "prefix_triggered": False,
            "natural_triggered": False,
        }
        self.tracks.append(track)
        return track

    def _associate(
        self, anchors: list[dict[str, Any]], forward: int
    ) -> list[tuple[dict[str, Any], dict[str, Any], int | None]]:
        used: set[int] = set()
        result = []
        for anchor in sorted(anchors, key=lambda x: x["value_start"]):
            choices = []
            for track in self.tracks:
                if track["track_id"] in used:
                    continue
                misses = forward - int(track["last_forward"]) - 1
                distance = abs(int(anchor["value_start"]) - int(track["center"]))
                if misses <= self.config.max_track_misses and distance <= self.config.merge_distance:
                    choices.append((distance, -int(track["seen_count"]), track["track_id"], track))
            track = min(choices)[-1] if choices else self._new_track(anchor, forward)
            previous = int(track["center"]) if track["seen_count"] else None
            track["seen_count"] += 1
            track["last_forward"] = forward
            track["center"] = int(anchor["value_start"])
            track["center_history"].append(int(anchor["value_start"]))
            used.add(int(track["track_id"]))
            result.append((track, anchor, previous))
        return result

    def _score_region(
        self,
        observation: ForwardObservation,
        center: int,
    ) -> dict[str, Any] | None:
        prompt = observation.prompt_length
        generation = observation.committed_x[0, prompt:]
        committed_ids = self.baseline.steps[-1]["generation_token_ids_after"]
        logits = observation.logits[0, prompt:, :].float()
        total = generation.shape[0]
        left = max(0, center - prompt - self.config.region_radius)
        right = min(total - 1, center - prompt + self.config.region_radius)
        candidates: dict[str, dict[str, Any]] = {}
        with self._profile("candidate_token_scoring", observation.logits):
            partitions = torch.logsumexp(logits, dim=-1)
            alignment_rows = {}
            for name, ids in self._candidate_ids.items():
                alignment_tensors = []
                alignment_positions = []
                for start in range(left, min(right, total - len(ids)) + 1):
                    current = committed_ids[start:start + len(ids)]
                    if any(actual not in (observation.mask_id, expected) for actual, expected in zip(current, ids)):
                        continue
                    score = torch.zeros((), device=logits.device, dtype=torch.float32)
                    for offset, token_id in enumerate(ids):
                        score += logits[start + offset, token_id] - partitions[start + offset]
                    alignment_tensors.append(score / len(ids))
                    alignment_positions.append(prompt + start)
                if alignment_tensors:
                    alignment_rows[name] = (
                        torch.stack(alignment_tensors), alignment_positions
                    )
        with self._profile("region_scoring", observation.logits):
            for name, (stacked, alignment_positions) in alignment_rows.items():
                top_count = min(self.config.region_top_k, len(alignment_positions))
                top_values, top_indices = torch.topk(stacked, k=top_count)
                best_index = int(top_indices[0].detach().cpu())
                self._record_transfer("gpu_to_cpu", 8 + 4 * top_count)
                candidates[name] = {
                    "score": float(top_values.mean().detach().cpu()),
                    "best_position": alignment_positions[best_index],
                    "alignment_count": len(alignment_positions),
                }
        if not candidates:
            return None
        ranked = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        top1_name, top1 = ranked[0]
        top2 = ranked[1][1]["score"] if len(ranked) > 1 else None
        return {
            "predicted_tool": top1_name,
            "top1_score": top1["score"],
            "top2_score": top2,
            "margin": None if top2 is None else top1["score"] - top2,
            "best_position": top1["best_position"],
            "region_left": prompt + left,
            "region_right": prompt + right,
            "candidate_scores": candidates,
            "name_uncommitted_ratio": sum(
                token_id == observation.mask_id
                for token_id in committed_ids[
                    top1["best_position"] - prompt:
                    top1["best_position"] - prompt
                    + len(self._candidate_ids[top1_name])
                ]
            ) / float(len(self._candidate_ids[top1_name])),
        }

    def _prefix_and_natural(
        self, observation: ForwardObservation, center: int
    ) -> tuple[tuple[str, str, int] | None, tuple[str, int] | None]:
        prompt = observation.prompt_length
        generation = self.baseline.steps[-1]["generation_token_ids_after"]
        left = max(0, center - prompt - self.config.region_radius)
        right = min(len(generation) - 1, center - prompt + self.config.region_radius)
        best_prefix = None
        natural = None
        for start in range(left, right + 1):
            for name, ids in self._candidate_ids.items():
                closed = self._candidate_closed_ids[name]
                if start + len(closed) <= len(generation):
                    value = generation[start:start + len(closed)]
                    if value == list(closed):
                        natural = (name, prompt + start)
                # SentencePiece/BPE may merge the closing quote with the next
                # comma/brace (for example a token decoding to `",`).  Accept
                # the same complete name tokenization followed by any fully
                # committed token whose decoded surface starts with a quote.
                if start + len(ids) < len(generation):
                    value = generation[start:start + len(ids)]
                    boundary = generation[start + len(ids)]
                    if (
                        value == list(ids)
                        and boundary != observation.mask_id
                        and self.tokenizer.decode(
                            [boundary], skip_special_tokens=True
                        ).lstrip().startswith('"')
                    ):
                        natural = (name, prompt + start)
        # Prefixes must begin at THIS name field, not an arbitrary nearby
        # token. Read the whole committed run and reject contradictory suffix
        # islands rather than salvaging a candidate's shorter matching part.
        match = self._prefix_matcher.match(generation, center - prompt, observation.mask_id)
        if match is not None:
            if observation.cache_mode == "dual":
                match["generation_position"] += observation.sequence_offset - (
                    self._dual_prompt_offset if self._dual_prompt_offset is not None else observation.sequence_offset
                )
            self._prefix_evidence[center] = match
            best_prefix = (match["predicted_tool"], match["prefix"], center)
        return best_prefix, natural

    def _emit(
        self,
        *,
        track: dict[str, Any],
        source: str,
        tool: str,
        observation: ForwardObservation,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        event = {
            "track_id": int(track["track_id"]),
            "predicted_tool": tool,
            "source": source,
            "forward_step": observation.forward_index,
            "predict_time_s": self._relative_now(),
            **extra,
        }
        self.events.append(event)
        self.first_tool_events.setdefault(tool, event)
        if source == "latent_region":
            self.latent_events.append(event)
        self.fallback_commits.setdefault(int(track["track_id"]), event)
        return event

    def _observe_omtp(
        self, observation: ForwardObservation, detection_start_s: float
    ) -> None:
        with self._profile("candidate_token_scoring", observation.logits):
            temporary = self._temporary_ids(observation)
            temporary_ids = temporary.detach().cpu().tolist()
            self._record_transfer("gpu_to_cpu", temporary.numel() * temporary.element_size())
        with self._profile("anchor_detection", observation.logits):
            anchors = self._detect_anchors(observation, temporary_ids)
        with self._profile("candidate_tracking_merge"):
            associated = self._associate(anchors, observation.forward_index)
        active_rows = []
        for track, anchor, previous in associated:
            span = observation.forward_index - int(track["first_forward"]) + 1
            support = int(track["seen_count"]) / float(span)
            delta = None if previous is None else int(track["center"]) - previous
            position_ok = delta is None or abs(delta) <= self.config.position_tolerance
            validated = int(track["seen_count"]) >= self.config.min_seen and support >= self.config.min_support
            score = self._score_region(observation, int(track["center"])) if validated else None
            with self._profile("temporal_stability"):
                gate_passed = bool(
                    score is not None and self._confidence_gate(score)
                )
                if score is not None and position_ok and gate_passed:
                    predicted = score["predicted_tool"]
                    track["stable_count"] = int(track["stable_count"]) + 1 if track["last_prediction"] == predicted else 1
                    track["last_prediction"] = predicted
                else:
                    track["stable_count"] = 0
                    track["last_prediction"] = None
            row = {
                "forward_index": observation.forward_index,
                "block_index": observation.block_index,
                "block_step": observation.block_step,
                "forward_complete_time_s": observation.elapsed_s,
                "observer_detection_start_time_s": detection_start_s,
                "track_id": int(track["track_id"]),
                "anchor_start": anchor["anchor_start"],
                "anchor_value_start": anchor["value_start"],
                "anchor_score": anchor["anchor_score"],
                "anchor_observed_ratio": anchor["anchor_observed_ratio"],
                "anchor_variant": anchor["anchor_variant"],
                "position_delta": delta,
                "position_ok": position_ok,
                "seen_count": int(track["seen_count"]),
                "support_ratio": support,
                "validated": validated,
                "stable_count": int(track["stable_count"]),
                "confidence_gate_passed": gate_passed,
                **(score or {"predicted_tool": None, "top1_score": None, "top2_score": None, "margin": None, "best_position": None, "candidate_scores": {}}),
            }
            if score is not None and int(track["stable_count"]) >= self.config.stable_n and not track["latent_triggered"]:
                track["latent_triggered"] = True
                event = self._emit(
                    track=track,
                    source="latent_region",
                    tool=str(score["predicted_tool"]),
                    observation=observation,
                    extra={
                        "stable_count": int(track["stable_count"]),
                        "margin": score["margin"],
                        "top1_score": score["top1_score"],
                        "top2_score": score["top2_score"],
                        "best_position": score["best_position"],
                        "seen_count": int(track["seen_count"]),
                        "support_ratio": support,
                        "name_uncommitted_ratio": score[
                            "name_uncommitted_ratio"
                        ],
                    },
                )
                row["latent_trigger"] = event
            with self._profile("prefix_natural_detection"):
                prefix, natural = self._prefix_and_natural(
                    observation, int(track["center"])
                )
            if prefix is not None and not track["prefix_triggered"]:
                track["prefix_triggered"] = True
                event = self._emit(
                    track=track, source="prefix", tool=prefix[0], observation=observation,
                    extra={
                        "prefix": prefix[1], "best_position": prefix[2],
                        "prefix_validation": dict(self._prefix_evidence[prefix[2]]),
                    },
                )
                row["prefix_event"] = event
            if natural is not None and not track["natural_triggered"]:
                track["natural_triggered"] = True
                event = self._emit(
                    track=track, source="natural", tool=natural[0], observation=observation,
                    extra={"best_position": natural[1]},
                )
                row["natural_event"] = event
            active_rows.append(row)
        completed = self._relative_now()
        for row in active_rows:
            row["observer_detection_complete_time_s"] = completed
            self.trajectory.append(row)

        baseline_step = self.baseline.steps[-1]
        available = float(baseline_step["prediction_available_time_s"])
        for name in baseline_step["committed_tool_types"]:
            self.natural_available_by_tool.setdefault(name, available)

    def fallback_events(self) -> list[dict[str, Any]]:
        """Return the immutable first online event for every tracked occurrence."""

        return sorted(
            self.fallback_commits.values(), key=lambda event: event["predict_time_s"]
        )


__all__ = ["OMTPConfig", "OMTPObserver", "_json_object_depth"]
