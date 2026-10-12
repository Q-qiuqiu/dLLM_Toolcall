"""Performance-oriented OMTP observers preserving the original online policy."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch

from fastdllm_generate import ForwardObservation
from omtp_observer import OMTPConfig, OMTPObserver


class OptimizedOMTPObserver(OMTPObserver):
    """OMTP with fused argmax/row-max and batched local candidate scoring."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        candidate_names: Iterable[str],
        stable_n: int = 2,
        config: OMTPConfig | None = None,
        profile_enabled: bool = False,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            candidate_names=candidate_names,
            stable_n=stable_n,
            config=config,
            profile_enabled=profile_enabled,
        )
        self._cached_top_ids: torch.Tensor | None = None
        self._cached_row_max: torch.Tensor | None = None
        self._partition_cache: torch.Tensor | None = None
        self._token_surface_cache: dict[int, str] = {}
        self._cached_outer_depths: list[int] = []

    def _build_outer_depths(self, token_ids: list[int]) -> list[int]:
        depths = []
        depth = 0
        quoted = False
        escaped = False
        for token_id in token_ids:
            depths.append(depth)
            surface = self._token_surface_cache.get(token_id)
            if surface is None:
                surface = self.tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                self._token_surface_cache[token_id] = surface
            for char in surface:
                if escaped:
                    escaped = False
                elif char == "\\" and quoted:
                    escaped = True
                elif char == '"':
                    quoted = not quoted
                elif not quoted and char == "{":
                    depth += 1
                elif not quoted and char == "}":
                    depth = max(0, depth - 1)
        return depths

    def _observe_omtp(
        self, observation: ForwardObservation, detection_start_s: float
    ) -> None:
        self._cached_top_ids = None
        self._cached_row_max = None
        generation_length = observation.logits.shape[1] - observation.prompt_length
        self._partition_cache = torch.full(
            (generation_length,),
            torch.nan,
            device=observation.logits.device,
            dtype=torch.float32,
        )
        try:
            super()._observe_omtp(observation, detection_start_s)
        finally:
            self._cached_top_ids = None
            self._cached_row_max = None
            self._partition_cache = None

    def _ensure_top(self, observation: ForwardObservation) -> None:
        if self._cached_top_ids is not None:
            return
        logits = observation.logits[0, observation.prompt_length:, :]
        values, ids = torch.max(logits, dim=-1)
        self._cached_row_max = values.float()
        self._cached_top_ids = ids

    def _temporary_ids(self, observation: ForwardObservation) -> torch.Tensor:
        self._ensure_top(observation)
        generation = observation.committed_x[0, observation.prompt_length:]
        return torch.where(
            generation == observation.mask_id,
            self._cached_top_ids,
            generation,
        )

    def _detect_anchors(
        self, observation: ForwardObservation, temporary_ids: list[int]
    ) -> list[dict[str, Any]]:
        self._ensure_top(observation)
        prompt = observation.prompt_length
        generation = observation.committed_x[0, prompt:]
        committed_ids = self.baseline.steps[-1]["generation_token_ids_after"]
        logits = observation.logits[0, prompt:, :]
        row_max = self._cached_row_max
        length = generation.shape[0]
        by_start: dict[int, dict[str, Any]] = {}
        self._cached_outer_depths = self._build_outer_depths(temporary_ids)

        for variant in self._anchor_variants:
            pattern = tuple(variant["ids"])
            width = len(pattern)
            count = length - width + 1
            if count <= 0:
                continue
            if variant["speculative"]:
                starts = torch.arange(count, device=logits.device)
                offsets = torch.arange(width, device=logits.device)
                positions = starts[:, None] + offsets[None, :]
                token_ids = torch.tensor(pattern, device=logits.device)
                self._record_transfer("cpu_to_gpu", token_ids.numel() * token_ids.element_size())
                selected = logits[positions, token_ids[None, :]].float()
                scores = (
                    selected - row_max[positions]
                ).mean(dim=1)
                current = generation[positions]
                expected = token_ids[None, :]
                compatible = ((current == observation.mask_id) | (current == expected)).all(dim=1)
                observed = (current == expected).float().mean(dim=1)
                eligible = compatible & (
                    (scores >= self.config.anchor_min_logit_margin)
                    | (observed == 1.0)
                )
                indices = torch.nonzero(eligible, as_tuple=False).flatten()
                local_list = indices.detach().cpu().tolist()
                score_list = scores[indices].detach().cpu().tolist()
                observed_list = observed[indices].detach().cpu().tolist()
                self._record_transfer(
                    "gpu_to_cpu", len(local_list) * (8 + 4 + 4)
                )
            else:
                local_list = [
                    start
                    for start in range(count)
                    if committed_ids[start:start + width] == list(pattern)
                ]
                if local_list:
                    positions = torch.tensor(
                        [
                            [start + offset for offset in range(width)]
                            for start in local_list
                        ],
                        device=logits.device,
                    )
                    token_ids = torch.tensor(pattern, device=logits.device)
                    self._record_transfer("cpu_to_gpu", token_ids.numel() * token_ids.element_size())
                    selected = logits[positions, token_ids[None, :]].float()
                    score_list = (
                        selected - row_max[positions]
                    ).mean(dim=1).detach().cpu().tolist()
                    observed_list = [1.0] * len(local_list)
                    self._record_transfer(
                        "gpu_to_cpu", len(local_list) * (8 + 4)
                    )
                else:
                    score_list, observed_list = [], []

            for local, score_value, observed_value in zip(
                local_list, score_list, observed_list
            ):
                if (
                    int(local) >= len(self._cached_outer_depths)
                    or self._cached_outer_depths[int(local)] > 1
                ):
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
                if old is None or (
                    row["anchor_observed_ratio"], row["anchor_score"]
                ) > (old["anchor_observed_ratio"], old["anchor_score"]):
                    by_start[absolute] = row

        cluster_radius = max(len(row["ids"]) for row in self._anchor_variants)
        clusters: list[list[dict[str, Any]]] = []
        for row in sorted(by_start.values(), key=lambda item: item["anchor_start"]):
            if (
                not clusters
                or row["anchor_start"] - clusters[-1][-1]["anchor_start"]
                >= cluster_radius
            ):
                clusters.append([row])
            else:
                clusters[-1].append(row)
        ranked = [
            max(
                cluster,
                key=lambda item: (
                    item["anchor_observed_ratio"], item["anchor_score"]
                ),
            )
            for cluster in clusters
        ]
        selected_rows: list[dict[str, Any]] = []
        for row in ranked:
            if all(
                abs(row["anchor_start"] - old["anchor_start"])
                >= self.config.min_anchor_gap
                for old in selected_rows
            ):
                selected_rows.append(row)
        return selected_rows

    def _ensure_partitions(
        self, logits: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        unique = torch.unique(positions.flatten())
        missing = torch.isnan(self._partition_cache[unique])
        rows = unique[missing]
        if rows.numel():
            self._partition_cache[rows] = torch.logsumexp(
                logits[rows].float(), dim=-1
            )
        return self._partition_cache

    def _score_bounds(
        self,
        observation: ForwardObservation,
        *,
        left: int,
        right: int,
        top_k: int,
    ) -> dict[str, Any] | None:
        prompt = observation.prompt_length
        logits = observation.logits[0, prompt:, :].float()
        committed_ids = self.baseline.steps[-1]["generation_token_ids_after"]
        total = len(committed_ids)
        candidates: dict[str, dict[str, Any]] = {}
        by_length: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
        for name, ids in self._candidate_ids.items():
            by_length[len(ids)].append((name, ids))

        with self._profile("candidate_token_scoring", observation.logits):
            score_groups = []
            for width, group in by_length.items():
                starts = list(range(left, min(right, total - width) + 1))
                if not starts:
                    continue
                compatible = torch.tensor(
                    [
                        [
                            all(
                                actual in (observation.mask_id, expected)
                                for actual, expected in zip(
                                    committed_ids[start:start + width], ids
                                )
                            )
                            for start in starts
                        ]
                        for _, ids in group
                    ],
                    device=logits.device,
                    dtype=torch.bool,
                )
                self._record_transfer("cpu_to_gpu", compatible.numel())
                start_tensor = torch.tensor(starts, device=logits.device)
                offsets = torch.arange(width, device=logits.device)
                positions = start_tensor[:, None] + offsets[None, :]
                token_tensor = torch.tensor(
                    [list(ids) for _, ids in group], device=logits.device
                )
                self._record_transfer(
                    "cpu_to_gpu",
                    start_tensor.numel() * start_tensor.element_size()
                    + token_tensor.numel() * token_tensor.element_size(),
                )
                selected = logits[
                    positions[None, :, :], token_tensor[:, None, :]
                ]
                partitions = self._ensure_partitions(logits, positions)
                scores = (
                    selected - partitions[positions][None, :, :]
                ).mean(dim=-1)
                scores = torch.where(
                    compatible,
                    scores,
                    torch.full_like(scores, -torch.inf),
                )
                score_groups.append((group, start_tensor, scores))

        with self._profile("region_scoring", observation.logits):
            for group, start_tensor, scores in score_groups:
                for index, (name, _) in enumerate(group):
                    valid = torch.isfinite(scores[index])
                    count = int(valid.sum().item())
                    if not count:
                        continue
                    valid_scores = scores[index][valid]
                    valid_starts = start_tensor[valid]
                    count_top = min(top_k, count)
                    values, order = torch.topk(valid_scores, k=count_top)
                    best_position = int(valid_starts[order[0]].detach().cpu())
                    candidates[name] = {
                        "score": float(values.mean().detach().cpu()),
                        "best_position": prompt + best_position,
                        "alignment_count": count,
                    }
                    self._record_transfer(
                        "gpu_to_cpu", 8 + 4 * count_top
                    )
        if not candidates:
            return None
        ranked = sorted(
            candidates.items(), key=lambda item: item[1]["score"], reverse=True
        )
        top1_name, top1 = ranked[0]
        top2 = ranked[1][1]["score"] if len(ranked) > 1 else None
        start = top1["best_position"] - prompt
        width = len(self._candidate_ids[top1_name])
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
                token == observation.mask_id
                for token in committed_ids[start:start + width]
            ) / float(width),
        }

    def _score_region(
        self, observation: ForwardObservation, center: int
    ) -> dict[str, Any] | None:
        prompt = observation.prompt_length
        total = observation.logits.shape[1] - prompt
        left = max(0, center - prompt - self.config.region_radius)
        right = min(total - 1, center - prompt + self.config.region_radius)
        result = self._score_bounds(
            observation, left=left, right=right,
            top_k=self.config.region_top_k,
        )
        if result is not None:
            result["scoring_mode"] = "optimized_full_region"
        return result


__all__ = ["OptimizedOMTPObserver"]
