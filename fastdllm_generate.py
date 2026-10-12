"""Fast-dLLM v1 no-cache LLaDA generation with an optional observer hook.

The sampling and token-transfer code is copied from
``Fast-dLLM/v1/llada/generate.py`` at repository commit
``aadcf060e12b6413b494bf9aebb172cfe3cf911c``.  The only behavioral extension
is ``observation_callback``.  When it is ``None`` (the default), no timing,
device synchronization, decoding, or tensor copy is added to the generation
loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ForwardObservation:
    """Read-only snapshots for one forward and its unchanged token transfer."""

    forward_index: int
    block_index: int
    block_step: int
    elapsed_s: float
    commit_elapsed_s: float
    forward_time_s: float
    prompt_length: int
    block_start: int
    block_end: int
    mask_id: int
    x: torch.Tensor
    committed_x: torch.Tensor
    logits: torch.Tensor


ObservationCallback = Callable[[ForwardObservation], None]


def _synchronize_for(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_selected_token_probability(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    logits_fp32 = logits.float()
    selected_logits = torch.gather(
        logits_fp32, dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    log_partition = torch.logsumexp(logits_fp32, dim=-1)
    return torch.exp(selected_logits - log_partition)


def get_num_transfer_tokens(
    block_mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    remainder = total - base * steps
    result = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    columns = torch.arange(steps, device=block_mask_index.device).unsqueeze(0)
    return result + (columns < remainder.unsqueeze(1)).to(torch.long)


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: Optional[torch.Tensor],
    threshold: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        x0_p = get_selected_token_probability(logits, x0)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    negative_infinity = torch.tensor(
        torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype
    )
    confidence = torch.where(mask_index, x0_p, negative_infinity)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(
            1, max_conf_indices, True
        )
        return (x0, (transfer_index | force_mask) & mask_index)

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens is required when threshold is None")
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = torch.clamp(
        num_transfer_tokens.to(dtype=torch.long, device=confidence.device), min=0
    )
    _, sorted_indices = torch.sort(confidence, dim=1, descending=True)
    batch_size, length = confidence.shape
    columns = torch.arange(length, device=confidence.device).unsqueeze(0).expand(
        batch_size, length
    )
    select_sorted = columns < num_transfer_tokens.unsqueeze(1).expand(
        batch_size, length
    )
    transfer = torch.zeros(
        batch_size, length, device=confidence.device, dtype=torch.int8
    )
    transfer = transfer.scatter(1, sorted_indices, select_sorted.to(torch.int8))
    return x0, transfer.bool() & mask_index


def get_transfer_index_dynamic(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: Optional[torch.Tensor],
    factor: float = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del num_transfer_tokens
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        x0_p = get_selected_token_probability(logits, x0)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    remaining = mask_index.sum(dim=1, keepdim=True)
    for row in range(confidence.shape[0]):
        count = int(remaining[row].item())
        if count == 0:
            continue
        thresholds = [1 - factor / (n + 1) for n in range(1, count + 1)]
        thresholds[0] = -1
        sorted_confidence = torch.sort(
            confidence[row][mask_index[row]], descending=True
        )[0]
        top_i = 0
        for top_i in range(len(thresholds)):
            if sorted_confidence[top_i] < thresholds[top_i]:
                break
        if top_i == 0 or top_i == len(thresholds) - 1:
            top_i += 1
        _, selected = torch.topk(confidence[row], k=top_i)
        transfer_index[row, selected] = True
    return x0, transfer_index


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    observation_callback: Optional[ObservationCallback] = None,
) -> tuple[torch.Tensor, int]:
    """Run the Fast-dLLM v1 no-cache generation path.

    ``x`` is the sequence immediately after the forward and before this step's
    transfer; ``committed_x`` is the sequence after that unchanged transfer.
    The callback is delivered after transfer so observer-side decoding cannot
    inflate ``commit_elapsed_s``.  Callbacks must treat tensors as read-only.
    CUDA is synchronized only when the callback is enabled, making timestamps
    actual device-completion wall times rather than CPU enqueue times.
    """

    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("This experiment requires batch_size=1")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("steps must be divisible by the number of blocks")
    steps_per_block = steps // num_blocks

    generation_start = time.perf_counter()
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_length = int(prompt.shape[1])
    nfe = 0

    for block_index in range(num_blocks):
        block_start = prompt_length + block_index * block_length
        block_end = block_start + block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )
        block_step = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            mask_index[:, block_end:] = False

            observation_fields = None
            if observation_callback is None:
                logits = model(x).logits
            else:
                _synchronize_for(x)
                forward_start = time.perf_counter()
                logits = model(x).logits
                _synchronize_for(logits)
                forward_end = time.perf_counter()
                observation_fields = (x.clone(), forward_start, forward_end)

            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, block_step]
                    if threshold is None
                    else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    None,
                    factor,
                )
            x[transfer_index] = x0[transfer_index]
            if observation_callback is not None:
                _synchronize_for(x)
                commit_end = time.perf_counter()
                x_before, forward_start, forward_end = observation_fields
                observation_callback(
                    ForwardObservation(
                        forward_index=nfe,
                        block_index=block_index,
                        block_step=block_step,
                        elapsed_s=forward_end - generation_start,
                        commit_elapsed_s=commit_end - generation_start,
                        forward_time_s=forward_end - forward_start,
                        prompt_length=prompt_length,
                        block_start=block_start,
                        block_end=block_end,
                        mask_id=mask_id,
                        x=x_before,
                        committed_x=x.clone(),
                        logits=logits,
                    )
                )
            block_step += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            if block_step >= steps_per_block:
                raise RuntimeError(
                    "block still contains MASK after its configured steps"
                )
    return x, nfe


__all__ = [
    "ForwardObservation",
    "ObservationCallback",
    "generate",
    "get_num_transfer_tokens",
    "get_transfer_index",
]
