"""Native Hugging Face greedy AR generation with a read-only Natural callback.

The extra StoppingCriteria always returns False: it observes each newly
committed token, never forces a tool or changes the stopping/token policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from fastdllm_generate import ForwardObservation
from natural_observer import NaturalOnlyObserver
from run_experiment import _seed_everything, _sync


def resolve_eos_ids(model: Any, tokenizer: Any) -> list[int]:
    ids = []
    for source in (model.generation_config, model.config, tokenizer):
        value = getattr(source, "eos_token_id", None)
        if value is not None:
            ids.extend(value if isinstance(value, (list, tuple)) else [value])
    # Llama 3 Instruct's turn terminator, only when actually in this vocabulary.
    eot = tokenizer.get_vocab().get("<|eot_id|>")
    if eot is not None:
        ids.append(eot)
    return list(dict.fromkeys(int(token) for token in ids))


def greedy_config(model: Any, tokenizer: Any, max_new_tokens: int | None = None) -> GenerationConfig:
    """Generate until EOS by default, bounded only by the model's context.

    Setting no explicit length would let HF's default max_length=20 (or a
    checkpoint's saved limit) truncate the run. Override that with the actual
    architectural context limit, NOT an experimental output-token budget.
    An optional explicit budget is retained only for diagnostic ablations.
    """
    eos_ids = resolve_eos_ids(model, tokenizer)
    if not eos_ids:
        raise ValueError("natural AR generation requires checkpoint/tokenizer EOS IDs")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit < 1:
        raise ValueError("AR checkpoint must expose a finite positive context limit")
    length_options = {"max_length": context_limit} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
    return GenerationConfig(
        **length_options,
        do_sample=False, num_beams=1,
        use_cache=True, eos_token_id=eos_ids or None,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        return_dict_in_generate=False, output_scores=False,
    )


@dataclass
class ARGenerationResult:
    output: torch.Tensor
    forward_count: int
    generation_time_s: float
    first_token_time_s: float | None
    prefill_forward_time_s: float | None
    model_forward_time_s: float


class _ARMonitor(StoppingCriteria):
    def __init__(self, *, prompt_length, device, observer):
        self.prompt_length = prompt_length
        self.device = device
        self.observer = observer
        self.started = 0.0
        self.forward_count = 0
        self.token_count = 0
        self.forward_started = 0.0
        self.forward_complete_s = 0.0
        self.last_forward_time_s = 0.0
        self.forward_total_s = 0.0
        self.prefill_forward_s = None
        self.first_token_s = None

    def before_forward(self, module, inputs):
        _sync(self.device)
        self.forward_started = time.perf_counter()

    def after_forward(self, module, inputs, output):
        _sync(self.device)
        finished = time.perf_counter()
        self.forward_count += 1
        self.forward_complete_s = finished - self.started
        self.last_forward_time_s = finished - self.forward_started
        self.forward_total_s += self.last_forward_time_s
        if self.forward_count == 1:
            self.prefill_forward_s = self.last_forward_time_s

    def __call__(self, input_ids, scores, **kwargs):
        # Called by native generate AFTER token selection and append, including
        # the final EOS token. Synchronize before assigning device-complete time.
        _sync(self.device)
        committed_s = time.perf_counter() - self.started
        self.token_count = int(input_ids.shape[1]) - self.prompt_length
        if self.first_token_s is None:
            self.first_token_s = committed_s
        if self.observer is not None:
            self.observer(ForwardObservation(
                forward_index=self.forward_count, block_index=0,
                block_step=self.token_count - 1,
                elapsed_s=self.forward_complete_s, commit_elapsed_s=committed_s,
                forward_time_s=self.last_forward_time_s,
                prompt_length=self.prompt_length, block_start=self.prompt_length,
                block_end=int(input_ids.shape[1]), mask_id=-1,
                x=None, committed_x=input_ids, logits=None, cache_mode="ar_kv",
            ))
        # No GT/final-output-conditioned stopping, JSON completion stop, forced
        # names, logits processors or extra model forwards.
        return False


def generate_ar(
    *, model: Any, tokenizer: Any, prompt_ids: torch.Tensor,
    max_new_tokens: int | None = None, seed: int, device: torch.device,
    observer: NaturalOnlyObserver | None,
) -> ARGenerationResult:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("AR baseline requires batch_size=1")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    context_limit = int(model.config.max_position_embeddings)
    prompt_length = int(prompt_ids.shape[1])
    if prompt_length >= context_limit:
        raise ValueError("prompt leaves no space in the AR model context")
    if max_new_tokens is not None and prompt_length + max_new_tokens > context_limit:
        raise ValueError("explicit output budget exceeds the AR context")
    config = greedy_config(model, tokenizer, max_new_tokens)
    monitor = _ARMonitor(prompt_length=int(prompt_ids.shape[1]), device=device, observer=observer)
    _seed_everything(seed)
    # Both On/Off retain these timing hooks. Off disables only name detection,
    # not instrumentation. Always remove hooks, including failed requests.
    pre_hook = model.register_forward_pre_hook(monitor.before_forward)
    post_hook = model.register_forward_hook(monitor.after_forward)
    try:
        _sync(device)
        monitor.started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                generation_config=config,
                stopping_criteria=StoppingCriteriaList([monitor]),
            )
        _sync(device)
        elapsed = time.perf_counter() - monitor.started
    finally:
        pre_hook.remove()
        post_hook.remove()
    generated_count = int(output.shape[1] - prompt_ids.shape[1])
    if monitor.forward_count != generated_count or monitor.token_count != generated_count:
        raise RuntimeError("native AR forward/token callback count differs; inspect Transformers compatibility")
    return ARGenerationResult(
        output=output, forward_count=monitor.forward_count,
        generation_time_s=elapsed, first_token_time_s=monitor.first_token_s,
        prefill_forward_time_s=monitor.prefill_forward_s,
        model_forward_time_s=monitor.forward_total_s,
    )
