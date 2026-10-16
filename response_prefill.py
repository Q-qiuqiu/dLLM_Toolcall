"""Optional response-side structural prefill for model-family ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RESPONSE_MODES = ("native", "json_prefill")
JSON_PREFILL_TEXT = '[{"name":"'


@dataclass(frozen=True)
class ResponsePrefill:
    mode: str
    text: str
    token_ids: tuple[int, ...]


def resolve_response_prefill(
    tokenizer: Any, *, model_family: str, mode: str
) -> ResponsePrefill:
    """Resolve the committed generation prefix for an explicit response mode.

    ``json_prefill`` is intentionally restricted to iLLaDA.  It is an
    experimental decoder-initialization ablation, not a prompt change and not
    a model-native switch for disabling thinking.  The prefix contains JSON
    syntax only; candidate tool names remain unknown to this function.
    """

    if mode not in RESPONSE_MODES:
        raise ValueError(
            f"unsupported response mode {mode!r}; expected one of {RESPONSE_MODES}"
        )
    if mode == "native":
        return ResponsePrefill(mode=mode, text="", token_ids=())
    if model_family != "illada":
        raise ValueError(
            "--response-mode json_prefill is an iLLaDA-only ablation; "
            "use --response-mode native for other model families"
        )

    token_ids = tokenizer.encode(JSON_PREFILL_TEXT, add_special_tokens=False)
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        raise ValueError("tokenizer produced no tokens for the JSON prefill")
    normalized = tuple(int(token_id) for token_id in token_ids)
    if any(token_id < 0 for token_id in normalized):
        raise ValueError("tokenizer produced an invalid token id for the JSON prefill")
    return ResponsePrefill(
        mode=mode, text=JSON_PREFILL_TEXT, token_ids=normalized
    )


__all__ = [
    "JSON_PREFILL_TEXT", "RESPONSE_MODES", "ResponsePrefill",
    "resolve_response_prefill",
]
