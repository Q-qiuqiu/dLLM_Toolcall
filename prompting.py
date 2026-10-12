"""Ground-truth-free BFCL prompt construction."""

from __future__ import annotations

import json
from typing import Any

from bfcl_data import BFCLSample


SYSTEM_PROMPT = """You are a function-calling assistant. Select calls only from the provided candidate functions.
Return exactly one JSON array and nothing else. Each array item must have exactly this form:
{"name": "candidate.function_name", "arguments": {"parameter": "value"}}
Use one array item per call. The same function may be called multiple times. Do not include reasoning, thoughts, PLAN, markdown fences, or explanatory text. If several calls are requested, put every call in the single array."""


def _render_question(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )


def build_messages(sample: BFCLSample) -> list[dict[str, str]]:
    """Build model messages without consulting ``sample.ground_truth``."""

    function_json = json.dumps(
        sample.functions, ensure_ascii=False, separators=(",", ":")
    )
    user_content = (
        "Original request:\n"
        f"{_render_question(sample.messages)}\n\n"
        "Candidate function definitions (JSON):\n"
        f"{function_json}\n\n"
        "Return the tool-call JSON array now."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer: Any, sample: BFCLSample) -> str:
    return tokenizer.apply_chat_template(
        build_messages(sample), add_generation_prompt=True, tokenize=False
    )


__all__ = ["SYSTEM_PROMPT", "build_messages", "render_prompt"]
