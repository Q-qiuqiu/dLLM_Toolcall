"""Committed-only, candidate-unique tool prefixes and offline error attribution.

No final output or ground truth is used by ``CommittedPrefixMatcher.match``.
Final text is consulted only by ``audit_prefix_event`` after generation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


def closing_quote(text: str) -> int | None:
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index
    return None


def _outer_field_prefix(text: str) -> bool:
    match = re.search(r'"name"\s*:\s*"$', text)
    if match is None:
        return False
    depth = 0
    quoted = escaped = False
    for char in text[:match.start()]:
        if escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
    return not quoted and depth == 1


class CommittedPrefixMatcher:
    """Use the whole committed prefix, never just a candidate's matching part."""

    def __init__(self, tokenizer: Any, candidate_names: Iterable[str]) -> None:
        self.tokenizer = tokenizer
        self.names = list(dict.fromkeys(candidate_names))
        self.literals = {
            name: json.dumps(name, ensure_ascii=False)[1:-1] for name in self.names
        }
        self.forms = {
            name: (self._encode(value), self._encode(value + '"'))
            for name, value in self.literals.items()
        }
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    def _encode(self, text: str) -> tuple[int, ...]:
        try:
            return tuple(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return tuple(self.tokenizer.encode(text))

    def _compatible(
        self, generation: list[int], start: int, mask_id: int, name: str,
    ) -> bool:
        # Check committed suffix islands too, not only the contiguous prefix.
        # Different BPE segmentations can be conservatively rejected here;
        # that costs coverage, never licenses a contradictory prediction.
        for closed, expected in enumerate(self.forms[name]):
            if start + len(expected) > len(generation):
                continue
            if any(
                actual != mask_id and actual != wanted
                for actual, wanted in zip(generation[start:], expected)
            ):
                continue
            if not closed:
                end = start + len(expected)
                if end >= len(generation):
                    continue
                boundary = generation[end]
                if boundary != mask_id and not self.tokenizer.decode(
                    [boundary], skip_special_tokens=True
                ).startswith('"'):
                    continue
            return True
        return False

    def match(
        self, generation: list[int], start: int, mask_id: int,
    ) -> dict[str, Any] | None:
        if not 0 <= start < len(generation):
            return None
        run = []
        for token in generation[start:]:
            if token == mask_id or token in self.special_ids:
                break
            run.append(token)
        if not run:
            return None
        text = self.tokenizer.decode(run, skip_special_tokens=True)
        end_quote = closing_quote(text)
        if end_quote is not None:
            # A complete invalid field must not be reinterpreted as a shorter
            # valid name (e.g. "Audio-to-Audio Enhancement").
            literal = text[:end_quote]
            try:
                name = json.loads('"' + literal + '"')
            except json.JSONDecodeError:
                return None
            if name not in self.literals:
                return None
            return {
                "predicted_tool": name, "prefix": name,
                "prefix_literal": literal, "matching_candidate_count": 1,
                "complete_name": True, "validated_online": True,
                "generation_position": start,
            }
        # Uniqueness is tested against ALL candidate names in text space.
        # BPE tokenization differences must not hide another matching tool.
        matches = [name for name, literal in self.literals.items() if literal.startswith(text)]
        if len(matches) != 1:
            return None
        name = matches[0]
        if not self._compatible(generation, start, mask_id, name):
            return None
        return {
            "predicted_tool": name, "prefix": text,
            "prefix_literal": text, "matching_candidate_count": 1,
            "complete_name": False, "validated_online": True,
            "generation_position": start,
        }


def audit_prefix_event(
    event: dict[str, Any], *, generated_ids: list[int],
    tokenizer: Any, candidate_names: Iterable[str],
) -> dict[str, Any]:
    """Separate detector errors from later non-catalog name generation.

    An exception requires verified online evidence AND the final name at the
    SAME field position. An unrelated invalid name elsewhere is not evidence.
    This attribution never edits a trigger, its time, or raw fidelity metrics.
    """
    evidence = event.get("prefix_validation") or {}
    result = {
        "track_id": event.get("track_id"),
        "forward_step": event.get("forward_step"),
        "predict_time_s": event["predict_time_s"],
        "predicted_tool": event["predicted_tool"],
        "prefix": event.get("prefix"),
        "status": "unresolved", "final_field_name": None,
        "counts_as_prefix_error": None,
    }
    start = evidence.get("generation_position")
    if not evidence.get("validated_online") or not isinstance(start, int):
        return result
    if not 0 <= start < len(generated_ids):
        return result
    before = tokenizer.decode(generated_ids[:start], skip_special_tokens=True)
    # The opening name value quote must still end at this same token boundary.
    if not _outer_field_prefix(before):
        return result
    suffix = tokenizer.decode(generated_ids[start:], skip_special_tokens=True)
    end = closing_quote(suffix)
    if end is None:
        return result
    literal = suffix[:end]
    try:
        final_name = json.loads('"' + literal + '"')
    except json.JSONDecodeError:
        return result
    result["final_field_name"] = final_name
    if final_name == event["predicted_tool"]:
        result.update(status="correct", counts_as_prefix_error=False)
    elif (
        final_name not in set(candidate_names)
        and literal.startswith(evidence["prefix_literal"])
    ):
        result.update(status="model_invalid_name", counts_as_prefix_error=False)
    else:
        result.update(status="prediction_error", counts_as_prefix_error=True)
    return result
