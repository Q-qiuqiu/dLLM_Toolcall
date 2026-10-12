"""Strict final JSON parsing and conservative partial ``name`` extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class ParseResult:
    json_valid: bool
    format_valid: bool
    candidate_names_valid: bool
    calls: list[dict[str, Any]]
    error: str | None

    @property
    def valid(self) -> bool:
        """Backward-compatible strict validity across all three layers."""

        return self.json_valid and self.format_valid and self.candidate_names_valid

    @property
    def names(self) -> list[str]:
        return [str(call["name"]) for call in self.calls]


def parse_tool_calls(text: str, candidate_names: Iterable[str]) -> ParseResult:
    """Parse the experiment's exact output format; markdown is intentionally invalid."""

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as error:
        return ParseResult(
            False, False, False, [], f"json_decode_error: {error.msg}"
        )
    if not isinstance(value, list):
        return ParseResult(True, False, False, [], "top_level_not_array")
    candidates = set(candidate_names)
    calls: list[dict[str, Any]] = []
    candidate_names_valid = True
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return ParseResult(
                True, False, False, calls, f"call_{index}_not_object"
            )
        if set(item) != {"name", "arguments"}:
            return ParseResult(
                True, False, False, calls, f"call_{index}_wrong_fields"
            )
        if not isinstance(item["name"], str):
            return ParseResult(
                True, False, False, calls, f"call_{index}_name_not_string"
            )
        if not isinstance(item["arguments"], dict):
            return ParseResult(
                True, False, False, calls, f"call_{index}_arguments_not_object"
            )
        calls.append({"name": item["name"], "arguments": item["arguments"]})
        if item["name"] not in candidates:
            candidate_names_valid = False
    error = None if candidate_names_valid else "unknown_candidate_name"
    return ParseResult(True, True, candidate_names_valid, calls, error)


_NAME_FIELD = re.compile(
    r"(?P<key_quote>[\"'])name(?P=key_quote)\s*:\s*"
    r"(?P<value_quote>[\"'])(?P<name>[^\"'\r\n]+)(?P=value_quote)"
)


def extract_partial_names(text: str, candidate_names: Iterable[str]) -> list[str]:
    """Extract only complete JSON-like ``name`` fields, never arbitrary substrings."""

    candidates = set(candidate_names)
    parsed = parse_tool_calls(text, candidates)
    if parsed.format_valid:
        return [name for name in parsed.names if name in candidates]
    return [
        match.group("name")
        for match in _NAME_FIELD.finditer(text)
        if match.group("name") in candidates
    ]


def ordered_unique(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))


__all__ = [
    "ParseResult",
    "extract_partial_names",
    "ordered_unique",
    "parse_tool_calls",
]
