"""Offline-only JSON repair and evaluation for generated tool calls."""

from __future__ import annotations

from importlib.metadata import version
import time
from typing import Any, Iterable

from json_repair import repair_json

from evaluation import evaluate_final_calls
from tool_parser import parse_tool_calls


JSON_REPAIR_VERSION = version("json-repair")


def evaluate_json_repair(
    text: str,
    candidate_names: Iterable[str],
    sample: Any,
) -> dict[str, Any]:
    """Repair a completed generation and evaluate it without online feedback.

    Valid JSON is kept byte-for-byte.  Invalid JSON is repaired once, after
    generation, and then passed through the same strict tool-call parser used
    for the unmodified output.  The repaired result never enters an observer,
    model prompt, or online decision.
    """

    candidate_names = list(candidate_names)
    original = parse_tool_calls(text, candidate_names)
    attempted = not original.json_valid
    started = time.perf_counter()
    repair_exception = None
    try:
        repaired_text = (
            str(repair_json(text, ensure_ascii=False)) if attempted else text
        )
    except Exception as error:  # pragma: no cover - defensive library boundary
        repaired_text = text
        repair_exception = f"{type(error).__name__}: {error}"
    elapsed_s = time.perf_counter() - started

    parsed = parse_tool_calls(repaired_text, candidate_names)
    repaired_calls = parsed.calls if parsed.format_valid else []
    quality = evaluate_final_calls(sample, repaired_calls)
    return {
        "json_repair_library": "json-repair",
        "json_repair_version": JSON_REPAIR_VERSION,
        "json_repair_offline_only": True,
        "json_repair_attempted": attempted,
        "json_repair_changed": repaired_text != text,
        "json_repair_processing_time_s": elapsed_s,
        "json_repair_exception": repair_exception,
        "repaired_json_text": repaired_text,
        "repaired_json_valid": parsed.json_valid,
        "repaired_tool_call_format_valid": parsed.format_valid,
        "repaired_candidate_names_valid": parsed.candidate_names_valid,
        "repaired_output_valid": parsed.valid,
        "repaired_json_error": parsed.error,
        "repaired_tool_calls": repaired_calls,
        "repaired_tool_occurrences": [call["name"] for call in repaired_calls],
        "repaired_tool_types": list(dict.fromkeys(
            call["name"] for call in repaired_calls
        )),
        "repaired_tool_name_precision": quality["tool_name_precision"],
        "repaired_tool_name_recall": quality["tool_name_recall"],
        "repaired_tool_name_f1": quality["tool_name_f1"],
        "repaired_tool_set_exact_match": quality["tool_set_exact_match"],
        "repaired_tool_type_precision": quality["tool_type_precision"],
        "repaired_tool_type_recall": quality["tool_type_recall"],
        "repaired_tool_type_f1": quality["tool_type_f1"],
        "repaired_tool_call_count_match": quality["tool_call_count_match"],
        "repaired_ast_compatible_match": quality["bfcl_ast_compatible_match"],
    }


__all__ = ["JSON_REPAIR_VERSION", "evaluate_json_repair"]
