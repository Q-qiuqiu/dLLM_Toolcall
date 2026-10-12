"""BFCL single-turn data loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass(frozen=True)
class BFCLSample:
    sample_id: str
    messages: list[dict[str, str]]
    functions: list[dict[str, Any]]
    ground_truth: list[dict[str, dict[str, list[Any]]]]

    @property
    def candidate_names(self) -> list[str]:
        return [str(function["name"]) for function in self.functions]

    @property
    def ground_truth_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for call in self.ground_truth:
            if not isinstance(call, dict) or len(call) != 1:
                raise ValueError(
                    f"{self.sample_id}: malformed BFCL ground-truth call {call!r}"
                )
            name, allowed_arguments = next(iter(call.items()))
            calls.append({"name": name, "arguments": allowed_arguments})
        return calls

    @property
    def ground_truth_tool_types(self) -> set[str]:
        return {call["name"] for call in self.ground_truth_calls}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load either BFCL's JSONL-with-.json-extension or a JSON array."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected JSON array in {source}")
        return value
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {source}:{line_number}: {error}") from error
    return records


def _flatten_question(question: Any, sample_id: str) -> list[dict[str, str]]:
    """Normalize BFCL's ``[[message, ...]]`` single-turn representation."""

    while (
        isinstance(question, list)
        and len(question) == 1
        and isinstance(question[0], list)
    ):
        question = question[0]
    if isinstance(question, str):
        return [{"role": "user", "content": question}]
    if not isinstance(question, list):
        raise ValueError(f"{sample_id}: question must be a string or message list")
    messages: list[dict[str, str]] = []
    for item in question:
        if not isinstance(item, dict) or "content" not in item:
            raise ValueError(f"{sample_id}: malformed question message {item!r}")
        messages.append(
            {"role": str(item.get("role", "user")), "content": str(item["content"])}
        )
    return messages


def iter_samples(
    question_path: str | Path,
    answer_path: str | Path,
    *,
    limit: Optional[int] = None,
    start: int = 0,
) -> Iterator[BFCLSample]:
    questions = load_records(question_path)
    answers = load_records(answer_path)
    answers_by_id = {str(row["id"]): row for row in answers}
    selected = questions[start : None if limit is None else start + limit]
    for row in selected:
        sample_id = str(row["id"])
        if sample_id not in answers_by_id:
            raise ValueError(f"No possible_answer row for {sample_id}")
        functions = row.get("function")
        if not isinstance(functions, list) or not functions:
            raise ValueError(f"{sample_id}: function must be a non-empty list")
        ground_truth = answers_by_id[sample_id].get("ground_truth")
        if not isinstance(ground_truth, list):
            raise ValueError(f"{sample_id}: ground_truth must be a list")
        yield BFCLSample(
            sample_id=sample_id,
            messages=_flatten_question(row.get("question"), sample_id),
            functions=functions,
            ground_truth=ground_truth,
        )


def validate_dataset(
    question_path: str | Path, answer_path: str | Path
) -> dict[str, Any]:
    samples = list(iter_samples(question_path, answer_path))
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in question data")
    return {
        "sample_count": len(samples),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
        "max_candidates": max((len(sample.functions) for sample in samples), default=0),
        "max_calls": max((len(sample.ground_truth) for sample in samples), default=0),
    }


__all__ = ["BFCLSample", "iter_samples", "load_records", "validate_dataset"]
