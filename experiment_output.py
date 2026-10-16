"""Single-file experiment records; no generation-time file I/O or evaluation."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from prefix_detection import audit_prefix_event


@contextmanager
def open_result_file(directory: str | Path):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty results directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "samples.jsonl").open("x", encoding="utf-8") as handle:
        yield handle


def write_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def sample_identity(sample: Any, seed: int) -> dict[str, Any]:
    result = {
        "record_type": "sample", "sample_id": sample.sample_id, "seed": seed,
        "candidate_tools": sample.candidate_names,
        "ground_truth_tools": sorted(sample.ground_truth_tool_types),
        "ground_truth_calls": sample.ground_truth_calls,
    }
    if hasattr(sample, "domain"):
        result.update(
            benchmark=f"taskbench_{sample.domain}", graph_type=sample.graph_type,
            dependency_edges=sample.dependencies,
        )
    else:
        result["benchmark"] = "bfcl_parallel_multiple"
    return result


def compact_sample(
    *, sample: Any, seed: int, observer: Any, tokenizer: Any,
    generated_ids: list[int], raw_output: str, completion: str,
    generation_time_s: float, forward_count: int, prompt_token_count: int,
    observer_off_time_s: float | None, output_identical: bool | None,
    nfe_identical: bool | None,
) -> dict[str, Any]:
    # These dictionaries were populated causally during the callback. We do
    # not select timestamps or tool names using the final output / GT here.
    events = []
    for original in sorted(observer.first_tool_events.values(), key=lambda e: e["predict_time_s"]):
        event = {
            "tool": original["predicted_tool"], "source": original["source"],
            "time_s": original["predict_time_s"], "forward": original["forward_step"],
        }
        if original["source"] == "prefix":
            event["prefix"] = original["prefix"]
            event["prefix_validation"] = original["prefix_validation"]
            event["offline_prefix_audit"] = audit_prefix_event(
                original, generated_ids=generated_ids, tokenizer=tokenizer,
                candidate_names=sample.candidate_names,
            )
        if "margin" in original:
            event["margin"] = original["margin"]
        events.append(event)
    eos = getattr(tokenizer, "eos_token_id", None)
    stop_ids = {x for x in (eos, getattr(tokenizer, "pad_token_id", None)) if x is not None}
    first_stop = next((i for i, token in enumerate(generated_ids) if token in stop_ids), len(generated_ids))
    record = {
        **sample_identity(sample, seed),
        "raw_output": raw_output,
        "generation_time_s": generation_time_s,
        "forward_count": forward_count,
        "prompt_token_count": prompt_token_count,
        "output_token_count": first_stop,
        "generated_canvas_token_count": len(generated_ids),
        "eos_detected": eos is not None and eos in generated_ids,
        "observer_time_s": observer.total_processing_s,
        "first_tool_events": events,
        "natural_first_times": {
            name: {"time_s": e["predict_time_s"], "forward": e["forward_step"]}
            for name, e in observer.natural_first_events.items()
        },
        "parity": {
            "checked": output_identical is not None,
            "output_identical": output_identical,
            "nfe_identical": nfe_identical,
            "observer_off_generation_time_s": observer_off_time_s,
        },
        "error": None,
    }
    if completion != raw_output:
        record["output_before_eos"] = completion
    return record


def failure_sample(sample: Any, seed: int, error: Exception) -> dict[str, Any]:
    return {
        **sample_identity(sample, seed), "raw_output": "",
        "generation_time_s": None, "forward_count": None,
        "output_token_count": None, "eos_detected": None,
        "observer_time_s": None, "first_tool_events": [],
        "natural_first_times": {}, "parity": {"checked": False},
        "error": f"{type(error).__name__}: {error}",
    }
