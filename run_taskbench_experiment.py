"""Run fixed Optimized OMTP + Fallback on a frozen TaskBench manifest."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch

from run_experiment import _git_head, _load_model
from run_omtp_experiment import _run_sample, _runtime_failure_record
from taskbench_data import (
    TaskBenchSample, iter_taskbench_samples, sha256_file,
    validate_taskbench_manifest,
)
from taskbench_evaluation import evaluate_taskbench_methods
from taskbench_report import build_taskbench_report
from tool_parser import extract_partial_names
from evaluation import set_quality
from json_repair_evaluation import evaluate_json_repair


FIXED = {
    "gen_length": 128,
    "block_length": 128,
    "steps": 32,
    "temperature": 0.0,
    "seed": 20261009,
    "stable_n": 2,
    "region_radius": 4,
    "region_top_k": 2,
    "min_seen": 2,
    "min_support": 0.5,
    "min_top1_score": None,
    "min_margin": None,
    "profile_observer": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tool-catalog", required=True)
    parser.add_argument("--model-path", default="/data/labshare/Param/llada")
    parser.add_argument("--fastdllm-root", default="/data/home/yzx/Fast-dLLM")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache-mode", choices=("none", "dual"), default="none",
        help="Fast-dLLM v1 no-cache or Dual Cache path; all OMTP settings remain fixed.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--parity-samples", type=int, default=1,
        help="Observer-off/on parity checks for the first N requests.",
    )
    return parser.parse_args()


def _fixed_runtime_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(cli), **FIXED)


def _augment(
    record: dict[str, Any], sample: TaskBenchSample, observer: Any | None,
    tokenizer: Any | None, *, gen_length: int = 128,
    expected_nfe: int = 32,
) -> dict[str, Any]:
    latent_events = list(observer.latent_events) if observer is not None else []
    methods = evaluate_taskbench_methods(
        latent_events=latent_events,
        fallback_events=record["omtp_fallback_events"],
        natural_events=record["natural_baseline"]["events"],
        ground_truth_occurrences=[
            call["name"] for call in sample.ground_truth_calls
        ],
        final_occurrences=record["final_tool_occurrences"],
    )
    output_tokens = None
    if tokenizer is not None and record.get("raw_output"):
        output_tokens = len(tokenizer(
            record["raw_output"], add_special_tokens=False
        ).input_ids)
    raw_tools = extract_partial_names(
        record.get("raw_output", ""), sample.candidate_names
    )
    raw_quality = set_quality(raw_tools, sample.ground_truth_tool_types)
    record.update({
        "benchmark": f"taskbench_{sample.domain}",
        "cache_mode": record.get("cache_mode"),
        "source_id": sample.source_id,
        "source_seed": sample.source_seed,
        "graph_type": sample.graph_type,
        "ground_truth_calls": sample.ground_truth_calls,
        "dependency_edges": sample.dependencies,
        "reference_steps": sample.reference_steps,
        "ground_truth_call_count": sample.tool_call_count,
        "ground_truth_tool_type_count": sample.tool_type_count,
        "dependency_edge_count": len(sample.dependencies),
        "latent_region_events": latent_events,
        "taskbench_methods": methods,
        "decoded_output_token_count": output_tokens,
        # This is deliberately labelled a heuristic: special-token removal can
        # make decoded/re-tokenized length differ slightly from the generation
        # canvas.
        "likely_length_truncated": bool(
            not record.get("tool_call_format_valid")
            and output_tokens is not None and output_tokens >= gen_length - 2
        ),
        "generation_length": gen_length,
        "expected_forward_count": expected_nfe,
        "final_raw_tool_types": sorted(set(raw_tools)),
        "final_raw_tool_precision": raw_quality["precision"],
        "final_raw_tool_recall": raw_quality["recall"],
        "final_raw_tool_f1": raw_quality["f1"],
        "final_raw_tool_set_exact_match": raw_quality["exact"],
    })
    record.update(evaluate_json_repair(
        record.get("evaluated_output_before_eos", record.get("raw_output", "")),
        sample.candidate_names,
        sample,
    ))
    return record


def _metadata(
    args: argparse.Namespace, output: Path, device: torch.device,
) -> None:
    value = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "args": vars(args),
        "cache_mode": args.cache_mode,
        "fixed_generation_and_omtp_parameters": {
            key: getattr(args, key) for key in (
                "gen_length", "block_length", "steps", "temperature", "seed",
                "stable_n", "region_radius", "region_top_k", "min_seen",
                "min_support", "min_top1_score", "min_margin",
                "profile_observer",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "fastdllm_commit": _git_head(args.fastdllm_root),
        "omtp_core_sha256": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in (
                "omtp_observer.py", "optimized_omtp_observer.py",
                "omtp_evaluation.py", "run_omtp_experiment.py",
                "prompting.py", "fastdllm_generate.py",
            )
        },
        "manifest_sha256": sha256_file(args.manifest),
        "tool_catalog_sha256": sha256_file(args.tool_catalog),
        "dataset_validation": validate_taskbench_manifest(
            args.manifest, args.tool_catalog
        ),
        "online_method": {
            "name": "optimized_omtp_with_prefix_natural_fallback",
            "core_implementation_reused_without_changes": True,
            "candidate_scope": (
                f"complete {Path(args.tool_catalog).resolve().parent.name} "
                "tool catalog"
            ),
            "ground_truth_used_online": False,
            "extra_model_forwards_per_request": 0,
        },
        "post_generation_json_repair": {
            "library": "json-repair",
            "offline_only": True,
            "applied_only_after_generation": True,
            "feeds_online_prediction": False,
            "strict_tool_call_validation_after_repair": True,
        },
        "prompt_excludes": [
            "ground_truth_calls", "dependency_edges", "reference_steps"
        ],
        "timing_semantics": (
            "Request time zero is before generation canvas construction; model "
            "forward work is CUDA-synchronized by the existing callback path, "
            "and online event timestamps include observer detection/scoring."
        ),
        "parity_method": (
            "Selected requests use an observer-off warmup, measured observer-off "
            "run, then measured observer-on run with the identical seed."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_events(handle: Any, sample_id: str, events: list[dict[str, Any]]) -> None:
    for event in events:
        handle.write(json.dumps(
            {"sample_id": sample_id, **event}, ensure_ascii=False
        ) + "\n")


def main() -> None:
    cli = parse_args()
    if cli.parity_samples < 0:
        raise ValueError("--parity-samples must be non-negative")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args = _fixed_runtime_args(cli)
    samples = list(iter_taskbench_samples(args.manifest, args.tool_catalog))
    if not samples:
        raise ValueError("empty TaskBench manifest")
    output = Path(args.output_dir)
    names = (
        "taskbench_samples.jsonl", "samples.jsonl", "steps.jsonl",
        "omtp_events.jsonl", "latent_events.jsonl", "omtp_fallback_events.jsonl",
        "natural_events.jsonl", "omtp_trajectory.jsonl",
        "position_records.jsonl", "failures.jsonl", "metadata.json",
        "taskbench_method_comparison.csv", "taskbench_group_summary.csv",
        "taskbench_position_summary.csv", "taskbench_generation_summary.csv",
    )
    for name in names:
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, tokenizer = _load_model(args, device)
    _metadata(args, output, device)
    paths = {
        name: (output / name).open("w", encoding="utf-8")
        for name in (
            "taskbench_samples.jsonl", "samples.jsonl", "steps.jsonl",
            "omtp_events.jsonl", "latent_events.jsonl", "omtp_fallback_events.jsonl",
            "natural_events.jsonl", "omtp_trajectory.jsonl",
            "position_records.jsonl", "failures.jsonl",
        )
    }
    try:
        for index, sample in enumerate(samples):
            print(f"[{index + 1}/{len(samples)}] {sample.sample_id}", flush=True)
            observer = None
            try:
                record, observer = _run_sample(
                    args, sample, index, model, tokenizer, device
                )
                record = _augment(
                    record, sample, observer, tokenizer,
                    gen_length=args.gen_length, expected_nfe=args.steps,
                )
                record["cache_mode"] = args.cache_mode
            except Exception as error:
                print(
                    f"  FAILED: {type(error).__name__}: {error}",
                    file=sys.stderr, flush=True,
                )
                record = _augment(
                    _runtime_failure_record(sample, error), sample, None, None,
                    gen_length=args.gen_length, expected_nfe=args.steps,
                )
                record["cache_mode"] = args.cache_mode
                record["runtime_error"] = f"{type(error).__name__}: {error}"

            serialized = json.dumps(record, ensure_ascii=False) + "\n"
            paths["taskbench_samples.jsonl"].write(serialized)
            paths["samples.jsonl"].write(serialized)
            _write_events(paths["omtp_events.jsonl"], sample.sample_id,
                          record["omtp_events"])
            _write_events(paths["latent_events.jsonl"], sample.sample_id,
                          record["latent_region_events"])
            _write_events(paths["omtp_fallback_events.jsonl"], sample.sample_id,
                          record["omtp_fallback_events"])
            _write_events(paths["natural_events.jsonl"], sample.sample_id,
                          record["natural_baseline"]["events"])
            _write_events(paths["steps.jsonl"], sample.sample_id, record["steps"])
            _write_events(paths["omtp_trajectory.jsonl"], sample.sample_id,
                          record["omtp_trajectory"])
            for method, value in record["taskbench_methods"].items():
                for row in value["vs_ground_truth"]["position_records"]:
                    paths["position_records.jsonl"].write(json.dumps({
                        "sample_id": sample.sample_id,
                        "graph_type": sample.graph_type,
                        "method": method,
                        **row,
                    }, ensure_ascii=False) + "\n")

            failures = []
            if record.get("runtime_error"): failures.append("runtime_error")
            if not record["final_json_valid"]: failures.append("invalid_json")
            if not record["tool_set_exact_match"]: failures.append("final_tool_set_mismatch")
            if record["likely_length_truncated"]: failures.append("likely_length_truncated")
            if record["observer_output_identical"] is False: failures.append("observer_changed_output")
            if record["observer_nfe_identical"] is False: failures.append("observer_changed_nfe")
            if failures:
                paths["failures.jsonl"].write(json.dumps({
                    "sample_id": sample.sample_id,
                    "graph_type": sample.graph_type,
                    "failure_types": failures,
                    "error": record.get("runtime_error"),
                    "raw_output": record["raw_output"],
                }, ensure_ascii=False) + "\n")
            for handle in paths.values():
                handle.flush()
    finally:
        for handle in paths.values():
            handle.close()
    build_taskbench_report(output)
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
