"""Run TaskBench with OMTP + fallback or Natural-only; write only samples.jsonl."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import platform
import sys

import torch

from experiment_output import failure_sample, open_result_file, write_record
from natural_observer import add_observer_arguments, validate_observer_mode
from prompting import PROMPT_PROFILES
from reliable_omtp_observer import add_reliability_arguments, reliability_options
from response_prefill import RESPONSE_MODES
from run_experiment import _git_head, _load_model
from run_omtp_experiment import _run_sample
from taskbench_data import iter_taskbench_samples, sha256_file, validate_taskbench_manifest
from tool_format import OUTPUT_SCHEMAS


OMTP_FIXED = {
    "temperature": 0.0, "seed": 20261009, "stable_n": 2,
    "region_radius": 4, "region_top_k": 2, "min_seen": 2,
    "min_support": 0.5, "min_top1_score": None, "min_margin": None,
    "profile_observer": False,
}
GENERATION_PROFILES = {
    "fixed128": {"gen_length": 128, "block_length": 128, "steps": 32},
    "diagnostic256": {"gen_length": 256, "block_length": 256, "steps": 64},
}
# Public compatibility constant used by tests and historical analysis tools.
FIXED = {**GENERATION_PROFILES["fixed128"], **OMTP_FIXED}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tool-catalog", required=True)
    parser.add_argument("--model-path", default="/data/labshare/Param/llada")
    parser.add_argument("--fastdllm-root", default="/data/home/yzx/Fast-dLLM")
    parser.add_argument("--output-dir", required=True)
    add_observer_arguments(parser)
    parser.add_argument("--generation-profile", choices=tuple(GENERATION_PROFILES), default="fixed128")
    parser.add_argument("--prompt-profile", choices=PROMPT_PROFILES, default="compact_json_v3")
    parser.add_argument("--cache-mode", choices=("none", "dual"), default="none")
    parser.add_argument("--response-mode", choices=RESPONSE_MODES, default="native")
    parser.add_argument("--output-schema", choices=OUTPUT_SCHEMAS, default="standard")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--parity-samples", type=int, default=1)
    parser.add_argument("--debug-output", action="store_true", help="Include full trajectories in the same JSONL; still no extra files.")
    add_reliability_arguments(parser)
    return parser.parse_args()


def _fixed_runtime_args(cli: argparse.Namespace) -> argparse.Namespace:
    profile = getattr(cli, "generation_profile", "fixed128")
    args = argparse.Namespace(**vars(cli), **OMTP_FIXED, **GENERATION_PROFILES[profile])
    args.min_top1_score = getattr(cli, "omtp_min_top1_score", None)
    args.min_margin = getattr(cli, "omtp_min_margin", None)
    return args


def _metadata(args: argparse.Namespace, device: torch.device) -> dict:
    source = Path(__file__).resolve().parent
    value = {
        "record_type": "metadata", "schema_version": "tool_first_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv, "args": vars(args),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "fastdllm_commit": _git_head(args.fastdllm_root),
        "manifest_sha256": sha256_file(args.manifest),
        "tool_catalog_sha256": sha256_file(args.tool_catalog),
        "dataset_validation": validate_taskbench_manifest(args.manifest, args.tool_catalog),
        "code_sha256": {
            name: sha256_file(source / name) for name in (
                "run_taskbench_experiment.py", "run_omtp_experiment.py",
                "omtp_observer.py", "optimized_omtp_observer.py", "observer.py", "natural_observer.py",
                "reliable_omtp_observer.py", "prefix_detection.py",
                "experiment_output.py", "prompting.py", "taskbench_data.py",
                "fastdllm_generate.py", "model_adapter.py", "response_prefill.py",
                "tool_format.py", "tool_parser.py",
            )
        },
        "online_method": {
            "name": "optimized_omtp_prefix_natural_first_per_tool",
            "reliability_options": asdict(reliability_options(args)),
            "min_top1_score": args.min_top1_score, "min_margin": args.min_margin,
            "commit_policy": "first causal event per tool identity across all tracks and sources",
            "prefix_policy": "whole committed text prefix unique among full catalog; no committed suffix conflicts",
            "ground_truth_used_online": False, "extra_model_forwards": 0,
        },
        "timing_semantics": {
            "zero": "before generation canvas construction; includes model prompt processing",
            "excludes": ["prompt rendering/tokenization", "model loading", "post-generation evaluation", "file serialization"],
            "events": "after causal detection/scoring; natural scan precedes region scoring",
            "natural_baseline": "same observer-on request; not a separately measured observer-off request",
            "cuda": "generation callback path synchronizes model/transfer work",
        },
        "offline_evaluation": {
            "json_repair_and_ast": "not run automatically; raw text and reference calls retained",
            "prefix_model_invalid_name": "separate attribution, not a detector error; physical false-start cost remains",
        },
        "prompt_excludes": ["ground_truth_calls", "dependency_edges", "reference_steps"],
    }
    if getattr(args, "observer_mode", "omtp") == "natural":
        value["online_method"] = {
            "name": "natural_only", "observer_implementation": "committed_tokens_only",
            "detection": "shared detect_committed_tools, identical to OMTP Natural recognition",
            "ground_truth_used_online": False, "extra_model_forwards": 0,
            "logits_accessed": False, "prefix_or_region_scoring": False,
        }
        value["timing_semantics"].update(
            events="after committed-token D2H copying, sparse decoding and detection",
            natural_baseline="independent Natural-only request, without OMTP work",
            remaining_overhead="minimal committed-token observer and existing synchronized callback; NOT observer-off",
        )
    return value


def main() -> None:
    cli = parse_args()
    if cli.parity_samples < 0:
        raise ValueError("--parity-samples must be non-negative")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args = _fixed_runtime_args(cli)
    validate_observer_mode(args)
    samples = list(iter_taskbench_samples(args.manifest, args.tool_catalog))
    if not samples:
        raise ValueError("empty TaskBench manifest")
    device = torch.device(args.device)
    with open_result_file(args.output_dir) as handle:
        model, tokenizer = _load_model(args, device)
        write_record(handle, _metadata(args, device))
        for index, sample in enumerate(samples):
            print(f"[{index + 1}/{len(samples)}] {sample.sample_id}", flush=True)
            try:
                record, _ = _run_sample(args, sample, index, model, tokenizer, device)
            except Exception as error:
                print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                record = failure_sample(sample, args.seed + index, error)
            write_record(handle, record)
    print(f"Results written to {Path(args.output_dir).resolve() / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
