"""Run no-cache LLaDA on BFCL parallel_multiple with per-forward observation."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any
from collections.abc import Sequence

import numpy as np
import torch

from bfcl_data import BFCLSample, iter_samples, validate_dataset
from evaluation import analyze_observer, evaluate_final_calls
from fastdllm_generate import generate, generate_with_dual_cache
from model_adapter import load_model_and_tokenizer, runtime_spec
from observer import ToolCallObserver
from prompting import render_prompt
from report import build_summary
from simulation import simulate_warmup
from tool_parser import extract_partial_names, ordered_unique, parse_tool_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--model-path", default="/data/labshare/Param/llada")
    parser.add_argument("--fastdllm-root", default="/data/home/yzx/Fast-dLLM")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--stable-n", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--observer-top-k", type=int, choices=(1, 2), default=1)
    parser.add_argument("--save-full-logits", action="store_true")
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=0,
        help="Also run observer-off generation for the first N samples.",
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_completion(tokenizer: Any, token_ids: list[int]) -> str:
    eos_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    stop = next(
        (index for index, token_id in enumerate(token_ids) if token_id in eos_ids),
        len(token_ids),
    )
    return tokenizer.decode(token_ids[:stop], skip_special_tokens=True)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _load_model(args: argparse.Namespace, device: torch.device):
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    raw_memory = getattr(args, "max_memory_json", None)
    max_memory = None
    if raw_memory:
        max_memory = {int(k) if k.isdecimal() else k: v for k, v in json.loads(raw_memory).items()}
    model, tokenizer, spec = load_model_and_tokenizer(
        model_path=args.model_path,
        fastdllm_root=args.fastdllm_root,
        dtype=dtype,
        device=device,
        cache_mode=getattr(args, "cache_mode", "none"),
        device_map=getattr(args, "device_map", None), max_memory=max_memory,
    )
    args.model_path_resolved = spec.resolved_path
    args.model_family = spec.family
    args.model_runtime = spec.to_dict()
    if getattr(model, "hf_device_map", None) is not None:
        args.model_device_map = {k: str(v) for k, v in model.hf_device_map.items()}
    return model, tokenizer


def _generate_once(
    *,
    args: argparse.Namespace,
    model: Any,
    prompt_ids: torch.Tensor,
    mask_id: int,
    observer: ToolCallObserver | None,
    seed: int,
    device: torch.device,
    generation_prefix_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, int, float]:
    _seed_everything(seed)
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        cache_mode = getattr(args, "cache_mode", "none")
        spec = runtime_spec(model)
        if cache_mode == "dual" and not spec.supports_dual_cache:
            raise ValueError(
                f"Dual Cache is not supported for model family {spec.family!r}; "
                "use --cache-mode none"
            )
        generate_fn = {
            "none": generate,
            "dual": generate_with_dual_cache,
        }.get(cache_mode)
        if generate_fn is None:
            raise ValueError(f"unsupported cache mode: {cache_mode}")
        output, nfe = generate_fn(
            model=model,
            prompt=prompt_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            remasking="low_confidence",
            mask_id=mask_id,
            threshold=None,
            factor=None,
            observation_callback=observer,
            generation_prefix_ids=generation_prefix_ids,
        )
    _sync(device)
    return output, nfe, time.perf_counter() - started


def _run_sample(
    *,
    args: argparse.Namespace,
    sample: BFCLSample,
    sample_index: int,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    rendered = render_prompt(tokenizer, sample)
    prompt_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
    spec = runtime_spec(model)
    if prompt_ids.shape[1] + args.gen_length > spec.max_sequence_length:
        raise ValueError(
            f"{sample.sample_id}: prompt+generation exceeds model context: "
            f"{prompt_ids.shape[1]}+{args.gen_length}"
        )
    mask_id = spec.mask_token_id
    full_logits_dir = (
        output_dir / "full_logits" / sample.sample_id
        if args.save_full_logits
        else None
    )
    observer = ToolCallObserver(
        tokenizer=tokenizer,
        candidate_names=sample.candidate_names,
        stable_n=args.stable_n,
        top_k=args.observer_top_k,
        full_logits_dir=full_logits_dir,
    )
    sample_seed = args.seed + sample_index

    baseline_time = None
    baseline_output = None
    if sample_index < args.parity_samples:
        # One unmeasured observer-off pass warms model kernels/allocators.  A
        # cold baseline compared with a warm observed pass can otherwise report
        # a physically meaningless negative observer overhead.
        _generate_once(
            args=args,
            model=model,
            prompt_ids=prompt_ids,
            mask_id=mask_id,
            observer=None,
            seed=sample_seed,
            device=device,
        )
        baseline_output, _, baseline_time = _generate_once(
            args=args,
            model=model,
            prompt_ids=prompt_ids,
            mask_id=mask_id,
            observer=None,
            seed=sample_seed,
            device=device,
        )

    output, nfe, generation_time = _generate_once(
        args=args,
        model=model,
        prompt_ids=prompt_ids,
        mask_id=mask_id,
        observer=observer,
        seed=sample_seed,
        device=device,
    )
    generated_ids = output[0, prompt_ids.shape[1] :].detach().cpu().tolist()
    eos_id = tokenizer.eos_token_id
    first_eos_position = next(
        (
            index
            for index, token_id in enumerate(generated_ids)
            if eos_id is not None and token_id == eos_id
        ),
        None,
    )
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    raw_output_with_special_tokens = tokenizer.decode(
        generated_ids, skip_special_tokens=False
    )
    evaluated_output = _decode_completion(tokenizer, generated_ids)
    parsed = parse_tool_calls(evaluated_output, sample.candidate_names)
    predicted_calls = parsed.calls if parsed.format_valid else []
    final_partial_names = extract_partial_names(
        evaluated_output, sample.candidate_names
    )
    final_types = ordered_unique(
        [call["name"] for call in predicted_calls]
        if parsed.format_valid
        else final_partial_names
    )
    final_metrics = evaluate_final_calls(sample, predicted_calls)
    timing = analyze_observer(observer, sample.ground_truth_tool_types)
    natural_all = timing["natural_all_detect_time_s"]
    trigger_time = timing["predict_all_detect_time_s"]
    trigger_correct = timing["predict_trigger_set_exact"]
    lead_time = (
        natural_all - trigger_time
        if natural_all is not None and trigger_time is not None and trigger_correct
        else None
    )
    warmup_window = (
        generation_time - trigger_time
        if trigger_time is not None and trigger_correct
        else None
    )
    trigger_available_time = timing["predict_all_available_time_s"]
    actual_warmup_window = (
        generation_time - trigger_available_time
        if trigger_available_time is not None and trigger_correct
        else None
    )
    actual_lead_time = (
        natural_all - trigger_available_time
        if natural_all is not None
        and trigger_available_time is not None
        and trigger_correct
        else None
    )
    stable_correct_time = timing["offline_stable_correct_trigger_time_s"]
    offline_stable_correct_lead = (
        natural_all - stable_correct_time
        if natural_all is not None and stable_correct_time is not None
        else None
    )
    warmup = simulate_warmup(
        expected_types=sample.ground_truth_tool_types,
        natural_first_times=timing["natural_first_times_by_tool"],
        stable_events=timing["stable_events"],
        generation_time_s=generation_time,
    )
    parity_equal = (
        bool(torch.equal(baseline_output, output))
        if baseline_output is not None
        else None
    )
    model_forward_time = sum(step["forward_time_s"] for step in observer.steps)
    record: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "ground_truth_tools": [call["name"] for call in sample.ground_truth_calls],
        "ground_truth_tool_types": sorted(sample.ground_truth_tool_types),
        "generated_tool_calls": predicted_calls,
        "raw_output": raw_output,
        "raw_output_with_special_tokens": raw_output_with_special_tokens,
        "evaluated_output_before_eos": evaluated_output,
        "generated_token_ids": generated_ids,
        "first_eos_position": first_eos_position,
        "eos_at_last_position": first_eos_position == len(generated_ids) - 1,
        "final_partial_call_names": final_partial_names,
        "final_predicted_tool_types": final_types,
        "final_json_valid": parsed.json_valid,
        "tool_call_format_valid": parsed.format_valid,
        "candidate_names_valid": parsed.candidate_names_valid,
        "final_json_error": parsed.error,
        **final_metrics,
        "generation_time_s": generation_time,
        "model_forward_time_s": model_forward_time,
        "observer_processing_time_s": observer.total_processing_s,
        **timing,
        "prediction_lead_time_s": lead_time,
        "actual_prediction_lead_time_s": actual_lead_time,
        "offline_stable_correct_lead_time_s": offline_stable_correct_lead,
        "available_warmup_window_s": warmup_window,
        "actual_available_warmup_window_s": actual_warmup_window,
        "forward_count": nfe,
        "prompt_token_count": int(prompt_ids.shape[1]),
        "warmup_simulation": warmup,
        "observer_off_generation_time_s": baseline_time,
        "observer_output_identical": parity_equal,
        "observer_wall_overhead_ratio": (
            generation_time / baseline_time - 1
            if baseline_time not in (None, 0)
            else None
        ),
        "steps": observer.steps,
    }
    return record


def _write_metadata(
    args: argparse.Namespace, output_dir: Path, device: torch.device
) -> None:
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "fastdllm_commit": _git_head(args.fastdllm_root),
        "questions_sha256": _sha256(args.questions),
        "answers_sha256": _sha256(args.answers),
        "dataset_validation": validate_dataset(args.questions, args.answers),
        "timing_semantics": {
            "forward_complete_time_s": "CUDA-synchronized end of model(x).logits; includes earlier observer work in wall-clock timeline, excludes current observer work",
            "commit_complete_time_s": "CUDA-synchronized token transfer completion before current observer work",
            "generation_time_s": "observed end-to-end generation including callbacks",
            "model_forward_time_s": "sum of synchronized model-forward intervals",
        },
        "warmup_assumption": "Tools start independently in parallel; no device contention or Docker scheduling overhead.",
        "ast_metric": "Local BFCL-compatible unordered AST checker; not the external bfcl_eval package.",
        "parity_method": "For parity samples: one unmeasured observer-off warmup, then measured observer-off baseline, then measured observer-on generation.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.temperature != 0:
        raise ValueError("The requested first experiment requires temperature=0")
    if args.gen_length % args.block_length:
        raise ValueError("gen-length must be divisible by block-length")
    if args.steps % (args.gen_length // args.block_length):
        raise ValueError("steps must be divisible by the number of blocks")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.parity_samples < 0:
        raise ValueError("parity-samples cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "results"
    ) / datetime.now().strftime("%Y%m%d_%H%M%S")
    for protected in ("samples.jsonl", "steps.jsonl", "failures.jsonl", "metadata.json"):
        if (output_dir / protected).exists():
            raise FileExistsError(
                f"Refusing to overwrite existing run file: {output_dir / protected}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model(args, device)
    _write_metadata(args, output_dir, device)
    samples = list(
        iter_samples(args.questions, args.answers, limit=args.limit, start=args.start)
    )
    with (
        (output_dir / "samples.jsonl").open("w", encoding="utf-8") as samples_file,
        (output_dir / "steps.jsonl").open("w", encoding="utf-8") as steps_file,
        (output_dir / "failures.jsonl").open("w", encoding="utf-8") as failures_file,
    ):
        for sample_index, sample in enumerate(samples):
            print(f"[{sample_index + 1}/{len(samples)}] {sample.sample_id}", flush=True)
            try:
                record = _run_sample(
                    args=args,
                    sample=sample,
                    sample_index=sample_index,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    output_dir=output_dir,
                )
            except Exception as error:
                failure = {
                    "sample_id": sample.sample_id,
                    "failure_types": ["runtime_error"],
                    "error": f"{type(error).__name__}: {error}",
                }
                failures_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failures_file.flush()
                # Runtime failures remain in the sample denominator with null
                # timings; they are never silently dropped from aggregation.
                record = {
                    "sample_id": sample.sample_id,
                    "ground_truth_tools": [
                        call["name"] for call in sample.ground_truth_calls
                    ],
                    "ground_truth_tool_types": sorted(
                        sample.ground_truth_tool_types
                    ),
                    "generated_tool_calls": [],
                    "raw_output": "",
                    "raw_output_with_special_tokens": "",
                    "evaluated_output_before_eos": "",
                    "generated_token_ids": [],
                    "first_eos_position": None,
                    "eos_at_last_position": False,
                    "final_partial_call_names": [],
                    "final_predicted_tool_types": [],
                    "final_json_valid": False,
                    "tool_call_format_valid": False,
                    "candidate_names_valid": False,
                    "final_json_error": failure["error"],
                    "tool_name_precision": 0.0,
                    "tool_name_recall": 0.0,
                    "tool_name_f1": 0.0,
                    "tool_set_exact_match": False,
                    "tool_type_precision": 0.0,
                    "tool_type_recall": 0.0,
                    "tool_type_f1": 0.0,
                    "tool_call_count_match": False,
                    "bfcl_ast_compatible_match": False,
                    "generation_time_s": None,
                    "model_forward_time_s": None,
                    "observer_processing_time_s": None,
                    "natural_first_detect_time_s": None,
                    "natural_last_detect_time_s": None,
                    "natural_all_detect_time_s": None,
                    "predict_first_detect_time_s": None,
                    "predict_all_detect_time_s": None,
                    "predict_all_available_time_s": None,
                    "predict_trigger_tool_types": [],
                    "predict_trigger_set_exact": False,
                    "predict_trigger_precision": 0.0,
                    "predict_trigger_recall": 0.0,
                    "predict_trigger_false_positives": [],
                    "predict_trigger_false_negatives": sorted(
                        sample.ground_truth_tool_types
                    ),
                    "oracle_correct_detect_time_s": None,
                    "offline_stable_correct_trigger_time_s": None,
                    "natural_first_times_by_tool": {},
                    "online_natural_first_times_all_candidates": {},
                    "predict_first_times_by_tool": {},
                    "online_predict_first_times_all_candidates": {},
                    "stable_events": [],
                    "prediction_lead_time_s": None,
                    "actual_prediction_lead_time_s": None,
                    "offline_stable_correct_lead_time_s": None,
                    "available_warmup_window_s": None,
                    "actual_available_warmup_window_s": None,
                    "forward_count": None,
                    "prompt_token_count": None,
                    "warmup_simulation": [],
                    "observer_off_generation_time_s": None,
                    "observer_output_identical": None,
                    "observer_wall_overhead_ratio": None,
                    "runtime_error": failure["error"],
                    "steps": [],
                }
                samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                samples_file.flush()
                print(f"  FAILED: {failure['error']}", file=sys.stderr, flush=True)
                continue
            samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            samples_file.flush()
            for step in record["steps"]:
                steps_file.write(
                    json.dumps(
                        {"sample_id": sample.sample_id, **step}, ensure_ascii=False
                    )
                    + "\n"
                )
            steps_file.flush()
            failure_types = []
            if not record["final_json_valid"]:
                failure_types.append("invalid_json")
            if not record["tool_call_format_valid"]:
                failure_types.append("invalid_tool_call_format")
            elif not record["candidate_names_valid"]:
                failure_types.append("unknown_candidate_name")
            if not record["tool_set_exact_match"]:
                failure_types.append("final_tool_set_mismatch")
            if record["oracle_correct_detect_time_s"] is None:
                failure_types.append("no_correct_intermediate_prediction")
            if record["predict_all_detect_time_s"] is None:
                failure_types.append("no_online_stable_trigger")
            if not record["bfcl_ast_compatible_match"]:
                failure_types.append("ast_mismatch")
            if record["observer_output_identical"] is False:
                failure_types.append("observer_changed_output")
            if failure_types:
                failures_file.write(
                    json.dumps(
                        {
                            "sample_id": sample.sample_id,
                            "failure_types": failure_types,
                            "final_json_error": record["final_json_error"],
                            "raw_output": record["raw_output"],
                            "evaluated_output_before_eos": record[
                                "evaluated_output_before_eos"
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                failures_file.flush()
    build_summary(output_dir)
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
