"""Run BFCL parallel_multiple with the fixed OMTP + fallback policy."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import platform
import sys
from typing import Any

import torch

from bfcl_data import BFCLSample, iter_samples, validate_dataset
from evaluation import evaluate_final_calls
from omtp_evaluation import (
    METHOD_NAME, NATURAL_BASELINE_NAME, evaluate_omtp_fallback,
)
from omtp_observer import OMTPConfig, OMTPObserver
from optimized_omtp_observer import OptimizedOMTPObserver
from reliable_omtp_observer import ReliableOMTPObserver, add_reliability_arguments, reliability_options
from prefix_detection import audit_prefix_event
from experiment_output import compact_sample, failure_sample, open_result_file, write_record
from model_adapter import runtime_spec
from prompting import render_prompt
from response_prefill import RESPONSE_MODES, resolve_response_prefill
from tool_format import OUTPUT_SCHEMAS, model_call_items, parameter_aliases
from run_experiment import (
    _decode_completion, _generate_once, _git_head, _load_model, _sha256,
)
from tool_parser import extract_partial_names, ordered_unique, parse_tool_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--model-path", default="/data/labshare/Param/llada")
    parser.add_argument("--fastdllm-root", default="/data/home/yzx/Fast-dLLM")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--cache-mode", choices=("none", "dual"), default="none",
        help="Fast-dLLM v1 generation path; OMTP itself is unchanged.",
    )
    parser.add_argument(
        "--response-mode", choices=RESPONSE_MODES, default="native",
        help=(
            "native leaves decoding unchanged; json_prefill is an explicit "
            "iLLaDA-only structural-prefix ablation."
        ),
    )
    parser.add_argument(
        "--output-schema", choices=OUTPUT_SCHEMAS, default="standard",
        help=(
            "standard emits name/arguments objects; flat_name_v1 keeps name "
            "and places parameters directly in each call object."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--stable-n", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--region-radius", type=int, default=4)
    parser.add_argument("--region-top-k", type=int, default=2)
    parser.add_argument("--min-seen", type=int, default=2)
    parser.add_argument("--min-support", type=float, default=0.5)
    parser.add_argument("--parity-samples", type=int, default=1)
    parser.add_argument(
        "--profile-observer", action="store_true",
        help="Synchronize at OMTP module boundaries and record a diagnostic profile.",
    )
    # Keep legacy spellings while exposing the same reliability switches as TaskBench.
    add_reliability_arguments(parser)
    parser.add_argument("--min-top1-score", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    parser.add_argument("--device-map", choices=("auto",), default=None)
    parser.add_argument("--max-memory-json", default=None, help='Optional Accelerate limits, e.g. {"0":"3GiB","cpu":"32GiB"}.')
    parser.add_argument("--debug-output", action="store_true", help="Include full forward trajectories inside samples.jsonl.")
    args = parser.parse_args()
    for field in ("min_top1_score", "min_margin"):
        modern = getattr(args, "omtp_" + field)
        legacy = getattr(args, field)
        if modern is not None and legacy is not None and modern != legacy:
            parser.error(f"conflicting values for {field}")
        setattr(args, field, modern if modern is not None else legacy)
    return args


def _run_sample(
    args: argparse.Namespace,
    sample: BFCLSample,
    sample_index: int,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *, observer_factory: Any = None,
) -> tuple[dict[str, Any], OMTPObserver]:
    prompt = render_prompt(
        tokenizer,
        sample,
        getattr(args, "prompt_profile", "original"),
        getattr(args, "output_schema", "standard"),
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    spec = runtime_spec(model)
    response_prefill = resolve_response_prefill(
        tokenizer,
        model_family=spec.family,
        mode=getattr(args, "response_mode", "native"),
    )
    if prompt_ids.shape[1] + args.gen_length > spec.max_sequence_length:
        raise ValueError(f"{sample.sample_id}: prompt plus generation exceeds context")
    mask_id = spec.mask_token_id
    reliability = reliability_options(args)
    observer_class = ReliableOMTPObserver if reliability.enabled else OptimizedOMTPObserver
    observer = observer_class(
        **({"reliability": reliability} if reliability.enabled else {}),
        tokenizer=tokenizer,
        candidate_names=sample.candidate_names,
        stable_n=args.stable_n,
        config=OMTPConfig(
            stable_n=args.stable_n,
            region_radius=args.region_radius,
            region_top_k=args.region_top_k,
            min_seen=args.min_seen,
            min_support=args.min_support,
            min_top1_score=args.min_top1_score,
            min_margin=args.min_margin,
        ),
        profile_enabled=args.profile_observer,
        record_history=getattr(args, "debug_output", False),
    )
    if observer_factory is not None:
        observer = observer_factory(observer)
    sample_seed = args.seed + sample_index
    baseline_output = None
    baseline_time = None
    baseline_nfe = None
    if sample_index < args.parity_samples:
        _generate_once(args=args, model=model, prompt_ids=prompt_ids, mask_id=mask_id, observer=None, seed=sample_seed, device=device, generation_prefix_ids=response_prefill.token_ids)
        baseline_output, baseline_nfe, baseline_time = _generate_once(
            args=args, model=model, prompt_ids=prompt_ids, mask_id=mask_id,
            observer=None, seed=sample_seed, device=device,
            generation_prefix_ids=response_prefill.token_ids,
        )
    output, nfe, generation_time = _generate_once(
        args=args, model=model, prompt_ids=prompt_ids, mask_id=mask_id,
        observer=observer, seed=sample_seed, device=device,
        generation_prefix_ids=response_prefill.token_ids,
    )
    generated_ids = output[0, prompt_ids.shape[1]:].detach().cpu().tolist()
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    evaluated = _decode_completion(tokenizer, generated_ids)
    compact = compact_sample(
        sample=sample, seed=sample_seed, observer=observer, tokenizer=tokenizer,
        generated_ids=generated_ids, raw_output=raw_output, completion=evaluated,
        generation_time_s=generation_time, forward_count=nfe,
        prompt_token_count=int(prompt_ids.shape[1]),
        observer_off_time_s=baseline_time,
        output_identical=(bool(torch.equal(baseline_output, output)) if baseline_output is not None else None),
        nfe_identical=(baseline_nfe == nfe if baseline_nfe is not None else None),
    )
    compact["remaining_mask_count"] = generated_ids.count(mask_id)
    if not getattr(args, "debug_output", False):
        return compact, observer
    output_schema = getattr(args, "output_schema", "standard")
    parsed = parse_tool_calls(
        evaluated,
        sample.candidate_names,
        output_schema=output_schema,
        functions=sample.functions,
    )
    predicted_calls = parsed.calls if parsed.format_valid else []
    partial_names = extract_partial_names(evaluated, sample.candidate_names)
    final_occurrences = (
        [call["name"] for call in predicted_calls]
        if parsed.format_valid else partial_names
    )
    final_tools = ordered_unique(final_occurrences)
    gt_occurrences = [call["name"] for call in sample.ground_truth_calls]
    online = evaluate_omtp_fallback(
        observer,
        final_tools=final_tools,
        ground_truth_tools=sample.ground_truth_tool_types,
        final_occurrences=final_occurrences,
        ground_truth_occurrences=gt_occurrences,
    )
    omtp_gt = online["vs_ground_truth"]
    natural_gt = online["natural_baseline"]["vs_ground_truth"]
    record = {
        "sample_id": sample.sample_id,
        "method": METHOD_NAME,
        "cache_mode": args.cache_mode,
        "output_schema": output_schema,
        "model_output_tool_calls": model_call_items(evaluated),
        "flat_conversion_applied": output_schema == "flat_name_v1",
        "flat_conversion_valid": (
            parsed.valid if output_schema == "flat_name_v1" else None
        ),
        "flat_conversion_error": (
            parsed.error if output_schema == "flat_name_v1" else None
        ),
        "converted_standard_tool_calls": predicted_calls,
        "parameter_aliases": parameter_aliases(sample.functions),
        "response_mode": response_prefill.mode,
        "generation_prefix_text": response_prefill.text,
        "generation_prefix_token_ids": list(response_prefill.token_ids),
        "generation_prefix_token_count": len(response_prefill.token_ids),
        "generation_prefix_is_model_generated": False,
        "observer_implementation": "reliable_optimized" if reliability.enabled else "optimized",
        "observer_reliability_options": vars(reliability),
        "observer_confidence_gates": {
            "min_top1_score": args.min_top1_score,
            "min_margin": args.min_margin,
        },
        "candidate_tools": sample.candidate_names,
        "final_tools": final_tools,
        "ground_truth_tools": sorted(sample.ground_truth_tool_types),
        "ground_truth_tool_occurrences": gt_occurrences,
        "final_tool_occurrences": final_occurrences,
        "generated_tool_calls": predicted_calls,
        "raw_output": raw_output,
        "evaluated_output_before_eos": evaluated,
        "final_json_valid": parsed.json_valid,
        "tool_call_format_valid": parsed.format_valid,
        "candidate_names_valid": parsed.candidate_names_valid,
        "final_json_error": parsed.error,
        **evaluate_final_calls(sample, predicted_calls),
        "omtp_events": observer.events,
        "prefix_event_audit": [
            audit_prefix_event(
                event, generated_ids=generated_ids, tokenizer=tokenizer,
                candidate_names=sample.candidate_names,
            )
            for event in observer.events if event["source"] == "prefix"
        ],
        "omtp_fallback_events": online["events"],
        "per_tool_predictions": online["per_tool_predictions"],
        "natural_detection_times": online["natural_detection_times"],
        "generation_time_s": generation_time,
        "omtp_tool_precision": omtp_gt["tool_precision"],
        "omtp_tool_recall": omtp_gt["tool_recall"],
        "omtp_tool_f1": omtp_gt["tool_f1"],
        "omtp_tool_set_exact_match": omtp_gt["tool_set_exact_match"],
        "all_tool_types_prefetched": omtp_gt["all_tool_types_prefetched"],
        "all_tool_types_prefetched_time_s": omtp_gt[
            "oracle_all_correct_coverage_time_s"
        ],
        "first_correct_prediction_time_s": omtp_gt["oracle_first_correct_prediction_time_s"],
        "all_correct_prediction_time_s": omtp_gt["oracle_exact_set_time_s"],
        "natural_first_detection_time_s": natural_gt[
            "oracle_first_correct_prediction_time_s"
        ],
        "natural_all_detection_time_s": natural_gt["oracle_exact_set_time_s"],
        "natural_all_tool_types_detection_time_s": natural_gt[
            "oracle_all_correct_coverage_time_s"
        ],
        "omtp_lead_vs_natural_s": omtp_gt[
            "paired_exact_set_lead_vs_natural_s"
        ],
        "all_tool_occurrences_prefetched": omtp_gt[
            "all_tool_occurrences_prefetched"
        ],
        "tool_occurrence_multiset_exact_match": omtp_gt[
            "tool_occurrence_multiset_exact_match"
        ],
        "position_accuracy": omtp_gt["position_accuracy"],
        "position_exact_match": omtp_gt["position_exact_match"],
        "all_occurrences_prediction_time_s": omtp_gt[
            "oracle_all_occurrences_covered_time_s"
        ],
        "all_occurrences_lead_vs_natural_s": omtp_gt[
            "paired_all_occurrences_lead_vs_natural_s"
        ],
        "omtp_evaluation": {
            "vs_final": online["vs_final"],
            "vs_ground_truth": online["vs_ground_truth"],
        },
        "natural_baseline": online["natural_baseline"],
        "position_comparison": online["position_comparison"],
        "forward_count": nfe,
        "model_forward_time_s": sum(step["forward_time_s"] for step in observer.steps),
        "observer_processing_time_s": observer.total_processing_s,
        "observer_mean_processing_time_s": observer.total_processing_s / nfe if nfe else None,
        "omtp_processing_time_s": observer.omtp_processing_s,
        "observer_profile": observer.profile_summary(),
        "observer_off_generation_time_s": baseline_time,
        "observer_output_identical": (
            bool(torch.equal(baseline_output, output)) if baseline_output is not None else None
        ),
        "observer_nfe_identical": (
            baseline_nfe == nfe if baseline_nfe is not None else None
        ),
        "observer_wall_overhead_ratio": (
            generation_time / baseline_time - 1 if baseline_time not in (None, 0) else None
        ),
        "prompt_token_count": int(prompt_ids.shape[1]),
        "steps": observer.steps,
        "omtp_trajectory": observer.trajectory,
    }
    compact["debug"] = record
    return compact, observer


def _metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    value = {
        "record_type": "metadata", "schema_version": "tool_first_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "fastdllm_commit": _git_head(args.fastdllm_root),
        "questions_sha256": _sha256(args.questions),
        "answers_sha256": _sha256(args.answers),
        "dataset_validation": validate_dataset(args.questions, args.answers),
        "algorithm": {
            "name": "OMTP+fallback",
            "cache_mode": args.cache_mode,
            "response_mode": getattr(args, "response_mode", "native"),
            "output_schema": getattr(args, "output_schema", "standard"),
            "observer_implementation": "optimized",
            "reliability_options": vars(reliability_options(args)),
            "confidence_gates": {"min_top1_score": args.min_top1_score, "min_margin": args.min_margin},
            "anchor": "JSON name logits anchor, outer tool-call object only",
            "score": "candidate mean log-probability, region top-2 mean",
            "online_parameters": {
                "stable_n": args.stable_n, "region_radius": args.region_radius,
                "region_top_k": args.region_top_k, "min_seen": args.min_seen,
                "min_support": args.min_support,
            },
            "ground_truth_used_online": False,
            "extra_model_forwards_per_request": 0,
            "fallback_policy": "first causal event per tool identity across all tracks and sources",
            "analysis_baseline": NATURAL_BASELINE_NAME,
        },
        "timing_semantics": "All times start before generation canvas construction (including prompt processing by model); OMTP trigger timestamps are sampled after the causal detection/scoring work that produced the event. CUDA model/transfer work is synchronized by the existing callback path.",
        "parity_method": "For selected samples: unmeasured observer-off warmup, measured observer-off run, measured observer-on run with identical seed.",
    }
    source = Path(__file__).resolve().parent
    value["code_sha256"] = {
        name: _sha256(source / name) for name in (
            "run_omtp_experiment.py", "omtp_observer.py", "optimized_omtp_observer.py",
            "reliable_omtp_observer.py", "prefix_detection.py", "experiment_output.py",
            "observer.py", "fastdllm_generate.py", "prompting.py", "tool_parser.py",
        )
    }
    return value


def main() -> None:
    args = parse_args()
    if args.temperature != 0 or args.gen_length % args.block_length:
        raise ValueError("OMTP requires temperature=0 and divisible block length")
    if args.steps % (args.gen_length // args.block_length):
        raise ValueError("steps must be divisible by block count")
    if args.limit < 1 or args.parity_samples < 0:
        raise ValueError("invalid limit/parity-samples")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    samples = list(iter_samples(args.questions, args.answers, limit=args.limit, start=args.start))
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
