"""Run BFCL parallel_multiple with the fixed OMTP + fallback policy."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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
from omtp_report import build_omtp_report
from prompting import render_prompt
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
    parser.add_argument("--min-top1-score", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    return parser.parse_args()


def _run_sample(
    args: argparse.Namespace,
    sample: BFCLSample,
    sample_index: int,
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> tuple[dict[str, Any], OMTPObserver]:
    prompt = render_prompt(tokenizer, sample)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if prompt_ids.shape[1] + args.gen_length > getattr(model.config, "max_sequence_length", 4096):
        raise ValueError(f"{sample.sample_id}: prompt plus generation exceeds context")
    mask_id = getattr(model.config, "mask_token_id", None) or 126336
    observer = OptimizedOMTPObserver(
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
    )
    sample_seed = args.seed + sample_index
    baseline_output = None
    baseline_time = None
    baseline_nfe = None
    if sample_index < args.parity_samples:
        _generate_once(args=args, model=model, prompt_ids=prompt_ids, mask_id=mask_id, observer=None, seed=sample_seed, device=device)
        baseline_output, baseline_nfe, baseline_time = _generate_once(
            args=args, model=model, prompt_ids=prompt_ids, mask_id=mask_id,
            observer=None, seed=sample_seed, device=device,
        )
    output, nfe, generation_time = _generate_once(
        args=args, model=model, prompt_ids=prompt_ids, mask_id=mask_id,
        observer=observer, seed=sample_seed, device=device,
    )
    generated_ids = output[0, prompt_ids.shape[1]:].detach().cpu().tolist()
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    evaluated = _decode_completion(tokenizer, generated_ids)
    parsed = parse_tool_calls(evaluated, sample.candidate_names)
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
        "observer_implementation": "optimized",
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
    return record, observer


def _metadata(args: argparse.Namespace, output: Path, device: torch.device) -> None:
    value = {
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
            "observer_implementation": "optimized",
            "anchor": "JSON name logits anchor, outer tool-call object only",
            "score": "candidate mean log-probability, region top-2 mean",
            "online_parameters": {
                "stable_n": args.stable_n, "region_radius": args.region_radius,
                "region_top_k": args.region_top_k, "min_seen": args.min_seen,
                "min_support": args.min_support,
            },
            "ground_truth_used_online": False,
            "extra_model_forwards_per_request": 0,
            "fallback_policy": "first event per occurrence track; latent_region, then unique prefix, then natural",
            "analysis_baseline": NATURAL_BASELINE_NAME,
        },
        "timing_semantics": "All times start before generation canvas construction (including prompt processing by model); OMTP trigger timestamps are sampled after the causal detection/scoring work that produced the event. CUDA model/transfer work is synchronized by the existing callback path.",
        "parity_method": "For selected samples: unmeasured observer-off warmup, measured observer-off run, measured observer-on run with identical seed.",
    }
    (output / "metadata.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _runtime_failure_record(sample: BFCLSample, error: Exception) -> dict[str, Any]:
    def empty_result() -> dict[str, Any]:
        return {
            "predicted_tools": [], "all_tool_types_prefetched": False,
            "tool_precision": 0.0, "tool_recall": 0.0,
            "tool_f1": 0.0, "tool_set_exact_match": False,
            "false_positive_count": 0,
            "false_negative_count": len(sample.ground_truth_tool_types),
            "false_positives": [],
            "false_negatives": sorted(sample.ground_truth_tool_types),
            "correct_trigger_coverage": 0.0,
            "online_first_trigger_time_s": None,
            "oracle_first_correct_prediction_time_s": None,
            "oracle_all_correct_coverage_time_s": None,
            "oracle_exact_set_time_s": None,
            "first_times_by_tool": {}, "trigger_count": 0,
            "paired_exact_set_lead_vs_natural_s": None,
            "predicted_tool_occurrences": [],
            "expected_tool_occurrences": [],
            "occurrence_precision": 0.0, "occurrence_recall": 0.0,
            "occurrence_f1": 0.0,
            "all_tool_occurrences_prefetched": False,
            "tool_occurrence_count_match": False,
            "tool_occurrence_multiset_exact_match": False,
            "oracle_all_occurrences_covered_time_s": None,
            "oracle_exact_occurrence_multiset_time_s": None,
            "natural_all_occurrences_covered_time_s": None,
            "paired_all_occurrences_lead_vs_natural_s": None,
            "position_correct_count": 0, "position_total": 0,
            "position_accuracy": 0.0, "position_exact_match": False,
            "position_records": [],
        }
    message = f"{type(error).__name__}: {error}"
    return {
        "sample_id": sample.sample_id, "method": METHOD_NAME,
        "cache_mode": None,
        "observer_implementation": "optimized",
        "candidate_tools": sample.candidate_names,
        "final_tools": [], "ground_truth_tools": sorted(sample.ground_truth_tool_types),
        "final_tool_occurrences": [],
        "ground_truth_tool_occurrences": [call["name"] for call in sample.ground_truth_calls],
        "generated_tool_calls": [], "raw_output": "", "evaluated_output_before_eos": "",
        "final_json_valid": False, "tool_call_format_valid": False,
        "candidate_names_valid": False, "final_json_error": message,
        "tool_name_precision": 0.0, "tool_name_recall": 0.0, "tool_name_f1": 0.0,
        "tool_set_exact_match": False, "tool_type_precision": 0.0,
        "tool_type_recall": 0.0, "tool_type_f1": 0.0,
        "tool_call_count_match": False, "bfcl_ast_compatible_match": False,
        "omtp_events": [], "omtp_fallback_events": [], "per_tool_predictions": [],
        "natural_detection_times": {}, "generation_time_s": None,
        "omtp_tool_precision": 0.0, "omtp_tool_recall": 0.0, "omtp_tool_f1": 0.0,
        "omtp_tool_set_exact_match": False, "first_correct_prediction_time_s": None,
        "all_tool_types_prefetched": False,
        "all_tool_types_prefetched_time_s": None,
        "all_correct_prediction_time_s": None,
        "natural_first_detection_time_s": None,
        "natural_all_detection_time_s": None,
        "natural_all_tool_types_detection_time_s": None,
        "omtp_lead_vs_natural_s": None,
        "all_tool_occurrences_prefetched": False,
        "tool_occurrence_multiset_exact_match": False,
        "position_accuracy": 0.0, "position_exact_match": False,
        "all_occurrences_prediction_time_s": None,
        "all_occurrences_lead_vs_natural_s": None,
        "omtp_evaluation": {
            "vs_final": empty_result(),
            "vs_ground_truth": empty_result(),
        },
        "natural_baseline": {
            "method": NATURAL_BASELINE_NAME, "events": [],
            "vs_final": empty_result(),
            "vs_ground_truth": empty_result(),
        },
        "position_comparison": {"vs_final": [], "vs_ground_truth": []},
        "forward_count": None, "model_forward_time_s": None,
        "observer_processing_time_s": None, "observer_mean_processing_time_s": None,
        "omtp_processing_time_s": None, "observer_off_generation_time_s": None,
        "observer_output_identical": None, "observer_nfe_identical": None,
        "observer_wall_overhead_ratio": None, "prompt_token_count": None,
        "observer_profile": {}, "runtime_error": message, "steps": [],
        "omtp_trajectory": [],
    }


def main() -> None:
    args = parse_args()
    if args.temperature != 0 or args.gen_length % args.block_length:
        raise ValueError("OMTP smoke requires temperature=0 and divisible block length")
    if args.steps % (args.gen_length // args.block_length):
        raise ValueError("steps must be divisible by block count")
    if args.limit < 1 or args.parity_samples < 0:
        raise ValueError("invalid limit/parity-samples")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir)
    protected = (
        "samples.jsonl", "steps.jsonl", "failures.jsonl", "metadata.json",
        "omtp_events.jsonl", "omtp_trajectory.jsonl", "omtp_samples.jsonl",
        "omtp_fallback_events.jsonl", "natural_events.jsonl",
        "position_records.jsonl", "omtp_position_summary.csv",
        "omtp_summary.csv", "omtp_comparison.csv", "omtp_diagnostics.csv",
    )
    for name in protected:
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, tokenizer = _load_model(args, device)
    _metadata(args, output, device)
    samples = list(iter_samples(args.questions, args.answers, limit=args.limit, start=args.start))
    paths = {name: (output / name).open("w", encoding="utf-8") for name in (
        "samples.jsonl", "steps.jsonl", "failures.jsonl", "omtp_events.jsonl",
        "omtp_fallback_events.jsonl", "natural_events.jsonl",
        "position_records.jsonl",
        "omtp_trajectory.jsonl", "omtp_samples.jsonl",
    )}
    try:
        for index, sample in enumerate(samples):
            print(f"[{index + 1}/{len(samples)}] {sample.sample_id}", flush=True)
            try:
                record, observer = _run_sample(args, sample, index, model, tokenizer, device)
            except Exception as error:
                failure = {"sample_id": sample.sample_id, "failure_types": ["runtime_error"], "error": f"{type(error).__name__}: {error}"}
                paths["failures.jsonl"].write(json.dumps(failure, ensure_ascii=False) + "\n")
                paths["failures.jsonl"].flush()
                print(f"  FAILED: {failure['error']}", file=sys.stderr, flush=True)
                record = _runtime_failure_record(sample, error)
                record["cache_mode"] = args.cache_mode
                serialized = json.dumps(record, ensure_ascii=False) + "\n"
                paths["samples.jsonl"].write(serialized)
                paths["omtp_samples.jsonl"].write(serialized)
                continue
            serialized = json.dumps(record, ensure_ascii=False) + "\n"
            paths["samples.jsonl"].write(serialized)
            paths["omtp_samples.jsonl"].write(serialized)
            for step in record["steps"]:
                paths["steps.jsonl"].write(json.dumps({"sample_id": sample.sample_id, **step}, ensure_ascii=False) + "\n")
            for event in observer.events:
                paths["omtp_events.jsonl"].write(json.dumps({"sample_id": sample.sample_id, **event}, ensure_ascii=False) + "\n")
            for event in record["omtp_fallback_events"]:
                paths["omtp_fallback_events.jsonl"].write(json.dumps({"sample_id": sample.sample_id, **event}, ensure_ascii=False) + "\n")
            for event in record["natural_baseline"]["events"]:
                paths["natural_events.jsonl"].write(json.dumps({"sample_id": sample.sample_id, **event}, ensure_ascii=False) + "\n")
            for reference, rows in record["position_comparison"].items():
                for row in rows:
                    paths["position_records.jsonl"].write(json.dumps({
                        "sample_id": sample.sample_id,
                        "reference": reference,
                        **row,
                    }, ensure_ascii=False) + "\n")
            for row in observer.trajectory:
                paths["omtp_trajectory.jsonl"].write(json.dumps({"sample_id": sample.sample_id, **row}, ensure_ascii=False) + "\n")
            failures = []
            if not record["final_json_valid"]: failures.append("invalid_json")
            if not record["tool_set_exact_match"]: failures.append("final_tool_set_mismatch")
            if not record["omtp_tool_set_exact_match"]: failures.append("omtp_ground_truth_set_mismatch")
            if record["observer_output_identical"] is False: failures.append("observer_changed_output")
            if record["observer_nfe_identical"] is False: failures.append("observer_changed_nfe")
            if failures:
                paths["failures.jsonl"].write(json.dumps({
                    "sample_id": sample.sample_id, "failure_types": failures,
                    "raw_output": record["raw_output"],
                    "omtp_events": record["omtp_events"],
                }, ensure_ascii=False) + "\n")
            for handle in paths.values():
                handle.flush()
    finally:
        for handle in paths.values():
            handle.close()
    build_omtp_report(output)
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
