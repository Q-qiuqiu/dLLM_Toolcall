"""Llama/Hugging Face AR Natural baseline for TaskBench and BFCL (one JSONL)."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ar_generate import generate_ar, resolve_eos_ids
from bfcl_data import iter_samples, validate_dataset
from experiment_output import compact_sample, failure_sample, open_result_file, write_record
from natural_observer import NaturalOnlyObserver
from prompting import PROMPT_PROFILES, build_messages, render_prompt
from run_experiment import _sha256
from taskbench_data import iter_taskbench_samples, validate_taskbench_manifest
from tool_format import OUTPUT_SCHEMAS
from tool_parser import extract_partial_names, ordered_unique, parse_tool_calls


DEFAULT_MODEL = "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("taskbench", "bfcl"), required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--tool-catalog")
    parser.add_argument("--questions")
    parser.add_argument("--answers")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Optional diagnostic budget only. Omit for natural EOS generation, bounded only by model context.",
    )
    parser.add_argument("--prompt-profile", choices=PROMPT_PROFILES, default=None)
    parser.add_argument("--output-schema", choices=OUTPUT_SCHEMAS, default=None)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager"), default="sdpa",
        help="Native PyTorch SDPA by default; eager explicitly reproduces the old server attention backend.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--parity-samples", type=int, default=1)
    parser.add_argument("--debug-output", action="store_true")
    args = parser.parse_args()
    needed = ("manifest", "tool_catalog") if args.benchmark == "taskbench" else ("questions", "answers")
    unused = ("questions", "answers") if args.benchmark == "taskbench" else ("manifest", "tool_catalog")
    for name in needed:
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required for {args.benchmark}")
    if any(getattr(args, name) for name in unused):
        parser.error("do not mix TaskBench and BFCL input flags")
    if (args.max_new_tokens is not None and args.max_new_tokens < 1) or args.start < 0 or args.parity_samples < 0 or (args.limit is not None and args.limit < 1):
        parser.error("invalid length/start/limit/parity-samples")
    args.prompt_profile = args.prompt_profile or ("compact_json_dep_v4" if args.benchmark == "taskbench" else "original")
    args.output_schema = args.output_schema or ("flat_name_v1" if args.benchmark == "taskbench" else "standard")
    args.observer_mode = "natural"
    args.inference_backend = "huggingface_ar"
    args.temperature = 0.0
    args.batch_size = 1
    args.use_cache = True
    args.generation_stop_policy = "eos_or_context_limit" if args.max_new_tokens is None else "eos_or_explicit_token_budget"
    return args


def load_ar_model(args, device):
    root = Path(args.model_path).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"local Hugging Face checkpoint missing: {root}")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("AR tokenizer has neither PAD nor EOS")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, torch_dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation, low_cpu_mem_usage=True,
    ).to(device).eval()
    model.config.use_cache = True
    args.model_path_resolved = str(root)
    args.model_family = "huggingface_ar"
    # No hash scan of multi-GB weights during the benchmark. Preserve index,
    # config/tokenizer hashes and shard file sizes instead.
    args.model_runtime = {
        "architecture": type(model).__name__, "model_type": model.config.model_type,
        "dtype": str(model.dtype), "use_cache": True,
        "attention_implementation": model.config._attn_implementation,
        "max_sequence_length": model.config.max_position_embeddings,
        "eos_token_ids": resolve_eos_ids(model, tokenizer),
        "file_sha256": {name: _sha256(root / name) for name in (
            "config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        ) if (root / name).is_file()},
        "weight_files": {p.name: p.stat().st_size for p in sorted(root.glob("*.safetensors"))},
        "revision": None,
    }
    revision = root / ".cache/huggingface/download/config.json.metadata"
    if revision.is_file():
        args.model_runtime["revision"] = revision.read_text().splitlines()[0].strip()
    return model, tokenizer


def run_ar_sample(args, sample, sample_index, model, tokenizer, device):
    # Reuse unchanged instruction/definitions; only the model-native chat
    # template differs. In particular, never duplicate Llama's BOS token.
    prompt = render_prompt(tokenizer, sample, args.prompt_profile, args.output_schema)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    context_limit = int(model.config.max_position_embeddings)
    if prompt_ids.shape[1] >= context_limit:
        raise ValueError(f"{sample.sample_id}: prompt leaves no space in AR context")
    if args.max_new_tokens is not None and prompt_ids.shape[1] + args.max_new_tokens > context_limit:
        raise ValueError(f"{sample.sample_id}: prompt plus output budget exceeds AR context")
    observer = NaturalOnlyObserver(
        tokenizer=tokenizer, candidate_names=sample.candidate_names,
        record_history=args.debug_output,
    )
    seed = args.seed + sample_index
    generate_kwargs = dict(
        model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens, seed=seed, device=device,
    )
    off = None
    if sample_index < args.parity_samples:
        generate_ar(**generate_kwargs, observer=None)  # unmeasured warmup
        off = generate_ar(**generate_kwargs, observer=None)
    on = generate_ar(**generate_kwargs, observer=observer)
    ids = on.output[0, prompt_ids.shape[1]:].detach().cpu().tolist()
    eos_ids = set(resolve_eos_ids(model, tokenizer))
    stop = next((i for i, token in enumerate(ids) if token in eos_ids), len(ids))
    context_reached = stop == len(ids) and int(on.output.shape[1]) >= context_limit
    if stop < len(ids):
        stop_reason = "eos"
    elif context_reached:
        stop_reason = "context_limit"
    elif args.max_new_tokens is not None and len(ids) >= args.max_new_tokens:
        stop_reason = "length_budget"
    else:
        stop_reason = "unexpected_stop"
    raw = tokenizer.decode(ids, skip_special_tokens=True)
    completion = tokenizer.decode(ids[:stop], skip_special_tokens=True)
    record = compact_sample(
        sample=sample, seed=seed, observer=observer, tokenizer=tokenizer,
        generated_ids=ids, raw_output=raw, completion=completion,
        generation_time_s=on.generation_time_s, forward_count=on.forward_count,
        prompt_token_count=int(prompt_ids.shape[1]),
        observer_off_time_s=off.generation_time_s if off else None,
        output_identical=bool(torch.equal(off.output, on.output)) if off else None,
        nfe_identical=off.forward_count == on.forward_count if off else None,
    )
    parsed = parse_tool_calls(completion, sample.candidate_names,
                              output_schema=args.output_schema, functions=sample.functions)
    final = ordered_unique(extract_partial_names(raw, sample.candidate_names))
    def all_time(tools):
        # Offline annotation only. Never feeds generation or detection.
        return max(observer.natural_first_events[t]["predict_time_s"] for t in tools) if tools and set(tools) <= observer.natural_first_events.keys() else None
    record.update(
        inference_backend="huggingface_ar", observer_mode="natural", cache_mode="ar_kv",
        output_token_count=stop, generated_token_count=len(ids),
        eos_detected=stop < len(ids), remaining_mask_count=0,
        stop_reason=stop_reason,
        natural_generation_complete=stop < len(ids),
        context_limit_reached=context_reached,
        context_limit_tokens=context_limit,
        requested_max_new_tokens=args.max_new_tokens,
        generation_stop_policy="eos_or_context_limit" if args.max_new_tokens is None else "eos_or_explicit_token_budget",
        first_token_time_s=on.first_token_time_s,
        prefill_forward_time_s=on.prefill_forward_time_s,
        model_forward_time_s=on.model_forward_time_s,
        decode_forward_time_s=on.model_forward_time_s - (on.prefill_forward_time_s or 0.),
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        messages_sha256=hashlib.sha256(json.dumps(build_messages(sample, args.prompt_profile, args.output_schema), ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        final_tools=final, final_json_valid=parsed.json_valid,
        tool_call_format_valid=parsed.format_valid, candidate_names_valid=parsed.candidate_names_valid,
        final_parse_error=parsed.error,
        all_final_tools_natural_time_s=all_time(final),
        all_ground_truth_tools_natural_time_s=all_time(sample.ground_truth_tool_types),
        coverage_times_are_offline_annotations=True,
    )
    if args.debug_output:
        record["debug"] = {"method": "ar_natural", "steps": observer.steps}
    return record


def metadata(args, device):
    source = Path(__file__).resolve().parent
    value = {
        "record_type": "metadata", "schema_version": "tool_first_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv, "args": vars(args), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "online_method": {
            "name": "ar_natural_only", "backend": "AutoModelForCausalLM.generate",
            "detection": "same NaturalOnlyObserver/detect_committed_tools as LLaDA",
            "use_cache": True, "do_sample": False, "num_beams": 1,
            "extra_model_forwards": 0, "ground_truth_used_online": False,
            "stopping": "checkpoint EOS/turn-end; physical model context limit only unless an explicit diagnostic max_new_tokens is supplied; never tool/GT/JSON-driven",
            "default_output_token_budget": None,
            "requested_max_new_tokens": args.max_new_tokens,
        },
        "timing_semantics": {
            "zero": "before native generate setup; includes prefill and all generated tokens",
            "excludes": ["model loading", "prompt rendering/tokenization", "offline evaluation", "JSONL serialization"],
            "events": "after token commit synchronization + D2H + shared Natural detection",
            "forward_count": "actual model forward hooks; first call is full-prompt prefill, later calls use KV cache",
            "model_forward_time_s": "sum of synchronized forward wall times, excluding name detector and token selection",
            "observer_time_s": "Natural callback processing only; hook/synchronization overhead is in generation_time_s",
            "parity": "first N: warmup, measured detection-off, measured detection-on; timing hooks remain in both",
        },
        "fairness_notes": [
            "AR defaults to natural EOS completion, dLLM retains its fixed canvas; report the distinct stopping policies, coverage and output lengths.",
            "NFE differs by design: AR one per generated token, dLLM one per denoising step.",
            "Do not claim AR must be slower. Report actual timing and quality including coverage failures.",
        ],
        "code_sha256": {name: _sha256(source / name) for name in (
            "run_ar_experiment.py", "ar_generate.py", "natural_observer.py", "observer.py",
            "experiment_output.py", "prompting.py", "tool_parser.py", "tool_format.py",
            "bfcl_data.py", "taskbench_data.py",
        )},
    }
    if args.benchmark == "taskbench":
        value.update(
            manifest_sha256=_sha256(args.manifest), tool_catalog_sha256=_sha256(args.tool_catalog),
            dataset_validation=validate_taskbench_manifest(args.manifest, args.tool_catalog),
        )
    else:
        value.update(
            questions_sha256=_sha256(args.questions), answers_sha256=_sha256(args.answers),
            dataset_validation=validate_dataset(args.questions, args.answers),
        )
    return value


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run on a GPU-enabled terminal")
    device = torch.device(args.device)
    if args.benchmark == "taskbench":
        all_samples = list(iter_taskbench_samples(args.manifest, args.tool_catalog))
    else:
        all_samples = list(iter_samples(args.questions, args.answers))
    samples = all_samples[args.start:None if args.limit is None else args.start + args.limit]
    if not samples:
        raise ValueError("empty selected dataset")
    with open_result_file(args.output_dir) as handle:
        model, tokenizer = load_ar_model(args, device)
        write_record(handle, metadata(args, device))
        for index, sample in enumerate(samples):
            print(f"[{index + 1}/{len(samples)}] {sample.sample_id}", flush=True)
            try:
                record = run_ar_sample(args, sample, index, model, tokenizer, device)
            except Exception as error:
                print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                record = failure_sample(sample, args.seed + index, error)
                record.update(inference_backend="huggingface_ar", observer_mode="natural")
            write_record(handle, record)
    print(f"Results written to {Path(args.output_dir).resolve() / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
