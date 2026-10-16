"""Paired BFCL gate diagnostic: four observers, identical generation forwards.

Only accuracy and forward-index earliness are comparable. Observers run
sequentially in a shared callback; their wall timestamps are NOT independent
runtime measurements. One JSONL retains every variant's first-tool events.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

from bfcl_data import iter_samples
from experiment_output import compact_sample, failure_sample, open_result_file, write_record
from omtp_observer import OMTPConfig
from reliable_omtp_observer import ReliabilityOptions, ReliableOMTPObserver
from run_experiment import _load_model
from run_omtp_experiment import _metadata, _run_sample, parse_args
from summarize_experiment import summarize


VARIANTS = {
    "A_no_gates": {"strict_anchor": False, "min_top1_score": None, "min_margin": None},
    "B_anchor_only": {"strict_anchor": True, "min_top1_score": None, "min_margin": None},
    "C_confidence_only": {"strict_anchor": False, "min_top1_score": -5., "min_margin": .1},
    "D_full_gates": {"strict_anchor": True, "min_top1_score": -5., "min_margin": .1},
}


class ObserverGroup:
    def __init__(self, *, tokenizer, candidate_names, stable_n=2):
        self.observers = {
            name: ReliableOMTPObserver(
                tokenizer=tokenizer, candidate_names=candidate_names,
                reliability=ReliabilityOptions(strict_anchor=cfg["strict_anchor"]),
                config=OMTPConfig(stable_n=stable_n, min_top1_score=cfg["min_top1_score"], min_margin=cfg["min_margin"]),
                stable_n=stable_n, record_history=False,
            )
            for name, cfg in VARIANTS.items()
        }

    def __getattr__(self, name):
        return getattr(self.observers["D_full_gates"], name)

    def __call__(self, observation):
        original_x = observation.x.clone()
        original_committed = observation.committed_x.clone()
        for observer in self.observers.values():
            observer(observation)
            if not torch.equal(observation.x, original_x) or not torch.equal(observation.committed_x, original_committed):
                raise RuntimeError("observer mutated shared generation state")


def ablation_summary(records):
    result = {"sample_count": len(records), "variants": {}}
    for name in VARIANTS:
        rows = [record["ablation_variants"][name] for record in records]
        summary = summarize(rows)
        # Do not present the coupled callback wall times as fair comparisons.
        result["variants"][name] = {
            "accuracy": summary["accuracy"],
            "lead_forwards": summary["paired_lead_forwards"],
            "first_event_sources": summary["first_event_sources"],
            "runtime_failure_count": summary["runtime_failure_count"],
            "nfe_counts": summary["nfe_counts"],
            "parity_checked": summary["parity_checked"],
            "parity_failed": summary["parity_failed"],
        }
    return result


def main():
    args = parse_args()
    if args.cache_mode != "none" or args.debug_output:
        raise ValueError("gate diagnostic requires no cache and compact output")
    if args.region_radius != 4 or args.region_top_k != 2 or args.min_seen != 2 or args.min_support != .5:
        raise ValueError("diagnostic keeps the frozen region/tracking parameters")
    samples = list(iter_samples(args.questions, args.answers, start=args.start, limit=args.limit))
    device = torch.device(args.device)
    records = []
    with open_result_file(args.output_dir) as handle:
        model, tokenizer = _load_model(args, device)
        meta = _metadata(args, device)
        meta.update(
            ablation_variants=VARIANTS,
            timing_comparable=False,
            ablation_semantics="all variants see identical model logits and committed tokens; no extra forwards for variants",
            selected_sample_ids=[sample.sample_id for sample in samples],
        )
        meta["code_sha256"]["run_bfcl_gate_ablation.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_record(handle, meta)
        for index, sample in enumerate(samples):
            print(f"[{index + 1}/{len(samples)}] {sample.sample_id}", flush=True)
            group = ObserverGroup(tokenizer=tokenizer, candidate_names=sample.candidate_names, stable_n=args.stable_n)
            try:
                record, _ = _run_sample(
                    args, sample, index, model, tokenizer, device,
                    observer_factory=lambda default: group,
                )
                ids = group.steps[-1]["generation_token_ids_after"]
                variants = {}
                for name, observer in group.observers.items():
                    variant = compact_sample(
                        sample=sample, seed=record["seed"], observer=observer, tokenizer=tokenizer,
                        generated_ids=ids, raw_output=record["raw_output"],
                        completion=record.get("output_before_eos", record["raw_output"]),
                        generation_time_s=record["generation_time_s"], forward_count=record["forward_count"],
                        prompt_token_count=record["prompt_token_count"],
                        observer_off_time_s=record["parity"].get("observer_off_generation_time_s"),
                        output_identical=record["parity"].get("output_identical"),
                        nfe_identical=record["parity"].get("nfe_identical"),
                    )
                    seen = set(); latent = []
                    for event in observer.latent_events:
                        if event["predicted_tool"] not in seen:
                            latent.append({"tool": event["predicted_tool"], "forward": event["forward_step"], "margin": event["margin"]})
                            seen.add(event["predicted_tool"])
                    variant["latent_tool_events"] = latent
                    variants[name] = variant
                record["ablation_variants"] = variants
            except Exception as error:
                print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                record = failure_sample(sample, args.seed + index, error)
                record["ablation_variants"] = {name: dict(record) for name in VARIANTS}
            write_record(handle, record)
            records.append(record)
            print(json.dumps({
                name: {"tools": [e["tool"] for e in variant["first_tool_events"]], "error": variant["error"]}
                for name, variant in record["ablation_variants"].items()
            }, ensure_ascii=False), flush=True)
    print(json.dumps(ablation_summary(records), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
