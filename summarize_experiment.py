"""Summarize the single JSONL on stdout; never create report files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from tool_parser import extract_partial_names


def distribution(values):
    values = sorted(value for value in values if value is not None)
    def percentile(p):
        if not values:
            return None
        index = (len(values) - 1) * p
        low = int(index)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (index - low)
    return {
        "count": len(values), "mean": mean(values) if values else None,
        "p50": percentile(.5), "p95": percentile(.95),
    }


def summarize(rows):
    accuracy = {reference: Counter() for reference in ("final_raw_text", "ground_truth")}
    natural_times = []; online_times = []; leads = []; forward_leads = []
    missing = Counter(); sources = Counter(); prefix_status = Counter()
    model_invalid_name_mismatches = 0
    for row in rows:
        events = row["first_tool_events"]
        online = {e["tool"]: e for e in events}
        if len(online) != len(events):
            raise ValueError(f"{row['sample_id']}: duplicate first-tool events")
        final = set(extract_partial_names(row["raw_output"], row["candidate_tools"]))
        predicted = set(online)
        for ref, truth in (("final_raw_text", final), ("ground_truth", set(row["ground_truth_tools"]))):
            count = accuracy[ref]
            count["true_positive"] += len(predicted & truth)
            count["false_positive"] += len(predicted - truth)
            count["false_negative"] += len(truth - predicted)
            count["tasks_with_wrong_trigger"] += bool(predicted - truth)
            count["set_exact_match"] += predicted == truth
        for event in events:
            sources[event["source"]] += 1
            if event["source"] == "prefix":
                status = event.get("offline_prefix_audit", {}).get("status", "unresolved")
                prefix_status[status] += 1
                if status == "model_invalid_name" and event["tool"] not in final:
                    model_invalid_name_mismatches += 1
        for tool in final:
            n = row["natural_first_times"].get(tool)
            o = online.get(tool)
            if n is None or o is None:
                missing["both_missing" if n is None and o is None else "natural_only" if o is None else "online_only"] += 1
                continue
            natural_times.append(n["time_s"]); online_times.append(o["time_s"])
            leads.append(n["time_s"] - o["time_s"])
            forward_leads.append(n["forward"] - o["forward"])
    qualities = {}
    for reference, count in accuracy.items():
        tp, fp, fn = (count[key] for key in ("true_positive", "false_positive", "false_negative"))
        total = tp + fp
        quality = {
            **dict(count),
            "precision_micro": tp / total if total else None,
            "recall_micro": tp / (tp + fn) if tp + fn else None,
            "f1_micro": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "wrong_trigger_rate": fp / total if total else None,
            "wrong_tool_types_per_query": fp / len(rows) if rows else None,
            "set_exact_match_rate": count["set_exact_match"] / len(rows) if rows else None,
        }
        if reference == "final_raw_text":
            excluded = model_invalid_name_mismatches
            quality.update(
                model_invalid_name_trigger_count=excluded,
                # Excluded cases remain physical extra starts and raw FPs.
                # They are not relabelled as true positives.
                detector_error_count=fp - excluded,
                detector_error_rate_excluding_model_invalid_names=(
                    (fp - excluded) / (total - excluded) if total > excluded else None
                ),
            )
        qualities[reference] = quality
    return {
        "sample_count": len(rows),
        "runtime_failure_count": sum(bool(r.get("error")) for r in rows),
        "accuracy": qualities,
        "timing_reference": "final_raw_text, first occurrence per query/tool type",
        "natural_time_s": distribution(natural_times),
        "online_time_s": distribution(online_times),
        "paired_lead_s": distribution(leads),
        "paired_lead_forwards": distribution(forward_leads),
        "paired_missing": dict(missing),
        "wins_ties_losses": [sum(x > 1e-9 for x in leads), sum(abs(x) <= 1e-9 for x in leads), sum(x < -1e-9 for x in leads)],
        "generation_time_s": distribution([r.get("generation_time_s") for r in rows]),
        "observer_time_s": distribution([r.get("observer_time_s") for r in rows]),
        "nfe_counts": dict(Counter(r.get("forward_count") for r in rows)),
        "parity_checked": sum(r.get("parity", {}).get("checked", False) for r in rows),
        "parity_failed": sum(
            p.get("output_identical") is False or p.get("nfe_identical") is False
            for p in (r.get("parity", {}) for r in rows)
        ),
        "first_event_sources": dict(sources),
        "prefix_first_event_attribution": dict(prefix_status),
    }


def compare_natural_baseline(omtp_meta, omtp_rows, natural_meta, natural_rows):
    """Offline cross-run comparison, not total-observer subtraction.

    Keep negative leads and missing detections. Compare each final generated
    tool identity once per query; repeated occurrences do not inflate gains.
    Changed model outputs/NFE are reported and excluded from timing pairing.
    """
    left_args, right_args = omtp_meta["args"], natural_meta["args"]
    if left_args.get("observer_mode", "omtp") != "omtp":
        raise ValueError("--input must be the OMTP run")
    if right_args.get("observer_mode") != "natural":
        raise ValueError("--natural-baseline must have --observer-mode natural metadata")
    defaults = {"cache_mode": "none", "response_mode": "native",
                "output_schema": "standard", "prompt_profile": "original",
                "device_map": None, "max_memory_json": None, "debug_output": False}
    for key in (
        "gen_length", "block_length", "steps", "temperature", "seed", "dtype",
        "cache_mode", "response_mode", "output_schema", "prompt_profile",
        "device_map", "max_memory_json", "debug_output",
    ):
        if left_args.get(key, defaults.get(key)) != right_args.get(key, defaults.get(key)):
            raise ValueError(f"unfair pairing: configuration differs: {key}")
    for key in ("model_path_resolved", "model_family", "model_runtime"):
        if left_args.get(key) != right_args.get(key):
            raise ValueError(f"unfair pairing: model differs: {key}")
    warnings = []
    signatures = (
        "manifest_sha256", "tool_catalog_sha256", "questions_sha256", "answers_sha256",
        "fastdllm_commit", "gpu", "torch", "cuda",
    )
    for key in signatures:
        if key in omtp_meta or key in natural_meta:
            if key not in omtp_meta or key not in natural_meta:
                warnings.append(f"cannot verify metadata field: {key}")
            elif omtp_meta[key] != natural_meta[key]:
                raise ValueError(f"unfair pairing: metadata differs: {key}")
    for name in ("fastdllm_generate.py", "prompting.py", "model_adapter.py",
                 "response_prefill.py", "tool_format.py", "tool_parser.py"):
        left = omtp_meta.get("code_sha256", {}).get(name)
        right = natural_meta.get("code_sha256", {}).get(name)
        if left is not None and right is not None and left != right:
            raise ValueError(f"unfair pairing: generation/prompt code differs: {name}")
    def index(rows):
        mapping = {r["sample_id"]: r for r in rows}
        if len(mapping) != len(rows):
            raise ValueError("duplicate sample IDs cannot be paired")
        return mapping
    omtp, natural = index(omtp_rows), index(natural_rows)
    skipped = {"runtime_failure": [], "output_mismatch": [], "nfe_mismatch": []}
    coverage = Counter({key: 0 for key in ("both_detected", "omtp_only", "natural_only", "both_missing")})
    request_coverage = Counter({key: 0 for key in ("both_complete", "omtp_only", "natural_only", "both_incomplete", "empty_final_set")})
    natural_times, online_times, leads, forward_leads = [], [], [], []
    request_mean_leads, all_set_leads = [], []
    generation_natural, generation_omtp, generation_deltas = [], [], []
    matched, missing_prompt_hash, final_tool_count = 0, 0, 0
    for sid, row in omtp.items():
        baseline = natural.get(sid)
        if baseline is None:
            continue
        for key in ("seed", "candidate_tools", "benchmark", "prompt_token_count"):
            if row.get(key) != baseline.get(key) and not (row.get("error") or baseline.get("error")):
                raise ValueError(f"{sid}: unfair pairing: {key} differs")
        if row.get("error") or baseline.get("error"):
            skipped["runtime_failure"].append(sid)
            continue
        if "prompt_sha256" in row and "prompt_sha256" in baseline:
            if row["prompt_sha256"] != baseline["prompt_sha256"]:
                raise ValueError(f"{sid}: actual rendered prompts differ")
        else:
            missing_prompt_hash += 1
        if row["raw_output"] != baseline["raw_output"]:
            skipped["output_mismatch"].append(sid)
            continue
        if row["forward_count"] != baseline["forward_count"]:
            skipped["nfe_mismatch"].append(sid)
            continue
        if any(e["source"] != "natural" for e in baseline["first_tool_events"]):
            raise ValueError(f"{sid}: Natural-only run contains a non-natural trigger")
        matched += 1
        generation_omtp.append(row["generation_time_s"])
        generation_natural.append(baseline["generation_time_s"])
        generation_deltas.append(row["generation_time_s"] - baseline["generation_time_s"])
        final = set(extract_partial_names(row["raw_output"], row["candidate_tools"]))
        online = {e["tool"]: e for e in row["first_tool_events"]}
        detected = baseline["natural_first_times"]
        paired_in_query = []
        for tool in final:
            final_tool_count += 1
            n, o = detected.get(tool), online.get(tool)
            key = ("both_detected" if n is not None and o is not None else
                   "omtp_only" if o is not None else "natural_only" if n is not None else "both_missing")
            coverage[key] += 1
            if key != "both_detected":
                continue
            lead = n["time_s"] - o["time_s"]
            natural_times.append(n["time_s"])
            online_times.append(o["time_s"])
            leads.append(lead)
            forward_leads.append(n["forward"] - o["forward"])
            paired_in_query.append(lead)
        if paired_in_query:
            request_mean_leads.append(mean(paired_in_query))
        if not final:
            request_coverage["empty_final_set"] += 1
        else:
            all_o, all_n = final <= online.keys(), final <= detected.keys()
            key = ("both_complete" if all_o and all_n else "omtp_only" if all_o else
                   "natural_only" if all_n else "both_incomplete")
            request_coverage[key] += 1
            if all_o and all_n:
                all_set_leads.append(max(detected[t]["time_s"] for t in final) -
                                     max(online[t]["time_s"] for t in final))
    if missing_prompt_hash:
        warnings.append(f"{missing_prompt_hash} historical pairs lack rendered-prompt hashes; checked config/catalog/prompt code instead")
    ratio = mean(natural_times) / mean(online_times) if online_times and mean(online_times) > 0 else None
    return {
        "timing_definition": "Natural-only detection time minus OMTP-on first event time; both include actual observer overhead",
        "unit": "one query x final_raw_text tool identity, deduplicated; NOT only the first tool",
        "omtp_sample_count": len(omtp), "natural_sample_count": len(natural),
        "paired_identical_output_and_nfe_count": matched,
        "only_in_omtp": sorted(omtp.keys() - natural.keys()),
        "only_in_natural": sorted(natural.keys() - omtp.keys()),
        "excluded_sample_ids": skipped, "warnings": warnings,
        "final_tool_identity_count_in_matched_queries": final_tool_count,
        "tool_detection_coverage": dict(coverage),
        "request_complete_coverage": dict(request_coverage),
        "natural_only_detection_time_s": distribution(natural_times),
        "omtp_on_detection_time_s": distribution(online_times),
        "net_paired_lead_s": distribution(leads),
        "net_paired_lead_forwards": distribution(forward_leads),
        "per_query_mean_net_lead_s": distribution(request_mean_leads),
        "complete_set_net_paired_lead_s": distribution(all_set_leads),
        "tool_wins_ties_losses": [sum(x > 1e-9 for x in leads), sum(abs(x) <= 1e-9 for x in leads), sum(x < -1e-9 for x in leads)],
        "detection_time_ratio_natural_over_omtp": ratio,
        "ratio_is_not_end_to_end_tool_execution_speedup": True,
        "natural_only_generation_time_s": distribution(generation_natural),
        "omtp_on_generation_time_s": distribution(generation_omtp),
        "paired_generation_overhead_s": distribution(generation_deltas),
        "omtp_observer_time_s": distribution([r.get("observer_time_s") for r in omtp_rows]),
        "natural_only_observer_time_s": distribution([r.get("observer_time_s") for r in natural_rows]),
    }


def load_records(path):
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or records[0].get("schema_version") != "tool_first_v1":
        raise ValueError("expected a tool_first_v1 metadata header; historical reports are separate")
    return records[0], [r for r in records if r.get("record_type") == "sample"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="New-format samples.jsonl")
    parser.add_argument("--natural-baseline", help="Independent --observer-mode natural samples.jsonl; report net cross-run leads")
    args = parser.parse_args()
    meta, rows = load_records(args.input)
    report = summarize(rows)
    if args.natural_baseline:
        natural_meta, natural_rows = load_records(args.natural_baseline)
        report["independent_natural_baseline"] = {
            "natural_only_summary": summarize(natural_rows),
            "paired_comparison": compare_natural_baseline(meta, rows, natural_meta, natural_rows),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
