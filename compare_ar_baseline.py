"""Offline dLLM-versus-AR name detection comparison; print JSON, create no files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from statistics import mean

from summarize_experiment import distribution, load_records, summarize
from tool_parser import extract_partial_names


def _first_events(row, *, natural):
    if row.get("error"):
        return {}
    if natural:
        return {name: event["time_s"] for name, event in row["natural_first_times"].items()}
    events = row["first_tool_events"]
    if len({event["tool"] for event in events}) != len(events):
        raise ValueError(f"{row['sample_id']}: duplicate first-tool triggers")
    return {event["tool"]: event["time_s"] for event in events}


def _all_time(events, target):
    return max(events[tool] for tool in target) if target and target <= events.keys() else None


def _timing_pairs(pairs):
    """None is failure, never zero; negative dLLM advantages remain negative."""
    status = Counter({key: 0 for key in ("both_success", "dllm_only", "ar_only", "both_failed")})
    left, right, advantages = [], [], []
    for dllm, ar in pairs:
        key = ("both_success" if dllm is not None and ar is not None else
               "dllm_only" if dllm is not None else "ar_only" if ar is not None else "both_failed")
        status[key] += 1
        if key == "both_success":
            left.append(dllm)
            right.append(ar)
            advantages.append(ar - dllm)
    return {
        "eligible_count": len(pairs), "coverage": dict(status),
        "dllm_time_s_paired": distribution(left), "ar_time_s_paired": distribution(right),
        "dllm_lead_s_ar_minus_dllm": distribution(advantages),
        "dllm_wins_ties_losses": [sum(v > 1e-9 for v in advantages),
                                  sum(abs(v) <= 1e-9 for v in advantages),
                                  sum(v < -1e-9 for v in advantages)],
        "paired_detection_time_ratio_ar_over_dllm": mean(right) / mean(left) if left and mean(left) > 0 else None,
    }


def compare_ar(dllm_meta, dllm_rows, ar_meta, ar_rows):
    da, aa = dllm_meta["args"], ar_meta["args"]
    if aa.get("inference_backend") != "huggingface_ar":
        raise ValueError("--ar-input must be from run_ar_experiment.py")
    if da.get("inference_backend") == "huggingface_ar":
        raise ValueError("--dllm-input must be a dLLM run")
    for key in ("seed", "dtype", "temperature", "debug_output"):
        if da.get(key, False if key == "debug_output" else None) != aa.get(key, False if key == "debug_output" else None):
            raise ValueError(f"unfair AR pairing: {key} differs")
    for key, default in (("prompt_profile", "original"), ("output_schema", "standard")):
        if da.get(key, default) != aa.get(key, default):
            raise ValueError(f"unfair AR pairing: {key} differs")
    if da.get("response_mode", "native") != "native":
        raise ValueError("AR comparison requires the dLLM native response mode, without prefilled output")
    signatures = ("manifest_sha256", "tool_catalog_sha256") if "manifest_sha256" in dllm_meta else ("questions_sha256", "answers_sha256")
    for key in signatures:
        if not dllm_meta.get(key) or dllm_meta[key] != ar_meta.get(key):
            raise ValueError(f"unfair AR pairing: data signature differs/missing: {key}")
    for name in ("prompting.py", "tool_parser.py", "tool_format.py"):
        dl = dllm_meta.get("code_sha256", {}).get(name)
        ar = ar_meta.get("code_sha256", {}).get(name)
        if dl and ar and dl != ar:
            raise ValueError(f"unfair AR pairing: prompt/parser code differs: {name}")
    warnings = [
        "Different models/tokenizers/native chat templates; AR may use natural EOS while dLLM uses a fixed generation canvas.",
        "Negative lead means AR was faster. No tool execution or end-to-end prefetch speedup is measured.",
        "Do not compare AR NFE to denoising NFE as equal computational work.",
    ]
    if aa.get("max_new_tokens") is None:
        warnings.append("AR uses EOS completion without an experimental token budget; dLLM uses a fixed canvas. Compare the same target tool set and report output lengths/completion failures.")
    elif da.get("gen_length") != aa.get("max_new_tokens"):
        warnings.append("Generation token budgets differ; do not treat this as an equal-budget comparison.")
    for key in ("gpu", "torch", "cuda"):
        if dllm_meta.get(key) != ar_meta.get(key):
            warnings.append(f"hardware/runtime metadata differs: {key}")
    def index(rows):
        result = {r["sample_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate sample IDs")
        return result
    dllm, ar = index(dllm_rows), index(ar_rows)
    common = [sid for sid in dllm if sid in ar]
    set_relation = Counter()
    final_sets = {}
    for sid in common:
        left, right = dllm[sid], ar[sid]
        for key in ("seed", "candidate_tools", "benchmark", "ground_truth_tools"):
            if left.get(key) != right.get(key):
                raise ValueError(f"{sid}: unfair AR pairing: {key} differs")
        if left.get("messages_sha256") and right.get("messages_sha256") and left["messages_sha256"] != right["messages_sha256"]:
            raise ValueError(f"{sid}: semantic messages differ")
        dl_set = set(extract_partial_names(left["raw_output"], left["candidate_tools"]))
        ar_set = set(extract_partial_names(right["raw_output"], right["candidate_tools"]))
        final_sets[sid] = dl_set, ar_set
        set_relation["same_nonempty" if dl_set == ar_set and dl_set else
                     "both_empty" if not dl_set and not ar_set else "different_sets"] += 1
    if any(not dllm[sid].get("messages_sha256") for sid in common):
        warnings.append("Historical dLLM results lack semantic-message hashes; checked prompt profile/source and full catalog hashes.")
    comparisons = {}
    for mode in ("online", "natural"):
        common_tool_pairs, same_set_pairs, gt_pairs = [], [], []
        own_dllm, own_ar = [], []
        kth = {str(k): [] for k in range(1, 5)}
        for sid in common:
            dl_row, ar_row = dllm[sid], ar[sid]
            dl_events = _first_events(dl_row, natural=mode == "natural")
            ar_events = _first_events(ar_row, natural=True)
            dl_set, ar_set = final_sets[sid]
            # Identical targets within each pair, verified only offline.
            for tool in sorted(dl_set & ar_set):
                common_tool_pairs.append((dl_events.get(tool), ar_events.get(tool)))
            if dl_set and dl_set == ar_set:
                same_set_pairs.append((_all_time(dl_events, dl_set), _all_time(ar_events, ar_set)))
            gt = set(dl_row["ground_truth_tools"])
            if gt:
                gt_pairs.append((_all_time(dl_events, gt), _all_time(ar_events, gt)))
                dl_correct = sorted(dl_events[t] for t in gt if t in dl_events)
                ar_correct = sorted(ar_events[t] for t in gt if t in ar_events)
                for k in range(1, 5):
                    if len(gt) >= k:
                        kth[str(k)].append((dl_correct[k - 1] if len(dl_correct) >= k else None,
                                            ar_correct[k - 1] if len(ar_correct) >= k else None))
            own_dllm.append(_all_time(dl_events, dl_set))
            own_ar.append(_all_time(ar_events, ar_set))
        comparisons[f"dllm_{mode}_vs_ar_natural"] = {
            "common_final_tool_identities": _timing_pairs(common_tool_pairs),
            "identical_nonempty_final_tool_set_complete_time": _timing_pairs(same_set_pairs),
            "ground_truth_complete_coverage_time": _timing_pairs(gt_pairs),
            "ground_truth_kth_correct_detection_by_time": {k: _timing_pairs(v) for k, v in kth.items()},
            "own_final_tool_set_complete_time_descriptive_only": {
                "dllm": distribution(own_dllm), "ar": distribution(own_ar),
                "note": "Targets differ unless final sets are identical; not an equal-work comparison.",
            },
            "coverage_is_not_exact_match": "extra false triggers do not prevent coverage; precision/exact-match are reported separately",
        }
    def syntax_valid(row):
        try:
            json.loads(row["raw_output"])
            return True
        except (ValueError, TypeError):
            return False
    return {
        "dllm_summary": summarize(dllm_rows), "ar_summary": summarize(ar_rows),
        "paired_query_count": len(common), "final_tool_set_relation": dict(set_relation),
        "only_in_dllm": sorted(dllm.keys() - ar.keys()), "only_in_ar": sorted(ar.keys() - dllm.keys()),
        "runtime_failed_sample_ids": {"dllm": [r["sample_id"] for r in dllm_rows if r.get("error")],
                                      "ar": [r["sample_id"] for r in ar_rows if r.get("error")]},
        "raw_json_syntax_valid_count": {"dllm": sum(syntax_valid(r) for r in dllm_rows),
                                        "ar": sum(syntax_valid(r) for r in ar_rows)},
        "ar_ttft_s": distribution([r.get("first_token_time_s") for r in ar_rows]),
        "ar_prefill_forward_time_s": distribution([r.get("prefill_forward_time_s") for r in ar_rows]),
        "ar_stop_reason_counts": dict(Counter(r.get("stop_reason", "unrecorded") for r in ar_rows)),
        "ar_natural_completion_count": sum(r.get("stop_reason") == "eos" for r in ar_rows),
        "ar_context_limit_failure_count": sum(r.get("stop_reason") == "context_limit" for r in ar_rows),
        "ar_generated_token_count": distribution([r.get("generated_token_count") for r in ar_rows]),
        "comparisons": comparisons, "warnings": warnings,
        "analysis_is_offline": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dllm-input", required=True)
    parser.add_argument("--ar-input", required=True)
    args = parser.parse_args()
    dm, dr = load_records(args.dllm_input)
    am, ar = load_records(args.ar_input)
    print(json.dumps(compare_ar(dm, dr, am, ar), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
