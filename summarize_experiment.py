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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="New-format samples.jsonl")
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or rows[0].get("schema_version") != "tool_first_v1":
        raise ValueError("expected a tool_first_v1 metadata header; historical reports are separate")
    print(json.dumps(summarize([r for r in rows if r.get("record_type") == "sample"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
