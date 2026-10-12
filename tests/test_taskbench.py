import json

from json_repair_evaluation import evaluate_json_repair
from prompting import build_messages
from run_taskbench_experiment import FIXED, _fixed_runtime_args
from taskbench_data import iter_taskbench_samples, validate_taskbench_manifest
from taskbench_evaluation import FALLBACK, LATENT, NATURAL, evaluate_taskbench_methods
from taskbench_report import build_taskbench_report


def _files(tmp_path):
    catalog = tmp_path / "tool_desc.json"
    catalog.write_text(json.dumps({"nodes": [
        {"id": "tool_a", "desc": "A", "parameters": [
            {"name": "when", "type": "date", "desc": "date"}
        ]},
        {"id": "tool_b", "desc": "B", "parameters": []},
        {"id": "tool_decoy", "desc": "decoy", "parameters": []},
    ]}), encoding="utf-8")
    manifest = tmp_path / "split.jsonl"
    manifest.write_text(json.dumps({
        "sample_id": "taskbench_x", "source_id": "x", "source_seed": 7,
        "graph_type": "chain", "instruction": "Do A and then B.",
        "ground_truth_calls": [
            {"name": "tool_a", "arguments": {"when": "GT_SECRET_2042"}},
            {"name": "tool_b", "arguments": {}},
        ],
        "dependencies": [{"source": "tool_a", "target": "tool_b"}],
        "reference_steps": ["REFERENCE_SECRET_9988"],
    }) + "\n", encoding="utf-8")
    return catalog, manifest


def _event(tool, when, position, source):
    return {
        "predicted_tool": tool, "predict_time_s": when,
        "best_position": position, "track_id": position,
        "forward_step": int(when * 10), "source": source,
    }


def test_taskbench_adapter_uses_full_catalog_and_prompt_has_no_gt_leak(tmp_path):
    catalog, manifest = _files(tmp_path)
    sample = next(iter_taskbench_samples(manifest, catalog))
    assert sample.candidate_names == ["tool_a", "tool_b", "tool_decoy"]
    assert sample.functions[0]["parameters"]["properties"]["when"]["type"] == "string"
    rendered = json.dumps(build_messages(sample), ensure_ascii=False)
    assert "tool_decoy" in rendered
    assert "GT_SECRET_2042" not in rendered
    assert "REFERENCE_SECRET_9988" not in rendered
    assert '"source"' not in rendered and '"target"' not in rendered
    audit = validate_taskbench_manifest(manifest, catalog)
    assert audit["sample_count"] == 1
    assert audit["candidate_tool_count"] == 3


def test_three_methods_are_separate_and_duplicate_calls_are_not_collapsed():
    natural = [
        _event("tool_a", 3.0, 10, "natural_decoding"),
        _event("tool_a", 3.5, 20, "natural_decoding"),
        _event("tool_b", 4.0, 30, "natural_decoding"),
    ]
    latent = [
        _event("tool_a", 1.0, 10, "latent_region"),
        _event("tool_b", 1.5, 30, "latent_region"),
    ]
    fallback = latent + [_event("tool_a", 2.0, 20, "prefix")]
    result = evaluate_taskbench_methods(
        latent_events=latent, fallback_events=fallback, natural_events=natural,
        ground_truth_occurrences=["tool_a", "tool_a", "tool_b"],
        final_occurrences=["tool_a", "tool_a", "tool_b"],
    )
    assert set(result) == {NATURAL, LATENT, FALLBACK}
    assert result[LATENT]["vs_ground_truth"]["tool_set_exact_match"]
    assert not result[LATENT]["vs_ground_truth"]["all_tool_occurrences_prefetched"]
    assert result[FALLBACK]["vs_ground_truth"]["all_tool_occurrences_prefetched"]
    assert result[FALLBACK]["vs_ground_truth"]["paired_coverage_lead_vs_natural_s"] == 2.5


def test_taskbench_cli_parameters_are_fixed():
    class Args:
        manifest = "m"; tool_catalog = "t"; model_path = "p"
        fastdllm_root = "f"; output_dir = "o"; device = "cuda"
        dtype = "bfloat16"; parity_samples = 1
    args = _fixed_runtime_args(Args())
    assert args.gen_length == args.block_length == 128
    assert args.steps == 32 and args.stable_n == 2
    assert args.min_top1_score is None and args.min_margin is None
    assert FIXED["region_radius"] == 4 and FIXED["region_top_k"] == 2


def test_json_repair_recovers_missing_separator(tmp_path):
    catalog, manifest = _files(tmp_path)
    sample = next(iter_taskbench_samples(manifest, catalog))
    broken = (
        '[{"name":"tool_a","arguments":{"when":"tomorrow"}} '
        '{"name":"tool_b","arguments":{}}]'
    )
    result = evaluate_json_repair(broken, sample.candidate_names, sample)
    assert result["json_repair_offline_only"]
    assert result["json_repair_attempted"]
    assert result["repaired_json_valid"]
    assert result["repaired_output_valid"]
    assert result["repaired_tool_set_exact_match"]


def test_json_repair_does_not_hide_wrong_tool_schema(tmp_path):
    catalog, manifest = _files(tmp_path)
    sample = next(iter_taskbench_samples(manifest, catalog))
    broken = '[{"name":"tool_a","arguments":"when":"tomorrow"}]'
    result = evaluate_json_repair(broken, sample.candidate_names, sample)
    assert result["repaired_json_valid"]
    assert not result["repaired_tool_call_format_valid"]
    assert not result["repaired_output_valid"]


def test_taskbench_report_writes_tables_and_figures(tmp_path):
    natural = [
        _event("tool_a", 3.0, 10, "natural_decoding"),
        _event("tool_b", 4.0, 20, "natural_decoding"),
    ]
    latent = [
        _event("tool_a", 1.0, 10, "latent_region"),
        _event("tool_b", 1.5, 20, "latent_region"),
    ]
    methods = evaluate_taskbench_methods(
        latent_events=latent, fallback_events=latent, natural_events=natural,
        ground_truth_occurrences=["tool_a", "tool_b"],
        final_occurrences=["tool_a", "tool_b"],
    )
    sample = {
        "sample_id": "x", "graph_type": "chain",
        "ground_truth_call_count": 2, "ground_truth_tool_type_count": 2,
        "taskbench_methods": methods, "final_json_valid": True,
        "tool_set_exact_match": True, "final_raw_tool_set_exact_match": True,
        "tool_call_count_match": True, "likely_length_truncated": False,
        "forward_count": 32, "observer_output_identical": True,
        "expected_forward_count": 32,
        "observer_nfe_identical": True, "generation_time_s": 4.2,
        "observer_processing_time_s": .4,
        "observer_mean_processing_time_s": .0125,
        "observer_off_generation_time_s": 3.8,
        "observer_wall_overhead_ratio": .105, "prompt_token_count": 2800,
    }
    (tmp_path / "taskbench_samples.jsonl").write_text(
        json.dumps(sample) + "\n", encoding="utf-8"
    )
    build_taskbench_report(tmp_path)
    for name in (
        "taskbench_method_comparison.csv", "taskbench_group_summary.csv",
        "taskbench_position_summary.csv", "taskbench_generation_summary.csv",
        "figures/method_accuracy.png", "figures/paired_coverage_lead.png",
        "figures/position_1_4_timing.png", "figures/chain_dag_f1.png",
        "figures/observer_and_generation_quality.png",
    ):
        assert (tmp_path / name).exists()
