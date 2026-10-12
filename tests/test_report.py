import csv
import json

from report import build_summary


def test_summary_and_figure_directory_are_created(tmp_path):
    sample = {
        "sample_id": "x",
        "ground_truth_tool_types": ["a"],
        "final_json_valid": True,
        "tool_call_format_valid": True,
        "candidate_names_valid": True,
        "tool_name_precision": 1.0,
        "tool_name_recall": 1.0,
        "tool_name_f1": 1.0,
        "tool_set_exact_match": True,
        "tool_call_count_match": True,
        "bfcl_ast_compatible_match": True,
        "predict_trigger_set_exact": True,
        "generation_time_s": 3.0,
        "model_forward_time_s": 2.0,
        "observer_processing_time_s": 0.1,
        "natural_all_detect_time_s": 2.0,
        "predict_all_detect_time_s": 1.0,
        "predict_all_available_time_s": 1.1,
        "oracle_correct_detect_time_s": 1.0,
        "offline_stable_correct_trigger_time_s": 1.0,
        "prediction_lead_time_s": 1.0,
        "actual_prediction_lead_time_s": 0.9,
        "offline_stable_correct_lead_time_s": 1.0,
        "available_warmup_window_s": 2.0,
        "actual_available_warmup_window_s": 1.9,
        "warmup_simulation": [
            {
                "startup_duration_s": duration,
                "ready_time_savings_s": 1.0,
                "natural_ready_by_generation_end": False,
                "prediction_ready_by_generation_end": True,
                "wrong_preload_cost_s": 0.0,
            }
            for duration in (1.0, 1.5, 2.0)
        ],
        "steps": [
            {"predicted_tool_types": ["a"]},
        ],
    }
    (tmp_path / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    rows = build_summary(tmp_path)
    assert rows
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "figures").is_dir()
    with (tmp_path / "summary.csv").open() as handle:
        assert any(row["metric"] == "tool_name_f1" for row in csv.DictReader(handle))
