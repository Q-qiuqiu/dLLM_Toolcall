import torch
import json

from fastdllm_generate import ForwardObservation
from omtp_evaluation import (
    METHOD_NAME, evaluate_events, evaluate_occurrences, evaluate_omtp_fallback,
)
from omtp_observer import OMTPObserver, _json_object_depth
from omtp_report import build_omtp_report
from optimized_omtp_observer import OptimizedOMTPObserver


class CharacterTokenizer:
    def __init__(self, text):
        chars = sorted(set(text + ' \t\n'))
        self.alphabet = {index + 1: char for index, char in enumerate(chars)}
        self.reverse = {char: token for token, char in self.alphabet.items()}
        self.eos_token_id = None
        self.pad_token_id = None
        self.all_special_ids = []

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.reverse[char] for char in text]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(self.alphabet.get(int(token), "") for token in ids)


def test_json_depth_rejects_nested_argument_name():
    assert _json_object_depth('[{"name":"x",') == 1
    assert _json_object_depth('[{"name":"x","arguments":{"name":') == 2


def test_occurrence_metrics_do_not_collapse_duplicates_or_positions():
    events = [
        {"predicted_tool": "b", "best_position": 10, "track_id": 0,
         "predict_time_s": 1.0, "forward_step": 2, "source": "latent_region"},
        {"predicted_tool": "a", "best_position": 20, "track_id": 1,
         "predict_time_s": 1.2, "forward_step": 3, "source": "latent_region"},
    ]
    swapped = evaluate_occurrences(events, ["a", "b"])
    assert swapped["tool_occurrence_multiset_exact_match"]
    assert swapped["position_accuracy"] == 0.0
    assert not swapped["position_exact_match"]

    missing_duplicate = evaluate_occurrences(events, ["a", "a", "b"])
    assert not missing_duplicate["all_tool_occurrences_prefetched"]
    assert missing_duplicate["occurrence_recall"] == 2 / 3


def test_broad_tool_type_coverage_ignores_counts_and_allows_extras():
    events = [
        {"predicted_tool": tool, "predict_time_s": index + 1.0}
        for index, tool in enumerate(("weather", "air", "unneeded"))
    ]
    result = evaluate_events(events, {"weather", "air"})
    assert result["all_tool_types_prefetched"]
    assert not result["tool_set_exact_match"]


def test_omtp_tracks_occurrences_and_stable_commits_without_mutation():
    target = '[{"name":"weather","arguments":{"name":"x"}},{"name":"weather","arguments":{}}]'
    tokenizer = CharacterTokenizer(target)
    target_ids = torch.tensor(tokenizer.encode(target))
    mask_id, vocab = 126, 127
    observer = OMTPObserver(tokenizer=tokenizer, candidate_names=["weather"], stable_n=2)
    for forward in (1, 2, 3):
        before = torch.full((1, len(target_ids)), mask_id, dtype=torch.long)
        after = before.clone()
        committed = len(target_ids) if forward == 3 else forward
        after[:, :committed] = target_ids[:committed]
        logits = torch.full((1, len(target_ids), vocab), -8.0)
        logits[0, torch.arange(len(target_ids)), target_ids] = 8.0
        snapshot = after.clone()
        observer(ForwardObservation(
            forward_index=forward, block_index=0, block_step=forward - 1,
            elapsed_s=float(forward), commit_elapsed_s=float(forward) + 0.1,
            forward_time_s=0.5, prompt_length=0, block_start=0,
            block_end=len(target_ids), mask_id=mask_id, x=before,
            committed_x=after, logits=logits,
        ))
        assert torch.equal(after, snapshot)
    assert len(observer.latent_events) == 2
    assert {event["predicted_tool"] for event in observer.latent_events} == {"weather"}
    assert all(event["forward_step"] == 3 for event in observer.latent_events)
    # The nested arguments.name field must not create a third occurrence track.
    assert len({event["track_id"] for event in observer.latent_events}) == 2
    assert observer.baseline.stable_events == []
    evaluated = evaluate_omtp_fallback(
        observer, final_tools=["weather"], ground_truth_tools=["weather"],
        final_occurrences=["weather", "weather"],
        ground_truth_occurrences=["weather", "weather"],
    )
    result = evaluated["vs_ground_truth"]
    assert evaluated["method"] == METHOD_NAME
    assert result["tool_set_exact_match"]
    assert result["tool_precision"] == result["tool_recall"] == 1.0
    assert result["all_tool_occurrences_prefetched"]
    assert result["tool_occurrence_multiset_exact_match"]
    assert result["position_accuracy"] == 1.0
    assert result["position_exact_match"]
    assert evaluated["events"] == observer.fallback_events()
    natural = evaluated["natural_baseline"]
    assert natural["method"] == "natural_decoding"
    assert natural["events"][0]["forward_step"] == 3
    assert natural["vs_ground_truth"]["tool_set_exact_match"]


def test_optimized_omtp_preserves_original_events():
    target = '[{"name":"weather","arguments":{}},{"name":"weather","arguments":{}}]'
    tokenizer = CharacterTokenizer(target)
    target_ids = torch.tensor(tokenizer.encode(target))
    observers = [
        OMTPObserver(tokenizer=tokenizer, candidate_names=["weather"], stable_n=2),
        OptimizedOMTPObserver(tokenizer=tokenizer, candidate_names=["weather"], stable_n=2),
    ]
    for observer in observers:
        for forward in (1, 2, 3):
            before = torch.full((1, len(target_ids)), 126, dtype=torch.long)
            after = before.clone(); after[:, :forward] = target_ids[:forward]
            logits = torch.full((1, len(target_ids), 127), -8.0)
            logits[0, torch.arange(len(target_ids)), target_ids] = 8.0
            observer(ForwardObservation(
                forward_index=forward, block_index=0, block_step=forward - 1,
                elapsed_s=float(forward), commit_elapsed_s=float(forward) + .1,
                forward_time_s=.5, prompt_length=0, block_start=0,
                block_end=len(target_ids), mask_id=126, x=before,
                committed_x=after, logits=logits,
            ))
    signature = lambda observer: [
        (event["track_id"], event["predicted_tool"], event["forward_step"])
        for event in observer.latent_events
    ]
    assert signature(observers[0]) == signature(observers[1])


def test_confidence_gate_can_disable_latent_trigger():
    from omtp_observer import OMTPConfig
    target = '[{"name":"weather","arguments":{}}]'
    tokenizer = CharacterTokenizer(target)
    target_ids = torch.tensor(tokenizer.encode(target))
    observer = OptimizedOMTPObserver(
        tokenizer=tokenizer, candidate_names=["weather"], stable_n=2,
        config=OMTPConfig(stable_n=2, min_top1_score=0.1),
    )
    for forward in (1, 2, 3):
        before = torch.full((1, len(target_ids)), 126, dtype=torch.long)
        logits = torch.full((1, len(target_ids), 127), -8.0)
        logits[0, torch.arange(len(target_ids)), target_ids] = 8.0
        observer(ForwardObservation(
            forward_index=forward, block_index=0, block_step=forward - 1,
            elapsed_s=float(forward), commit_elapsed_s=float(forward) + .1,
            forward_time_s=.5, prompt_length=0, block_start=0,
            block_end=len(target_ids), mask_id=126, x=before,
            committed_x=before.clone(), logits=logits,
        ))
    assert observer.latent_events == []


def test_omtp_report_writes_all_artifacts(tmp_path):
    result = {
        "predicted_tools": ["weather"], "all_tool_types_prefetched": True,
        "tool_precision": 1.0,
        "tool_recall": 1.0, "tool_f1": 1.0, "tool_set_exact_match": True,
        "false_positive_count": 0, "false_negative_count": 0,
        "false_positives": [], "false_negatives": [],
        "correct_trigger_coverage": 1.0, "online_first_trigger_time_s": 1.0,
        "oracle_first_correct_prediction_time_s": 1.0,
        "oracle_all_correct_coverage_time_s": 1.0,
        "oracle_exact_set_time_s": 1.0,
        "natural_exact_set_time_s": 3.0,
        "paired_exact_set_lead_vs_natural_s": 2.0,
        "first_times_by_tool": {"weather": 1.0}, "trigger_count": 1,
        "occurrence_precision": 1.0, "occurrence_recall": 1.0,
        "occurrence_f1": 1.0, "all_tool_occurrences_prefetched": True,
        "tool_occurrence_count_match": True,
        "tool_occurrence_multiset_exact_match": True,
        "position_accuracy": 1.0, "position_exact_match": True,
        "oracle_all_occurrences_covered_time_s": 1.0,
        "oracle_exact_occurrence_multiset_time_s": 1.0,
        "natural_all_occurrences_covered_time_s": 3.0,
        "paired_all_occurrences_lead_vs_natural_s": 2.0,
        "position_records": [],
    }
    events = [{"source": "latent_region", "predicted_tool": "weather", "forward_step": 2, "predict_time_s": 1.0}]
    sample = {
        "sample_id": "x", "ground_truth_tools": ["weather"],
        "omtp_evaluation": {"vs_final": result, "vs_ground_truth": result},
        "natural_baseline": {
            "method": "natural_decoding", "events": events,
            "vs_final": result, "vs_ground_truth": result,
        },
        "position_comparison": {
            key: [{
                "position": 1, "expected_tool": "weather",
                "omtp_predicted_tool": "weather", "omtp_correct": True,
                "omtp_predict_time_s": 1.0, "omtp_forward_step": 2,
                "omtp_source": "latent_region", "omtp_track_id": 0,
                "omtp_sequence_position": 4,
                "natural_predicted_tool": "weather", "natural_correct": True,
                "natural_detect_time_s": 3.0, "natural_forward_step": 3,
                "lead_vs_natural_s": 2.0,
            }]
            for key in ("vs_final", "vs_ground_truth")
        },
        "omtp_fallback_events": events,
        "per_tool_predictions": [{
            "tool": "weather", "T_latent": 1.0, "T_prefix": None,
            "T_natural": 3.0, "T_online": 1.0,
            "latent_lead_vs_natural_s": 2.0,
            "online_lead_vs_natural_s": 2.0,
        }],
        "forward_count": 2,
        "observer_processing_time_s": .1, "observer_mean_processing_time_s": .05,
        "omtp_processing_time_s": .04, "generation_time_s": 3.0,
        "observer_off_generation_time_s": 2.8,
        "observer_wall_overhead_ratio": 3 / 2.8 - 1,
        "observer_output_identical": True, "observer_nfe_identical": True,
    }
    (tmp_path / "omtp_samples.jsonl").write_text(json.dumps(sample) + "\n")
    build_omtp_report(tmp_path)
    assert (tmp_path / "omtp_summary.csv").exists()
    assert (tmp_path / "omtp_comparison.csv").exists()
    assert (tmp_path / "omtp_position_summary.csv").exists()
    assert (tmp_path / "omtp_diagnostics.csv").exists()
    assert (tmp_path / "figures" / "omtp_fallback_quality.png").exists()
