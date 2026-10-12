import torch
import json

from fastdllm_generate import ForwardObservation
from omtp_evaluation import evaluate_omtp
from omtp_observer import OMTPObserver, _json_object_depth
from omtp_report import build_omtp_report
from optimized_omtp_observer import OptimizedOMTPObserver, SelectiveOMTPObserver


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


def test_omtp_tracks_occurrences_and_stable_commits_without_mutation():
    target = '[{"name":"weather","arguments":{"name":"x"}},{"name":"weather","arguments":{}}]'
    tokenizer = CharacterTokenizer(target)
    target_ids = torch.tensor(tokenizer.encode(target))
    mask_id, vocab = 126, 127
    observer = OMTPObserver(tokenizer=tokenizer, candidate_names=["weather"], stable_n=2)
    for forward in (1, 2, 3):
        before = torch.full((1, len(target_ids)), mask_id, dtype=torch.long)
        after = before.clone()
        after[:, :forward] = target_ids[:forward]
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
    assert observer.per_tool_argmax_events[0]["forward_step"] == 2

    evaluated = evaluate_omtp(
        observer, final_tools=["weather"], ground_truth_tools=["weather"]
    )
    latent = evaluated["methods"]["omtp_latent_region_n2"]["vs_ground_truth"]
    assert latent["tool_set_exact_match"]
    assert latent["tool_precision"] == latent["tool_recall"] == 1.0


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
    observer = SelectiveOMTPObserver(
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
        "predicted_tools": ["weather"], "tool_precision": 1.0,
        "tool_recall": 1.0, "tool_f1": 1.0, "tool_set_exact_match": True,
        "false_positive_count": 0, "false_negative_count": 0,
        "false_positives": [], "false_negatives": [],
        "correct_trigger_coverage": 1.0, "online_first_trigger_time_s": 1.0,
        "oracle_first_correct_prediction_time_s": 1.0,
        "oracle_all_correct_coverage_time_s": 1.0,
        "oracle_exact_set_time_s": 1.0,
        "paired_exact_set_lead_vs_natural_s": 2.0,
        "first_times_by_tool": {"weather": 1.0}, "trigger_count": 1,
    }
    events = [{"source": "latent_region", "predicted_tool": "weather", "forward_step": 2, "predict_time_s": 1.0}]
    methods = {}
    from omtp_evaluation import METHODS
    for method in METHODS:
        methods[method] = {"events": events, "vs_final": result, "vs_ground_truth": result}
    sample = {
        "sample_id": "x", "ground_truth_tools": ["weather"],
        "method_comparison": methods, "forward_count": 2,
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
    assert (tmp_path / "figures" / "method_precision_recall_f1.png").exists()
