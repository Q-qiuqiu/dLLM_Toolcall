import torch

from fastdllm_generate import ForwardObservation
from reliable_omtp_observer import ReliabilityOptions, ReliableOMTPObserver
from test_omtp import CharacterTokenizer


def _observe(observer, tokenizer, target, forwards=(1, 2, 3)):
    ids = torch.tensor(tokenizer.encode(target))
    for forward in forwards:
        before = torch.full((1, len(ids)), 126, dtype=torch.long)
        after = ids.unsqueeze(0).clone()
        logits = torch.full((1, len(ids), 127), -8.0)
        logits[0, torch.arange(len(ids)), ids] = 8.0
        observer(ForwardObservation(
            forward_index=forward, block_index=0, block_step=forward - 1,
            elapsed_s=float(forward), commit_elapsed_s=forward + .1,
            forward_time_s=.5, prompt_length=0, block_start=0,
            block_end=len(ids), mask_id=126, x=before,
            committed_x=after, logits=logits,
        ))
        assert torch.equal(after[0], ids)


def test_boundary_rejects_short_candidate_inside_long_name():
    target = '[{"name":"Image Search (by Image)","input_1":"x"}]'
    tokenizer = CharacterTokenizer(target)
    observer = ReliableOMTPObserver(
        tokenizer=tokenizer,
        candidate_names=["Image Search", "Image Search (by Image)"],
        reliability=ReliabilityOptions(name_boundary=True),
    )
    _observe(observer, tokenizer, target)
    assert observer.latent_events
    assert {event["predicted_tool"] for event in observer.latent_events} == {
        "Image Search (by Image)"
    }


def test_consecutive_stability_resets_across_missing_anchor():
    tokenizer = CharacterTokenizer('[{"name":"weather"}]')
    observer = ReliableOMTPObserver(
        tokenizer=tokenizer, candidate_names=["weather"],
        reliability=ReliabilityOptions(consecutive_stability=True),
    )
    anchor = {"value_start": 10}
    track = observer._associate([anchor], 1)[0][0]
    track["stable_count"] = 1
    track["last_prediction"] = "weather"
    observer._associate([], 2)
    recovered = observer._associate([anchor], 3)[0][0]
    assert recovered["track_id"] == track["track_id"]
    assert recovered["stable_count"] == 0
    assert recovered["last_prediction"] is None


def test_strict_anchor_rejects_name_inside_argument_text():
    target = '[{"name":"weather","input":"name":"air"}]'
    tokenizer = CharacterTokenizer(target)
    observer = ReliableOMTPObserver(
        tokenizer=tokenizer, candidate_names=["weather", "air"],
        reliability=ReliabilityOptions(strict_anchor=True),
    )
    _observe(observer, tokenizer, target)
    assert len(observer.latent_events) == 1
    assert observer.latent_events[0]["predicted_tool"] == "weather"


def test_natural_append_retains_wrong_commit_and_earlier_causal_detection():
    tokenizer = CharacterTokenizer('[{"name":"wrong right other"}]')
    observer = ReliableOMTPObserver(
        tokenizer=tokenizer, candidate_names=["wrong", "right", "other"],
        reliability=ReliabilityOptions(append_natural=True),
    )
    wrong = {"track_id": 0, "source": "latent_region", "predicted_tool": "wrong", "predict_time_s": 1.0}
    natural = {"track_id": 0, "source": "natural", "predicted_tool": "right", "predict_time_s": 2.0}
    later = {"track_id": 1, "source": "latent_region", "predicted_tool": "right", "predict_time_s": 3.0}
    observer.fallback_commits = {0: wrong, 1: later}
    observer.events = [wrong, natural, later]
    result = observer.fallback_events()
    assert wrong in result and natural in result
    assert next(event for event in result if event["predicted_tool"] == "right")["predict_time_s"] == 2.0
