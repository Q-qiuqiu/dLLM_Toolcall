from types import SimpleNamespace

import torch

from fastdllm_generate import ForwardObservation
from observer import ToolCallObserver
from simulation import simulate_warmup


class CharacterTokenizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.alphabet[token_id] for token_id in token_ids if token_id in self.alphabet)


def test_argmax_prediction_and_stable_trigger():
    target = '[{"name":"weather","arguments":{}}]'
    alphabet = {index + 1: char for index, char in enumerate(sorted(set(target)))}
    reverse = {char: index for index, char in alphabet.items()}
    target_ids = torch.tensor([reverse[char] for char in target])
    mask_id = 63
    vocab = 64
    tokenizer = CharacterTokenizer(alphabet)
    observer = ToolCallObserver(
        tokenizer=tokenizer, candidate_names=["weather"], stable_n=2
    )
    for forward in (1, 2):
        before = torch.full((1, len(target_ids)), mask_id, dtype=torch.long)
        after = before.clone()
        after[:, :forward] = target_ids[:forward]
        logits = torch.full((1, len(target_ids), vocab), -5.0)
        logits[0, torch.arange(len(target_ids)), target_ids] = 5.0
        observer(
            ForwardObservation(
                forward_index=forward,
                block_index=0,
                block_step=forward - 1,
                elapsed_s=float(forward),
                commit_elapsed_s=float(forward) + 0.1,
                forward_time_s=0.5,
                prompt_length=0,
                block_start=0,
                block_end=len(target_ids),
                mask_id=mask_id,
                x=before,
                committed_x=after,
                logits=logits,
            )
        )
    assert observer.steps[0]["predicted_tool_types"] == ["weather"]
    assert not observer.steps[0]["stable_trigger"]
    assert observer.steps[1]["stable_trigger"]
    assert observer.stable_events[0].tool_types == ["weather"]


def test_warmup_simulation_counts_wrong_and_falls_back_to_natural():
    rows = simulate_warmup(
        expected_types={"a", "b"},
        natural_first_times={"a": 4.0, "b": 5.0},
        stable_events=[{"time_s": 1.0, "tool_types": ["a", "wrong"]}],
        generation_time_s=6.0,
    )
    one_second = rows[0]
    assert one_second["natural_all_tools_ready_time_s"] == 6.0
    assert one_second["prediction_all_tools_ready_time_s"] == 6.0
    assert one_second["wrong_preload_types"] == ["wrong"]
    assert one_second["wrong_preload_cost_s"] == 1.0
