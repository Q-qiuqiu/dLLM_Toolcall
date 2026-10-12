from types import SimpleNamespace

import torch

from fastdllm_generate import generate


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def __call__(self, x):
        batch, length = x.shape
        vocab = 32
        logits = torch.full((batch, length, vocab), -10.0)
        for position in range(length):
            token = 3 + position
            logits[:, position, token] = 10.0 + position
        return SimpleNamespace(logits=logits)


def test_observer_off_and_on_produce_identical_tokens():
    model = FakeModel()
    prompt = torch.tensor([[1, 2]])
    baseline, baseline_nfe = generate(
        model, prompt, steps=2, gen_length=4, block_length=4, mask_id=31
    )
    observations = []
    observed, observed_nfe = generate(
        model,
        prompt,
        steps=2,
        gen_length=4,
        block_length=4,
        mask_id=31,
        observation_callback=observations.append,
    )
    assert torch.equal(baseline, observed)
    assert baseline_nfe == observed_nfe == 2
    assert len(observations) == 2
    assert observations[0].forward_index == 1
    assert observations[0].block_index == 0
    assert observations[0].block_step == 0
    assert (observations[0].x[:, 2:] == 31).sum().item() == 4
    assert (observations[0].committed_x[:, 2:] == 31).sum().item() == 2
    assert observations[0].elapsed_s <= observations[0].commit_elapsed_s


def test_block_metadata_and_forward_count():
    observations = []
    output, nfe = generate(
        FakeModel(),
        torch.tensor([[1]]),
        steps=4,
        gen_length=4,
        block_length=2,
        mask_id=31,
        observation_callback=observations.append,
    )
    assert nfe == 4
    assert [(item.block_index, item.block_step) for item in observations] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
