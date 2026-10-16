from types import SimpleNamespace

import torch

from fastdllm_generate import generate, generate_with_dual_cache


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


class DualFakeModel:
    """Position-independent logits make cache/no-cache parity unambiguous."""

    def __init__(self):
        self.device = torch.device("cpu")

    def __call__(
        self, x, past_key_values=None, use_cache=False,
        replace_position=None,
    ):
        del replace_position
        batch, length = x.shape
        logits = torch.full((batch, length, 32), -10.0)
        logits[:, :, 7] = 10.0
        cache_length = length
        if past_key_values is not None:
            cache_length += int(past_key_values[0][0].shape[2])
        cache = ((torch.zeros(batch, 1, cache_length, 1),),)
        return SimpleNamespace(
            logits=logits,
            past_key_values=cache if use_cache else None,
        )


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


def test_generation_prefix_is_committed_inside_canvas_and_preserves_parity():
    model = FakeModel()
    prompt = torch.tensor([[1, 2]])
    prefix = [20, 21]
    baseline, baseline_nfe = generate(
        model, prompt, steps=2, gen_length=4, block_length=4,
        mask_id=31, generation_prefix_ids=prefix,
    )
    observations = []
    observed, observed_nfe = generate(
        model, prompt, steps=2, gen_length=4, block_length=4,
        mask_id=31, generation_prefix_ids=prefix,
        observation_callback=observations.append,
    )
    assert torch.equal(baseline, observed)
    assert baseline_nfe == observed_nfe == 2
    assert observed[0, 2:4].tolist() == prefix
    assert observations[0].x[0, 2:4].tolist() == prefix
    assert int((observations[0].x[:, 2:] == 31).sum()) == 2


def test_generation_prefix_rejects_mask_and_full_canvas():
    model = FakeModel()
    prompt = torch.tensor([[1]])
    for prefix in ([31], [10, 11, 12, 13]):
        try:
            generate(
                model, prompt, steps=2, gen_length=4, block_length=4,
                mask_id=31, generation_prefix_ids=prefix,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation prefix was accepted")


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


def test_dual_cache_matches_no_cache_without_extra_forwards():
    model = DualFakeModel()
    prompt = torch.tensor([[1, 2, 3]])
    baseline, baseline_nfe = generate(
        model, prompt, steps=4, gen_length=8, block_length=8, mask_id=31
    )
    observations = []
    cached, cached_nfe = generate_with_dual_cache(
        model,
        prompt,
        steps=4,
        gen_length=8,
        block_length=8,
        mask_id=31,
        observation_callback=observations.append,
    )
    assert torch.equal(cached, baseline)
    assert baseline_nfe == cached_nfe == len(observations) == 4
    assert all(row.logits.shape[1] == 8 for row in observations)
    assert all(row.x.shape[1] == 8 for row in observations)
    assert all(row.committed_x.shape[1] == 8 for row in observations)
    assert all(row.prompt_length == 0 for row in observations)
    assert all(row.sequence_offset == 3 for row in observations)
    assert all(row.cache_mode == "dual" for row in observations)
    assert [row.block_step for row in observations] == [0, 1, 2, 3]
