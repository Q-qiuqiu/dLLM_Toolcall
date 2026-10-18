import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import natural_observer as natural_module
import run_omtp_experiment as runner
from fastdllm_generate import ForwardObservation
from natural_observer import NaturalOnlyObserver, validate_observer_mode
from observer import ToolCallObserver
from run_taskbench_experiment import _fixed_runtime_args
from summarize_experiment import compare_natural_baseline
from test_experiment_output import JsonModel, PromptTokenizer
from test_omtp import CharacterTokenizer


def observation(tokenizer, text, *, prompt='', mask_positions=(), forward=1):
    ids = tokenizer.encode(prompt + text)
    after = torch.tensor([ids])
    for index in mask_positions:
        after[0, len(prompt) + index] = 126
    logits = torch.full((1, len(ids), 127), -8.)
    logits[0, torch.arange(len(ids)), torch.tensor(ids)] = 8.
    return ForwardObservation(
        forward_index=forward, block_index=0, block_step=forward - 1,
        elapsed_s=forward - .1, commit_elapsed_s=float(forward),
        forward_time_s=.1, prompt_length=len(prompt), block_start=len(prompt),
        block_end=len(ids), mask_id=126, x=after.clone(), committed_x=after,
        logits=logits,
    )


def test_shared_natural_detector_masks_prompt_duplicates_and_no_logits(monkeypatch):
    text = '[{"name":"weather","arguments":{}},{"name":"weather"},{"name":"air"}]'
    prompt = '[{"name":"prompt_only"}]'
    tokenizer = CharacterTokenizer(text + prompt)
    candidates = ['weather', 'air', 'prompt_only']
    full = ToolCallObserver(tokenizer=tokenizer, candidate_names=candidates)
    natural = NaturalOnlyObserver(tokenizer=tokenizer, candidate_names=candidates, record_history=True)
    gap = text.index('weather') + 2
    for step, masked in enumerate(([gap], []), 1):
        obs = observation(tokenizer, text, prompt=prompt, mask_positions=masked, forward=step)
        snapshot = obs.committed_x.clone()
        full(obs)
        # Natural must not access x/logits or perform another top-k operation.
        with monkeypatch.context() as patch:
            patch.setattr(torch, 'topk', lambda *a, **kw: pytest.fail('Natural used topk'))
            natural(replace(obs, x=None, logits=None))
        assert torch.equal(obs.committed_x, snapshot)
        assert natural.steps[-1]['committed_tool_types'] == full.steps[-1]['committed_tool_types']
        assert natural.steps[-1]['committed_call_names'] == full.steps[-1]['committed_call_names']
    assert natural.steps[-1]['committed_call_names'] == ['weather', 'weather', 'air']
    assert set(natural.first_tool_events) == {'weather', 'air'}
    assert len(natural.first_tool_events) == 2
    assert 'prompt_only' not in natural.natural_first_events
    assert natural.forward_count == 2


def test_mask_gap_never_concatenates_a_fake_natural_name():
    text = '[{"name":"weather"}]'
    tokenizer = CharacterTokenizer(text)
    natural = NaturalOnlyObserver(tokenizer=tokenizer, candidate_names=['weather'])
    natural(observation(tokenizer, text, mask_positions=[text.index('weather') + 2]))
    assert natural.first_tool_events == {}
    natural(observation(tokenizer, text, forward=2))
    assert natural.first_tool_events['weather']['forward_step'] == 2


def test_trigger_includes_copy_decode_detection_time(monkeypatch):
    text = '[{"name":"weather"}]'
    tokenizer = CharacterTokenizer(text)
    natural = NaturalOnlyObserver(tokenizer=tokenizer, candidate_names=['weather'])
    ticks = iter([100., 100.25, 100.3])
    monkeypatch.setattr(natural_module.time, 'perf_counter', lambda: next(ticks))
    natural(observation(tokenizer, text, forward=3))
    assert natural.first_tool_events['weather']['predict_time_s'] == 3.25
    assert natural.total_processing_s == pytest.approx(.3)


@pytest.mark.parametrize('debug', [False, True])
def test_real_cpu_32_nfe_parity_and_no_omtp_construction(monkeypatch, debug):
    target = '[{"name":"weather","arguments":{}}]'
    tokenizer = PromptTokenizer(target)
    model = JsonModel(tokenizer, target)
    model.ids += [125] * (128 - len(model.ids))
    sample = SimpleNamespace(
        sample_id='toy', candidate_names=['weather'],
        messages=[{'role': 'user', 'content': 'weather'}],
        functions=[{'name': 'weather', 'description': 'weather', 'parameters': {'type': 'object', 'properties': {}}}],
        ground_truth_calls=[{'name': 'weather', 'arguments': {}}],
        ground_truth_tool_types={'weather'}, ground_truth=[{'weather': {}}],
    )
    args = _fixed_runtime_args(argparse.Namespace(
        generation_profile='fixed128', parity_samples=1, cache_mode='none',
        response_mode='native', output_schema='standard', prompt_profile='original',
        debug_output=False, observer_mode='omtp',
    ))
    omtp_record, _ = runner._run_sample(args, sample, 0, model, tokenizer, torch.device('cpu'))
    before_calls = model.calls
    args.observer_mode, args.debug_output = 'natural', debug
    def forbid_omtp(**kwargs):
        pytest.fail('Natural-only instantiated an OMTP observer')
    monkeypatch.setattr(runner, 'OptimizedOMTPObserver', forbid_omtp)
    monkeypatch.setattr(runner, 'ReliableOMTPObserver', forbid_omtp)
    record, observer = runner._run_sample(args, sample, 0, model, tokenizer, torch.device('cpu'))
    assert isinstance(observer, NaturalOnlyObserver)
    assert record['observer_mode'] == 'natural'
    assert record['raw_output'] == omtp_record['raw_output'] == target
    assert record['forward_count'] == 32
    assert model.calls - before_calls == 96  # 32 warmup + 32 off + 32 natural
    assert record['parity']['output_identical'] and record['parity']['nfe_identical']
    assert record['prompt_sha256'] == omtp_record['prompt_sha256']
    assert record['natural_first_times']['weather']['forward'] == omtp_record['natural_first_times']['weather']['forward']
    assert all(e['source'] == 'natural' for e in record['first_tool_events'])
    assert record['first_tool_events'][0]['time_s'] == record['natural_first_times']['weather']['time_s']
    assert 0 <= record['observer_time_s'] < record['generation_time_s']
    assert len(observer.steps) == (32 if debug else 0)


@pytest.mark.parametrize('flag,value', [
    ('omtp_strict_anchor', True), ('omtp_min_margin', 0.),
    ('min_top1_score', -5.), ('profile_observer', True),
])
def test_natural_rejects_unused_omtp_options(flag, value):
    with pytest.raises(ValueError, match='remove OMTP'):
        validate_observer_mode(argparse.Namespace(observer_mode='natural', **{flag: value}))


def paired_fixture():
    metadata = {'args': {'observer_mode': 'omtp', 'steps': 32, 'gen_length': 128,
                         'block_length': 128, 'temperature': 0., 'seed': 1,
                         'dtype': 'bfloat16'}}
    baseline_meta = {'args': {**metadata['args'], 'observer_mode': 'natural'}}
    row = {
        'sample_id': 'x', 'seed': 1, 'candidate_tools': ['weather', 'air', 'wrong'],
        'raw_output': '[{"name":"weather"},{"name":"weather"},{"name":"air"}]',
        'ground_truth_tools': ['weather', 'air'], 'error': None,
        'first_tool_events': [
            {'tool': 'weather', 'source': 'latent_region', 'time_s': 2., 'forward': 2},
            {'tool': 'air', 'source': 'natural', 'time_s': 5., 'forward': 4},
            {'tool': 'wrong', 'source': 'latent_region', 'time_s': 1., 'forward': 1},
        ],
        'natural_first_times': {'weather': {'time_s': 4., 'forward': 3},
                                'air': {'time_s': 5., 'forward': 4}},
        'generation_time_s': 6., 'observer_time_s': 1., 'forward_count': 32,
    }
    baseline = {
        **row, 'first_tool_events': [
            {'tool': 'weather', 'source': 'natural', 'time_s': 3., 'forward': 3},
            {'tool': 'air', 'source': 'natural', 'time_s': 4., 'forward': 4},
        ],
        'natural_first_times': {'weather': {'time_s': 3., 'forward': 3},
                                'air': {'time_s': 4., 'forward': 4}},
        'generation_time_s': 5., 'observer_time_s': .01,
    }
    return metadata, [row], baseline_meta, [baseline]


def test_cross_run_net_lead_all_tool_types_negative_leads_and_not_total_subtraction():
    report = compare_natural_baseline(*paired_fixture())
    assert report['net_paired_lead_s']['count'] == 2  # weather twice still counts once
    assert report['net_paired_lead_s']['mean'] == 0.  # +1 for weather, -1 for air
    assert report['tool_wins_ties_losses'] == [1, 0, 1]
    assert report['complete_set_net_paired_lead_s']['mean'] == -1.
    assert report['paired_generation_overhead_s']['mean'] == 1.
    assert report['tool_detection_coverage']['both_detected'] == 2


def test_cross_run_preserves_missing_and_rejects_unfair_config():
    omtp_meta, omtp_rows, natural_meta, natural_rows = paired_fixture()
    natural_rows[0]['natural_first_times'].pop('air')
    report = compare_natural_baseline(omtp_meta, omtp_rows, natural_meta, natural_rows)
    assert report['tool_detection_coverage']['omtp_only'] == 1
    assert report['net_paired_lead_s']['count'] == 1
    natural_meta['args']['steps'] = 64
    with pytest.raises(ValueError, match='configuration differs: steps'):
        compare_natural_baseline(omtp_meta, omtp_rows, natural_meta, natural_rows)


@pytest.mark.parametrize('mismatch', ['raw_output', 'forward_count', 'error'])
def test_cross_run_reports_output_nfe_and_runtime_failures(mismatch):
    omtp_meta, omtp_rows, natural_meta, natural_rows = paired_fixture()
    natural_rows[0][mismatch] = {'raw_output': '[]', 'forward_count': 31, 'error': 'failed'}[mismatch]
    report = compare_natural_baseline(omtp_meta, omtp_rows, natural_meta, natural_rows)
    assert report['paired_identical_output_and_nfe_count'] == 0
    assert report['net_paired_lead_s']['count'] == 0
    assert sum(len(ids) for ids in report['excluded_sample_ids'].values()) == 1


def test_cross_run_actual_prompt_mismatch_rejected():
    omtp_meta, omtp_rows, natural_meta, natural_rows = paired_fixture()
    omtp_rows[0]['prompt_sha256'] = 'old'
    natural_rows[0]['prompt_sha256'] = 'different'
    with pytest.raises(ValueError, match='actual rendered prompts differ'):
        compare_natural_baseline(omtp_meta, omtp_rows, natural_meta, natural_rows)
