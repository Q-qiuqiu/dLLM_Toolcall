import sys
from types import SimpleNamespace

import pytest
import torch

from run_bfcl_gate_ablation import ObserverGroup, VARIANTS
from run_omtp_experiment import parse_args, _run_sample
from run_taskbench_experiment import _fixed_runtime_args
from test_experiment_output import PromptTokenizer, JsonModel


def test_bfcl_exposes_reliability_and_legacy_gate_aliases(monkeypatch):
    base = ['run_omtp_experiment.py', '--questions', 'q', '--answers', 'a', '--output-dir', 'o']
    monkeypatch.setattr(sys, 'argv', base + ['--omtp-strict-anchor', '--omtp-min-top1-score', '-5', '--omtp-min-margin', '.1'])
    args = parse_args()
    assert args.omtp_strict_anchor and args.min_top1_score == -5 and args.min_margin == .1
    monkeypatch.setattr(sys, 'argv', base + ['--min-top1-score', '-4', '--min-margin', '.2'])
    args = parse_args()
    assert args.min_top1_score == -4 and args.min_margin == .2
    monkeypatch.setattr(sys, 'argv', base + ['--min-margin', '.2', '--omtp-min-margin', '.1'])
    with pytest.raises(SystemExit):
        parse_args()


def test_four_policies_share_real_cpu_forwards_and_keep_parity():
    target = '[{"name":"weather","arguments":{}}]'
    tokenizer = PromptTokenizer(target)
    model = JsonModel(tokenizer, target)
    sample = SimpleNamespace(
        sample_id='toy', candidate_names=['weather', 'air'],
        messages=[{'role': 'user', 'content': 'weather'}],
        functions=[{'name': name, 'description': name, 'parameters': {'type': 'object', 'properties': {}}} for name in ('weather', 'air')],
        ground_truth_calls=[{'name': 'weather', 'arguments': {}}],
        ground_truth_tool_types={'weather'}, ground_truth=[{'weather': {}}],
    )
    args = _fixed_runtime_args(SimpleNamespace(
        generation_profile='fixed128', parity_samples=1, cache_mode='none',
        response_mode='native', output_schema='standard', prompt_profile='original', debug_output=False,
    ))
    args.gen_length = args.block_length = len(tokenizer.encode(target))
    args.steps = 4
    group = ObserverGroup(tokenizer=tokenizer, candidate_names=sample.candidate_names)
    record, _ = _run_sample(args, sample, 0, model, tokenizer, torch.device('cpu'), observer_factory=lambda _: group)
    assert record['raw_output'] == target
    assert model.calls == 12  # warmup, off, on: four variants do not multiply NFE
    assert record['parity']['output_identical'] and record['parity']['nfe_identical']
    assert set(group.observers) == set(VARIANTS)
    assert all(o.forward_count == 4 for o in group.observers.values())
    assert all('weather' in o.natural_first_events for o in group.observers.values())
