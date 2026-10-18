import argparse
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ar_generate import generate_ar, greedy_config, resolve_eos_ids
from compare_ar_baseline import compare_ar
from natural_observer import NaturalOnlyObserver
from run_ar_experiment import parse_args, run_ar_sample
from test_experiment_output import PromptTokenizer


class ARTokenizer(PromptTokenizer):
    bos_token_id = 124

    def get_vocab(self):
        return {**self.reverse, '<|eot_id|>': self.eos_token_id}

    def apply_chat_template(self, messages, **kwargs):
        return '  '


class ScriptedLlama(LlamaForCausalLM):
    """Actual tiny Llama/KV/native generate, deterministic test-only logits."""
    def __init__(self, tokenizer, target):
        super().__init__(LlamaConfig(
            vocab_size=127, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=512, bos_token_id=124, eos_token_id=125,
            pad_token_id=125, attn_implementation='eager',
        ))
        self.target_ids = tokenizer.encode(target) + [125]
        self.input_widths = []
        self._offset = 0

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                cache_position=None, **kwargs):
        self.input_widths.append(input_ids.shape[1])
        if cache_position is None or int(cache_position[0]) == 0:
            self._offset = 0
        output = super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, cache_position=cache_position, **kwargs,
        )
        output.logits.fill_(-100.)
        output.logits[:, -1, self.target_ids[min(self._offset, len(self.target_ids) - 1)]] = 100.
        self._offset += 1
        return output


def fixture_model():
    target = '[{"name":"weather"},{"name":"weather"},{"name":"air"}]'
    tokenizer = ARTokenizer(target)
    model = ScriptedLlama(tokenizer, target).eval()
    prompt_ids = torch.tensor([tokenizer.encode('  ')])
    return target, tokenizer, model, prompt_ids


def test_native_hf_kv_eos_nfe_online_times_and_parity():
    target, tokenizer, model, prompt = fixture_model()
    observer = NaturalOnlyObserver(tokenizer=tokenizer, candidate_names=['weather', 'air'], record_history=True)
    kw = dict(model=model, tokenizer=tokenizer, prompt_ids=prompt,
              max_new_tokens=128, seed=20261009, device=torch.device('cpu'))
    on = generate_ar(**kw, observer=observer)
    off = generate_ar(**kw, observer=None)
    assert torch.equal(on.output, off.output)
    assert tokenizer.decode(on.output[0, prompt.shape[1]:].tolist()) == target
    assert on.forward_count == off.forward_count == len(model.target_ids)
    assert len(observer.steps) == len(model.target_ids)
    assert model.input_widths[:len(model.target_ids)] == [2] + [1] * (len(model.target_ids) - 1)
    assert all(e['source'] == 'natural' for e in observer.first_tool_events.values())
    assert set(observer.first_tool_events) == {'weather', 'air'}  # duplicate calls deduplicated
    assert observer.first_tool_events['weather']['forward_step'] == target.index('weather') + len('weather') + 1
    assert observer.first_tool_events['weather']['predict_time_s'] < observer.first_tool_events['air']['predict_time_s'] < on.generation_time_s
    assert 0 < on.prefill_forward_time_s <= on.first_token_time_s <= on.generation_time_s
    assert len(model._forward_hooks) == len(model._forward_pre_hooks) == 0


def test_budget_is_not_extended_or_stopped_by_detected_tool():
    _, tokenizer, model, prompt = fixture_model()
    observer = NaturalOnlyObserver(tokenizer=tokenizer, candidate_names=['weather'])
    result = generate_ar(model=model, tokenizer=tokenizer, prompt_ids=prompt,
                         max_new_tokens=25, seed=1, device=torch.device('cpu'), observer=observer)
    assert result.forward_count == 25
    assert 'weather' in observer.first_tool_events
    assert result.output[0, -1].item() != 125


def test_generation_config_discards_sampling_defaults_but_keeps_eos():
    _, tokenizer, model, _ = fixture_model()
    model.generation_config.do_sample = True
    model.generation_config.temperature = .6
    model.generation_config.top_p = .9
    model.generation_config.eos_token_id = [123, 125]
    assert resolve_eos_ids(model, tokenizer) == [123, 125]
    config = greedy_config(model, tokenizer, 128)
    assert config.do_sample is False and config.use_cache is True
    assert config.num_beams == 1 and config.max_new_tokens == 128
    assert config.temperature == 1. and config.top_p == 1.


def test_default_is_eos_not_hf_20_or_checkpoint_saved_length():
    _, tokenizer, model, prompt = fixture_model()
    model.generation_config.max_length = 10
    config = greedy_config(model, tokenizer)
    assert config.max_new_tokens is None
    assert config.max_length == model.config.max_position_embeddings
    result = generate_ar(model=model, tokenizer=tokenizer, prompt_ids=prompt,
                         seed=1, device=torch.device('cpu'), observer=None)
    assert result.forward_count == len(model.target_ids) > 20
    assert result.output[0, -1].item() == tokenizer.eos_token_id


def test_eos_generation_continues_beyond_128_with_identical_early_prefix():
    target = '[{"name":"weather","notes":"' + ' ' * 140 + '"},{"name":"air"}]'
    tokenizer = ARTokenizer(target)
    model = ScriptedLlama(tokenizer, target).eval()
    prompt = torch.tensor([tokenizer.encode('  ')])
    kwargs = dict(model=model, tokenizer=tokenizer, prompt_ids=prompt, seed=1, device=torch.device('cpu'))
    capped = generate_ar(**kwargs, max_new_tokens=128, observer=None)
    observer = NaturalOnlyObserver(tokenizer=tokenizer, candidate_names=['weather', 'air'])
    natural = generate_ar(**kwargs, observer=observer)
    assert natural.forward_count == len(model.target_ids) > 128
    assert torch.equal(capped.output, natural.output[:, :capped.output.shape[1]])
    assert natural.output[0, -1].item() == tokenizer.eos_token_id
    assert observer.natural_first_events['air']['forward_step'] > 128


def test_cli_no_output_budget_by_default(monkeypatch):
    monkeypatch.setattr('sys.argv', ['run_ar_experiment.py', '--benchmark', 'taskbench',
                                    '--manifest', 'manifest', '--tool-catalog', 'catalog',
                                    '--output-dir', 'output'])
    args = parse_args()
    assert args.max_new_tokens is None
    assert args.generation_stop_policy == 'eos_or_context_limit'


def test_context_limit_is_failure_not_eos_completion():
    _, tokenizer, model, _ = fixture_model()
    model.config.max_position_embeddings = 32
    sample = SimpleNamespace(
        sample_id='context', candidate_names=['weather', 'air'],
        messages=[{'role': 'user', 'content': 'weather'}],
        functions=[{'name': n, 'description': n, 'parameters': {'type': 'object', 'properties': {}}}
                   for n in ('weather', 'air')],
        ground_truth_calls=[{'name': n, 'arguments': {}} for n in ('weather', 'air')],
        ground_truth_tool_types={'weather', 'air'},
    )
    args = argparse.Namespace(prompt_profile='original', output_schema='standard',
                              max_new_tokens=None, seed=1, parity_samples=0, debug_output=False)
    row = run_ar_sample(args, sample, 0, model, tokenizer, torch.device('cpu'))
    assert row['stop_reason'] == 'context_limit'
    assert row['context_limit_reached'] is True
    assert row['natural_generation_complete'] is False
    assert row['eos_detected'] is False
    assert row['generated_token_count'] == 30
    assert row['all_ground_truth_tools_natural_time_s'] is None


def test_eos_mode_rejects_full_prompt_and_invalid_explicit_budget():
    _, tokenizer, model, _ = fixture_model()
    kwargs = dict(model=model, tokenizer=tokenizer, seed=1, device=torch.device('cpu'), observer=None)
    with pytest.raises(ValueError, match='no space'):
        generate_ar(**kwargs, prompt_ids=torch.ones((1, 512), dtype=torch.long))
    with pytest.raises(ValueError, match='exceeds'):
        generate_ar(**kwargs, prompt_ids=torch.ones((1, 2), dtype=torch.long), max_new_tokens=512)


def test_failed_native_generate_removes_hooks(monkeypatch):
    _, tokenizer, model, prompt = fixture_model()
    def fail(**kwargs):
        raise RuntimeError('test')
    monkeypatch.setattr(model, 'generate', fail)
    with pytest.raises(RuntimeError, match='test'):
        generate_ar(model=model, tokenizer=tokenizer, prompt_ids=prompt,
                    max_new_tokens=128, seed=1, device=torch.device('cpu'), observer=None)
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_ar_compact_record_reuses_schema_and_full_set_time():
    target, tokenizer, model, _ = fixture_model()
    sample = SimpleNamespace(
        sample_id='toy', candidate_names=['weather', 'air'],
        messages=[{'role': 'user', 'content': 'weather and air'}],
        functions=[{'name': n, 'parameters': {'type': 'object', 'properties': {}}, 'description': n}
                   for n in ('weather', 'air')],
        ground_truth_calls=[{'name': n, 'arguments': {}} for n in ('weather', 'air')],
        ground_truth_tool_types={'weather', 'air'},
    )
    args = argparse.Namespace(prompt_profile='compact_json_dep_v4', output_schema='flat_name_v1',
                              max_new_tokens=128, seed=20261009, parity_samples=1, debug_output=False)
    row = run_ar_sample(args, sample, 0, model, tokenizer, torch.device('cpu'))
    assert row['raw_output'] == target
    assert row['inference_backend'] == 'huggingface_ar'
    assert row['parity']['checked'] and row['parity']['output_identical'] and row['parity']['nfe_identical']
    assert row['eos_detected'] and row['stop_reason'] == 'eos'
    assert row['natural_generation_complete'] is True
    assert row['context_limit_reached'] is False
    assert row['output_token_count'] + 1 == row['generated_token_count'] == row['forward_count']
    assert row['all_final_tools_natural_time_s'] == row['natural_first_times']['air']['time_s']
    assert row['all_ground_truth_tools_natural_time_s'] == row['all_final_tools_natural_time_s']
    assert row['final_json_valid'] and row['tool_call_format_valid']
    json.dumps(row, allow_nan=False)


def comparison_fixture():
    da = {'seed': 1, 'dtype': 'bfloat16', 'temperature': 0., 'gen_length': 128}
    aa = {**da, 'max_new_tokens': 128, 'inference_backend': 'huggingface_ar'}
    dm = {'args': da, 'questions_sha256': 'q', 'answers_sha256': 'a'}
    am = {'args': aa, 'questions_sha256': 'q', 'answers_sha256': 'a'}
    row = {
        'sample_id': 'x', 'seed': 1, 'candidate_tools': ['weather', 'air'],
        'benchmark': 'bfcl_parallel_multiple', 'ground_truth_tools': ['air', 'weather'],
        'raw_output': '[{"name":"weather"},{"name":"air"}]', 'error': None,
        'first_tool_events': [{'tool': 'weather', 'source': 'latent_region', 'time_s': 1., 'forward': 1},
                              {'tool': 'air', 'source': 'natural', 'time_s': 4., 'forward': 4}],
        'natural_first_times': {'weather': {'time_s': 3., 'forward': 3}, 'air': {'time_s': 4., 'forward': 4}},
        'generation_time_s': 5., 'observer_time_s': .1, 'forward_count': 32,
    }
    ar = {**row, 'first_tool_events': [{'tool': 'weather', 'source': 'natural', 'time_s': 2., 'forward': 10},
                                     {'tool': 'air', 'source': 'natural', 'time_s': 3., 'forward': 20}],
          'natural_first_times': {'weather': {'time_s': 2., 'forward': 10}, 'air': {'time_s': 3., 'forward': 20}}}
    return dm, [row], am, [ar]


def test_ar_comparison_negative_leads_no_nfe_equality_requirement():
    report = compare_ar(*comparison_fixture())
    result = report['comparisons']['dllm_online_vs_ar_natural']
    tools = result['common_final_tool_identities']
    assert tools['dllm_lead_s_ar_minus_dllm']['mean'] == 0.  # +1 and -1
    assert tools['dllm_wins_ties_losses'] == [1, 0, 1]
    assert result['identical_nonempty_final_tool_set_complete_time']['dllm_lead_s_ar_minus_dllm']['mean'] == -1.
    assert report['comparisons']['dllm_natural_vs_ar_natural']['common_final_tool_identities']['dllm_lead_s_ar_minus_dllm']['mean'] == -1.


def test_ar_fewer_tools_not_mislabelled_a_successful_full_set_comparison():
    dm, dr, am, ar = comparison_fixture()
    ar[0]['raw_output'] = '[{"name":"weather"}]'
    ar[0]['natural_first_times'].pop('air')
    ar[0]['first_tool_events'].pop()
    report = compare_ar(dm, dr, am, ar)
    result = report['comparisons']['dllm_online_vs_ar_natural']
    assert result['identical_nonempty_final_tool_set_complete_time']['eligible_count'] == 0
    assert result['ground_truth_complete_coverage_time']['coverage']['dllm_only'] == 1
    assert report['final_tool_set_relation'] == {'different_sets': 1}


def test_comparison_accepts_eos_mode_and_reports_context_failures():
    dm, dr, am, ar = comparison_fixture()
    am['args']['max_new_tokens'] = None
    ar[0].update(stop_reason='context_limit', natural_generation_complete=False, generated_token_count=8000)
    report = compare_ar(dm, dr, am, ar)
    assert report['ar_context_limit_failure_count'] == 1
    assert report['ar_natural_completion_count'] == 0
    assert any('without an experimental token budget' in warning for warning in report['warnings'])


def test_ar_comparison_never_silently_pairs_different_prompts():
    dm, dr, am, ar = comparison_fixture()
    am['args']['prompt_profile'] = 'strict_json_v2'
    with pytest.raises(ValueError, match='prompt_profile differs'):
        compare_ar(dm, dr, am, ar)


def test_ar_cli_real_loop_single_jsonl_and_failures_retained(tmp_path, monkeypatch):
    import run_ar_experiment as runner
    target, tokenizer, model, _ = fixture_model()
    q = tmp_path / 'questions.jsonl'
    a = tmp_path / 'answers.jsonl'
    functions = [{'name': name, 'description': name,
                  'parameters': {'type': 'object', 'properties': {}}} for name in ('weather', 'air')]
    q.write_text(''.join(json.dumps({'id': sid, 'question': [[{'role': 'user', 'content': 'test'}]], 'function': functions}) + '\n' for sid in ('ok', 'fail')))
    a.write_text(''.join(json.dumps({'id': sid, 'ground_truth': [{'weather': {}}, {'air': {}}]}) + '\n' for sid in ('ok', 'fail')))
    args = argparse.Namespace(benchmark='bfcl', questions=str(q), answers=str(a), model_path='fake',
                              output_dir=str(tmp_path / 'result'), prompt_profile='original', output_schema='standard',
                              max_new_tokens=128, seed=1, parity_samples=1, debug_output=False,
                              start=0, limit=None, device='cpu', dtype='float32', observer_mode='natural',
                              inference_backend='huggingface_ar', temperature=0.)
    monkeypatch.setattr(runner, 'parse_args', lambda: args)
    monkeypatch.setattr(runner, 'load_ar_model', lambda args, device: (model, tokenizer))
    original = runner.run_ar_sample
    def run(args, sample, *rest):
        if sample.sample_id == 'fail':
            raise RuntimeError('test failure')
        return original(args, sample, *rest)
    monkeypatch.setattr(runner, 'run_ar_sample', run)
    runner.main()
    files = list((tmp_path / 'result').iterdir())
    assert [p.name for p in files] == ['samples.jsonl']
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert records[0]['online_method']['backend'] == 'AutoModelForCausalLM.generate'
    assert records[1]['raw_output'] == target and records[1]['parity']['output_identical']
    assert records[2]['error'] == 'RuntimeError: test failure' and records[2]['generation_time_s'] is None
