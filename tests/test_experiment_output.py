import argparse
import json
from types import SimpleNamespace

import pytest
import torch

from experiment_output import open_result_file, write_record
from model_adapter import ModelRuntimeSpec
from omtp_observer import OMTPObserver
from run_omtp_experiment import _run_sample
from run_taskbench_experiment import _fixed_runtime_args
from summarize_experiment import summarize
from test_omtp import CharacterTokenizer


def test_single_file_metadata_and_no_overwrite(tmp_path):
    out = tmp_path / 'run'
    with open_result_file(out) as handle:
        write_record(handle, {'record_type': 'metadata', 'schema_version': 'tool_first_v1'})
        write_record(handle, {'record_type': 'sample', 'sample_id': 'x'})
    assert [p.name for p in out.iterdir()] == ['samples.jsonl']
    assert len((out / 'samples.jsonl').read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        with open_result_file(out):
            pass


def test_identity_first_trigger_keeps_errors_and_ignores_duplicate_tracks():
    tokenizer = CharacterTokenizer('[{"name":"weather air wrong"}]')
    observer = OMTPObserver(tokenizer=tokenizer, candidate_names=['weather', 'air', 'wrong'])
    observer._relative_now = lambda: float(len(observer.events) + 1)
    observation = SimpleNamespace(forward_index=1)
    for track, source, tool in (
        (0, 'latent_region', 'wrong'), (0, 'prefix', 'weather'),
        (1, 'latent_region', 'weather'), (1, 'natural', 'air'),
    ):
        observer._emit(track={'track_id': track}, source=source, tool=tool,
                       observation=observation, extra={})
    assert list(observer.first_tool_events) == ['wrong', 'weather', 'air']
    assert observer.first_tool_events['weather']['source'] == 'prefix'
    assert observer.first_tool_events['weather']['predict_time_s'] == 2.


class PromptTokenizer(CharacterTokenizer):
    def __init__(self, target):
        super().__init__(target + 'air')
        self.eos_token_id = 125
        self.pad_token_id = 125
        self.all_special_ids = [125]

    def apply_chat_template(self, messages, **kwargs):
        return ' '

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.encode(text)
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)


class JsonModel:
    device = torch.device('cpu')

    def __init__(self, tokenizer, target):
        self.ids = tokenizer.encode(target)
        self.calls = 0
        self._dllm_toolcall_runtime_spec = ModelRuntimeSpec(
            requested_path='fake', resolved_path='fake', family='llada',
            architecture='LLaDAModelLM', model_type='llada', mask_token_id=126,
            eos_token_id=125, pad_token_id=125, vocab_size=127,
            max_sequence_length=4096, supports_dual_cache=True,
            source_revision=None, config_sha256='fake', weight_index_sha256=None,
        )

    def __call__(self, x):
        self.calls += 1
        logits = torch.full((1, x.shape[1], 127), -8.)
        for pos in range(x.shape[1]):
            token = self.ids[pos - 1] if pos > 0 else 1
            logits[0, pos, token] = 8. + pos / 100
        return SimpleNamespace(logits=logits)


def test_real_cpu_generation_compact_debug_and_observer_off_parity():
    target = '[{"name":"weather","arguments":{}}]'
    tokenizer = PromptTokenizer(target)
    model = JsonModel(tokenizer, target)
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
        debug_output=False,
    ))
    args.gen_length = args.block_length = len(tokenizer.encode(target))
    args.steps = 4
    compact, observer = _run_sample(args, sample, 0, model, tokenizer, torch.device('cpu'))
    assert compact['raw_output'] == target
    assert compact['forward_count'] == 4
    assert model.calls == 12  # warmup + observer off + observer on; no extra OMTP forward
    assert compact['parity']['output_identical'] and compact['parity']['nfe_identical']
    assert len(observer.steps) == 1 and observer.trajectory == []
    assert 'debug' not in compact and 'steps' not in compact
    assert len(compact['first_tool_events']) == 1
    event = compact['first_tool_events'][0]
    assert event['time_s'] <= compact['natural_first_times']['weather']['time_s']
    args.debug_output = True
    debug, history = _run_sample(args, sample, 0, model, tokenizer, torch.device('cpu'))
    assert debug['raw_output'] == compact['raw_output']
    assert debug['forward_count'] == compact['forward_count']
    assert len(history.steps) == 4 and 'debug' in debug
    assert [(e['tool'], e['source'], e['forward']) for e in debug['first_tool_events']] == [
        (e['tool'], e['source'], e['forward']) for e in compact['first_tool_events']
    ]
    json.dumps(compact, allow_nan=False)


def test_summary_separates_model_invalid_name_from_physical_false_start():
    row = {
        'sample_id': 'x', 'candidate_tools': ['weather', 'air'],
        'raw_output': '[{"name":"weather"},{"name":"air enhancement"}]',
        'ground_truth_tools': ['weather'],
        'first_tool_events': [
            {'tool': 'weather', 'source': 'natural', 'time_s': 2., 'forward': 4},
            {'tool': 'air', 'source': 'prefix', 'time_s': 1., 'forward': 2,
             'offline_prefix_audit': {'status': 'model_invalid_name'}},
        ],
        'natural_first_times': {'weather': {'time_s': 2., 'forward': 4}},
        'generation_time_s': 3., 'observer_time_s': .1, 'forward_count': 4,
    }
    report = summarize([row])
    quality = report['accuracy']['final_raw_text']
    assert quality['false_positive'] == 1
    assert quality['model_invalid_name_trigger_count'] == 1
    assert quality['detector_error_count'] == 0
    assert quality['wrong_trigger_rate'] == .5
    assert quality['detector_error_rate_excluding_model_invalid_names'] == 0
    assert report['paired_lead_s']['mean'] == 0.


@pytest.mark.parametrize('mode', ['omtp', 'natural'])
def test_taskbench_cli_only_one_jsonl_and_failures_are_retained(tmp_path, monkeypatch, mode):
    import run_taskbench_experiment as runner
    target = '[{"name":"weather","arguments":{}}]'
    tokenizer = PromptTokenizer(target)
    model = JsonModel(tokenizer, target)
    catalog = tmp_path / 'tool_desc.json'
    catalog.write_text(json.dumps({'nodes': [
        {'id': name, 'desc': name, 'parameters': []} for name in ('weather', 'air')
    ]}))
    manifest = tmp_path / 'formal.jsonl'
    records = [{
        'sample_id': sid, 'source_id': sid, 'source_seed': 1,
        'graph_type': 'chain', 'instruction': 'weather',
        'ground_truth_calls': [{'name': name, 'arguments': {}} for name in ('weather', 'air')],
        'dependencies': [{'source': 'weather', 'target': 'air'}], 'reference_steps': [],
    } for sid in ('ok', 'fail')]
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    output = tmp_path / 'results'
    cli = argparse.Namespace(
        manifest=str(manifest), tool_catalog=str(catalog), output_dir=str(output),
        model_path='fake', fastdllm_root=str(tmp_path), device='cpu', dtype='float32',
        generation_profile='fixed128', prompt_profile='original', cache_mode='none',
        response_mode='native', output_schema='standard', parity_samples=1, debug_output=False,
        observer_mode=mode,
    )
    # Generate a 128-position canvas with EOS after the fixed target.
    model.ids += [125] * (128 - len(model.ids))
    monkeypatch.setattr(runner, 'parse_args', lambda: cli)
    monkeypatch.setattr(runner, '_load_model', lambda args, device: (model, tokenizer))
    original = runner._run_sample
    def run(args, sample, *rest):
        if sample.sample_id == 'fail':
            raise RuntimeError('intentional test failure')
        return original(args, sample, *rest)
    monkeypatch.setattr(runner, '_run_sample', run)
    runner.main()
    assert [p.name for p in output.iterdir()] == ['samples.jsonl']
    values = [json.loads(line) for line in (output / 'samples.jsonl').read_text().splitlines()]
    assert len(values) == 3
    assert values[0]['schema_version'] == 'tool_first_v1'
    assert (values[0]['online_method']['name'] == 'natural_only') == (mode == 'natural')
    assert values[1]['observer_mode'] == mode
    assert values[1]['forward_count'] == 32
    assert values[1]['parity']['output_identical']
    assert values[2]['generation_time_s'] is None
    assert values[2]['error'] == 'RuntimeError: intentional test failure'
    assert values[2]['seed'] == 20261010
