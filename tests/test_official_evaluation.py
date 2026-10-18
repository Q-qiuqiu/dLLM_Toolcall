"""Official-source integration and conversion regression tests (CPU only)."""

import json
import pytest

from bfcl_data import BFCLSample
from official_eval_adapters import (
    AdaptedCalls, adapt_calls, check_bfcl, enable_local_dependencies,
    node_projection, resource_preflight, taskbench_prediction, verify_sources,
)

enable_local_dependencies()


def test_pinned_sources_verify():
    assert verify_sources()['bfcl_revision'].startswith('58f57e9')


def test_official_bfcl_is_unordered_but_duplicate_sensitive():
    function = {'name': 'math.f', 'parameters': {
        'type': 'dict', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}}
    sample = BFCLSample('test', [], [function], [{'math.f': {'x': [1]}}, {'math.f': {'x': [2]}}])
    correct = AdaptedCalls([{'name': 'math.f', 'arguments': {'x': 2}},
                            {'name': 'math.f', 'arguments': {'x': 1}}], None)
    wrong = AdaptedCalls([{'name': 'math.f', 'arguments': {'x': 1}}] * 2, None)
    assert check_bfcl(sample, correct)['valid']
    assert not check_bfcl(sample, wrong)['valid']


def test_adapter_does_not_hide_wrong_or_missing_parameters():
    functions = [{'name': 'f', 'parameters': {'properties': {'x': {'type': 'string'}}, 'required': ['x']}}]
    r = adapt_calls('[{"name":"f","unknown":3}]', 'flat_name_v1', functions)
    assert r.error is None and r.calls[0]['arguments'] == {'unknown': 3}
    r = adapt_calls('[{"name":"wrong","arguments":{"x":3}}]', 'standard', functions)
    assert r.error is None and r.calls[0]['name'] == 'wrong'
    assert r.calls[0]['arguments']['x'] == 3


def test_flat_alias_and_wrapper_are_reversible_not_answer_assisted():
    functions = [{'name': 'f', 'parameters': {'properties': {'name': {'type': 'string'}, 'x': {'type': 'integer'}}}}]
    r = adapt_calls('[{"name":"f","arguments":{"arg_name":"a","x":2}}]', 'flat_name_v1', functions)
    assert r.calls == [{'name': 'f', 'arguments': {'name': 'a', 'x': 2}}]
    assert adapt_calls('[{"name":"f","x":1,"arguments":{"x":2}}]', 'flat_name_v1', functions).error


def test_invalid_json_is_failure_not_partial_call_recovery():
    r = adapt_calls('[{"name":"f","arguments":{}},', 'standard', [])
    assert r.error and r.calls == []


def test_resource_adapter_restores_positions_and_preserves_names():
    calls = [{'name': 'Image-to-Text', 'arguments': {'input_2': '<node-1>', 'input_1': 'a.jpg'}},
             {'name': 'UNKNOWN TOOL', 'arguments': {}}]
    native = taskbench_prediction(calls, 'resource')
    assert native['task_nodes'][0]['arguments'] == ['a.jpg', '<node-1>']
    assert native['task_nodes'][1]['task'] == 'UNKNOWN TOOL'
    assert resource_preflight(native['task_nodes']) is None  # official permits forward refs
    assert node_projection(native)['task_nodes'][0]['task'] == 'Image-to-Text'


def test_invalid_resource_references_cannot_reuse_ground_truth_argument():
    assert resource_preflight([{'task': 'f', 'arguments': ['<node-object>']}])
    assert resource_preflight([{'task': 'f', 'arguments': ['<node-5>']}])
    assert resource_preflight([{'task': 'f', 'arguments': [123]}])
    assert resource_preflight([{'task': 'f', 'arguments': ['<node--1>']}])


def test_temporal_dependency_is_only_derived_from_generated_references():
    calls = [{'name': 'a', 'arguments': {'x': 'literal'}},
             {'name': 'b', 'arguments': {'x': '<node-0>'}},
             {'name': 'c', 'arguments': {}}]
    result = taskbench_prediction(calls, 'temporal')
    assert result['task_links'] == [{'source': 'a', 'target': 'b'}]
    assert result['task_nodes'][1]['arguments'] == [{'name': 'x', 'value': '<node-0>'}]


@pytest.mark.parametrize('dependency_type', ['temporal', 'resource'])
def test_real_taskbench_evaluator_perfect_and_empty(tmp_path, dependency_type):
    from evaluate_official_results import invoke_taskbench, save_jsonl
    labels = []
    for sample_id, graph in [('c', 'chain'), ('d', 'dag')]:
        calls = [{'name': 'f', 'arguments': {'input_1': 'a.txt'}},
                 {'name': 'g', 'arguments': {'input_1': '<node-0>'}}]
        native = taskbench_prediction(calls, dependency_type)
        labels.append({'id': sample_id, 'type': graph, **native})
    catalog = {'nodes': [{'id': 'f', 'output-type': ['text']}, {'id': 'g', 'output-type': ['text']}]}
    save_jsonl(tmp_path / 'data.json', labels)
    for stem, empty in [('perfect', False), ('empty', True)]:
        predictions = [{'id': r['id'], 'result': {'task_nodes': [] if empty else r['task_nodes'],
                                                 'task_links': [] if empty else r['task_links']}}
                       for r in labels]
        save_jsonl(tmp_path / 'predictions' / (stem + '.json'), predictions)
        scores = invoke_taskbench(tmp_path, stem, catalog, dependency_type, tmp_path / (stem + '.log'))
        for result in scores.values():
            assert result['node_supports'] == result['all_samples']
            for key in ['node_micro_f1_no_matching', 'link_binary_f1',
                        'argument_task_argname_binary_f1_no_matching',
                        'argument_task_argname_value_binary_f1_no_matching']:
                assert result[key] == (0 if empty else 1)


def test_ground_truth_annotation_errors_are_declared_not_filled():
    from evaluate_official_results import ground_truth_error
    gt = {'task_nodes': [{'task': 'send_email', 'arguments': [{'name': 'content', 'value_from_node': 'a'}]}]}
    before = json.dumps(gt)
    assert ground_truth_error(gt, 'temporal')
    assert json.dumps(gt) == before
    assert ground_truth_error({'task_nodes': [{'task': 'f'}]}, 'resource')


@pytest.mark.parametrize('domain', ['bfcl', 'dailylifeapis', 'huggingface', 'multimedia'])
def test_ar_paths_use_complete_eos_runs_without_changing_default(domain, tmp_path):
    from evaluate_official_results import AR_METHOD, run_paths
    default = run_paths(tmp_path, domain)
    expanded = run_paths(tmp_path, domain, include_ar=True)
    assert len(default) == 3 and len(expanded) == 4
    assert expanded[:3] == default
    assert expanded[3][0] == AR_METHOD
    assert 'llama3_ar_natural_eos_' in expanded[3][1].parent.name


def test_ar_cache_is_not_mislabeled_as_dllm_nocache_or_dual():
    from evaluate_official_results import inference_cache_label
    assert inference_cache_label({'args': {'inference_backend': 'huggingface_ar'}}) == 'ar_kv'
    assert inference_cache_label({'args': {'cache_mode': 'dual'}}) == 'dual'
    assert inference_cache_label({'args': {}}) == 'none'
