"""Deterministic JSON adapters and byte-verified official evaluator loaders.

No model inference, answer-assisted correction, or scoring-function patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from download_official_evaluators import ROOT, SOURCE_ROOT, BFCL_ROOT, TASKBENCH_ROOT
from tool_format import parameter_aliases


DEPENDENCY_ROOT = ROOT / '.official_eval_deps'
REGISTRY_NAME = 'llada-offline'
WRAPPERS = {'arguments', 'args', 'parameters', 'params', 'inputs', 'input'}
NODE_REFERENCE = re.compile(r'^<node-(\d+)>$')


def enable_local_dependencies():
    if DEPENDENCY_ROOT.is_dir() and str(DEPENDENCY_ROOT) not in sys.path:
        sys.path.insert(0, str(DEPENDENCY_ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    path = SOURCE_ROOT / 'sources.json'
    if not path.exists():
        raise RuntimeError('Run download_official_evaluators.py first')
    manifest = json.loads(path.read_text())
    for source in manifest['files']:
        if sha256(ROOT / source['path']) != source['sha256']:
            raise ValueError(f'official source changed: {source["path"]}')
    return manifest


@lru_cache(maxsize=1)
def load_bfcl():
    verify_sources()
    sys.path.insert(0, str(BFCL_ROOT))
    # The official registry imports dozens of inference providers. The checker
    # only reads underscore_to_dot. Register this JSON-output model without
    # importing/invoking any provider; all scoring source bytes stay unchanged.
    name = 'bfcl_eval.constants.model_config'
    if name in sys.modules:
        raise RuntimeError('BFCL registry already imported; use a fresh offline process')
    bridge = ModuleType(name)
    bridge.MODEL_CONFIG_MAPPING = {REGISTRY_NAME: SimpleNamespace(underscore_to_dot=False)}
    bridge.__doc__ = 'Offline registry bridge: preserve original dotted function names.'
    sys.modules[name] = bridge
    checker = importlib.import_module('bfcl_eval.eval_checker.ast_eval.ast_checker')
    enums = importlib.import_module('bfcl_eval.constants.enums')
    return checker, enums.Language.PYTHON


@lru_cache(maxsize=1)
def load_taskbench():
    verify_sources()
    enable_local_dependencies()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    path = TASKBENCH_ROOT / 'evaluate.py'
    spec = importlib.util.spec_from_file_location('pinned_taskbench_official_evaluate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class AdaptedCalls:
    calls: list[dict[str, Any]]
    error: str | None


def adapt_calls(text, schema, functions):
    """Structural adaptation only; let official scoring judge missing/bad args.

    Unlike normalize_flat_arguments, do NOT drop unknown arguments, reject
    missing required parameters, or filter bad types/names before scoring.
    """
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as error:
        return AdaptedCalls([], 'json_parse_error:' + str(error))
    if not isinstance(value, list):
        return AdaptedCalls([], 'top_level_not_array')
    aliases = parameter_aliases(functions)
    by_name = {f['name']: f for f in functions}
    calls = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(item.get('name'), str):
            return AdaptedCalls([], f'call_{index}:missing_or_invalid_name')
        name = item['name']
        if schema == 'standard':
            if set(item) != {'name', 'arguments'} or not isinstance(item['arguments'], dict):
                return AdaptedCalls([], f'call_{index}:invalid_standard_structure')
            arguments = item['arguments']
        elif schema == 'flat_name_v1':
            arguments = {}
            properties = by_name.get(name, {}).get('parameters', {}).get('properties', {})
            conflict = []

            def collect(mapping, outer=False):
                for key, val in mapping.items():
                    if outer and key == 'name':
                        continue
                    original = aliases.get(name, {}).get(key, key)
                    if key in WRAPPERS and original not in properties and isinstance(val, dict):
                        collect(val)
                    elif original in arguments and arguments[original] != val:
                        conflict.append(original)
                    else:
                        arguments[original] = val

            collect(item, outer=True)
            if conflict:
                return AdaptedCalls([], f'call_{index}:conflicting_arguments:{conflict}')
        else:
            raise ValueError(f'unsupported output schema: {schema}')
        calls.append({'name': name, 'arguments': arguments})
    return AdaptedCalls(calls, None)


def check_bfcl(sample, adapted):
    if adapted.error:
        return {'valid': False, 'error_type': 'output_adapter_error', 'error': [adapted.error]}
    module, language = load_bfcl()
    return module.ast_checker(sample.functions,
                              [{c['name']: c['arguments']} for c in adapted.calls],
                              sample.ground_truth, language, 'parallel_multiple', REGISTRY_NAME)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def taskbench_prediction(calls, dependency_type):
    nodes = []
    links = set()
    for index, call in enumerate(calls):
        arguments = call['arguments']
        if dependency_type == 'temporal':
            native_args = [{'name': key, 'value': value} for key, value in arguments.items()]
        else:
            # Reverses input_1/input_2 positional aliases without inventing
            # missing inputs or looking at the sample's expected tool nodes.
            keys = sorted(arguments, key=lambda key: (0, int(key[6:]))
                          if key.startswith('input_') and key[6:].isdigit() else (1, key))
            native_args = [arguments[key] for key in keys]
        nodes.append({'task': call['name'], 'arguments': native_args})
        for value in _strings(arguments):
            match = NODE_REFERENCE.fullmatch(value)
            if match:
                source = int(match.group(1))
                if 0 <= source < len(calls) and source != index:
                    links.add((calls[source]['name'], call['name']))
    return {'task_nodes': nodes, 'task_links': [{'source': a, 'target': b} for a, b in sorted(links)]}


def resource_preflight(nodes):
    """Prevent upstream except/pass from reusing a previous (even GT) argument.

    No values are repaired here. A non-adaptable resource prediction is scored
    as an empty output, never silently excluded. Original calls are audited.
    Self references follow the official script's explicit skip rule. Forward
    references are allowed by the official evaluator and remain unchanged.
    """
    for node_index, node in enumerate(nodes):
        for arg_index, argument in enumerate(node.get('arguments', [])):
            try:
                if isinstance(argument, dict):
                    argument = list(argument.values())[0]
                if isinstance(argument, list):
                    argument = ' '.join(argument)
                if not isinstance(argument, str):
                    raise ValueError('resource_argument_not_string')
                if '<node-' in argument:
                    start = argument.index('<node-') + 6
                    end = argument.index('>')
                    source = int(argument[start:end])
                    if not 0 <= source < len(nodes):
                        raise ValueError('reference_index_out_of_range')
                else:
                    argument.strip("'")
            except (ValueError, TypeError, IndexError) as error:
                return f'node_{node_index}_arg_{arg_index}:{type(error).__name__}:{error}'
    return None


def node_projection(native):
    return {'task_nodes': [{'task': n['task'], 'arguments': []} for n in native['task_nodes']],
            'task_links': []}
