"""Fetch immutable official scoring sources, without inference/API SDKs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
BFCL_REVISION = '58f57e9124ea981403792dd51e00a6577e621fae'
TASKBENCH_REVISION = 'd12764a8fa650f49f50507a02c3e707c71c67621'
SOURCE_ROOT = ROOT / 'third_party' / 'official_eval'
BFCL_ROOT = SOURCE_ROOT / ('bfcl_' + BFCL_REVISION[:7])
TASKBENCH_ROOT = SOURCE_ROOT / ('taskbench_' + TASKBENCH_REVISION[:7])


def sources():
    base = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_REVISION}/'
    paths = [
        'constants/enums.py', 'constants/type_mappings.py', 'constants/model_config.py',
        'eval_checker/ast_eval/ast_checker.py',
        'eval_checker/ast_eval/type_convertor/java_type_converter.py',
        'eval_checker/ast_eval/type_convertor/js_type_converter.py',
    ]
    items = [(BFCL_ROOT / 'bfcl_eval' / p, base + 'berkeley-function-call-leaderboard/bfcl_eval/' + p)
             for p in paths]
    items += [(BFCL_ROOT / 'LICENSE', base + 'LICENSE')]
    items += [(TASKBENCH_ROOT / 'evaluate.py',
               f'https://huggingface.co/datasets/microsoft/Taskbench/raw/{TASKBENCH_REVISION}/evaluate.py')]
    return items


def fetch_official_sources():
    manifest_path = SOURCE_ROOT / 'sources.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    known = {r['path']: r for r in old['files']} if old else {}

    def fetch(item):
        path, url = item
        relative = str(path.relative_to(ROOT))
        if path.exists():
            data = path.read_bytes()
            expected = known.get(relative, {}).get('sha256')
            if expected is None:
                # Resume an interrupted first fetch by comparing to the same
                # immutable remote source, never overwriting an unknown file.
                with urlopen(url, timeout=45) as response:
                    expected = hashlib.sha256(response.read()).hexdigest()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'unverified existing third-party file: {path}')
        else:
            with urlopen(url, timeout=45) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(data)
        return dict(path=relative, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, sources()))
    manifest = dict(bfcl_revision=BFCL_REVISION, taskbench_revision=TASKBENCH_REVISION, files=files)
    if old is None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    elif old != manifest:
        raise ValueError('source manifest changed')
    print(json.dumps(manifest, indent=2))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    fetch_official_sources()


if __name__ == '__main__':
    main()
