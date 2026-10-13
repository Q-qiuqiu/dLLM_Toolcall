"""Download or verify the pinned local tool models and write a SHA manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from model_specs import MODELS_ROOT, MODEL_SPECS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(file.relative_to(path)),
            "size_bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in sorted(path.rglob("*"))
        if file.is_file() and ".cache" not in file.parts
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        for spec in MODEL_SPECS.values():
            snapshot_download(
                repo_id=spec["repo_id"], revision=spec["revision"],
                allow_patterns=spec["allow_patterns"],
                local_dir=spec["local_dir"],
            )
    models = {}
    for name, spec in MODEL_SPECS.items():
        path = Path(spec["local_dir"])
        if not path.is_dir():
            raise FileNotFoundError(path)
        files = _files(path)
        if not files:
            raise RuntimeError(f"no local model files under {path}")
        models[name] = {
            "repo_id": spec["repo_id"], "revision": spec["revision"],
            "local_dir": str(path.resolve()),
            "taskbench_tools": spec["taskbench_tools"],
            "semantic_note": spec["semantic_note"],
            "file_count": len(files),
            "total_size_bytes": sum(row["size_bytes"] for row in files),
            "files": files,
        }
    target = MODELS_ROOT / "model_manifest.json"
    target.write_text(json.dumps({"models": models}, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
