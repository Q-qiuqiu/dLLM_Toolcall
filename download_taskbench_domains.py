"""Download or verify the two pinned official TaskBench resource domains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


REPO_ID = "microsoft/Taskbench"
REVISION = "d12764a8fa650f49f50507a02c3e707c71c67621"
DOMAINS = ("data_huggingface", "data_multimedia")
FILES = ("improved.parquet", "tool_desc.json", "graph_desc.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/taskbench")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
        for domain in DOMAINS:
            for filename in FILES:
                downloaded = Path(hf_hub_download(
                    repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                    filename=f"{domain}/{filename}",
                ))
                target = root / domain / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                # Avoid links into a mutable cache: the manifest hashes the
                # exact standalone files consumed by this experiment.
                shutil.copy2(downloaded, target)

    rows = []
    for domain in DOMAINS:
        for filename in FILES:
            path = root / domain / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({
                "domain": domain, "file": filename,
                "path": str(path.resolve()), "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "repo_id": REPO_ID, "repo_type": "dataset",
        "revision": REVISION, "files": rows,
    }
    target = root / "source_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
