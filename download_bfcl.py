"""Download a pinned BFCL parallel_multiple question/answer pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from bfcl_data import validate_dataset


# BFCL V4 release commit cited by the upstream repository.  The non-live
# parallel_multiple data is the original BFCL category carried into V4.
DEFAULT_REVISION = "58f57e9124ea981403792dd51e00a6577e621fae"
BASE = "https://raw.githubusercontent.com/ShishirPatil/gorilla/{revision}/berkeley-function-call-leaderboard/bfcl_eval/data"
FILES = {
    "questions": (
        "BFCL_v4_parallel_multiple.json",
        "BFCL_v4_parallel_multiple.json",
    ),
    "answers": (
        "possible_answer/BFCL_v4_parallel_multiple.json",
        "BFCL_v4_parallel_multiple_answers.json",
    ),
}
EXPECTED_SHA256 = {
    "questions": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
    "answers": "5ebf24f458c1f16300c05505d83d6f0a1b68b79be273a033febd0d4f840507e3",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/bfcl")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"revision": args.revision, "files": {}}
    paths = {}
    for label, (relative, output_name) in FILES.items():
        url = f"{BASE.format(revision=args.revision)}/{relative}"
        with urlopen(url) as response:
            payload = response.read()
        digest = _sha256(payload)
        if args.revision == DEFAULT_REVISION and digest != EXPECTED_SHA256[label]:
            raise ValueError(
                f"SHA-256 mismatch for {label}: expected "
                f"{EXPECTED_SHA256[label]}, got {digest}"
            )
        target = output / output_name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        target.write_bytes(payload)
        paths[label] = target
        metadata["files"][label] = {
            "url": url,
            "path": str(target.resolve()),
            "sha256": digest,
            "bytes": len(payload),
        }
    metadata["validation"] = validate_dataset(paths["questions"], paths["answers"])
    (output / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
