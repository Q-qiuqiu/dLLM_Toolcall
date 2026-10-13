"""Audit candidate tool names against the frozen LLaDA tokenizer and OMTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from transformers import AutoTokenizer

from omtp_observer import OMTPObserver
from taskbench_data import load_tool_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-catalog", required=True)
    parser.add_argument("--tokenizer", default="/data/labshare/Param/llada")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    functions = load_tool_catalog(args.tool_catalog)
    names = [function["name"] for function in functions]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    observer = OMTPObserver(tokenizer=tokenizer, candidate_names=names)
    rows = []
    token_sequences = {}
    for name in names:
        ids = observer._candidate_ids[name]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        token_sequences.setdefault(ids, []).append(name)
        rows.append({
            "name": name,
            "token_ids": list(ids),
            "token_count": len(ids),
            "decoded": decoded,
            "roundtrip_exact": decoded == name,
            "has_space": " " in name,
            "has_hyphen": "-" in name,
            "has_parenthesis": bool(re.search(r"[()]", name)),
        })
    character_prefix_collisions = [
        {"short": left, "long": right}
        for left in names for right in names
        if left != right and right.startswith(left)
    ]
    token_prefix_collisions = [
        {"short": left, "long": right}
        for left in names for right in names
        if left != right
        and observer._candidate_ids[right][:len(observer._candidate_ids[left])]
        == observer._candidate_ids[left]
    ]
    value = {
        "tool_catalog": str(Path(args.tool_catalog).resolve()),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tool_count": len(names),
        "observer_construction_ok": True,
        "empty_tokenization_count": sum(not row["token_ids"] for row in rows),
        "roundtrip_failure_count": sum(not row["roundtrip_exact"] for row in rows),
        "duplicate_token_sequence_groups": [
            group for group in token_sequences.values() if len(group) > 1
        ],
        "character_prefix_collisions": character_prefix_collisions,
        "token_prefix_collisions": token_prefix_collisions,
        "tools": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value[key] for key in (
        "tool_count", "observer_construction_ok", "empty_tokenization_count",
        "roundtrip_failure_count", "duplicate_token_sequence_groups",
        "character_prefix_collisions", "token_prefix_collisions",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
