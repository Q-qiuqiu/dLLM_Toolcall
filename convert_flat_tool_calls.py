"""Convert strict flat_name_v1 JSON into canonical name/arguments calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from taskbench_data import load_tool_catalog
from tool_parser import parse_tool_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="Path containing one flat JSON array, or - for stdin.",
    )
    parser.add_argument(
        "--tool-catalog", required=True,
        help="TaskBench tool_desc.json used to resolve candidates and aliases.",
    )
    parser.add_argument(
        "--output", default="-",
        help="Canonical JSON destination; defaults to stdout.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def convert_text(text: str, tool_catalog: str | Path) -> list[dict]:
    functions = load_tool_catalog(tool_catalog)
    parsed = parse_tool_calls(
        text,
        [str(function["name"]) for function in functions],
        output_schema="flat_name_v1",
        functions=functions,
    )
    if not parsed.valid:
        raise ValueError(f"flat tool-call conversion failed: {parsed.error}")
    return parsed.calls


def main() -> None:
    args = parse_args()
    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(
        encoding="utf-8"
    )
    converted = convert_text(text, args.tool_catalog)
    rendered = json.dumps(converted, ensure_ascii=False, indent=args.indent) + "\n"
    if args.output == "-":
        sys.stdout.write(rendered)
    else:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
