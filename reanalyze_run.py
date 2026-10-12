"""Re-evaluate an existing run without repeating model inference."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from bfcl_data import iter_samples
from evaluation import evaluate_final_calls
from report import build_summary
from tool_parser import extract_partial_names, ordered_unique, parse_tool_calls


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _failure_types(record: dict[str, Any]) -> list[str]:
    failures = []
    if record.get("runtime_error"):
        failures.append("runtime_error")
    if not record["final_json_valid"]:
        failures.append("invalid_json")
    if not record["tool_call_format_valid"]:
        failures.append("invalid_tool_call_format")
    elif not record["candidate_names_valid"]:
        failures.append("unknown_candidate_name")
    if not record["tool_set_exact_match"]:
        failures.append("final_tool_set_mismatch")
    if record.get("oracle_correct_detect_time_s") is None:
        failures.append("no_correct_intermediate_prediction")
    if record.get("predict_all_detect_time_s") is None:
        failures.append("no_online_stable_trigger")
    if not record["bfcl_ast_compatible_match"]:
        failures.append("ast_mismatch")
    if record.get("observer_output_identical") is False:
        failures.append("observer_changed_output")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    args = parser.parse_args()

    source = Path(args.source_run)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    sample_by_id = {
        sample.sample_id: sample
        for sample in iter_samples(args.questions, args.answers)
    }
    records = _load_jsonl(source / "samples.jsonl")
    with (
        (output / "samples.jsonl").open("w", encoding="utf-8") as samples_file,
        (output / "failures.jsonl").open("w", encoding="utf-8") as failures_file,
    ):
        for record in records:
            sample = sample_by_id[record["sample_id"]]
            evaluated_output = record.get(
                "evaluated_output_before_eos", record.get("raw_output", "")
            )
            parsed = parse_tool_calls(evaluated_output, sample.candidate_names)
            generated_ids = record.get("generated_token_ids", [])
            first_eos_position = next(
                (
                    index
                    for index, token_id in enumerate(generated_ids)
                    if token_id == 126081
                ),
                None,
            )
            predicted_calls = parsed.calls if parsed.format_valid else []
            partial_names = extract_partial_names(
                evaluated_output, sample.candidate_names
            )
            final_types = ordered_unique(
                [call["name"] for call in predicted_calls]
                if parsed.format_valid
                else partial_names
            )
            record.update(
                {
                    "generated_tool_calls": predicted_calls,
                    "final_partial_call_names": partial_names,
                    "final_predicted_tool_types": final_types,
                    "final_json_valid": parsed.json_valid,
                    "tool_call_format_valid": parsed.format_valid,
                    "candidate_names_valid": parsed.candidate_names_valid,
                    "final_json_error": parsed.error,
                    "first_eos_position": first_eos_position,
                    "eos_at_last_position": (
                        bool(generated_ids)
                        and first_eos_position == len(generated_ids) - 1
                    ),
                    **evaluate_final_calls(sample, predicted_calls),
                }
            )
            samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            failures = _failure_types(record)
            if failures:
                failures_file.write(
                    json.dumps(
                        {
                            "sample_id": record["sample_id"],
                            "failure_types": failures,
                            "final_json_error": record["final_json_error"],
                            "raw_output": record.get("raw_output", ""),
                            "evaluated_output_before_eos": evaluated_output,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    shutil.copyfile(source / "steps.jsonl", output / "steps.jsonl")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["reanalyzed_at"] = datetime.now().astimezone().isoformat()
    metadata["reanalyzed_from"] = str(source.resolve())
    metadata["reanalyze_command"] = sys.argv
    metadata["metric_corrections"] = [
        "Separated JSON syntax, unified tool-call format, and candidate-name validity.",
        "Preserved structurally valid unknown calls for name/count metrics.",
        "Applied BFCL Python int-to-float coercion before AST value matching.",
    ]
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_summary(output)
    print(f"Reanalyzed {len(records)} samples into {output.resolve()}")


if __name__ == "__main__":
    main()
