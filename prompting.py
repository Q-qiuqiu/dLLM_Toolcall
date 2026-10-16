"""Ground-truth-free BFCL prompt construction."""

from __future__ import annotations

import json
from typing import Any

from bfcl_data import BFCLSample
from tool_format import OUTPUT_SCHEMAS, adapt_functions_for_output_schema


SYSTEM_PROMPT = """You are a function-calling assistant. Select calls only from the provided candidate functions.
Return exactly one JSON array and nothing else. Each array item must have exactly this form:
{"name": "candidate.function_name", "arguments": {"parameter": "value"}}
Use one array item per call. The same function may be called multiple times. Do not include reasoning, thoughts, PLAN, markdown fences, or explanatory text. If several calls are requested, put every call in the single array."""

STRICT_JSON_SYSTEM_PROMPT = """You are a function-call compiler. Select calls only from the provided candidate functions.
Return exactly one minified, syntactically valid JSON array and nothing else.
Every array element must contain exactly two keys in this order: "name" and "arguments".
Copy each tool name verbatim from the candidate definitions. Tool names may contain spaces, hyphens, and parentheses; never normalize, abbreviate, translate, or rewrite them.
Use only parameter keys declared by that tool's JSON Schema. Copy explicit values from the request verbatim. Any string value that you must infer must be at most 32 characters and must not be an explanatory sentence.
Use one array element per required call. Do not add duplicate calls unless the request requires the same tool more than once.
Separate adjacent objects with commas. Never emit trailing commas, comments, markdown fences, escaped newline sequences, reasoning, thoughts, PLAN, explanations, duplicate keys, or alternative key names.
The first output character must be [ and the last output character must be ]. Stop immediately after the closing ]."""

STRICT_JSON_FINAL_CONTRACT = """FINAL OUTPUT CONTRACT:
- Output one minified JSON array only.
- Each item must match {"name":"<EXACT_CANDIDATE_NAME>","arguments":{"<schema_key>":"<short_value>"}}.
- Angle-bracket strings above are syntax placeholders, not valid tool names or values.
- Copy candidate names exactly, including spaces, hyphens, and parentheses.
- Copy explicit request values unchanged; keep every inferred string at most 32 characters.
- Do not explain argument values. Use commas between objects and stop after ]."""

COMPACT_JSON_SYSTEM_PROMPT = """You are a function-calling assistant. Choose the functions needed for the request from the candidate function definitions and provide their inputs.
Respond with a valid JSON array. Represent each function call as an object with a "name" field and an "arguments" object that follows the selected function's parameter schema."""

COMPACT_JSON_FINAL_INSTRUCTION = "Generate the JSON array for the request."

FLAT_NAME_SYSTEM_PROMPT = """You are a function-calling assistant. Choose the functions needed for the request from the candidate tool signatures and provide their inputs.
Respond with a valid JSON array. Represent each call as one flat object whose first field is "name", followed directly by the input fields from that tool's signature.
Format example: the signature tool_a(city:string, date:string) is called as [{"name":"tool_a","city":"Beijing","date":"tomorrow"}]."""

FLAT_NAME_DEPENDENCY_SYSTEM_PROMPT = """You are a function-calling assistant. Choose the functions needed for the request from the candidate tool signatures and provide their inputs.
Respond with a valid JSON array. Represent each call as one flat object whose first field is "name", followed directly by the input fields from that tool's signature.
Array items are numbered from 0. When an input uses the output of an earlier call, write "<node-N>", where N is that earlier item's index.
Format examples: the signature tool_a(city:string) is called as [{"name":"tool_a","city":"Beijing"}]. A two-step call is [{"name":"Image-to-Text","input_1":"example.jpg"},{"name":"Summarization","input_1":"<node-0>"}]."""

PROMPT_PROFILES = (
    "original", "strict_json_v2", "compact_json_v3", "compact_json_dep_v4",
)


def _render_question(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )


def _standardize_schema(value: Any) -> Any:
    """Return a prompt-only JSON-Schema copy using standard object types."""

    if isinstance(value, dict):
        return {
            key: (
                "object" if key == "type" and item == "dict"
                else _standardize_schema(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_standardize_schema(item) for item in value]
    return value


def _schema_type_text(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "any"
    kind = str(schema.get("type", "any"))
    if kind == "array" and isinstance(schema.get("items"), dict):
        return f"array<{_schema_type_text(schema['items'])}>"
    return kind


def _flat_function_signatures(functions: list[dict[str, Any]]) -> str:
    """Render complete candidate definitions without a parameters wrapper."""

    rows = []
    for function in functions:
        parameters = function.get("parameters", {})
        properties = (
            parameters.get("properties", {})
            if isinstance(parameters, dict) else {}
        )
        required = set(parameters.get("required", [])) if isinstance(parameters, dict) else set()
        fields = []
        if isinstance(properties, dict):
            for key, schema in properties.items():
                marker = "" if key in required else "?"
                fields.append(f"{key}{marker}:{_schema_type_text(schema)}")
        signature = f"{function['name']}({', '.join(fields)})"
        description = str(function.get("description", "")).strip()
        rows.append(f"- {signature}: {description}" if description else f"- {signature}")
    return "\n".join(rows)


def build_messages(
    sample: BFCLSample,
    prompt_profile: str = "original",
    output_schema: str = "standard",
) -> list[dict[str, str]]:
    """Build model messages without consulting ``sample.ground_truth``."""

    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError(f"unknown prompt profile: {prompt_profile}")
    if output_schema not in OUTPUT_SCHEMAS:
        raise ValueError(f"unknown output schema: {output_schema}")
    if output_schema == "flat_name_v1" and prompt_profile not in {
        "compact_json_v3", "compact_json_dep_v4",
    }:
        raise ValueError(
            "flat_name_v1 requires compact_json_v3 or compact_json_dep_v4"
        )
    if prompt_profile == "compact_json_dep_v4" and output_schema != "flat_name_v1":
        raise ValueError("compact_json_dep_v4 requires output_schema=flat_name_v1")
    functions = (
        _standardize_schema(sample.functions)
        if prompt_profile in {
            "strict_json_v2", "compact_json_v3", "compact_json_dep_v4",
        }
        else sample.functions
    )
    functions = adapt_functions_for_output_schema(functions, output_schema)
    if output_schema == "flat_name_v1":
        definition_label = "Candidate tool signatures:"
        function_text = _flat_function_signatures(functions)
    else:
        definition_label = "Candidate function definitions (JSON):"
        function_text = json.dumps(
            functions, ensure_ascii=False, separators=(",", ":")
        )
    user_content = (
        "Original request:\n"
        f"{_render_question(sample.messages)}\n\n"
        f"{definition_label}\n"
        f"{function_text}\n\n"
        + {
            "original": "Return the tool-call JSON array now.",
            "strict_json_v2": STRICT_JSON_FINAL_CONTRACT,
            "compact_json_v3": COMPACT_JSON_FINAL_INSTRUCTION,
            "compact_json_dep_v4": COMPACT_JSON_FINAL_INSTRUCTION,
        }[prompt_profile]
    )
    return [
        {
            "role": "system",
            "content": {
                "original": SYSTEM_PROMPT,
                "strict_json_v2": STRICT_JSON_SYSTEM_PROMPT,
                "compact_json_v3": (
                    FLAT_NAME_SYSTEM_PROMPT
                    if output_schema == "flat_name_v1"
                    else COMPACT_JSON_SYSTEM_PROMPT
                ),
                "compact_json_dep_v4": FLAT_NAME_DEPENDENCY_SYSTEM_PROMPT,
            }[prompt_profile],
        },
        {"role": "user", "content": user_content},
    ]


def render_prompt(
    tokenizer: Any,
    sample: BFCLSample,
    prompt_profile: str = "original",
    output_schema: str = "standard",
) -> str:
    messages = build_messages(sample, prompt_profile, output_schema)
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception as error:
        # iLLaDA's native template accepts alternating user/assistant turns
        # but no system role. Preserve the instruction by folding it into the
        # first user turn; other template failures must remain visible.
        if (
            "System role not supported" not in str(error)
            or len(messages) < 2
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
        ):
            raise
        merged = [{
            "role": "user",
            "content": messages[0]["content"] + "\n\n" + messages[1]["content"],
        }, *messages[2:]]
        return tokenizer.apply_chat_template(
            merged, add_generation_prompt=True, tokenize=False
        )


__all__ = [
    "COMPACT_JSON_FINAL_INSTRUCTION", "COMPACT_JSON_SYSTEM_PROMPT",
    "FLAT_NAME_DEPENDENCY_SYSTEM_PROMPT", "FLAT_NAME_SYSTEM_PROMPT",
    "PROMPT_PROFILES", "STRICT_JSON_FINAL_CONTRACT", "STRICT_JSON_SYSTEM_PROMPT",
    "SYSTEM_PROMPT", "build_messages", "render_prompt",
]
