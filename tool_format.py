"""Reversible model-output schemas for canonical tool calls."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable


OUTPUT_SCHEMAS = ("standard", "flat_name_v1")
RESERVED_FLAT_FIELDS = frozenset({"name"})


def _properties(function: dict[str, Any]) -> dict[str, Any]:
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    return properties if isinstance(properties, dict) else {}


def parameter_aliases(
    functions: Iterable[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Map model-facing flattened keys back to original parameter keys.

    Only collisions with the reserved outer ``name`` field are renamed.  The
    alias is chosen against all original keys, so the mapping is reversible
    even if a future tool already declares ``arg_name``.
    """

    result: dict[str, dict[str, str]] = {}
    for function in functions:
        tool_name = str(function.get("name", ""))
        properties = _properties(function)
        original_keys = {str(key) for key in properties}
        used: set[str] = set()
        aliases: dict[str, str] = {}
        for original in properties:
            original = str(original)
            model_key = original
            if model_key in RESERVED_FLAT_FIELDS:
                model_key = f"arg_{model_key}"
                while model_key in original_keys or model_key in used:
                    model_key = f"arg_{model_key}"
            aliases[model_key] = original
            used.add(model_key)
        result[tool_name] = aliases
    return result


def adapt_functions_for_output_schema(
    functions: Iterable[dict[str, Any]], output_schema: str
) -> list[dict[str, Any]]:
    """Return prompt-only function definitions with reversible flat aliases."""

    if output_schema not in OUTPUT_SCHEMAS:
        raise ValueError(f"unsupported output schema: {output_schema!r}")
    copied = deepcopy(list(functions))
    if output_schema == "standard":
        return copied
    aliases_by_tool = parameter_aliases(copied)
    for function in copied:
        tool_name = str(function.get("name", ""))
        aliases = aliases_by_tool.get(tool_name, {})
        original_to_model = {original: model for model, original in aliases.items()}
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            parameters["properties"] = {
                original_to_model.get(str(key), str(key)): value
                for key, value in properties.items()
            }
        required = parameters.get("required")
        if isinstance(required, list):
            parameters["required"] = [
                original_to_model.get(str(key), str(key)) for key in required
            ]
    return copied


def convert_flat_calls(
    value: Any,
    *,
    candidate_names: Iterable[str],
    functions: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None, bool]:
    """Convert valid flattened call objects to canonical name/arguments calls.

    Returns ``(calls, format_error, candidate_names_valid)``.  Parameter-schema
    correctness remains the responsibility of the existing AST evaluator.
    """

    if not isinstance(value, list):
        return [], "top_level_not_array", False
    functions = list(functions)
    candidates = set(candidate_names)
    aliases_by_tool = parameter_aliases(functions)
    calls: list[dict[str, Any]] = []
    candidate_names_valid = True
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return calls, f"call_{index}_not_object", False
        if "name" not in item:
            return calls, f"call_{index}_missing_name", False
        name = item["name"]
        if not isinstance(name, str):
            return calls, f"call_{index}_name_not_string", False
        if name not in candidates:
            candidate_names_valid = False
        function = next(
            (function for function in functions if function.get("name") == name),
            None,
        )
        if function is None:
            calls.append({
                "name": name,
                "arguments": {
                    str(key): value for key, value in item.items()
                    if key != "name"
                },
            })
            continue
        arguments, argument_error = normalize_flat_arguments(
            item,
            function=function,
            aliases=aliases_by_tool.get(name, {}),
        )
        if argument_error is not None:
            return calls, f"call_{index}_{argument_error}", False
        calls.append({"name": name, "arguments": arguments})
    return calls, None, candidate_names_valid


def _schema_value_ok(value: Any, schema: dict[str, Any]) -> bool:
    expected = str(schema.get("type", "any"))
    if expected in {"any", ""}:
        return True
    if expected == "string":
        return type(value) is str
    if expected == "integer":
        return type(value) is int
    if expected in {"float", "number"}:
        return type(value) in {int, float}
    if expected == "boolean":
        return type(value) is bool
    if expected in {"array", "tuple"}:
        if not isinstance(value, list):
            return False
        items = schema.get("items")
        return not isinstance(items, dict) or all(
            _schema_value_ok(item, items) for item in value
        )
    if expected in {"object", "dict"}:
        return isinstance(value, dict)
    return True


def normalize_flat_arguments(
    item: dict[str, Any],
    *,
    function: dict[str, Any],
    aliases: dict[str, str],
) -> tuple[dict[str, Any], str | None]:
    """Recover recognizable arguments from direct or wrongly nested fields.

    A nested mapping is treated as a compatibility wrapper when descendants
    resolve to declared parameter names with valid JSON types. Unknown extra
    fields are discarded from the canonical call; conflicting recognized
    values and missing required inputs remain hard failures.
    """

    properties = _properties(function)
    required_value = function.get("parameters", {}).get("required", [])
    required = {str(key) for key in required_value} if isinstance(required_value, list) else set()
    collected: dict[str, Any] = {}
    conflicts: set[str] = set()

    def visit(mapping: dict[str, Any], path: tuple[str, ...]) -> bool:
        found_here = False
        for raw_key, value in mapping.items():
            model_key = str(raw_key)
            if not path and model_key == "name":
                continue
            original_key = aliases.get(model_key, model_key)
            schema = properties.get(original_key)
            if isinstance(schema, dict) and _schema_value_ok(value, schema):
                if original_key in collected and collected[original_key] != value:
                    conflicts.add(original_key)
                else:
                    collected[original_key] = value
                found_here = True
                continue

            if isinstance(value, dict):
                nested_found = visit(value, path + (model_key,))
                if nested_found:
                    found_here = True
                    continue
                if not value and not properties:
                    found_here = True
                    continue

        return found_here

    visit(item, ())
    if conflicts:
        return {}, "conflicting_arguments:" + ",".join(sorted(conflicts))
    missing = sorted(required - collected.keys())
    if missing:
        return {}, "missing_required_arguments:" + ",".join(missing)
    return collected, None


def model_call_items(text: str) -> list[Any]:
    """Return the unconverted JSON array items for auditable result records."""

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


__all__ = [
    "OUTPUT_SCHEMAS", "RESERVED_FLAT_FIELDS",
    "adapt_functions_for_output_schema", "convert_flat_calls",
    "model_call_items", "normalize_flat_arguments", "parameter_aliases",
]
