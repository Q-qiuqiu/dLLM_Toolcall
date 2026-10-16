from bfcl_data import BFCLSample
from evaluation import bfcl_compatible_ast_match, evaluate_final_calls
from tool_parser import extract_partial_names, parse_tool_calls


def _sample():
    return BFCLSample(
        sample_id="parallel_multiple_test",
        messages=[{"role": "user", "content": "weather twice"}],
        functions=[
            {
                "name": "get_weather",
                "description": "weather",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        ground_truth=[
            {"get_weather": {"city": ["Beijing"]}},
            {"get_weather": {"city": ["Shanghai"]}},
        ],
    )


def test_strict_json_and_conservative_partial_name_context():
    candidates = ["get_weather"]
    valid = '[{"name":"get_weather","arguments":{"city":"Beijing"}}]'
    assert parse_tool_calls(valid, candidates).valid
    assert extract_partial_names(valid, candidates) == ["get_weather"]
    assert extract_partial_names(
        '{"arguments":{"note":"get_weather"}}', candidates
    ) == []
    assert extract_partial_names('{"name":"get_weather', candidates) == []
    assert not parse_tool_calls("```json\n" + valid + "\n```", candidates).valid


def test_json_syntax_format_and_candidate_validity_are_separate():
    parsed = parse_tool_calls(
        '[{"name":"invented","arguments":{}}]', ["get_weather"]
    )
    assert parsed.json_valid
    assert parsed.format_valid
    assert not parsed.candidate_names_valid
    assert not parsed.valid
    assert parsed.calls == [{"name": "invented", "arguments": {}}]


def test_flat_name_schema_converts_multiple_arguments_and_duplicates():
    functions = [{
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "date": {"type": "string"},
            },
        },
    }]
    parsed = parse_tool_calls(
        '[{"name":"get_weather","location":"Beijing","date":"today"},'
        '{"name":"get_weather","location":"Shanghai","date":"tomorrow"}]',
        ["get_weather"],
        output_schema="flat_name_v1",
        functions=functions,
    )
    assert parsed.valid
    assert parsed.calls == [
        {
            "name": "get_weather",
            "arguments": {"location": "Beijing", "date": "today"},
        },
        {
            "name": "get_weather",
            "arguments": {"location": "Shanghai", "date": "tomorrow"},
        },
    ]


def test_flat_name_schema_reverses_name_parameter_alias():
    functions = [{
        "name": "book_hotel",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "name": {"type": "string"},
            },
        },
    }]
    parsed = parse_tool_calls(
        '[{"name":"book_hotel","date":"tomorrow","arg_name":"Hilton"}]',
        ["book_hotel"],
        output_schema="flat_name_v1",
        functions=functions,
    )
    assert parsed.valid
    assert parsed.calls == [{
        "name": "book_hotel",
        "arguments": {"date": "tomorrow", "name": "Hilton"},
    }]


def test_flat_name_schema_recovers_common_wrong_nesting():
    functions = [{
        "name": "edit",
        "parameters": {
            "type": "object",
            "properties": {
                "input_1": {"type": "string"},
                "input_2": {"type": "string"},
            },
            "required": ["input_1", "input_2"],
        },
    }]
    for text in (
        '[{"name":"edit","parameters":{"input_1":"x","input_2":"y"}}]',
        '[{"name":"edit","args":{"input_1":"x","input_2":"y"}}]',
        '[{"name":"edit","input_1":{"input_1":"x","input_2":"y"}}]',
    ):
        parsed = parse_tool_calls(
            text, ["edit"], output_schema="flat_name_v1", functions=functions
        )
        assert parsed.valid
        assert parsed.calls == [{
            "name": "edit", "arguments": {"input_1": "x", "input_2": "y"}
        }]


def test_flat_name_schema_rejects_missing_invalid_and_conflicting_arguments():
    functions = [{
        "name": "edit",
        "parameters": {
            "type": "object",
            "properties": {"input_1": {"type": "string"}},
            "required": ["input_1"],
        },
    }]
    for text in (
        '[{"name":"edit"}]',
        '[{"name":"edit","input_1":42}]',
        '[{"name":"edit","input_1":"x","args":{"input_1":"y"}}]',
    ):
        parsed = parse_tool_calls(
            text, ["edit"], output_schema="flat_name_v1", functions=functions
        )
        assert parsed.json_valid and not parsed.format_valid


def test_duplicate_calls_and_unordered_ast_match():
    calls = [
        {"name": "get_weather", "arguments": {"city": "Shanghai"}},
        {"name": "get_weather", "arguments": {"city": "Beijing"}},
    ]
    sample = _sample()
    assert bfcl_compatible_ast_match(calls, sample)
    metrics = evaluate_final_calls(sample, calls)
    assert metrics["tool_set_exact_match"]
    assert metrics["tool_call_count_match"]
    assert metrics["tool_name_f1"] == 1.0


def test_set_match_does_not_imply_call_count_match():
    sample = _sample()
    one_call = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
    metrics = evaluate_final_calls(sample, one_call)
    assert metrics["tool_set_exact_match"]
    assert not metrics["tool_call_count_match"]
    assert not metrics["bfcl_ast_compatible_match"]


def test_bfcl_python_float_schema_accepts_integer_json_number():
    sample = BFCLSample(
        sample_id="float",
        messages=[],
        functions=[
            {
                "name": "area",
                "parameters": {
                    "type": "dict",
                    "properties": {"radius": {"type": "float"}},
                    "required": ["radius"],
                },
            }
        ],
        ground_truth=[{"area": {"radius": [5.0]}}],
    )
    assert bfcl_compatible_ast_match(
        [{"name": "area", "arguments": {"radius": 5}}], sample
    )
