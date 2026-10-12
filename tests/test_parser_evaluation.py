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
