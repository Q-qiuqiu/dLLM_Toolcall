import json

from bfcl_data import BFCLSample, iter_samples, validate_dataset
from prompting import build_messages


def test_loads_official_nested_question_and_joins_answer(tmp_path):
    questions = tmp_path / "questions.json"
    answers = tmp_path / "answers.json"
    questions.write_text(
        json.dumps(
            {
                "id": "parallel_multiple_0",
                "question": [[{"role": "user", "content": "Do both."}]],
                "function": [
                    {
                        "name": "a",
                        "description": "A",
                        "parameters": {"type": "dict", "properties": {}},
                    },
                    {
                        "name": "b",
                        "description": "B",
                        "parameters": {"type": "dict", "properties": {}},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps(
            {
                "id": "parallel_multiple_0",
                "ground_truth": [{"a": {}}, {"b": {}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample = next(iter_samples(questions, answers))
    assert sample.messages == [{"role": "user", "content": "Do both."}]
    assert sample.candidate_names == ["a", "b"]
    assert sample.ground_truth_tool_types == {"a", "b"}
    assert validate_dataset(questions, answers)["sample_count"] == 1


def test_prompt_does_not_read_ground_truth():
    sample = BFCLSample(
        sample_id="x",
        messages=[{"role": "user", "content": "Call what is needed"}],
        functions=[
            {
                "name": "candidate",
                "description": "desc",
                "parameters": {"type": "dict", "properties": {}},
            }
        ],
        ground_truth=[{"SECRET_GROUND_TRUTH_NAME": {}}],
    )
    rendered = json.dumps(build_messages(sample))
    assert "candidate" in rendered
    assert "SECRET_GROUND_TRUTH_NAME" not in rendered
