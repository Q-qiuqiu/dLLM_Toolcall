import pytest

from response_prefill import JSON_PREFILL_TEXT, resolve_response_prefill


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert text == JSON_PREFILL_TEXT
        assert not add_special_tokens
        return [101, 102, 103]


def test_native_response_mode_has_no_prefill_for_any_family():
    value = resolve_response_prefill(
        FakeTokenizer(), model_family="llada", mode="native"
    )
    assert value.text == "" and value.token_ids == ()


def test_illada_json_prefill_contains_structure_only():
    value = resolve_response_prefill(
        FakeTokenizer(), model_family="illada", mode="json_prefill"
    )
    assert value.text == '[{"name":"'
    assert value.token_ids == (101, 102, 103)


def test_json_prefill_rejects_non_illada_model():
    with pytest.raises(ValueError, match="iLLaDA-only"):
        resolve_response_prefill(
            FakeTokenizer(), model_family="llada", mode="json_prefill"
        )
