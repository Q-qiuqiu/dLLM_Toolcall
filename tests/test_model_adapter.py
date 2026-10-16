import json
from types import SimpleNamespace

import pytest
import torch

from model_adapter import (
    detect_model_family,
    load_model_and_tokenizer,
    resolve_mask_token_id,
    resolve_max_sequence_length,
    resolve_model_path,
)


def test_resolve_model_path_accepts_direct_and_single_nested_checkpoint(tmp_path):
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_path(direct) == direct.resolve()

    wrapper = tmp_path / "wrapper"
    nested = wrapper / "checkpoint"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_path(wrapper) == nested.resolve()


def test_detect_supported_model_families():
    assert detect_model_family({
        "model_type": "llada", "architectures": ["LLaDAModelLM"],
    }) == ("llada", "LLaDAModelLM")
    assert detect_model_family({
        "model_type": "illada", "architectures": ["ILLaDAForCausalLM"],
    }) == ("illada", "ILLaDAForCausalLM")
    with pytest.raises(ValueError, match="unsupported model architecture"):
        detect_model_family({
            "model_type": "llada", "architectures": ["LLaDAMoEModel"],
        })


class _Tokenizer:
    mask_token_id = None
    unk_token_id = 3
    unk_token = "<[UNK]>"

    def convert_tokens_to_ids(self, token):
        return {"<[MASK]>": 5}.get(token, self.unk_token_id)


def test_illada_mask_and_context_fallbacks():
    config = SimpleNamespace(mask_token_id=None, max_position_embeddings=8192)
    assert resolve_mask_token_id(config, _Tokenizer(), "illada") == 5
    assert resolve_max_sequence_length(config) == 8192


def test_configured_llada_mask_and_context_take_precedence():
    config = SimpleNamespace(mask_token_id=126336, max_sequence_length=4096)
    assert resolve_mask_token_id(config, _Tokenizer(), "llada") == 126336
    assert resolve_max_sequence_length(config) == 4096


def test_illada_dual_cache_is_rejected_before_loading_weights(tmp_path):
    checkpoint = tmp_path / "illada"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "illada",
        "architectures": ["ILLaDAForCausalLM"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Dual Cache is not supported"):
        load_model_and_tokenizer(
            model_path=checkpoint,
            fastdllm_root=tmp_path / "fastdllm",
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            cache_mode="dual",
        )
