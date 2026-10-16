"""Model-family detection and runtime compatibility for diffusion LMs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


@dataclass(frozen=True)
class ModelRuntimeSpec:
    requested_path: str
    resolved_path: str
    family: str
    architecture: str
    model_type: str
    mask_token_id: int
    eos_token_id: int | None
    pad_token_id: int | None
    vocab_size: int
    max_sequence_length: int
    supports_dual_cache: bool
    source_revision: str | None
    config_sha256: str
    weight_index_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_path(path: str | Path) -> Path:
    """Resolve a checkpoint directory or a wrapper with one checkpoint child."""

    requested = Path(path).expanduser().resolve()
    if (requested / "config.json").is_file():
        return requested
    if not requested.is_dir():
        raise FileNotFoundError(f"model path is not a directory: {requested}")
    candidates = sorted(
        child for child in requested.iterdir()
        if child.is_dir() and (child / "config.json").is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"expected config.json in {requested} or exactly one checkpoint child; "
        f"found {len(candidates)} candidates"
    )


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"model config must be a JSON object: {path / 'config.json'}")
    return value


def _sha256_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_revision(path: Path) -> str | None:
    metadata = path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not metadata.is_file():
        return None
    first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    if len(first_line) == 40 and all(char in "0123456789abcdef" for char in first_line.lower()):
        return first_line
    return None


def detect_model_family(config: dict[str, Any]) -> tuple[str, str]:
    architectures = config.get("architectures") or []
    architecture = str(architectures[0]) if architectures else ""
    model_type = str(config.get("model_type", ""))
    if architecture == "LLaDAModelLM":
        return "llada", architecture
    if model_type == "illada" or architecture == "ILLaDAForCausalLM":
        return "illada", architecture
    raise ValueError(
        "unsupported model architecture: "
        f"model_type={model_type!r}, architectures={architectures!r}"
    )


def _token_id(tokenizer: Any, token: str) -> int | None:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or not isinstance(value, int) or value < 0:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and value == unk and token != getattr(tokenizer, "unk_token", None):
        return None
    return value


def resolve_mask_token_id(config: Any, tokenizer: Any, family: str) -> int:
    configured = getattr(config, "mask_token_id", None)
    if configured is not None:
        return int(configured)
    tokenizer_mask = getattr(tokenizer, "mask_token_id", None)
    if tokenizer_mask is not None:
        return int(tokenizer_mask)
    candidates = ("<[MASK]>", "[MASK]") if family == "illada" else ("[MASK]",)
    for token in candidates:
        value = _token_id(tokenizer, token)
        if value is not None:
            return value
    raise ValueError(f"could not resolve MASK token id for model family {family!r}")


def resolve_max_sequence_length(config: Any) -> int:
    for name in ("max_sequence_length", "max_position_embeddings"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("model config has no usable maximum sequence length")


def load_model_and_tokenizer(
    *, model_path: str | Path, fastdllm_root: str | Path,
    dtype: torch.dtype, device: torch.device, cache_mode: str = "none",
    device_map: str | None = None, max_memory: dict[Any, Any] | None = None,
) -> tuple[Any, Any, ModelRuntimeSpec]:
    """Load a supported checkpoint with its native tokenizer and model class."""

    from transformers import AutoModel, AutoTokenizer

    resolved = resolve_model_path(model_path)
    raw_config = _read_config(resolved)
    family, architecture = detect_model_family(raw_config)
    if cache_mode not in {"none", "dual"}:
        raise ValueError(f"unsupported cache mode: {cache_mode!r}")
    if cache_mode == "dual" and family != "llada":
        raise ValueError(
            f"Dual Cache is not supported for model family {family!r}; "
            "use --cache-mode none"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=True, local_files_only=True
    )
    placement = {"device_map": device_map, "max_memory": max_memory} if device_map is not None else {}

    if family == "llada":
        llada_dir = Path(fastdllm_root).resolve() / "v1" / "llada"
        if not (llada_dir / "model" / "modeling_llada.py").exists():
            raise FileNotFoundError(
                f"Fast-dLLM v1 LLaDA implementation not found: {llada_dir}"
            )
        if str(llada_dir) not in sys.path:
            sys.path.insert(0, str(llada_dir))
        from model.modeling_llada import LLaDAModelLM

        model = LLaDAModelLM.from_pretrained(
            resolved, torch_dtype=dtype, local_files_only=True,
            **placement,
        )
        supports_dual_cache = True
    else:
        model = AutoModel.from_pretrained(
            resolved, trust_remote_code=True, torch_dtype=dtype,
            local_files_only=True,
            **placement,
        )
        supports_dual_cache = False

    model = (model.to(device) if device_map is None else model).eval()
    mask_token_id = resolve_mask_token_id(model.config, tokenizer, family)
    max_sequence_length = resolve_max_sequence_length(model.config)
    vocab_size = int(getattr(model.config, "vocab_size"))
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError(
            f"MASK token id {mask_token_id} is outside vocab size {vocab_size}"
        )
    spec = ModelRuntimeSpec(
        requested_path=str(Path(model_path).expanduser()),
        resolved_path=str(resolved),
        family=family,
        architecture=architecture,
        model_type=str(raw_config.get("model_type", "")),
        mask_token_id=mask_token_id,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        supports_dual_cache=supports_dual_cache,
        source_revision=_download_revision(resolved),
        config_sha256=_sha256_if_file(resolved / "config.json") or "",
        weight_index_sha256=_sha256_if_file(
            resolved / "model.safetensors.index.json"
        ),
    )
    setattr(model, "_dllm_toolcall_runtime_spec", spec)
    return model, tokenizer, spec


def runtime_spec(model: Any) -> ModelRuntimeSpec:
    value = getattr(model, "_dllm_toolcall_runtime_spec", None)
    if not isinstance(value, ModelRuntimeSpec):
        raise ValueError("model was not loaded through the dLLM_Toolcall adapter")
    return value


__all__ = [
    "ModelRuntimeSpec", "detect_model_family", "load_model_and_tokenizer",
    "resolve_mask_token_id", "resolve_max_sequence_length",
    "resolve_model_path", "runtime_spec",
]
