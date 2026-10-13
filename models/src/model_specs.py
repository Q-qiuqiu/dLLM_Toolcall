"""Pinned local tool-model specifications."""

from __future__ import annotations

from pathlib import Path


MODELS_ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = {
    "whisper": {
        "repo_id": "openai/whisper-tiny",
        "revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        "local_dir": MODELS_ROOT / "whisper-tiny",
        "input": MODELS_ROOT / "inputs" / "whisper_10s.wav",
        "default_port": 18101,
        "taskbench_tools": {
            "data_huggingface": ["Automatic Speech Recognition"],
            "data_multimedia": ["Audio-to-Text"],
        },
        "semantic_note": "Speech recognition / audio transcription.",
        "allow_patterns": [
            "*.json", "*.txt", "*.model", "model.safetensors",
        ],
    },
    "vit": {
        "repo_id": "google/vit-base-patch16-224",
        "revision": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
        "local_dir": MODELS_ROOT / "vit-base-patch16-224",
        "input": MODELS_ROOT / "inputs" / "test_image_224.png",
        "default_port": 18102,
        "taskbench_tools": {"data_huggingface": ["Image Classification"]},
        "semantic_note": "ImageNet image classification.",
        "allow_patterns": ["*.json", "model.safetensors"],
    },
    "blip": {
        "repo_id": "Salesforce/blip-image-captioning-base",
        "revision": "82a37760796d32b1411fe092ab5d4e227313294b",
        "local_dir": MODELS_ROOT / "blip-image-captioning-base",
        "input": MODELS_ROOT / "inputs" / "test_image_224.png",
        "default_port": 18103,
        "taskbench_tools": {"data_huggingface": ["Image-to-Text"]},
        "semantic_note": (
            "Image captioning only. It is not a complete implementation of "
            "TaskBench Multimedia Image-to-Text, whose primary semantics are OCR."
        ),
        "allow_patterns": ["*.json", "*.txt", "pytorch_model.bin"],
    },
}


__all__ = ["MODELS_ROOT", "MODEL_SPECS"]
