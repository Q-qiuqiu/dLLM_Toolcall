from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import wave


ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC = ROOT / "models" / "src"
sys.path.insert(0, str(MODEL_SRC))

from benchmark_cold_start import _stats  # noqa: E402
from model_specs import MODEL_SPECS  # noqa: E402


def test_pinned_model_manifest_and_fixed_inputs() -> None:
    manifest = json.loads(
        (ROOT / "models" / "model_manifest.json").read_text(encoding="utf-8")
    )["models"]
    assert set(manifest) == set(MODEL_SPECS) == {"whisper", "vit", "blip"}
    for name, spec in MODEL_SPECS.items():
        row = manifest[name]
        assert row["revision"] == spec["revision"]
        assert row["repo_id"] == spec["repo_id"]
        assert row["file_count"] > 0
        assert row["total_size_bytes"] > 0
        assert Path(spec["local_dir"]).is_dir()
        assert Path(spec["input"]).is_file()

    with wave.open(str(MODEL_SPECS["whisper"]["input"]), "rb") as audio:
        assert audio.getframerate() == 16_000
        assert audio.getnchannels() == 1
        assert audio.getnframes() == 160_000


def test_cold_start_summary_statistics() -> None:
    result = _stats([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["mean"] == 2.5
    assert result["p50"] == 2.5
    assert math.isclose(result["p95"], 3.85)
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["std"] is not None
