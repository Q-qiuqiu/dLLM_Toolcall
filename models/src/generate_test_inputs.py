"""Create deterministic offline image/audio inputs for service benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from PIL import Image, ImageDraw

from model_specs import MODELS_ROOT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-wav",
        help="Optional speech WAV. It is converted/padded/truncated to 10 s at 16 kHz.",
    )
    args = parser.parse_args()
    output = MODELS_ROOT / "inputs"
    output.mkdir(parents=True, exist_ok=True)

    image_path = output / "test_image_224.png"
    yy, xx = np.mgrid[0:224, 0:224]
    pixels = np.stack((xx, yy, (xx + yy) // 2), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 55, 176, 170), outline="white", width=5)
    draw.ellipse((80, 75, 144, 139), fill=(240, 180, 30), outline="black", width=3)
    image.save(image_path)

    audio_path = output / "whisper_10s.wav"
    sample_rate = 16000
    samples = None
    source = "deterministic synthetic multi-tone audio"
    if args.source_wav:
        import soundfile as sf
        from scipy.signal import resample_poly
        samples, source_rate = sf.read(args.source_wav, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if source_rate != sample_rate:
            samples = resample_poly(samples, sample_rate, source_rate).astype(np.float32)
        source = str(Path(args.source_wav).resolve())
    if samples is None:
        timeline = np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
        envelope = np.minimum(1.0, np.minimum(timeline * 4, (10 - timeline) * 4))
        samples = 0.12 * envelope * (
            np.sin(2 * np.pi * 220 * timeline)
            + 0.5 * np.sin(2 * np.pi * 440 * timeline)
        )
    needed = sample_rate * 10
    samples = np.pad(samples[:needed], (0, max(0, needed - len(samples))))
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    manifest = {
        "image": {
            "path": str(image_path.resolve()), "size": [224, 224],
            "sha256": _sha256(image_path), "generation": "deterministic RGB gradient/shapes",
        },
        "audio": {
            "path": str(audio_path.resolve()), "sample_rate": sample_rate,
            "duration_s": 10.0, "sha256": _sha256(audio_path), "source": source,
        },
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
