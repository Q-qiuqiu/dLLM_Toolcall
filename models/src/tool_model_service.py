"""Local-only HTTP service for one real tool model.

The service does not listen until the model has loaded and one real inference
has completed successfully.  It has no OMTP or LLaDA integration.
"""

from __future__ import annotations

PROCESS_ENTRY_PERF_S = __import__("time").perf_counter()
PROCESS_ENTRY_EPOCH_S = __import__("time").time()

import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any

from model_specs import MODEL_SPECS


class ToolModelRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.spec = MODEL_SPECS[args.model]
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.device: Any = None
        self.dtype: Any = None
        self.stages: dict[str, float] = {}
        self.memory: dict[str, float | None] = {}
        self.self_test: dict[str, Any] | None = None
        self.inference_count = 0
        self.log_path = Path(args.log_file).resolve() if args.log_file else None

    def log(self, event: str, **values: Any) -> None:
        row = {
            "event": event, "epoch_s": time.time(),
            "elapsed_from_process_entry_s": time.perf_counter() - PROCESS_ENTRY_PERF_S,
            **values,
        }
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _measure(self, name: str, function):
        started = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - started
        self.stages[name] = elapsed
        self.log("stage_complete", stage=name, duration_s=elapsed)
        return value

    def import_dependencies(self) -> None:
        def load():
            import torch
            import transformers
            import numpy as np
            import soundfile as sf
            from PIL import Image
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            import uvicorn
            self.torch = torch
            self.transformers = transformers
            self.np = np
            self.sf = sf
            self.Image = Image
            self.FastAPI = FastAPI
            self.HTTPException = HTTPException
            self.BaseModel = BaseModel
            self.uvicorn = uvicorn
        self._measure("dependency_import_s", load)

    def initialize_device(self) -> None:
        torch = self.torch
        if self.args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
            torch.cuda.init()
            self.device = torch.device("cuda:0")
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self.device = torch.device("cpu")
        if self.args.dtype == "float16" and self.device.type == "cpu":
            raise ValueError("float16 service requires CUDA; use --dtype float32 on CPU")
        self.dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.args.dtype]

    def _local_path(self) -> str:
        path = Path(self.spec["local_dir"]).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return str(path)

    def load_processor(self) -> None:
        local = self._local_path()
        t = self.transformers
        if self.args.model == "whisper":
            self.processor = t.AutoProcessor.from_pretrained(local, local_files_only=True)
        elif self.args.model == "vit":
            self.processor = t.AutoImageProcessor.from_pretrained(local, local_files_only=True)
        else:
            self.processor = t.BlipProcessor.from_pretrained(local, local_files_only=True)

    def load_model_cpu(self) -> None:
        local = self._local_path()
        t = self.transformers
        kwargs = {"local_files_only": True, "torch_dtype": self.dtype}
        if self.args.model == "whisper":
            self.model = t.WhisperForConditionalGeneration.from_pretrained(local, **kwargs)
        elif self.args.model == "vit":
            self.model = t.ViTForImageClassification.from_pretrained(local, **kwargs)
        else:
            self.model = t.BlipForConditionalGeneration.from_pretrained(local, **kwargs)
        self.model.eval()

    def move_model(self) -> None:
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _memory_snapshot(self) -> dict[str, float | None]:
        if self.device.type != "cuda":
            return {
                "allocated_mb": None, "reserved_mb": None,
                "peak_allocated_mb": None, "device_used_mb": None,
            }
        torch = self.torch
        free, total = torch.cuda.mem_get_info(self.device)
        scale = 1024 ** 2
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / scale,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / scale,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / scale,
            "device_used_mb": (total - free) / scale,
        }

    def _prepare(self, input_path: Path) -> dict[str, Any]:
        if self.args.model == "whisper":
            audio, rate = self.sf.read(input_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate != 16000:
                raise ValueError(f"Whisper input must be 16 kHz, got {rate}")
            values = self.processor(
                audio, sampling_rate=rate, return_tensors="pt",
                return_attention_mask=True,
            )
            return {
                key: value.to(
                    device=self.device,
                    dtype=self.dtype if value.is_floating_point() else value.dtype,
                )
                for key, value in values.items()
            }
        image = self.Image.open(input_path).convert("RGB")
        values = self.processor(images=image, return_tensors="pt")
        return {
            key: value.to(
                device=self.device,
                dtype=self.dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in values.items()
        }

    def infer(self, input_path: str | None = None) -> dict[str, Any]:
        path = Path(input_path or self.spec["input"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        total_started = time.perf_counter()
        prepare_started = time.perf_counter()
        inputs = self._prepare(path)
        preprocess_s = time.perf_counter() - prepare_started
        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            if self.args.model == "whisper":
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.args.max_new_tokens,
                    language="en", task="transcribe",
                )
                output: Any = {
                    "text": self.processor.batch_decode(
                        output_ids, skip_special_tokens=True
                    )[0]
                }
            elif self.args.model == "vit":
                logits = self.model(**inputs).logits
                index = int(logits.argmax(dim=-1).item())
                output = {
                    "class_id": index,
                    "label": str(self.model.config.id2label.get(index, index)),
                }
            else:
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.args.max_new_tokens
                )
                output = {
                    "caption": self.processor.decode(
                        output_ids[0], skip_special_tokens=True
                    )
                }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_s = time.perf_counter() - inference_started
        self.inference_count += 1
        return {
            "model": self.args.model,
            "input_path": str(path),
            "preprocess_time_s": preprocess_s,
            "gpu_inference_time_s": inference_s,
            "service_total_time_s": time.perf_counter() - total_started,
            "output": output,
            "inference_count": self.inference_count,
        }

    def startup(self) -> None:
        self.log("process_entry", pid=os.getpid())
        self.import_dependencies()
        self._measure("cuda_initialization_s", self.initialize_device)
        self._measure("processor_load_s", self.load_processor)
        self._measure("model_load_cpu_s", self.load_model_cpu)
        self._measure("model_to_device_s", self.move_model)
        self.self_test = self._measure("first_test_inference_s", self.infer)
        self.memory = self._memory_snapshot()
        self.stages["service_ready_from_process_entry_s"] = (
            time.perf_counter() - PROCESS_ENTRY_PERF_S
        )
        self.log(
            "ready", stages=self.stages, memory=self.memory,
            self_test_output=self.self_test["output"],
        )

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.self_test is not None,
            "model": self.args.model,
            "repo_id": self.spec["repo_id"],
            "revision": self.spec["revision"],
            "model_dir": str(Path(self.spec["local_dir"]).resolve()),
            "device": str(self.device), "dtype": self.args.dtype,
            "software": {
                "torch": self.torch.__version__,
                "torch_compiled_cuda": self.torch.version.cuda,
                "transformers": self.transformers.__version__,
            },
            "pid": os.getpid(),
            "process_entry_epoch_s": PROCESS_ENTRY_EPOCH_S,
            "startup_stages": self.stages,
            "memory": self.memory,
            "self_test": self.self_test,
            "semantic_note": self.spec["semantic_note"],
        }

    def app(self):
        app = self.FastAPI(title=f"Local {self.args.model} tool service")
        runtime = self

        @app.get("/health")
        def health():
            return runtime.health()

        @app.post("/infer")
        def infer(payload: dict[str, Any] | None = None):
            try:
                return runtime.infer((payload or {}).get("input_path"))
            except Exception as error:
                raise runtime.HTTPException(status_code=500, detail=str(error)) from error

        @app.post("/shutdown")
        def shutdown():
            runtime.log("shutdown_requested")
            threading.Timer(
                0.1, lambda: os.kill(os.getpid(), signal.SIGTERM)
            ).start()
            return {"stopping": True, "pid": os.getpid()}

        return app


def parse_args(forced_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    if forced_model is None:
        parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--log-file")
    args = parser.parse_args()
    if forced_model is not None:
        args.model = forced_model
    if args.port is None:
        args.port = int(MODEL_SPECS[args.model]["default_port"])
    return args


def main(forced_model: str | None = None) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args(forced_model)
    runtime = ToolModelRuntime(args)
    runtime.startup()
    runtime.uvicorn.run(
        runtime.app(), host=args.host, port=args.port,
        log_level="warning", access_log=False,
    )
    runtime.log("service_exited")


if __name__ == "__main__":
    main()
