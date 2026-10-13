"""Externally benchmark repeated real process-level cold starts for one model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from model_specs import MODEL_SPECS


HERE = Path(__file__).resolve().parent


def _http_json(url: str, payload: dict[str, Any] | None = None,
               timeout: float = 5.0) -> dict[str, Any]:
    # This runner only talks to its loopback child service.  Explicitly bypass
    # cluster-wide HTTP(S)/SOCKS proxy variables; otherwise urllib may route
    # 127.0.0.1 to the proxy when NO_PROXY is not configured in the parent.
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def _gpu_query(index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi", f"--id={index}",
        "--query-gpu=name,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    value = subprocess.check_output(command, text=True).strip().splitlines()[0]
    name, driver, total, used = [part.strip() for part in value.split(",")]
    return {
        "name": name, "driver_version": driver,
        "total_mb": float(total), "used_mb": float(used),
    }


def _wait_ready(port: int, process: subprocess.Popen, timeout_s: float):
    deadline = time.perf_counter() + timeout_s
    last_error = None
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited before Ready, code={process.returncode}")
        try:
            value = _http_json(f"http://127.0.0.1:{port}/health", timeout=1)
            if value.get("ready"):
                return value
        except (URLError, TimeoutError, ConnectionError) as error:
            last_error = error
        time.sleep(0.05)
    raise TimeoutError(f"Ready timeout after {timeout_s}s; last error: {last_error}")


def _stop(port: int, process: subprocess.Popen, timeout_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        _http_json(f"http://127.0.0.1:{port}/shutdown", {}, timeout=5)
    except Exception:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_gpu_release(index: int, baseline_mb: float, timeout_s: float) -> float:
    deadline = time.perf_counter() + timeout_s
    last = float("inf")
    while time.perf_counter() < deadline:
        last = _gpu_query(index)["used_mb"]
        if last <= baseline_mb + 64:
            return last
        time.sleep(0.25)
    raise TimeoutError(
        f"GPU memory did not return near baseline: baseline={baseline_mb}, last={last} MiB"
    )


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] if lo == hi else (
        values[lo] + (values[hi] - values[lo]) * (rank - lo)
    )


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def _flatten_metrics(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    simple = (
        "external_spawn_to_ready_s", "external_spawn_to_process_entry_s",
        "shutdown_and_gpu_release_s", "gpu_baseline_used_mb",
        "gpu_after_ready_used_mb", "gpu_after_exit_used_mb",
    )
    for key in simple:
        metrics[key] = [float(row[key]) for row in rows]
    stage_names = sorted({
        key for row in rows for key in row["health"]["startup_stages"]
    })
    for key in stage_names:
        metrics[f"startup_{key}"] = [
            float(row["health"]["startup_stages"][key]) for row in rows
        ]
    memory_names = sorted({
        key for row in rows for key, value in row["health"]["memory"].items()
        if value is not None
    })
    for key in memory_names:
        metrics[f"ready_memory_{key}"] = [
            float(row["health"]["memory"][key]) for row in rows
        ]
    metrics["first_inference_gpu_s"] = [
        float(row["health"]["self_test"]["gpu_inference_time_s"])
        for row in rows
    ]
    metrics["first_inference_total_s"] = [
        float(row["health"]["self_test"]["service_total_time_s"])
        for row in rows
    ]
    for key in ("gpu_inference_time_s", "preprocess_time_s", "service_total_time_s", "client_roundtrip_s"):
        metrics[f"steady_{key}"] = [
            float(inference[key])
            for row in rows for inference in row["steady_inferences"]
        ]
    return metrics


def _write_summary(output: Path, model: str, rows: list[dict[str, Any]]) -> None:
    metrics = _flatten_metrics(rows)
    summary = {
        "model": model, "completed_cold_starts": len(rows),
        "os_file_cache_cleared": False,
        "metrics": {name: _stats(values) for name, values in metrics.items()},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    csv_rows = [
        {"model": model, "metric": name, **_stats(values)}
        for name, values in metrics.items()
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].hist(metrics["external_spawn_to_ready_s"], bins=min(15, len(rows)))
        axes[0].set_title("Process start to Ready"); axes[0].set_xlabel("seconds")
        axes[1].hist(metrics["steady_gpu_inference_time_s"], bins=min(15, max(1, len(metrics["steady_gpu_inference_time_s"]))))
        axes[1].set_title("Ready-state model inference"); axes[1].set_xlabel("seconds")
        figure.suptitle(model); figure.tight_layout()
        figure.savefig(output / "timing_distributions.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=30,
                        help="Total desired cold starts; supports --resume.")
    parser.add_argument("--steady-requests", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--gpu-release-timeout-s", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.steady_requests < 1:
        raise ValueError("repeats and steady-requests must be positive")
    spec = MODEL_SPECS[args.model]
    if not Path(spec["local_dir"]).is_dir() or not Path(spec["input"]).is_file():
        raise FileNotFoundError("download model and generate fixed inputs first")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"; logs.mkdir(exist_ok=True)
    runs_path = output / "cold_start_runs.jsonl"
    rows = []
    if runs_path.exists():
        if not args.resume:
            raise FileExistsError(f"use --resume or choose a new output: {runs_path}")
        rows = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()]
    gpu = _gpu_query(args.gpu_index)
    # Importing torch in the external runner is outside every measured service
    # lifecycle.  It records the build's CUDA version without adding work to
    # the child service's spawn-to-Ready interval.
    import torch
    port = args.port or int(spec["default_port"])
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model": args.model, "repo_id": spec["repo_id"],
        "revision": spec["revision"], "local_dir": str(Path(spec["local_dir"]).resolve()),
        "fixed_input": str(Path(spec["input"]).resolve()),
        "semantic_note": spec["semantic_note"],
        "python": sys.version, "platform": platform.platform(),
        "software": {
            "torch": torch.__version__,
            "torch_compiled_cuda": torch.version.cuda,
            "transformers": package_version("transformers"),
        },
        "gpu_index": args.gpu_index, "gpu": gpu,
        "dtype": args.dtype, "steady_requests_per_cold_start": args.steady_requests,
        "process_level_cold_start": True, "os_file_cache_cleared": False,
        "omtp_connected": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    script = HERE / f"serve_{args.model}.py"
    for run_index in range(len(rows), args.repeats):
        baseline = _gpu_query(args.gpu_index)
        log_path = logs / f"run_{run_index + 1:03d}.log"
        service_log = logs / f"run_{run_index + 1:03d}_service.jsonl"
        command = [
            sys.executable, str(script), "--port", str(port),
            "--device", "cuda", "--dtype", args.dtype,
            "--log-file", str(service_log),
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
        })
        started_perf = time.perf_counter(); started_epoch = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command, stdout=log_handle, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True,
            )
        try:
            health = _wait_ready(port, process, args.ready_timeout_s)
            ready_s = time.perf_counter() - started_perf
            after_ready = _gpu_query(args.gpu_index)
            steady = []
            for _ in range(args.steady_requests):
                request_started = time.perf_counter()
                value = _http_json(f"http://127.0.0.1:{port}/infer", {}, timeout=120)
                value["client_roundtrip_s"] = time.perf_counter() - request_started
                steady.append(value)
            shutdown_started = time.perf_counter()
            _stop(port, process)
            after_exit = _wait_gpu_release(
                args.gpu_index, baseline["used_mb"], args.gpu_release_timeout_s
            )
            row = {
                "run_index": run_index + 1, "success": True,
                "external_spawn_to_ready_s": ready_s,
                "external_spawn_to_process_entry_s": (
                    float(health["process_entry_epoch_s"]) - started_epoch
                ),
                "shutdown_and_gpu_release_s": time.perf_counter() - shutdown_started,
                "gpu_baseline_used_mb": baseline["used_mb"],
                "gpu_after_ready_used_mb": after_ready["used_mb"],
                "gpu_after_exit_used_mb": after_exit,
                "health": health, "steady_inferences": steady,
                "service_log": str(service_log.resolve()),
            }
            with runs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            _write_summary(output, args.model, rows)
            print(
                f"[{run_index + 1}/{args.repeats}] {args.model} "
                f"Ready={ready_s:.3f}s steady={statistics.mean(x['gpu_inference_time_s'] for x in steady):.4f}s",
                flush=True,
            )
        except Exception:
            _stop(port, process)
            raise
    print(f"Completed {len(rows)} cold starts: {output.resolve()}")


if __name__ == "__main__":
    main()
