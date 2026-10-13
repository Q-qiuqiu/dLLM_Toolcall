# Independent local tool-model cold-start benchmark

This directory is intentionally independent from LLaDA/OMTP.  No prediction
event starts a service, and no TaskBench ground truth is used here.

Pinned weights are stored directly under `models/`; lifecycle and benchmark
programs are under `models/src/`.  `model_manifest.json` records revisions,
sizes, and SHA256 hashes.  BLIP is image captioning and is **not** labelled as a
complete implementation of Multimedia `Image-to-Text` (OCR semantics).

Generate/verify assets:

```bash
cd /data/home/yzx/dLLM_Toolcall
/home/yzx/miniconda3/envs/llada/bin/python models/src/download_models.py --verify-only
/home/yzx/miniconda3/envs/llada/bin/python models/src/generate_test_inputs.py
```

Start one service on a dedicated RTX 3090:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/yzx/miniconda3/envs/llada/bin/python models/src/serve_whisper.py
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/yzx/miniconda3/envs/llada/bin/python models/src/serve_vit.py
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/yzx/miniconda3/envs/llada/bin/python models/src/serve_blip.py
```

Each process loads only local files, initializes CUDA, runs one real test
inference, and only then starts HTTP.  `/health` therefore means Ready, not
merely that a port exists.  `/infer` measures preprocessing, synchronized model
inference, and total service latency separately.  Stop with, for example:

```bash
/home/yzx/miniconda3/envs/llada/bin/python models/src/stop_service.py --port 18101
```

Run two validation starts first, then an independent formal 30-start run:

```bash
/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model whisper --gpu-index 0 --repeats 2 --steady-requests 3 \
  --output-dir models/results/whisper_smoke2

/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model whisper --gpu-index 0 --repeats 30 --steady-requests 3 \
  --output-dir models/results/whisper_cold30
```

Use `vit` and `blip` in the same command.  A run is process-level cold: the
previous process exits and GPU memory must return close to its pre-run baseline.
The OS filesystem cache is deliberately not cleared.  Partial runs can continue
with `--resume` and the same total `--repeats`.

The complete recommended sequence is:

```bash
cd /data/home/yzx/dLLM_Toolcall

# Functional smoke: two fresh processes per model.
/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model whisper --gpu-index 0 --repeats 2 --steady-requests 3 \
  --output-dir models/results/whisper_smoke2
/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model vit --gpu-index 0 --repeats 2 --steady-requests 3 \
  --output-dir models/results/vit_smoke2
/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model blip --gpu-index 0 --repeats 2 --steady-requests 3 \
  --output-dir models/results/blip_smoke2

# Formal measurement: use new output directories, not the smoke directories.
/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model whisper --gpu-index 0 --repeats 30 --steady-requests 3 \
  --output-dir models/results/whisper_cold30
/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model vit --gpu-index 0 --repeats 30 --steady-requests 3 \
  --output-dir models/results/vit_cold30
/home/yzx/miniconda3/envs/llada/bin/python models/src/benchmark_cold_start.py \
  --model blip --gpu-index 0 --repeats 30 --steady-requests 3 \
  --output-dir models/results/blip_cold30

/home/yzx/miniconda3/envs/llada/bin/python models/src/compare_models.py \
  --whisper models/results/whisper_cold30 \
  --vit models/results/vit_cold30 \
  --blip models/results/blip_cold30 \
  --output-dir models/results/comparison
```

Do not run these commands on a GPU occupied by LLaDA or another service.  The
runner records the physical GPU name and baseline/Ready/exit memory.  It refuses
to silently reuse an existing result unless `--resume` is provided, and before
the next repetition it requires GPU memory to return within 64 MiB of that
repetition's baseline.

Outputs for each model are:

- `cold_start_runs.jsonl`: every successful process lifecycle, health payload,
  synchronized first inference, and every Ready-state inference.
- `logs/run_NNN.log` and `logs/run_NNN_service.jsonl`: subprocess and structured
  stage logs.
- `metadata.json`: pinned revision, local input, hardware, dtype, and cold-start
  semantics.
- `summary.csv`, `summary.json`, and `timing_distributions.png`: count, mean,
  P50, P95, minimum, maximum, and sample standard deviation.

`external_spawn_to_ready_s` is the authoritative complete wall-clock Ready
latency.  Internal stages report dependency import, CUDA initialization,
processor loading, CPU weight loading, transfer to GPU, and first real test
inference.  Stages are sequential in the current service, but the external
measurement remains authoritative and should not be reconstructed by adding
rounded fields.  `first_inference_gpu_s` and `steady_gpu_inference_time_s` use
CUDA synchronization; preprocessing and client round-trip are reported
separately.  These are warm-filesystem, process-level cold starts: model download
is excluded and the operating-system page cache is not flushed.
