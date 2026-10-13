"""Combine independent cold-start summaries into one table and figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--whisper", required=True)
parser.add_argument("--vit", required=True)
parser.add_argument("--blip", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
summaries = []
for name in ("whisper", "vit", "blip"):
    value = json.loads((Path(getattr(args, name)) / "summary.json").read_text())
    row = {"model": name, "cold_starts": value["completed_cold_starts"]}
    for metric in (
        "external_spawn_to_ready_s", "first_inference_gpu_s",
        "steady_gpu_inference_time_s", "ready_memory_device_used_mb",
        "ready_memory_peak_allocated_mb",
    ):
        for stat in ("mean", "p50", "p95", "min", "max", "std"):
            row[f"{metric}_{stat}"] = value["metrics"][metric][stat]
    summaries.append(row)
output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
    writer.writeheader(); writer.writerows(summaries)
(output / "model_comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
names = [row["model"] for row in summaries]
for axis, metric, title in zip(axes, (
    "external_spawn_to_ready_s_mean", "steady_gpu_inference_time_s_mean",
    "ready_memory_device_used_mb_mean",
), ("Cold start to Ready", "Steady inference", "GPU memory at Ready")):
    axis.bar(names, [row[metric] for row in summaries])
    axis.set_title(title); axis.tick_params(axis="x", rotation=15)
figure.tight_layout(); figure.savefig(output / "model_comparison.png", dpi=180)
print(output.resolve())
