"""Latency of a TFLite model, following the project protocol.

Protocol: batch 1, input 224 x 224, 1 CPU thread, 50 warm-up runs,
then 200 timed runs; we report the median in ms. Always compare numbers
measured on the same machine type.

    python -m src.benchmark_latency baseline_v1 --variant int8
    python -m src.benchmark_latency path/to/model.tflite
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from src.tflite_utils import TFLiteModel

REPO_ROOT = Path(__file__).resolve().parents[1]


def benchmark(path, threads: int = 1, warmup: int = 50, runs: int = 200, seed: int = 0) -> dict:
    model = TFLiteModel(path, num_threads=threads)
    model.set_batch(1)
    shape, dtype = model.input_shape, model.input_dtype
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 255, shape)
    model.set_raw_input(x.astype(np.float32) if dtype == np.float32 else x.astype(dtype))
    for _ in range(warmup):
        model.invoke_raw()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model.invoke_raw()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "model": str(path),
        "median_ms": round(statistics.median(times), 3),
        "mean_ms": round(statistics.fmean(times), 3),
        "p90_ms": round(times[int(0.9 * len(times)) - 1], 3),
        "min_ms": round(times[0], 3),
        "threads": threads,
        "runs": runs,
        "machine": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="exp_id (with --variant) or a .tflite path")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    if args.target.endswith(".tflite"):
        path, out = Path(args.target), None
    else:
        run_dir = REPO_ROOT / "runs" / args.target
        path, out = run_dir / f"model_{args.variant}.tflite", run_dir / f"latency_{args.variant}.json"
    res = benchmark(path, args.threads, args.warmup, args.runs)
    print(json.dumps(res, indent=2))
    if out:
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
