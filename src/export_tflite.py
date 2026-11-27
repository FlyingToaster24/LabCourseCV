"""Convert a trained run to TFLite.

    python -m src.export_tflite baseline_v1                                  # fp32 only
    python -m src.export_tflite baseline_v1 --variants fp32 fp16 dynamic int8

Variants (method M3 compares them):
    fp32     plain float model (reference)
    fp16     weights stored as float16, compute in float32 on CPU
    dynamic  weights int8, activations float (dynamic-range quantization)
    int8     full-integer: weights and activations int8, input and output uint8.
             Calibrated on 300 training images. Input = raw pixels 0..255.

Writes runs/<exp_id>/model_<variant>.tflite and export.json (file sizes).
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
from pathlib import Path

import tensorflow as tf
import tf_keras as keras
import yaml

from src.data import make_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("fp32", "fp16", "dynamic", "int8")


def representative_dataset(img_size: int, n: int = 300, seed: int = 42):
    ds = make_dataset("train", img_size=img_size, batch_size=1, shuffle=False, cache=False, limit=n, seed=seed)

    def gen():
        for images, _ in ds:
            yield [tf.cast(images, tf.float32)]

    return gen


def convert(model: keras.Model, variant: str, img_size: int, calib_images: int = 300) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif variant == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset(img_size, calib_images)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
    elif variant != "fp32":
        raise ValueError(f"Unknown variant {variant!r}, choose from {VARIANTS}")
    return conv.convert()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_id")
    ap.add_argument("--variants", nargs="+", default=["fp32"], choices=VARIANTS)
    ap.add_argument("--calib-images", type=int, default=300, help="training images used to calibrate int8")
    args = ap.parse_args()

    run_dir = REPO_ROOT / "runs" / args.exp_id
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = keras.models.load_model(run_dir / "model.keras", compile=False)
    img_size = cfg["data"]["img_size"]

    info_path = run_dir / "export.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    for v in args.variants:
        print(f"Converting {args.exp_id} -> {v} ...")
        data = convert(model, v, img_size, args.calib_images)
        path = run_dir / f"model_{v}.tflite"
        path.write_bytes(data)
        info[v] = {"file": path.name, "size_mb": round(len(data) / 1e6, 3)}
        print(f"  {path} ({len(data) / 1e6:.2f} MB)")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Next: python -m src.evaluate {args.exp_id} --split val --variant {args.variants[0]}")


if __name__ == "__main__":
    main()
