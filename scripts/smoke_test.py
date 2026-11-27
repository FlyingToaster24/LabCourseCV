"""PC-pool smoke test: environment, GPU, data reading speed and a few training steps.

Everyone runs this once on a PC-pool machine (Step 1 task, due Fri 27 Nov):

    python scripts/smoke_test.py --data-root <path to NWPU-RESISC45 on the NAS or local disk>

and posts the SUMMARY block at the end in the group chat.
Use --weights none if the machine cannot download the ImageNet weights.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make "src" importable

import tensorflow as tf
import tf_keras as keras  # project convention: tf_keras (Keras 2), needed by TF-MOT later

from src.data import DEFAULT_DATA_ROOT, IMG_SIZE, load_class_names, load_split, make_dataset


def build_model(num_classes: int, img_size: int, weights) -> keras.Model:
    """Same structure as the baseline: raw pixels in, Rescaling inside the model."""
    inputs = keras.Input((img_size, img_size, 3))
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights=weights)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20, help="training steps per epoch (2 epochs are run)")
    ap.add_argument("--read-images", type=int, default=1000, help="images read to measure loading speed")
    ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    args = ap.parse_args()

    summary: dict[str, str] = {}

    print("== 1/4 Environment")
    gpus = tf.config.list_physical_devices("GPU")
    summary["machine"] = f"{platform.node()} ({platform.system()} {platform.release()})"
    summary["python / tf / tf_keras"] = f"{platform.python_version()} / {tf.__version__} / {keras.__version__}"
    summary["gpu"] = ", ".join(tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus) or "NONE (CPU only)"
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if not gpus:
        print("  WARNING: no GPU visible. Training will be slow. (Native Windows TensorFlow has no GPU support.)")

    print("== 2/4 Data")
    names = load_class_names()
    counts = {s: len(load_split(s)[0]) for s in ("train", "val", "test")}
    print(f"  data root: {Path(args.data_root).resolve()}")
    print(f"  {len(names)} classes, split sizes {counts}")
    ds = make_dataset("val", data_root=args.data_root, batch_size=args.batch_size, limit=args.read_images, cache=False)
    t0, n = time.perf_counter(), 0
    for images, _ in ds:
        n += int(images.shape[0])
    read_rate = n / (time.perf_counter() - t0)
    summary["data read speed"] = f"{read_rate:.0f} images/s ({n} images, uncached)"
    print(f"  {summary['data read speed']}")
    if read_rate < 200:
        print("  NOTE: slow reads. Fine with caching (only the first epoch reads from disk); "
              "otherwise copy the dataset to a local disk first.")

    print("== 3/4 Model")
    weights = None if args.weights == "none" else "imagenet"
    try:
        model = build_model(len(names), IMG_SIZE, weights)
    except Exception as exc:  # e.g. no internet for the weight download
        print(f"  could not load ImageNet weights ({exc}); continuing with random weights")
        weights = None
        model = build_model(len(names), IMG_SIZE, None)
    summary["model"] = f"MobileNetV2 + 45-class head, {model.count_params():,} params, weights={weights}"
    print(f"  {summary['model']}")

    print("== 4/4 Training steps")
    train = make_dataset("train", data_root=args.data_root, batch_size=args.batch_size,
                         limit=args.steps * args.batch_size, cache=True, drop_remainder=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    timer = EpochTimer()
    hist = model.fit(train, epochs=2, callbacks=[timer], verbose=2)
    step_time = timer.times[-1] / args.steps  # epoch 2: graph built, data cached
    summary["train speed"] = f"{step_time * 1000:.0f} ms/step at batch {args.batch_size} = {args.batch_size / step_time:.0f} images/s"
    est_epoch = counts["train"] / (args.batch_size / step_time) / 60
    summary["est. full epoch"] = f"{est_epoch:.1f} min (18,900 training images)"
    loss = hist.history["loss"]
    summary["loss epoch 1 -> 2"] = f"{loss[0]:.3f} -> {loss[-1]:.3f}"

    print("\n==== SUMMARY (paste this in the group chat) ====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("status: OK")


if __name__ == "__main__":
    main()
