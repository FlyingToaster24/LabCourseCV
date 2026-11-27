"""tf.data input pipeline for NWPU-RESISC45.

Every experiment loads data through this file, so all of us use the same
split, image size and decoding.

Output of every dataset:
    images  float32, shape (batch, 224, 224, 3), raw pixel values in [0, 255]
    labels  int32, shape (batch,), index into data/classes.txt

Normalisation is NOT done here. Every model starts with
    keras.layers.Rescaling(1 / 127.5, offset=-1)   # = mobilenet_v2.preprocess_input
so the exported TFLite model takes raw pixels. That keeps quantization
(representative dataset), the robustness corruptions and the on-device demo
simple. Use labels with SparseCategoricalCrossentropy.

Where the images are:
    Set the environment variable NWPU_ROOT to the dataset folder (the one with
    the 45 class folders), e.g. on the NAS or a local copy. Without it, the
    folder NWPU-RESISC45/ in the repo root is used.

Example:
    from src.data import get_datasets
    data = get_datasets(batch_size=64)
    model.fit(data.train, validation_data=data.val, epochs=20)
    model.evaluate(data.test)

Quick check from the repo root:
    python -m src.data
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"  # classes.txt and splits/*.csv (in git)
DEFAULT_DATA_ROOT = os.environ.get("NWPU_ROOT", str(REPO_ROOT / "NWPU-RESISC45"))
IMG_SIZE = 224
SPLITS = ("train", "val", "test")
AUTOTUNE = tf.data.AUTOTUNE


def load_class_names(data_dir: Union[str, Path] = DATA_DIR) -> list[str]:
    """The 45 class names; position in the list = label index."""
    path = Path(data_dir) / "classes.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/make_splits.py once (or git pull).")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_split(split: str, data_dir: Union[str, Path] = DATA_DIR) -> tuple[list[str], list[int]]:
    """Relative image paths and integer labels of one split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = Path(data_dir) / "splits" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/make_splits.py once (or git pull).")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["path"] for r in rows], [int(r["label"]) for r in rows]


def _read_file(path: tf.Tensor, label: tf.Tensor):
    return tf.io.read_file(path), label


def _make_decoder(img_size: int):
    def decode(jpeg_bytes: tf.Tensor, label: tf.Tensor):
        img = tf.io.decode_jpeg(jpeg_bytes, channels=3)  # uint8, 256 x 256
        if img_size == 256:
            img = tf.cast(img, tf.float32)
        else:
            img = tf.image.resize(img, (img_size, img_size), antialias=True)  # float32 in [0, 255]
        img = tf.clip_by_value(img, 0.0, 255.0)
        img.set_shape((img_size, img_size, 3))
        return img, label

    return decode


def make_dataset(
    split: str,
    data_root: Optional[Union[str, Path]] = None,
    img_size: int = IMG_SIZE,
    batch_size: int = 64,
    shuffle: Optional[bool] = None,
    augment: Optional[Callable] = None,
    cache: Union[bool, str] = True,
    limit: Optional[int] = None,
    seed: int = 42,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    """Batched dataset of one split.

    split      "train", "val" or "test"
    data_root  dataset folder; default NWPU_ROOT or <repo>/NWPU-RESISC45
    shuffle    default True for train, False otherwise
    augment    callable applied to each training batch of [0, 255] images,
               e.g. a keras.Sequential of Random* layers; called as
               augment(images, training=True)
    cache      True: keep the raw JPEG bytes in RAM after the first epoch
               (~250 MB for train), so later epochs do not read from the NAS.
               A string caches to that file instead. False: no cache.
    limit      use only this many images (random but fixed subset), for quick tests
    """
    root = Path(data_root or DEFAULT_DATA_ROOT)
    rel_paths, labels = load_split(split)
    if limit is not None and limit < len(rel_paths):
        idx = list(range(len(rel_paths)))
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
        rel_paths = [rel_paths[i] for i in idx]
        labels = [labels[i] for i in idx]

    first = root / rel_paths[0]
    if not first.exists():
        raise FileNotFoundError(
            f"Image not found: {first}\n"
            f"data_root={root} must be the folder that contains the 45 class folders. "
            f"Set NWPU_ROOT or pass data_root."
        )

    paths = [str(root / p) for p in rel_paths]
    if shuffle is None:
        shuffle = split == "train"

    ds = tf.data.Dataset.from_tensor_slices((paths, tf.constant(labels, dtype=tf.int32)))
    ds = ds.map(_read_file, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache(cache if isinstance(cache, str) else "")
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_make_decoder(img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    if augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


@dataclass
class Data:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def get_datasets(
    data_root: Optional[Union[str, Path]] = None,
    img_size: int = IMG_SIZE,
    batch_size: int = 64,
    augment: Optional[Callable] = None,
    cache: bool = True,
    seed: int = 42,
) -> Data:
    """Train (shuffled, optionally augmented), val and test datasets plus class names."""
    kw = dict(data_root=data_root, img_size=img_size, batch_size=batch_size, cache=cache, seed=seed)
    return Data(
        train=make_dataset("train", augment=augment, **kw),
        val=make_dataset("val", **kw),
        test=make_dataset("test", **kw),
        class_names=load_class_names(),
    )


if __name__ == "__main__":
    names = load_class_names()
    print(f"data root: {Path(DEFAULT_DATA_ROOT).resolve()}")
    print(f"classes:   {len(names)} ({names[0]} ... {names[-1]})")
    for s in SPLITS:
        paths, labels = load_split(s)
        print(f"{s:5s}:     {len(paths)} images, {len(set(labels))} classes")
    images, labels = next(iter(make_dataset("val", batch_size=8, cache=False)))
    print(f"batch:     images {images.shape} {images.dtype}, range [{float(tf.reduce_min(images)):.0f}, "
          f"{float(tf.reduce_max(images)):.0f}], labels {labels.numpy().tolist()}")
