"""Training-time data augmentation (method M1). Nothing here runs at inference.

Presets (config data.augment), applied to batches of raw [0, 255] images:
    none         no augmentation (baseline)
    geometric    flips + random 90 degree rotations + zoom-in up to 20 %
                 (aerial tiles have no "up", so all of these keep the label)
    photometric  brightness, contrast, saturation jitter
    geo_photo    both

Mixing (config data.mix), applied to batches with one-hot labels:
    mixup        blend two images and their labels
    cutmix       paste a box from another image, labels weighted by area
"""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf
import tf_keras as keras


def random_rot90(images: tf.Tensor) -> tf.Tensor:
    """Rotate each image by a random multiple of 90 degrees (lossless, square images)."""
    b = tf.shape(images)[0]
    k = tf.random.uniform([b], 0, 4, dtype=tf.int32)
    rotated = tf.stack([images] + [tf.image.rot90(images, i) for i in (1, 2, 3)], axis=0)
    return tf.gather_nd(rotated, tf.stack([k, tf.range(b)], axis=1))


def random_saturation(images: tf.Tensor, delta: float = 0.3) -> tf.Tensor:
    b = tf.shape(images)[0]
    factor = tf.random.uniform([b, 1, 1, 1], 1 - delta, 1 + delta)
    gray = tf.reduce_mean(images, axis=-1, keepdims=True)
    return gray + factor * (images - gray)


def build_augmenter(preset: str) -> Optional[Callable]:
    if preset in (None, "none"):
        return None
    geometric = preset in ("geometric", "geo_photo")
    photometric = preset in ("photometric", "geo_photo")
    if not (geometric or photometric):
        raise ValueError(f"Unknown data.augment preset {preset!r}")

    flip = keras.layers.RandomFlip("horizontal_and_vertical")
    zoom = keras.layers.RandomZoom(height_factor=(-0.2, 0.0), fill_mode="reflect")
    brightness = keras.layers.RandomBrightness(0.15, value_range=(0, 255))
    contrast = keras.layers.RandomContrast(0.2)

    def augment(images: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = images
        if geometric:
            x = flip(x, training=training)
            x = random_rot90(x)
            x = zoom(x, training=training)
        if photometric:
            x = brightness(x, training=training)
            x = contrast(x, training=training)
            x = random_saturation(x)
        return tf.clip_by_value(x, 0.0, 255.0)

    return augment


def _beta(alpha: float, n) -> tf.Tensor:
    g1 = tf.random.gamma([n], alpha)
    g2 = tf.random.gamma([n], alpha)
    return g1 / (g1 + g2)


def mixup(images: tf.Tensor, labels: tf.Tensor, alpha: float = 0.2):
    b = tf.shape(images)[0]
    lam = _beta(alpha, b)
    perm = tf.random.shuffle(tf.range(b))
    li = tf.reshape(lam, [-1, 1, 1, 1])
    images = li * images + (1 - li) * tf.gather(images, perm)
    ll = tf.reshape(lam, [-1, 1])
    labels = ll * labels + (1 - ll) * tf.gather(labels, perm)
    return images, labels


def cutmix(images: tf.Tensor, labels: tf.Tensor, alpha: float = 1.0):
    b = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    hf, wf = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
    lam = _beta(alpha, b)
    cut = tf.sqrt(1.0 - lam)
    ch, cw = cut * hf, cut * wf
    cy = tf.random.uniform([b], 0, hf)
    cx = tf.random.uniform([b], 0, wf)
    y0 = tf.clip_by_value(cy - ch / 2, 0, hf)
    y1 = tf.clip_by_value(cy + ch / 2, 0, hf)
    x0 = tf.clip_by_value(cx - cw / 2, 0, wf)
    x1 = tf.clip_by_value(cx + cw / 2, 0, wf)
    rows = tf.cast(tf.range(h), tf.float32)[None, :, None]
    cols = tf.cast(tf.range(w), tf.float32)[None, None, :]
    inside = ((rows >= y0[:, None, None]) & (rows < y1[:, None, None])
              & (cols >= x0[:, None, None]) & (cols < x1[:, None, None]))
    mask = tf.cast(inside, images.dtype)[..., None]  # 1 inside the pasted box
    perm = tf.random.shuffle(tf.range(b))
    images = images * (1 - mask) + tf.gather(images, perm) * mask
    lam_real = 1.0 - (y1 - y0) * (x1 - x0) / (hf * wf)
    ll = tf.reshape(lam_real, [-1, 1])
    labels = ll * labels + (1 - ll) * tf.gather(labels, perm)
    return images, labels


def build_mixer(mode: str, alpha: float) -> Optional[Callable]:
    if mode in (None, "none"):
        return None
    if mode == "mixup":
        return lambda x, y: mixup(x, y, alpha)
    if mode == "cutmix":
        return lambda x, y: cutmix(x, y, alpha)
    raise ValueError(f"Unknown data.mix {mode!r}")
