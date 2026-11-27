"""Run TFLite models from Python (float or integer-quantized), used by evaluate and benchmark."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def _interpreter_class():
    try:  # LiteRT, the successor of tf.lite (not available on every platform)
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter


class TFLiteModel:
    """model = TFLiteModel("runs/x/model_int8.tflite"); probs = model.predict(images_0_255)"""

    def __init__(self, path, num_threads: int | None = None):
        self.path = Path(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tf.lite.Interpreter deprecation notice
            self.interp = _interpreter_class()(model_path=str(self.path), num_threads=num_threads)
        self.interp.allocate_tensors()
        self._refresh()

    def _refresh(self):
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    @property
    def input_dtype(self):
        return self.inp["dtype"]

    @property
    def input_shape(self):
        return tuple(int(v) for v in self.inp["shape"])

    def _to_input(self, images: np.ndarray) -> np.ndarray:
        dtype = self.inp["dtype"]
        if dtype == np.float32:
            return images.astype(np.float32)
        scale, zero = self.inp["quantization"]
        info = np.iinfo(dtype)
        q = np.round(images / scale + zero) if scale else images
        return np.clip(q, info.min, info.max).astype(dtype)

    def _from_output(self, out: np.ndarray) -> np.ndarray:
        if self.out["dtype"] == np.float32:
            return out
        scale, zero = self.out["quantization"]
        return (out.astype(np.float32) - zero) * scale

    def set_batch(self, batch: int) -> None:
        if int(self.inp["shape"][0]) != batch:
            self.interp.resize_tensor_input(self.inp["index"], [batch, *self.inp["shape"][1:]])
            self.interp.allocate_tensors()
            self._refresh()

    def predict(self, images: np.ndarray) -> np.ndarray:
        """images: float array (B, H, W, 3) with raw pixels in [0, 255] -> class probabilities (B, C)."""
        self.set_batch(images.shape[0])
        self.interp.set_tensor(self.inp["index"], self._to_input(images))
        self.interp.invoke()
        return self._from_output(self.interp.get_tensor(self.out["index"]))

    def invoke_raw(self) -> None:
        """One inference on whatever is in the input tensor (for timing)."""
        self.interp.invoke()

    def set_raw_input(self, x: np.ndarray) -> None:
        self.interp.set_tensor(self.inp["index"], x)
