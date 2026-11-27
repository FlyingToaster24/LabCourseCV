"""Model builders. All models take raw pixels in [0, 255]."""

from __future__ import annotations

import tf_keras as keras


def build_model(cfg: dict, num_classes: int) -> tuple[keras.Model, keras.Model]:
    """Return (model, backbone). The backbone is returned so train.py can freeze/unfreeze it."""
    m, size = cfg["model"], cfg["data"]["img_size"]
    weights = None if str(m["weights"]).lower() == "none" else m["weights"]
    shape = (size, size, 3)

    inputs = keras.Input(shape, name="image")
    if m["arch"] == "mobilenet_v2":
        x = keras.layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(inputs)  # = mobilenet_v2.preprocess_input
        backbone = keras.applications.MobileNetV2(input_shape=shape, alpha=m["alpha"], include_top=False, weights=weights)
    elif m["arch"] in ("mobilenet_v3_small", "mobilenet_v3_large"):
        cls = keras.applications.MobileNetV3Small if m["arch"].endswith("small") else keras.applications.MobileNetV3Large
        x = inputs  # MobileNetV3 has its own rescaling layer inside (include_preprocessing=True)
        backbone = cls(input_shape=shape, alpha=m["alpha"], include_top=False, weights=weights, include_preprocessing=True)
    else:
        raise ValueError(f"Unknown model.arch {m['arch']!r}")

    x = backbone(x, training=False)  # BatchNorm stays in inference mode, also when the backbone is unfrozen
    x = keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = keras.layers.Dropout(m["dropout"], name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name=f"{m['arch']}_{m['alpha']}")
    return model, backbone


def set_backbone_trainable(backbone: keras.Model, unfreeze) -> None:
    """unfreeze: False (freeze all), "all", or an int N (train only the last N layers)."""
    if unfreeze is False:
        backbone.trainable = False
        return
    backbone.trainable = True
    if unfreeze != "all":
        n = int(unfreeze)
        for layer in backbone.layers[:-n] if n > 0 else backbone.layers:
            layer.trainable = False
