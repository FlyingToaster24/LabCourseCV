"""Experiment configuration: defaults + a YAML file + command-line overrides.

Every key and its default is listed in DEFAULTS below. A config file only
needs the keys that differ from the defaults.

    from src.config import load_config
    cfg = load_config("configs/baseline.yaml", overrides=["train.lr=3e-4", "exp_id=lr_3e-4"])
    cfg["train"]["lr"]  -> 0.0003
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULTS: dict[str, Any] = {
    "exp_id": None,            # unique name, also the run folder runs/<exp_id>
    "owner": "",               # who ran it
    "method": "baseline",      # baseline | augmentation | hparam | quantization | pruning | combined | other
    "notes": "",
    "seed": 42,
    "data": {
        "img_size": 224,
        "batch_size": 64,
        "augment": "none",     # none | geometric | photometric | geo_photo   (see src/augment.py)
        "mix": "none",         # none | mixup | cutmix
        "mix_alpha": 0.2,
        "cache": True,         # keep JPEG bytes in RAM after epoch 1
        "limit_train": None,   # e.g. 512 for a quick test
        "limit_val": None,
    },
    "model": {
        "arch": "mobilenet_v2",  # mobilenet_v2 | mobilenet_v3_small | mobilenet_v3_large
        "alpha": 1.0,            # width multiplier
        "weights": "imagenet",   # imagenet | none
        "dropout": 0.2,
    },
    "train": {
        "head_epochs": 5,        # phase A: backbone frozen, only the new head trains
        "head_lr": 1.0e-3,
        "epochs": 15,            # phase B: fine-tune (max epochs, early stopping may end it sooner)
        "lr": 1.0e-4,
        "optimizer": "adam",     # adam | adamw | sgd (momentum 0.9)
        "weight_decay": 0.0,
        "schedule": "constant",  # constant | cosine
        "warmup_epochs": 0,      # linear warm-up at the start of phase B (cosine only)
        "label_smoothing": 0.0,
        "unfreeze": "all",       # all | N (only the last N backbone layers train in phase B)
        "early_stopping_patience": 3,
    },
}


def _merge(base: dict, extra: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if k not in out:
            raise KeyError(f"Unknown config key '{path}{k}'. Valid keys: {sorted(out)}")
        if isinstance(out[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"Config key '{path}{k}' must be a section, got {v!r}")
            out[k] = _merge(out[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def _apply_override(cfg: dict, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)  # "3e-4" -> float, "none" -> str, "null" -> None, "true" -> bool
    if isinstance(value, str):
        try:
            value = float(value) if any(c in value.lower() for c in ".e") else int(value)
        except ValueError:
            pass
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Unknown config section in override {item!r}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key in override {item!r}")
    node[parts[-1]] = value


def load_config(path: Optional[str] = None, overrides: Optional[list[str]] = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
        if not cfg["exp_id"]:
            cfg["exp_id"] = Path(path).stem
    for item in overrides or []:
        _apply_override(cfg, item)
    if not cfg["exp_id"]:
        raise ValueError("exp_id is missing (set it in the config file or with exp_id=...)")
    return cfg


def save_config(cfg: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
