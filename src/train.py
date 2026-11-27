"""Train one experiment (two-phase fine-tuning).

    python -m src.train configs/baseline.yaml
    python -m src.train configs/baseline.yaml --set exp_id=hp_lr3e-4 --set train.lr=3e-4
    python -m src.train configs/baseline.yaml --quick        # pipeline check in a few minutes

Phase A trains only the new classification head (backbone frozen).
Phase B unfreezes the backbone (all or the last N layers) and fine-tunes;
the weights with the lowest validation loss are kept.

Writes runs/<exp_id>/:
    config.yaml          full effective config (defaults + file + --set)
    history.csv          per epoch: phase, lr, loss, accuracy, top5, val_loss, val_accuracy, val_top5
    model.keras          best model, without optimizer state
    train_summary.json   best epoch, val metrics, training time, params

Test-set numbers are NOT computed here: run src.evaluate when a configuration is final.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import tensorflow as tf
import tf_keras as keras

from src.augment import build_augmenter, build_mixer
from src.config import load_config, save_config
from src.data import load_class_names, load_split, make_dataset
from src.models import build_model, set_backbone_trainable

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
QUICK = ["data.limit_train=256", "data.limit_val=128", "data.batch_size=32",
         "train.head_epochs=1", "train.epochs=1"]


def make_optimizer(name: str, lr, weight_decay: float):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True,
                                    weight_decay=weight_decay or None)
    raise ValueError(f"Unknown train.optimizer {name!r}")


def make_lr(t: dict, steps_per_epoch: int):
    if t["schedule"] == "constant":
        return t["lr"]
    if t["schedule"] == "cosine":
        total = max(1, t["epochs"] * steps_per_epoch)
        warm = min(t["warmup_epochs"] * steps_per_epoch, total - 1)
        if warm > 0:
            return keras.optimizers.schedules.CosineDecay(0.0, total - warm, warmup_target=t["lr"], warmup_steps=warm)
        return keras.optimizers.schedules.CosineDecay(t["lr"], total)
    raise ValueError(f"Unknown train.schedule {t['schedule']!r}")


class HistoryLogger(keras.callbacks.Callback):
    """Writes one CSV row per epoch for both phases (file is updated after every epoch)."""

    FIELDS = ["epoch", "phase", "lr", "loss", "accuracy", "top5", "val_loss", "val_accuracy", "val_top5", "seconds"]

    def __init__(self, path: Path):
        super().__init__()
        self.path, self.phase, self.rows = path, "A", []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        lr = opt.learning_rate
        lr = float(lr(opt.iterations)) if callable(lr) else float(keras.backend.get_value(lr))
        row = {"epoch": epoch + 1, "phase": self.phase, "lr": lr, "seconds": round(time.perf_counter() - self._t, 1)}
        row.update({k: float(v) for k, v in (logs or {}).items() if k in self.FIELDS})
        self.rows.append(row)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML config, e.g. configs/baseline.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set train.lr=3e-4 (repeatable)")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline (exp_id gets _quick)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = ap.parse_args()

    overrides = (QUICK if args.quick else []) + args.overrides
    cfg = load_config(args.config, overrides)
    if args.quick:
        cfg["exp_id"] += "_quick"
    d, t = cfg["data"], cfg["train"]

    run_dir = RUNS_DIR / cfg["exp_id"]
    if run_dir.exists():
        if not (args.overwrite or args.quick):
            raise SystemExit(f"{run_dir} already exists. Pick a new exp_id or pass --overwrite.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")

    keras.utils.set_random_seed(cfg["seed"])
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Experiment {cfg['exp_id']}  |  TF {tf.__version__}  |  GPU: {len(gpus) or 'none (CPU only, slow)'}")

    # ---- data ----
    class_names = load_class_names()
    n_cls = len(class_names)
    one_hot = lambda x, y: (x, tf.one_hot(y, n_cls))
    train_ds = make_dataset("train", img_size=d["img_size"], batch_size=d["batch_size"],
                            augment=build_augmenter(d["augment"]), cache=d["cache"],
                            limit=d["limit_train"], seed=cfg["seed"]).map(one_hot)
    mixer = build_mixer(d["mix"], d["mix_alpha"])
    if mixer is not None:
        train_ds = train_ds.map(mixer, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = make_dataset("val", img_size=d["img_size"], batch_size=d["batch_size"], cache=d["cache"],
                          limit=d["limit_val"], seed=cfg["seed"]).map(one_hot)
    n_train = d["limit_train"] or len(load_split("train")[0])
    steps_per_epoch = math.ceil(n_train / d["batch_size"])

    # ---- model ----
    model, backbone = build_model(cfg, n_cls)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=t["label_smoothing"])
    metrics = lambda: [keras.metrics.CategoricalAccuracy(name="accuracy"),
                       keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
    logger = HistoryLogger(run_dir / "history.csv")
    best_weights = run_dir / "best.weights.h5"
    t0 = time.perf_counter()

    # ---- phase A: head only ----
    if t["head_epochs"] > 0:
        print(f"\nPhase A: backbone frozen, {t['head_epochs']} epochs, lr {t['head_lr']}")
        set_backbone_trainable(backbone, False)
        model.compile(optimizer=keras.optimizers.Adam(t["head_lr"]), loss=loss, metrics=metrics())
        model.fit(train_ds, validation_data=val_ds, epochs=t["head_epochs"], callbacks=[logger], verbose=2)

    # ---- phase B: fine-tune ----
    if t["epochs"] > 0:
        print(f"\nPhase B: unfreeze={t['unfreeze']}, up to {t['epochs']} epochs, {t['optimizer']} lr {t['lr']} "
              f"({t['schedule']}), label smoothing {t['label_smoothing']}")
        logger.phase = "B"
        set_backbone_trainable(backbone, t["unfreeze"])
        model.compile(optimizer=make_optimizer(t["optimizer"], make_lr(t, steps_per_epoch), t["weight_decay"]),
                      loss=loss, metrics=metrics())
        callbacks = [
            logger,
            keras.callbacks.ModelCheckpoint(str(best_weights), monitor="val_loss", save_best_only=True,
                                            save_weights_only=True, verbose=0),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stopping_patience"], verbose=1),
        ]
        model.fit(train_ds, validation_data=val_ds, initial_epoch=t["head_epochs"],
                  epochs=t["head_epochs"] + t["epochs"], callbacks=callbacks, verbose=2)
        model.load_weights(str(best_weights))
        best_weights.unlink()

    minutes = (time.perf_counter() - t0) / 60
    # Save an uncompiled copy so the file holds weights only (no optimizer state).
    export = keras.models.clone_model(model)
    export.set_weights(model.get_weights())
    export.save(run_dir / "model.keras")

    # ---- summary ----
    rows = logger.rows
    candidates = [r for r in rows if r["phase"] == "B"] or rows
    best = min(candidates, key=lambda r: r["val_loss"])
    summary = {
        "exp_id": cfg["exp_id"],
        "owner": cfg["owner"],
        "method": cfg["method"],
        "config": " ".join([args.config] + (["--quick"] if args.quick else []) + [f"--set {o}" for o in args.overrides]),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "best_val_top5": best["val_top5"],
        "epochs_run": len(rows),
        "train_minutes": round(minutes, 1),
        "params": int(model.count_params()),
        "gpu": [tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus],
        "tensorflow": tf.__version__,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone in {minutes:.1f} min. Best epoch {best['epoch']}: val_accuracy {best['val_accuracy']:.4f}, "
          f"val_top5 {best['val_top5']:.4f}")
    print(f"Saved {run_dir / 'model.keras'}")
    print(f"Next: python -m src.evaluate {cfg['exp_id']} --split val")


if __name__ == "__main__":
    main()
