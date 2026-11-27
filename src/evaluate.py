"""Evaluate a trained run (Keras model or one of its TFLite variants) and log the result.

    python -m src.evaluate baseline_v1 --split val                  # Keras model, validation set
    python -m src.evaluate baseline_v1 --split test                 # only for finished configurations
    python -m src.evaluate baseline_v1 --split test --variant int8  # TFLite variant (+ latency)

Computes top-1, top-5, macro-F1, per-class precision/recall/F1 and the confusion
matrix, measures latency for TFLite variants (benchmark_latency protocol) and
appends one row to results/results.csv.

Writes runs/<exp_id>/eval/:
    <variant>_<split>.json             metrics
    <variant>_<split>_confusion.csv    rows = true class, columns = predicted class
    <variant>_<split>_per_class.csv
    <variant>_<split>_preds.npz        labels + probabilities (for failure cases and ensembles)
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.data import load_class_names, make_dataset
from src.results import append_row

REPO_ROOT = Path(__file__).resolve().parents[1]


def compute_metrics(labels: np.ndarray, probs: np.ndarray, n_cls: int) -> tuple[dict, np.ndarray, list[dict]]:
    pred = probs.argmax(axis=1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / cm.sum(axis=0))
        recall = np.nan_to_num(tp / cm.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    metrics = {
        "n": int(len(labels)),
        "top1": float((pred == labels).mean()),
        "top5": float((top5 == labels[:, None]).any(axis=1).mean()),
        "macro_f1": float(f1.mean()),
    }
    per_class = [{"class_id": i, "precision": precision[i], "recall": recall[i], "f1": f1[i],
                  "support": int(cm[i].sum())} for i in range(n_cls)]
    return metrics, cm, per_class


def predict(run_dir: Path, variant: str, ds):
    labels, probs = [], []
    if variant == "keras":
        import tf_keras as keras
        model = keras.models.load_model(run_dir / "model.keras", compile=False)
        for x, y in ds:
            probs.append(model(x, training=False).numpy())
            labels.append(y.numpy())
        return np.concatenate(labels), np.concatenate(probs), model.count_params(), run_dir / "model.keras"
    from src.tflite_utils import TFLiteModel
    path = run_dir / f"model_{variant}.tflite"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m src.export_tflite {run_dir.name} --variants {variant}")
    model = TFLiteModel(path, num_threads=os.cpu_count())  # accuracy only; latency uses 1 thread
    for x, y in ds:
        probs.append(model.predict(x.numpy()))
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs), None, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_id")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--variant", default="keras", choices=["keras", "fp32", "fp16", "dynamic", "int8"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-latency", action="store_true", help="skip the latency measurement for TFLite variants")
    ap.add_argument("--no-log", action="store_true", help="do not append to results/results.csv")
    ap.add_argument("--notes", default="")
    ap.add_argument("--limit", type=int, default=None, help="quick check on a subset (never logged to results.csv)")
    args = ap.parse_args()

    run_dir = REPO_ROOT / "runs" / args.exp_id
    if not run_dir.exists():
        raise SystemExit(f"{run_dir} not found")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    summary_path = run_dir / "train_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    class_names = load_class_names()

    ds = make_dataset(args.split, img_size=cfg["data"]["img_size"], batch_size=args.batch_size,
                      shuffle=False, cache=False, limit=args.limit)
    print(f"Evaluating {args.exp_id} [{args.variant}] on {args.split} ...")
    labels, probs, params, model_path = predict(run_dir, args.variant, ds)
    metrics, cm, per_class = compute_metrics(labels, probs, len(class_names))
    params = params or summary.get("params")
    size_mb = model_path.stat().st_size / 1e6

    latency = None
    if args.variant != "keras" and not args.no_latency:
        from src.benchmark_latency import benchmark
        latency = benchmark(model_path)
        (run_dir / f"latency_{args.variant}.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")

    out = run_dir / "eval" / ("subset" if args.limit else "")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_{args.split}"
    result = {"exp_id": args.exp_id, "variant": args.variant, "split": args.split, **metrics,
              "params": params, "size_mb": round(size_mb, 3),
              "latency_pc_ms": latency["median_ms"] if latency else None, "latency": latency}
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / f"{stem}_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, cm):
            w.writerow([name] + row.tolist())
    with open(out / f"{stem}_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        w.writeheader()
        for r in sorted(per_class, key=lambda r: r["f1"]):
            w.writerow({"class": class_names[r["class_id"]], "precision": f"{r['precision']:.4f}",
                        "recall": f"{r['recall']:.4f}", "f1": f"{r['f1']:.4f}", "support": r["support"]})
    np.savez_compressed(out / f"{stem}_preds.npz", labels=labels, probs=probs.astype(np.float16))

    worst = sorted(per_class, key=lambda r: r["f1"])[:5]
    print(f"  top-1 {metrics['top1']:.4f} | top-5 {metrics['top5']:.4f} | macro-F1 {metrics['macro_f1']:.4f} "
          f"| {size_mb:.2f} MB" + (f" | {latency['median_ms']:.2f} ms/image (1 thread)" if latency else ""))
    print("  weakest classes: " + ", ".join(f"{class_names[r['class_id']]} ({r['f1']:.2f})" for r in worst))

    if args.limit:
        print("  (subset evaluation with --limit: not logged to results.csv)")
    elif not args.no_log:
        append_row({
            "exp_id": args.exp_id, "owner": cfg.get("owner", ""), "method": cfg.get("method", ""),
            "variant": args.variant, "split": args.split,
            "top1": metrics["top1"], "top5": metrics["top5"], "macro_f1": metrics["macro_f1"],
            "params": params, "size_mb": round(size_mb, 3),
            "latency_pc_ms": latency["median_ms"] if latency else None,
            "config": summary.get("config", ""), "notes": args.notes,
        })


if __name__ == "__main__":
    main()
