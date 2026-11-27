"""Poster figures for one run.

    python -m src.plots baseline_v1                         # uses the newest evaluation of the Keras model
    python -m src.plots baseline_v1 --variant keras --split test

Writes figures/<exp_id>/ (kept in git, PNG at 300 dpi for the A0 poster):
    curves.png               loss and accuracy per epoch, train vs validation, phase A/B marked
    confusion_<v>_<s>.png    row-normalised confusion matrix (45 x 45)
    top_confusions_<v>_<s>.png + .csv   the most frequent wrong predictions (failure cases)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

REPO_ROOT = Path(__file__).resolve().parents[1]

# Colours: validated categorical slots 1-2 and a single-hue sequential blue ramp.
TRAIN, VAL = "#257de0", "#f47230"
INK, INK_2, GRID = "#191919", "#666666", "#e5e5e5"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cce3fd", "#84b9f3", "#338ced", "#2370cd", "#0f4a96", "#0e3d7e"])

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_curves(run_dir: Path, out_dir: Path) -> Path:
    rows = read_csv(run_dir / "history.csv")
    ep = np.array([int(r["epoch"]) for r in rows])
    get = lambda k: np.array([float(r[k]) if r.get(k) else np.nan for r in rows])
    phase_b = [int(r["epoch"]) for r in rows if r["phase"] == "B"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Top-1 accuracy")]:
        ax.plot(ep, get(key), color=TRAIN, lw=2, marker="o", ms=4, label="train")
        ax.plot(ep, get(f"val_{key}"), color=VAL, lw=2, marker="o", ms=4, label="validation")
        if phase_b and phase_b[0] > 1:
            x = phase_b[0] - 0.5
            ax.axvline(x, color=INK_2, lw=1, ls="--")
            ax.text(x, 0.03, " fine-tuning starts", transform=ax.get_xaxis_transform(), color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(loc="best")
    fig.tight_layout()
    path = out_dir / "curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def load_confusion(run_dir: Path, stem: str) -> tuple[list[str], np.ndarray]:
    with open(run_dir / "eval" / f"{stem}_confusion.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    cm = np.array([[int(v) for v in r[1:]] for r in rows[1:]])
    return names, cm


def plot_confusion(names: list[str], cm: np.ndarray, out: Path, title: str) -> Path:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 11))
    # Square-root colour scale so that small confusions (a few %) stay visible next to the diagonal.
    im = ax.imshow(norm, cmap=SEQ, norm=PowerNorm(gamma=0.5, vmin=0, vmax=1))
    labels = [n.replace("_", " ") for n in names]
    ax.set_xticks(range(len(names)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0))
    cb.outline.set_visible(False)
    fig.savefig(out)
    plt.close(fig)
    return out


def top_confusions(names: list[str], cm: np.ndarray, k: int = 15) -> list[tuple[str, str, int, float]]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    support = cm.sum(axis=1)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [(names[i], names[j], int(off[i, j]), off[i, j] / max(support[i], 1)) for i, j in idx if off[i, j] > 0]


def plot_top_confusions(pairs, out: Path, title: str) -> Path:
    if not pairs:
        return out
    labels = [f"{t.replace('_', ' ')}  →  {p.replace('_', ' ')}" for t, p, _, _ in pairs][::-1]
    shares = [s for *_, s in pairs][::-1]
    counts = [c for _, _, c, _ in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(pairs) + 1.2))
    ax.barh(range(len(pairs)), shares, color=TRAIN, height=0.6)
    for y, (s, c) in enumerate(zip(shares, counts)):
        ax.text(s, y, f"  {s:.1%} ({c})", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(pairs)), labels, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of the true class predicted as the other class")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(shares) * 1.3)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= min(1.0, max(shares) * 1.3)])
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_id")
    ap.add_argument("--variant", default=None, help="keras | fp32 | fp16 | dynamic | int8 (default: newest evaluation)")
    ap.add_argument("--split", default=None, help="val | test (default: newest evaluation)")
    ap.add_argument("--out", default=None, help="output folder (default: figures/<exp_id>)")
    args = ap.parse_args()

    run_dir = REPO_ROOT / "runs" / args.exp_id
    out_dir = Path(args.out) if args.out else REPO_ROOT / "figures" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [plot_curves(run_dir, out_dir)] if (run_dir / "history.csv").exists() else []

    evals = sorted((run_dir / "eval").glob("*_confusion.csv"), key=lambda p: p.stat().st_mtime)
    if args.variant or args.split:
        evals = [p for p in evals if p.name.startswith(f"{args.variant or ''}")
                 and f"_{args.split or ''}" in p.name]
    if evals:
        stem = evals[-1].name.removesuffix("_confusion.csv")
        variant, split = stem.rsplit("_", 1)
        names, cm = load_confusion(run_dir, stem)
        acc = np.trace(cm) / cm.sum()
        made.append(plot_confusion(names, cm, out_dir / f"confusion_{stem}.png",
                                   f"{args.exp_id} ({variant}, {split} set): top-1 {acc:.1%}"))
        pairs = top_confusions(names, cm)
        with open(out_dir / f"top_confusions_{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true_class", "predicted_class", "count", "share_of_true_class"])
            w.writerows([(t, p, c, f"{s:.4f}") for t, p, c, s in pairs])
        made.append(plot_top_confusions(pairs, out_dir / f"top_confusions_{stem}.png",
                                        f"Most frequent confusions ({variant}, {split} set)"))
    else:
        print(f"No evaluation found in {run_dir / 'eval'}; run src.evaluate first for the confusion figures.")
    for p in made:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
