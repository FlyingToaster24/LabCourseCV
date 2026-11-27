"""Create the fixed train/val/test split files for NWPU-RESISC45.

Run this ONCE (Alper), commit the files in data/, and everyone else uses them.
Nobody re-runs it later, otherwise results stop being comparable.

What it writes:
    data/classes.txt          45 class names, line number = label index
    data/splits/train.csv     path,label,class   (path relative to the dataset root)
    data/splits/val.csv
    data/splits/test.csv
    data/splits/split_info.json   where the split came from, counts, date

Split source (--source):
    official  the standard 60/20/20 split from Google Research
              (remote_sensing_representations, also used by torchgeo and
              Hugging Face timm/resisc45). Needs internet.
    random    stratified 420/140/140 images per class, fixed seed.
    auto      (default) try official, fall back to random.

Usage (from the repo root):
    python scripts/make_splits.py --data-root NWPU-RESISC45
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import urllib.request
from pathlib import Path

OFFICIAL_URL = "https://storage.googleapis.com/remote_sensing_representations/resisc45-{split}.txt"
SPLITS = ("train", "val", "test")
PER_CLASS_RANDOM = {"train": 420, "val": 140, "test": 140}  # 60/20/20 of 700
EXPECTED_CLASSES = 45
EXPECTED_PER_CLASS = 700


def scan_dataset(root: Path) -> tuple[list[str], dict[str, str]]:
    """Return sorted class names and a map basename -> relative path.

    Only folders that directly contain .jpg files count as classes, so a
    stray nested copy of the dataset (e.g. NWPU-RESISC45/NWPU-RESISC45/) is ignored.
    """
    classes: list[str] = []
    index: dict[str, str] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        jpgs = sorted(f.name for f in folder.iterdir() if f.suffix.lower() == ".jpg")
        if not jpgs:
            print(f"  skipping folder without images: {folder.name}/")
            continue
        classes.append(folder.name)
        for name in jpgs:
            if name in index:
                sys.exit(f"Duplicate file name {name} in {folder.name}/ and {index[name]}")
            index[name] = f"{folder.name}/{name}"
    return classes, index


def official_split(index: dict[str, str]) -> dict[str, list[str]]:
    split = {}
    for s in SPLITS:
        url = OFFICIAL_URL.format(split=s)
        with urllib.request.urlopen(url, timeout=30) as resp:
            names = [ln.strip() for ln in resp.read().decode("utf-8").splitlines() if ln.strip()]
        missing = [n for n in names if n not in index]
        if missing:
            raise RuntimeError(f"{len(missing)} files from the official {s} list are not in the dataset, e.g. {missing[:3]}")
        split[s] = [index[n] for n in names]
    return split


def random_split(classes: list[str], index: dict[str, str], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for c in classes:
        files = sorted(p for p in index.values() if p.startswith(c + "/"))
        rng.shuffle(files)
        start = 0
        for s in SPLITS:
            n = PER_CLASS_RANDOM[s]
            split[s].extend(sorted(files[start:start + n]))
            start += n
    return split


def check(split: dict[str, list[str]], index: dict[str, str]) -> None:
    seen: set[str] = set()
    for s in SPLITS:
        overlap = seen.intersection(split[s])
        if overlap:
            raise RuntimeError(f"{len(overlap)} images appear in more than one split")
        seen.update(split[s])
    if len(seen) != len(index):
        print(f"  warning: splits cover {len(seen)} of {len(index)} images")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="NWPU-RESISC45", help="folder with one sub-folder per class")
    ap.add_argument("--out", default="data", help="output folder inside the repo")
    ap.add_argument("--source", choices=["auto", "official", "random"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)
    if not root.is_dir():
        sys.exit(f"Dataset folder not found: {root.resolve()}")
    if (out / "splits" / "train.csv").exists() and not args.force:
        sys.exit("Split files already exist. They must stay fixed for the whole project; use --force only if you really mean it.")

    print(f"Scanning {root.resolve()} ...")
    classes, index = scan_dataset(root)
    print(f"  {len(classes)} classes, {len(index)} images")
    if len(classes) != EXPECTED_CLASSES or len(index) != EXPECTED_CLASSES * EXPECTED_PER_CLASS:
        print(f"  warning: expected {EXPECTED_CLASSES} classes x {EXPECTED_PER_CLASS} images")

    source = args.source
    split = None
    if source in ("auto", "official"):
        try:
            split = official_split(index)
            source = "official"
        except Exception as exc:  # no internet, blocked host, changed files
            if args.source == "official":
                sys.exit(f"Official split failed: {exc}")
            print(f"  official split not available ({exc}); using the seeded random split")
    if split is None:
        split = random_split(classes, index, args.seed)
        source = "random"
    check(split, index)

    label_of = {c: i for i, c in enumerate(classes)}
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    counts = {}
    for s in SPLITS:
        with open(out / "splits" / f"{s}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "class"])
            for rel in split[s]:
                c = rel.split("/")[0]
                w.writerow([rel, label_of[c], c])
        counts[s] = len(split[s])
    info = {
        "source": source,
        "official_url": OFFICIAL_URL if source == "official" else None,
        "seed": args.seed if source == "random" else None,
        "counts": counts,
        "num_classes": len(classes),
        "created": dt.date.today().isoformat(),
    }
    (out / "splits" / "split_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}/classes.txt and {out}/splits/ ({source} split): {counts}")


if __name__ == "__main__":
    main()
