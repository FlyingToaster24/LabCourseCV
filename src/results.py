"""results/results.csv: the single source of truth for every number on the poster.

One row per measurement: (experiment, model variant, split). Rows are only
appended, never edited. Pull before you run, commit and push right after.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "results" / "results.csv"

COLUMNS = [
    "date",
    "exp_id",
    "owner",
    "method",
    "variant",             # keras | fp32 | fp16 | dynamic | int8 (TFLite variants)
    "split",               # val | test | a robustness test name
    "top1",
    "top5",
    "macro_f1",
    "params",
    "size_mb",             # file size of this variant
    "latency_pc_ms",       # median, TFLite, batch 1, 1 thread (see benchmark_latency.py)
    "latency_device_ms",
    "acc_corruptions_mean",
    "acc_ucmerced",
    "config",              # config file + overrides used for training
    "notes",
]


def append_row(row: dict, path: Path = RESULTS_CSV) -> None:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"Unknown result columns: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    row = {"date": dt.date.today().isoformat(), **row}
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: _fmt(row.get(k, "")) for k in COLUMNS})
    print(f"Appended to {path}: {row.get('exp_id')} / {row.get('variant')} / {row.get('split')}")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return "" if v is None else v
