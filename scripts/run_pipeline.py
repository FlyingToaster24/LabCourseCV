"""Run the whole chain for one config: train -> evaluate -> export -> evaluate TFLite -> figures.

    python scripts/run_pipeline.py configs/baseline.yaml                      # val only (while tuning)
    python scripts/run_pipeline.py configs/baseline.yaml --test --variants fp32 int8
    python scripts/run_pipeline.py configs/baseline.yaml --set train.lr=3e-4 --set exp_id=hp_lr3e-4
    python scripts/run_pipeline.py configs/baseline.yaml --check             # 10-minute end-to-end check

--test adds the test-set evaluation; use it only for finished configurations.
--check runs every step on a tiny training subset (run folder <exp_id>_quick, nothing is
logged to results.csv, figures go into the run folder) to catch problems before a long run.
Every step is a normal command you can also run on its own (printed before it runs).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_config  # noqa: E402


def run(args: list[str]) -> None:
    cmd = [sys.executable, "-m", *args]
    print("\n>>> python -m " + " ".join(args), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--test", action="store_true", help="also evaluate on the test set (final configurations only)")
    ap.add_argument("--variants", nargs="*", default=["fp32"], help="TFLite variants to export and evaluate")
    ap.add_argument("--skip-train", action="store_true", help="reuse an existing run folder")
    ap.add_argument("--check", action="store_true", help="quick end-to-end check, nothing logged")
    args = ap.parse_args()

    exp_id = load_config(args.config, args.overrides)["exp_id"]
    sets = [x for o in args.overrides for x in ("--set", o)]

    if args.check:
        exp_id += "_quick"
        run(["src.train", args.config, "--quick", *sets])
        run(["src.evaluate", exp_id, "--split", "val", "--no-log"])
        if args.variants:
            run(["src.export_tflite", exp_id, "--variants", *args.variants])
            for v in args.variants:
                run(["src.evaluate", exp_id, "--split", "val", "--variant", v, "--no-log"])
        run(["src.plots", exp_id, "--variant", "keras", "--split", "val",
             "--out", str(REPO_ROOT / "runs" / exp_id / "figures")])
        print(f"\nCheck passed: every step works (runs/{exp_id}/). Nothing was logged.")
        return

    if not args.skip_train:
        run(["src.train", args.config, *sets])
    splits = ["val"] + (["test"] if args.test else [])
    for split in splits:
        run(["src.evaluate", exp_id, "--split", split])
    if args.variants:
        run(["src.export_tflite", exp_id, "--variants", *args.variants])
        for v in args.variants:
            run(["src.evaluate", exp_id, "--split", splits[-1], "--variant", v])
    run(["src.plots", exp_id, "--variant", "keras", "--split", splits[-1]])
    print(f"\nDone: runs/{exp_id}/, figures/{exp_id}/, new rows in results/results.csv")


if __name__ == "__main__":
    main()
