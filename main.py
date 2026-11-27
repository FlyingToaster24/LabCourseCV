"""PyCharm entry point: edit the settings below and press Run (green triangle, configuration "main").

Does the same as
    python scripts/run_pipeline.py <CONFIG> [--set ...] [--test] --variants ...
and first installs missing packages (uv sync) if needed. Output appears in PyCharm's Run window;
the training log per epoch is in runs/<exp_id>/history.csv.

Training on a laptop CPU takes hours: keep the laptop plugged in and stop it from sleeping.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------- settings ----------------
CONFIG = "configs/baseline.yaml"
OVERRIDES: list[str] = []         # e.g. ["exp_id=hp_lr3e-4", "train.lr=3e-4"]
TEST = True                       # also evaluate on the test set (only for finished configurations)
VARIANTS = ["fp32", "int8"]       # TFLite variants to export and evaluate
CHECK_FIRST = True                # ~10-minute end-to-end check on a tiny subset before the real run
# ------------------------------------------

ROOT = Path(__file__).resolve().parent
REQUIRED = {"yaml": "pyyaml", "tensorflow": "tensorflow", "tf_keras": "tf-keras",
            "tensorflow_model_optimization": "tensorflow-model-optimization", "matplotlib": "matplotlib"}


def ensure_packages() -> None:
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"Missing packages: {', '.join(missing)}", flush=True)
    uv = shutil.which("uv")
    if uv:
        print(">>> uv sync", flush=True)
        subprocess.run([uv, "sync"], cwd=ROOT, check=True)
    else:
        print(">>> pip install (uv not found on PATH)", flush=True)
        subprocess.run([sys.executable, "-m", "ensurepip"], cwd=ROOT, check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=ROOT, check=True)


def pipeline(*extra: str) -> None:
    cmd = [sys.executable, "scripts/run_pipeline.py", CONFIG]
    cmd += [x for o in OVERRIDES for x in ("--set", o)]
    subprocess.run(cmd + list(extra), cwd=ROOT, check=True)


def main() -> None:
    ensure_packages()
    if CHECK_FIRST:
        print("\n===== End-to-end check on a tiny subset =====", flush=True)
        pipeline("--check", "--variants", *VARIANTS)
    print("\n===== Real run =====", flush=True)
    pipeline(*(["--test"] if TEST else []), "--variants", *VARIANTS)


if __name__ == "__main__":
    main()
