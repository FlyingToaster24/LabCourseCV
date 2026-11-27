# LabCourseCV: NWPU-RESISC45 scene classification on edge hardware

TUM Lab Course Mobile Computer Vision (W26/27). Group: Alper, Hasan, Denis, Kevin.
Plan, owners and deadlines: `Mobile CV Lab Project Plan (NWPU-RESISC45).md` (export of the shared plan doc).

## Setup

With uv (recommended):

```
uv sync
```

With plain pip:

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

GPU: TensorFlow has no GPU support on native Windows. On a Linux machine with an NVIDIA GPU (PC pool) install
`pip install "tensorflow[and-cuda]==2.21.*"` instead of plain `tensorflow`.

## Dataset

The images are not in git. Every machine needs a copy of NWPU-RESISC45 (the folder with the 45 class folders),
and the environment variable `NWPU_ROOT` points to it:

```
$env:NWPU_ROOT = "D:\data\NWPU-RESISC45"      # Windows PowerShell
export NWPU_ROOT=/path/to/NWPU-RESISC45        # Linux
```

Without `NWPU_ROOT`, the folder `NWPU-RESISC45/` in the repo root is used (ignored by git).

The train/val/test split is fixed for the whole project and lives in git:
`data/classes.txt` and `data/splits/{train,val,test}.csv` (18,900 / 6,300 / 6,300 images, stratified,
seed 42, see `data/splits/split_info.json`). Do not regenerate them.

## Workflow: one experiment

Every experiment is one YAML file in `configs/` (only the keys that differ from the defaults in
`src/config.py`), or an existing config plus `--set` overrides.

```
python scripts/run_pipeline.py configs/baseline.yaml                         # while tuning: validation only
python scripts/run_pipeline.py configs/baseline.yaml --test --variants fp32 int8   # finished configuration
python scripts/run_pipeline.py configs/baseline.yaml --set exp_id=hp_lr3e-4 --set train.lr=3e-4
```

The pipeline runs these steps, which also work on their own:

| Step | Command | Output |
| --- | --- | --- |
| Train | `python -m src.train configs/baseline.yaml` | `runs/<exp_id>/model.keras`, `history.csv`, `train_summary.json` |
| Evaluate | `python -m src.evaluate <exp_id> --split val` | metrics, confusion matrix, predictions in `runs/<exp_id>/eval/` + a row in `results/results.csv` |
| Export | `python -m src.export_tflite <exp_id> --variants fp32 fp16 dynamic int8` | `runs/<exp_id>/model_<variant>.tflite` |
| Evaluate TFLite | `python -m src.evaluate <exp_id> --split test --variant int8` | as above, plus latency |
| Latency only | `python -m src.benchmark_latency <exp_id> --variant int8` | `runs/<exp_id>/latency_int8.json` |
| Figures | `python -m src.plots <exp_id>` | `figures/<exp_id>/` (PNG, 300 dpi, for the poster) |

Quick pipeline check (a few minutes, also without GPU): `python -m src.train configs/baseline.yaml --quick`.

`runs/` stays on your machine (copy finished runs to the NAS to share models). `results/results.csv`
and `figures/` are in git: pull before you run, commit and push right after, so rows from different
people do not conflict.

## Conventions

- Use `import tf_keras as keras`, not `tf.keras`. TF-MOT pruning and quantization-aware training only work with
  Keras 2 (`tf_keras`) models, and Keras 3 `.keras` files cannot be loaded by `tf_keras`.
- The data pipeline returns raw pixels in [0, 255]. Every model starts with
  `keras.layers.Rescaling(1 / 127.5, offset=-1)` (same as `mobilenet_v2.preprocess_input`), so exported
  TFLite models take raw pixels; the int8 model takes uint8 pixels directly.
- Training uses one-hot labels (needed for label smoothing and MixUp/CutMix); evaluation uses integer labels.
- Tune on `val`; evaluate on `test` only for finished configurations.
- Latency protocol: TFLite, batch 1, 224 x 224, 1 CPU thread, 50 warm-up + 200 timed runs, median.
  Compare only numbers from the same machine type.
- Never commit datasets, model weights, TFLite files or virtual environments (see `.gitignore`).

## Layout

```
configs/        one YAML per experiment (baseline.yaml = the frozen reference)
data/           classes.txt, splits/*.csv (fixed, in git)
docs/poster/    poster text drafts, one file per section
figures/        poster figures per experiment (in git)
results/        results.csv, one row per (experiment, variant, split)
scripts/        make_splits.py, smoke_test.py, run_pipeline.py
src/            config, data, augment, models, train, evaluate, export_tflite, benchmark_latency, plots
```
