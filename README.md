# STILT Footprint Temporal Super-Resolution

Learn to reconstruct 15-minute STILT footprints from hourly ("subsampled") STILT footprints, so that in production STILT only needs to be run once per hour while the model fills in the three intermediate 15-minute frames. This repo trains and evaluates that model on **historical 15-minute STILT output** (which we subsample to hourly to create training pairs), so every intermediate frame has real ground truth to check against.

## Problem framing

For a target hourly interval `[t, t+1h]`, the model is given four **real**
hourly footprints:

```
t-1h    t    t+1h    t+2h
 |------|-----|-------|
        [ target interval ]
```

and predicts the three interior 15-minute footprints at `t+15m`, `t+30m`,
`t+45m`. Using frames on both sides of the target interval (not just the two
endpoints) gives the model information about how transport is *accelerating or turning*, not just its average direction — the same reason video frame
interpolation networks condition on more than two frames.

Only footprint-to-footprint information is used (no meteorology fields), per
project requirements.

## Data assumptions (edit in `config/config.yaml` if these change)

- One NetCDF file per day, e.g. `2024-03-14.nc`.
- Footprint variable `ma_foot`, dims `(recep_time, lat, lon)`.
- `recep_time` is nominally on a 15-minute grid, but **each day's start/end
  time is different** and is not known in advance — the dataset builder reads it from each file rather than assuming a fixed schedule.
- Grid spacing 0.1° x 0.1°; `lat`/`lon` are 1-D coordinate variables.
- Fixed receptor at `(lat=48.1509, lon=11.56861)`, not centered in the raw
  grid. We crop a `(2*crop_half+1) x (2*crop_half+1)` window (default
  41x41, `crop_half=20`) centered on the grid cell nearest the receptor.
- A candidate hourly boundary `t` is a valid training sample only if the    file has **continuous, gap-free** 15-minute coverage from `t-1h` through `t+2h`inclusive (i.e., >= 1h of margin on both sides of the target interval, as specified). Samples cannot span two days unless you explicitly enable cross-midnight stitching (off by default — see `config.yaml`).

## Project layout

```
config/config.yaml        Single source of truth for paths, crop size, model choice, training hyperparameters
src/data/index_builder.py Scans the raw-data folder once, writes a sample index (Parquet/CSV) of valid (file, t) pairs
src/data/dataset.py       PyTorch Dataset: reads the index, crops+normalizes on the fly, returns (inputs, targets)
src/data/transforms.py    Normalization (log1p + magnitude-weighted loss support) and its inverse
src/models/               One file per architecture, all registered in a common registry (models/__init__.py)
  layers.py                Shared conv blocks
  unet3d.py                 3-D U-Net baseline (treats the 4 input frames as a temporal depth axis)
  convlstm.py                ConvLSTM/ConvGRU seq2seq baseline
  flow_refine.py              Recommended default: learned warp (optical-flow-style) + residual refinement
src/losses.py              Magnitude-weighted reconstruction loss + optional mass-conservation penalty
src/train.py                Training loop: config-driven, checkpointing, TensorBoard logging, early stopping
src/evaluate.py              Test-set metrics (RMSE, weighted RMSE, spatial correlation, total-mass ratio) + example plots
scripts/                    Thin CLI wrappers around the above (what you actually run)
tests/                      Unit tests using synthetic NetCDF fixtures (no real data needed to run these)
```

## Why this architecture (`flow_refine`, the default)

STILT footprints move mostly by **advection** (mean wind) plus **turbulent
spreading**, not by unstructured pixel-value drift. A plain image-to-image
network (e.g. a naive U-Net mapping 4 frames -> 3 frames) has to *implicitly* learn to detect and reproduce that motion. `flow_refine` instead:

1. estimates a per-pixel motion field between the bracketing frames   explicitly,
2. warps the two nearest real frames toward each interior timestep in
proportion to how far in time it sits between them,
3. blends the two warped estimates and cleans up the result with a small
residual CNN (handles turbulent growth/decay that pure advection misses).

This tends to need less training data and degrades more gracefully than a
black-box regressor, which matters because generating STILT training data is
expensive. `unet3d` and `convlstm` are included as baselines to benchmark
against — swap architectures purely via `config.yaml`, no code changes needed.

## Quickstart

```bash
conda env create -f environment.yml
conda activate stilt-interp

# 1) One-time: scan the raw NetCDF folder and build the sample index
python scripts/build_index.py --config config/config.yaml

# 2) Train (architecture chosen by config.yaml -> model.name)
python scripts/train.py --config config/config.yaml

# 3) Evaluate on the held-out test split
python scripts/evaluate.py --config config/config.yaml --checkpoint runs/<run_id>/best.pt
```

Splitting is done **by day**, not by random sample, so no interval from a test day ever leaks (via overlapping bracket frames) into training — see
`src/data/index_builder.py::split_by_day`.

## Reproducibility

- All randomness (day split, weight init, DataLoader shuffling) is seeded from
  `config.yaml: seed`.
- The sample index is written once to disk (`index.parquet`) so every run
  trains/evaluates on the exact same samples.
- `runs/<run_id>/config.yaml` is a frozen copy of the config used for that run,
  saved automatically at the start of training.
