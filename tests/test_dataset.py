"""
Unit tests using small synthetic NetCDF files, so they run without any real
STILT data. Covers the two trickiest pieces of the pipeline:
  1. index_builder correctly finds valid target windows despite each day's
     footprint availability window differing.
  2. dataset cropping correctly centers on the receptor location and pads
     when the crop would run off the grid edge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.dataset import compute_crop_slices
from src.data.index_builder import build_index, split_by_day

RECEPTOR_LAT = 48.1509
RECEPTOR_LON = 11.56861


def _make_synthetic_day_file(
    path: Path, date: str, start_hour: float, n_steps: int, lat_size=60, lon_size=100
):
    """Creates one day's footprint file with `n_steps` 15-min steps starting at `start_hour`
    (fractional hours, e.g. 9.0 = 9:00, 12.25 = 12:15), on a grid where the receptor sits
    away from the array center (mirrors the real data)."""
    times = [
        pd.Timestamp(date) + pd.Timedelta(hours=start_hour) + pd.Timedelta(minutes=15 * i)
        for i in range(n_steps)
    ]
    # place receptor near, but not at, the grid center -- e.g. offset by a few cells
    lat = np.linspace(RECEPTOR_LAT - 3.5, RECEPTOR_LAT + 2.5, lat_size)
    lon = np.linspace(RECEPTOR_LON - 6.0, RECEPTOR_LON + 4.0, lon_size)

    rng = np.random.RandomState(0)
    data = rng.exponential(scale=1e-5, size=(n_steps, lat_size, lon_size)).astype(np.float32)

    ds = xr.Dataset(
        {"ma_foot": (("recep_time", "lat", "lon"), data)},
        coords={"recep_time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


def test_index_builder_handles_variable_daily_windows(tmp_path: Path):
    # Day 1: 09:00 to ~13:00 (17 steps) -> should yield some valid [t, t+1h] windows
    _make_synthetic_day_file(tmp_path / "2024-01-01.nc", "2024-01-01", start_hour=9.0, n_steps=17)
    # Day 2: 12:15 to ~16:00 (16 steps), different schedule -- should still work
    _make_synthetic_day_file(tmp_path / "2024-01-02.nc", "2024-01-02", start_hour=12.25, n_steps=16)
    # Day 3: too short to ever satisfy the 1h-before/1h-after margin -> zero valid windows
    _make_synthetic_day_file(tmp_path / "2024-01-03.nc", "2024-01-03", start_hour=10.0, n_steps=3)
    # Day 4: same schedule as day 1, just so we have enough days with samples for a 3-way split
    _make_synthetic_day_file(tmp_path / "2024-01-04.nc", "2024-01-04", start_hour=9.0, n_steps=17)

    cfg = {
        "data": {
            "raw_dir": str(tmp_path),
            "time_dim": "recep_time",
            "native_interval_minutes": 15,
            "allow_cross_midnight": False,
            "val_frac": 0.3,
            "test_frac": 0.3,
        },
        "seed": 0,
    }

    index_df = build_index(cfg)

    day1_targets = index_df[index_df["date"] == "2024-01-01"]["t"].tolist()
    day2_targets = index_df[index_df["date"] == "2024-01-02"]["t"].tolist()
    day3_targets = index_df[index_df["date"] == "2024-01-03"]["t"].tolist()

    # Day 1 covers 09:00-13:00; valid t on the hour with [t-1h, t+2h] inside that range: t=10:00, 11:00
    assert pd.Timestamp("2024-01-01 10:00:00").isoformat() in day1_targets
    assert pd.Timestamp("2024-01-01 11:00:00").isoformat() in day1_targets
    assert pd.Timestamp("2024-01-01 09:00:00").isoformat() not in day1_targets  # no 1h margin before

    # Day 2 covers 12:15-16:00 (not on-the-hour start); t=13:00 fails because t-1h=12:00 is
    # missing (file starts at 12:15), so the only valid target is t=14:00
    # (required range 13:00-16:00, fully inside 12:15-16:00).
    assert pd.Timestamp("2024-01-02 14:00:00").isoformat() in day2_targets
    assert pd.Timestamp("2024-01-02 13:00:00").isoformat() not in day2_targets

    # Day 3 is too short -> no valid windows at all
    assert len(day3_targets) == 0

    # split_by_day should not crash and should cover every sample exactly once
    split_df = split_by_day(index_df, val_frac=0.3, test_frac=0.3, seed=0)
    assert set(split_df["split"].unique()) <= {"train", "val", "test"}
    assert len(split_df) == len(index_df)


def test_crop_centers_on_receptor_and_pads_at_edges():
    lat = np.linspace(RECEPTOR_LAT - 0.5, RECEPTOR_LAT + 0.5, 15)  # receptor near center here
    lon = np.linspace(RECEPTOR_LON - 0.5, RECEPTOR_LON + 0.5, 15)

    lat_slice, lon_slice, pad = compute_crop_slices(lat, lon, RECEPTOR_LAT, RECEPTOR_LON, crop_half=5)
    assert pad == (0, 0, 0, 0)  # small crop, well within a 15x15 grid -> no padding needed
    assert (lat_slice.stop - lat_slice.start) == 11
    assert (lon_slice.stop - lon_slice.start) == 11

    # now request a crop bigger than the grid allows -> must pad, not crash or resize
    lat_slice, lon_slice, pad = compute_crop_slices(lat, lon, RECEPTOR_LAT, RECEPTOR_LON, crop_half=20)
    assert sum(pad) > 0
