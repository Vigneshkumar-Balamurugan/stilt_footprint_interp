"""
Build the sample index for the STILT footprint interpolation dataset.

Each raw file is one day of STILT output. Because the receptor's observation
window (and therefore the footprint's time coverage) shifts from day to day,
we cannot assume a fixed daily schedule -- we read each file's actual time
coordinate and search it directly.

A "sample" corresponds to one target hourly interval [t, t+1h]. It is valid
only if the file has *gap-free* 15-minute-resolution footprints covering the
full span [t - 1h, t + 2h] (this both gives us the 4 hourly bracket frames
at t-1h, t, t+1h, t+2h, and guarantees the 3 interior 15-min ground-truth
frames at t+15m, t+30m, t+45m all exist).

Usage:
    python scripts/build_index.py --config config/config.yaml
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

_TOL = pd.Timedelta(seconds=30)  # tolerance for matching nominal 15-min-grid timestamps


@dataclass
class DayFile:
    path: Path
    date: pd.Timestamp
    times: pd.DatetimeIndex  # sorted, as found in the file


def _read_day_file(path: Path, time_dim: str,var_name: str,) -> Optional[DayFile]:

    try:
        with xr.open_dataset(path) as ds:
            if var_name not in ds.variables: 
                logger.warning("Variable %s not found in %s, skipping.", path.name, var_name)
                return None

            if time_dim not in ds.variables:
                logger.warning("Time dimension %s not found in %s, skipping.", path.name, time_dim)
                return None

            raw_times = pd.to_datetime(ds[time_dim].values)

        times = pd.DatetimeIndex(sorted(raw_times))
        date = pd.Timestamp(path.stem.split("_")[1])  # expects filenames like MUC_20240101_column_ERA5.nc
        return DayFile(path=path, date=date, times=times)
    
    except Exception as e:
        logger.error("Error reading %s: %s", path.name, e)
        return None


def _times_contains(times: pd.DatetimeIndex, target: pd.Timestamp) -> bool:
    """True if `target` is present in `times` within a small float/rounding tolerance."""
    if len(times) == 0:
        return False
    pos = times.searchsorted(target)
    for cand_idx in (pos - 1, pos, pos + 1):
        if 0 <= cand_idx < len(times) and abs(times[cand_idx] - target) <= _TOL:
            return True
    return False


def _required_timestamps(t: pd.Timestamp, native_interval_min: int) -> List[pd.Timestamp]:
    """All 15-min-grid timestamps needed for target interval [t, t+1h]: t-1h .. t+2h inclusive."""
    n_steps = int((3 * 60) / native_interval_min) + 1  # 1h before + 1h target + 1h after -> 3h span
    start = t - pd.Timedelta(hours=1)
    return [start + pd.Timedelta(minutes=native_interval_min * i) for i in range(n_steps)]


def _candidate_hour_marks(times: pd.DatetimeIndex) -> List[pd.Timestamp]:
    return [t for t in times if t.minute == 0 and t.second == 0]


def find_valid_targets_single_file(day: DayFile, native_interval_min: int) -> List[pd.Timestamp]:
    """Valid target-interval start times `t` fully contained (with gap-free coverage) in one file."""
    valid = []
    for t in _candidate_hour_marks(day.times):
        required = _required_timestamps(t, native_interval_min)
        if all(_times_contains(day.times, req) for req in required):
            valid.append(t)
    return valid


def find_valid_targets_cross_midnight(
    day: DayFile, next_day: Optional[DayFile], native_interval_min: int
) -> List[pd.Timestamp]:
    """Like above, but a sample may draw its trailing frames from the next day's file."""
    combined_times = day.times
    if next_day is not None:
        combined_times = pd.DatetimeIndex(sorted(list(day.times) + list(next_day.times)))
    valid = []
    for t in _candidate_hour_marks(day.times):
        required = _required_timestamps(t, native_interval_min)
        if all(_times_contains(combined_times, req) for req in required):
            valid.append(t)
    return valid


def build_index(cfg: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    var_name = data_cfg["var_name"]
    raw_dir = Path(data_cfg["raw_dir"])
    time_dim = data_cfg["time_dim"]
    native_interval_min = data_cfg["native_interval_minutes"]
    allow_cross_midnight = data_cfg.get("allow_cross_midnight", False)

    nc_files = sorted(raw_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found under {raw_dir}")

    logger.info("Reading time coordinates from %d files under %s", len(nc_files), raw_dir)

    day_files = []
    for p in nc_files:
        day_file = _read_day_file(p, time_dim, var_name)
        if day_file is not None:
            day_files.append(day_file)
    
    day_files.sort(key=lambda d: d.date)
    by_date = {d.date.normalize(): d for d in day_files}

    rows = []
    for i, day in enumerate(day_files):
        if allow_cross_midnight:
            next_day = by_date.get(day.date.normalize() + pd.Timedelta(days=1))
            valid_ts = find_valid_targets_cross_midnight(day, next_day, native_interval_min)
        else:
            valid_ts = find_valid_targets_single_file(day, native_interval_min)

        for t in valid_ts:
            rows.append(
                {
                    "file_path": str(day.path),
                    "date": day.date.strftime("%Y-%m-%d"),
                    "t": t.isoformat(),
                }
            )
        logger.info("%s: %d valid target intervals", day.path.name, len(valid_ts))

    if not rows:
        raise RuntimeError(
            "No valid samples found anywhere. Check native_interval_minutes, time_dim, "
            "and whether files really contain gap-free 15-minute coverage."
        )

    df = pd.DataFrame(rows)
    logger.info("Total samples across all days: %d", len(df))
    return df


def split_by_day(
    index_df: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> pd.DataFrame:
    """
    Assign each *day* (not each sample) to train/val/test. This prevents leakage:
    samples from the same day share overlapping bracket frames, so splitting at
    the sample level would let near-duplicate information appear in both train
    and test.
    """
    rng = np.random.RandomState(seed)
    unique_days = sorted(index_df["date"].unique())
    rng.shuffle(unique_days)

    n_days = len(unique_days)
    n_test = max(1, int(round(n_days * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n_days * val_frac))) if val_frac > 0 else 0
    n_train = n_days - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"val_frac + test_frac too large for {n_days} available days "
            f"(got n_train={n_train}). Reduce val_frac/test_frac or add more data."
        )

    test_days = set(unique_days[:n_test])
    val_days = set(unique_days[n_test : n_test + n_val])
    train_days = set(unique_days[n_test + n_val :])

    def assign(d: str) -> str:
        if d in test_days:
            return "test"
        if d in val_days:
            return "val"
        return "train"

    index_df = index_df.copy()
    index_df["split"] = index_df["date"].map(assign)
    logger.info(
        "Day split -> train: %d days / %d samples | val: %d days / %d samples | test: %d days / %d samples",
        len(train_days),
        (index_df["split"] == "train").sum(),
        len(val_days),
        (index_df["split"] == "val").sum(),
        len(test_days),
        (index_df["split"] == "test").sum(),
    )
    return index_df
