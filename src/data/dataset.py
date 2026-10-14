"""
PyTorch Dataset for the STILT footprint interpolation task.

Each item:
  inputs  : FloatTensor (4, H, W) -- real footprints at t-1h, t, t+1h, t+2h
  targets : FloatTensor (3, H, W) -- real footprints at t+15m, t+30m, t+45m
  meta    : dict with the sample's date/time and file path (for plotting/debugging)

Values are cropped to a (2*crop_half+1) x (2*crop_half+1) window centered on
the grid cell nearest the fixed receptor location, then normalized (see
transforms.py).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from .transforms import normalize

logger = logging.getLogger(__name__)


def _nearest_index(coord: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coord - value)))


def compute_crop_slices(
    lat: np.ndarray, lon: np.ndarray, receptor_lat: float, receptor_lon: float, crop_half: int
) -> Tuple[slice, slice, Tuple[int, int, int, int]]:
    """
    Returns (lat_slice, lon_slice, pad) where pad = (pad_top, pad_bottom, pad_left, pad_right)
    -- non-zero only if the requested crop would run off the edge of the raw grid, in which
    case the caller should zero-pad by that amount to keep a fixed output size.
    """
    lat_idx = _nearest_index(lat, receptor_lat)
    lon_idx = _nearest_index(lon, receptor_lon)

    lat_lo, lat_hi = lat_idx - crop_half, lat_idx + crop_half + 1
    lon_lo, lon_hi = lon_idx - crop_half, lon_idx + crop_half + 1

    pad_top = max(0, -lat_lo)
    pad_left = max(0, -lon_lo)
    pad_bottom = max(0, lat_hi - lat.shape[0])
    pad_right = max(0, lon_hi - lon.shape[0])

    lat_slice = slice(max(0, lat_lo), min(lat.shape[0], lat_hi))
    lon_slice = slice(max(0, lon_lo), min(lon.shape[0], lon_hi))
    return lat_slice, lon_slice, (pad_top, pad_bottom, pad_left, pad_right)


class _FileCache:
    """Tiny LRU-ish cache of open xarray Datasets, one per worker process."""

    def __init__(self, max_open: int = 4):
        self.max_open = max_open
        self._store: Dict[str, xr.Dataset] = {}
        self._order: List[str] = []

    def get(self, path: str) -> xr.Dataset:
        if path in self._store:
            self._order.remove(path)
            self._order.append(path)
            return self._store[path]

        ds = xr.open_dataset(path, cache=False)
        self._store[path] = ds
        self._order.append(path)

        if len(self._order) > self.max_open:
            oldest = self._order.pop(0)
            self._store.pop(oldest).close()
        return ds

    def close_all(self) -> None:
        for ds in self._store.values():
            ds.close()
        self._store.clear()
        self._order.clear()


class StiltFootprintDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, cfg: Dict[str, Any], split: str):
        assert split in {"train", "val", "test"}
        self.rows = index_df[index_df["split"] == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"No samples found for split='{split}'.")

        data_cfg = cfg["data"]
        self.var_name = data_cfg["var_name"]
        self.time_dim = data_cfg["time_dim"]
        self.lat_name = data_cfg["lat_name"]
        self.lon_name = data_cfg["lon_name"]
        self.native_interval_min = data_cfg["native_interval_minutes"]
        self.receptor_lat = data_cfg["receptor_lat"]
        self.receptor_lon = data_cfg["receptor_lon"]
        self.crop_half = data_cfg["crop_half"]
        self.bracket_offsets_hours = data_cfg["bracket_offsets_hours"]
        self.interior_offsets_minutes = data_cfg["interior_offsets_minutes"]
        self.normalization = data_cfg["normalization"]
        self.log1p_scale = float(data_cfg["log1p_scale"])

        self._cache = _FileCache()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_crop(self, ds: xr.Dataset, timestamp: pd.Timestamp) -> np.ndarray:
        lat = ds[self.lat_name].values
        lon = ds[self.lon_name].values
        lat_slice, lon_slice, pad = compute_crop_slices(
            lat, lon, self.receptor_lat, self.receptor_lon, self.crop_half
        )
        frame = (
            ds[self.var_name]
            .sel({self.time_dim: timestamp}, method="nearest")
            .isel({self.lat_name: lat_slice, self.lon_name: lon_slice})
            .values.astype(np.float32)
        )
        pad_top, pad_bottom, pad_left, pad_right = pad
        if any(pad):
            frame = np.pad(
                frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0.0
            )
            logger.warning(
                "Crop ran off grid edge (pad=%s) for %s; padded with zeros. "
                "Check receptor_lat/lon and crop_half in config.",
                pad,
                timestamp,
            )
        expected = 2 * self.crop_half + 1
        assert frame.shape == (expected, expected), f"Unexpected crop shape {frame.shape}"
        return frame

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows.iloc[idx]
        t = pd.Timestamp(row["t"])
        ds = self._cache.get(row["file_path"])

        input_frames = []
        for h in self.bracket_offsets_hours:
            ts = t + pd.Timedelta(hours=h)
            input_frames.append(self._load_crop(ds, ts))

        target_frames = []
        for m in self.interior_offsets_minutes:
            ts = t + pd.Timedelta(minutes=m)
            target_frames.append(self._load_crop(ds, ts))

        inputs = np.stack(input_frames, axis=0)   # (4, H, W)
        targets = np.stack(target_frames, axis=0)  # (3, H, W)

        print("normalization:", self.normalization, type(self.normalization))
        print("log1p_scale:", self.log1p_scale, type(self.log1p_scale))

        inputs = normalize(inputs, self.normalization, self.log1p_scale)
        targets = normalize(targets, self.normalization, self.log1p_scale)

        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "meta": {"t": row["t"], "file_path": row["file_path"]},
        }

    def __del__(self):
        try:
            self._cache.close_all()
        except Exception:
            pass
