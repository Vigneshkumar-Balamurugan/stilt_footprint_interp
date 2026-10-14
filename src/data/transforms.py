"""
Normalization for STILT footprint values.

Footprints are heavy-tailed and mostly near-zero (most of the domain has ~0
sensitivity; a small near-field region has most of the mass), which is hard
for a network to regress directly in raw units. log1p compresses the dynamic
range while keeping zeros at zero and preserving monotonic ordering, and its
inverse (expm1) exactly recovers the original scale.

`log1p_scale` rescales the raw values before applying log1p -- pick it so
that typical non-zero footprint values end up roughly in the O(1)-O(10)
range after multiplying (tune to your actual ma_foot units/magnitude).
"""
from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray, method: str, log1p_scale: float = 1.0) -> np.ndarray:
    if method == "none":
        return x
    if method == "log1p":
        x = np.clip(x, a_min=0.0, a_max=None)  # footprints should be >= 0; guard against tiny negatives
        return np.log1p(x * log1p_scale) 
    raise ValueError(f"Unknown normalization method: {method}")


def denormalize(x: np.ndarray, method: str, log1p_scale: float = 1.0) -> np.ndarray:
    if method == "none":
        return x
    if method == "log1p":
        return np.expm1(x) / float(log1p_scale)
    raise ValueError(f"Unknown normalization method: {method}")
