"""
Model registry. Add a new architecture by writing a module with a
`build(cfg) -> nn.Module` function and registering it below -- nothing
elsewhere in the codebase needs to change (train.py / evaluate.py just call
`get_model(cfg)`).

Every model receives inputs of shape (B, 4, H, W) -- the four hourly bracket
frames at t-1h, t, t+1h, t+2h -- and must return outputs of shape (B, 3, H, W)
-- the three interior 15-minute frames at t+15m, t+30m, t+45m, in normalized
(log1p) space.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import torch.nn as nn

from . import convlstm, flow_refine, unet3d

_REGISTRY: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {
    "flow_refine": flow_refine.build,
    "unet3d": unet3d.build,
    "convlstm": convlstm.build,
}


def get_model(cfg: Dict[str, Any]) -> nn.Module:
    name = cfg["model"]["name"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](cfg)
