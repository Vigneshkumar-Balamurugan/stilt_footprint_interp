"""
Losses for footprint interpolation.

Plain pixel-wise MSE/L1 averaged uniformly over a (mostly near-zero) domain
under-weights the near-field region, which is small in area but is usually
what matters most for downstream flux attribution. `weighted_l1`/`weighted_mse`
weight each pixel by its (normalized) target magnitude so the loss pays more
attention to where the footprint actually has mass.

The optional mass-conservation term penalizes large changes in the *total*
footprint mass between consecutive predicted/real frames -- a soft physical
prior (footprints should spread/decay smoothly, not appear/disappear mass
between adjacent 15-minute steps) without hard-coding an exact conservation
law that doesn't strictly hold for this quantity.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _magnitude_weights(target: torch.Tensor, eps: float) -> torch.Tensor:
    w = target + eps
    # normalize so the mean weight per-sample is ~1 (keeps loss scale comparable across batches)
    w = w / w.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return w


class WeightedReconstructionLoss(nn.Module):
    def __init__(self, base: str = "l1", magnitude_weight_eps: float = 1e-3):
        super().__init__()
        assert base in {"l1", "mse"}
        self.base = base
        self.eps = magnitude_weight_eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = _magnitude_weights(target, self.eps)
        diff = pred - target
        per_pixel = diff.abs() if self.base == "l1" else diff.pow(2)
        return (per_pixel * weights).mean()


class MassConservationPenalty(nn.Module):
    """Penalizes large frame-to-frame changes in total (summed) footprint mass
    across the predicted interior sequence, relative to how much the real
    bracket frames change -- keeps predictions from having implausible jumps."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mass = pred.sum(dim=(2, 3))     # (B, 3)
        target_mass = target.sum(dim=(2, 3))  # (B, 3)
        pred_deltas = pred_mass[:, 1:] - pred_mass[:, :-1]
        target_deltas = target_mass[:, 1:] - target_mass[:, :-1]
        return (pred_deltas - target_deltas).abs().mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        loss_cfg = cfg["loss"]
        base = "l1" if loss_cfg["name"] == "weighted_l1" else "mse"
        self.recon = WeightedReconstructionLoss(base=base, magnitude_weight_eps=loss_cfg["magnitude_weight_eps"])
        self.mass_weight = loss_cfg.get("mass_conservation_weight", 0.0)
        self.mass_penalty = MassConservationPenalty() if self.mass_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon_loss = self.recon(pred, target)
        total = recon_loss
        parts = {"recon": recon_loss.detach()}
        if self.mass_penalty is not None:
            mass_loss = self.mass_penalty(pred, target)
            total = total + self.mass_weight * mass_loss
            parts["mass"] = mass_loss.detach()
        parts["total"] = total.detach()
        return {"loss": total, **parts}
