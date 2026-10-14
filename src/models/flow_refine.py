"""
Warp-then-refine architecture (recommended default).

STILT footprints evolve mostly through advection (mean-wind displacement) plus
turbulent spreading/decay. Rather than asking a black-box image-to-image net
to implicitly discover that motion, this model:

  1. Estimates a single bidirectional optical-flow-style field between the two
     bracketing hourly frames at t and t+1h (a small U-Net taking all 4 input
     frames as context, so it can see acceleration/curvature from t-1h/t+2h too).
  2. For each interior timestep at fractional position alpha in {0.25, 0.5, 0.75}
     within [t, t+1h], linearly scales that flow field by alpha (forward, from
     the t frame) and by (1 - alpha) (backward, from the t+1h frame) -- this is
     the standard trick from video frame interpolation (e.g. Super SloMo),
     assuming locally-linear motion within the hour.
  3. Warps both bracket frames toward the interior timestep, blends them with a
     learned pixel-wise weight, and cleans up the result with a small residual
     refinement CNN (this is what captures turbulent growth/decay that pure
     advection can't -- footprints don't just translate, their magnitude and
     spread change too).

Output is passed through softplus, since footprints are non-negative.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock, softplus_output

INTERIOR_ALPHAS = (0.25, 0.5, 0.75)  # t+15m, t+30m, t+45m as fractions of the [t, t+1h] interval


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp `img` (B,C,H,W) by per-pixel displacement `flow` (B,2,H,W) in pixel units (dx, dy)."""
    b, c, h, w = img.shape
    device = img.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(b, -1, -1, -1)  # (B,2,H,W)
    sample_grid = base_grid + flow

    grid_x = 2.0 * sample_grid[:, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * sample_grid[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (B,H,W,2)

    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class FlowEstimator(nn.Module):
    """Small 2-level U-Net predicting bidirectional flow between the t and t+1h frames,
    conditioned on all 4 bracket frames for extra context."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.in_conv = ConvBlock(4, hidden)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(hidden, hidden * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(hidden * 2, hidden * 4))
        self.up1 = nn.ConvTranspose2d(hidden * 4, hidden * 2, 2, stride=2)
        self.conv_up1 = ConvBlock(hidden * 4, hidden * 2)
        self.up2 = nn.ConvTranspose2d(hidden * 2, hidden, 2, stride=2)
        self.conv_up2 = ConvBlock(hidden * 2, hidden)
        self.out_conv = nn.Conv2d(hidden, 4, kernel_size=3, padding=1)  # (F01_x, F01_y, F10_x, F10_y)

    @staticmethod
    def _match_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return torch.cat([skip, x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        u1 = self.up1(s2)
        u1 = self.conv_up1(self._match_and_cat(u1, s1))
        u2 = self.up2(u1)
        u2 = self.conv_up2(self._match_and_cat(u2, s0))
        return self.out_conv(u2)


class RefineNet(nn.Module):
    """Blends the two warped estimates and corrects for non-advective changes
    (growth/decay of the footprint, not just its position)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        # inputs: warped_fwd(1) + warped_bwd(1) + blended(1) + alpha map(1) + 4 raw bracket frames = 8
        self.net = nn.Sequential(
            ConvBlock(8, hidden),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        warped_fwd: torch.Tensor,
        warped_bwd: torch.Tensor,
        blend_weight: torch.Tensor,
        alpha_map: torch.Tensor,
        raw_inputs: torch.Tensor,
    ) -> torch.Tensor:
        blended = blend_weight * warped_fwd + (1 - blend_weight) * warped_bwd
        x = torch.cat([warped_fwd, warped_bwd, blended, alpha_map, raw_inputs], dim=1)
        residual = self.net(x)
        return blended + residual


class FlowRefineNet(nn.Module):
    def __init__(self, flow_hidden: int = 32, refine_hidden: int = 32):
        super().__init__()
        self.flow_estimator = FlowEstimator(hidden=flow_hidden)
        self.blend_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)  # learned blend weight, per alpha reused
        self.refine_net = RefineNet(hidden=refine_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, H, W) = frames at [t-1h, t, t+1h, t+2h] in normalized space."""
        b, _, h, w = x.shape
        frame_t = x[:, 1:2]      # (B,1,H,W)
        frame_t1h = x[:, 2:3]    # (B,1,H,W)

        flow_field = self.flow_estimator(x)  # (B,4,H,W)
        f01 = flow_field[:, 0:2]  # t -> t+1h
        f10 = flow_field[:, 2:4]  # t+1h -> t

        raw_blend_logit = self.blend_head(x)  # (B,1,H,W), reused across alphas (scaled below)

        outputs = []
        for alpha in INTERIOR_ALPHAS:
            warped_fwd = warp(frame_t, alpha * f01)
            warped_bwd = warp(frame_t1h, (alpha - 1.0) * f10)  # (alpha - 1) = -(1 - alpha)

            # blend weight should favor the temporally closer frame; start from the
            # learned logit and nudge it by how close alpha is to each endpoint.
            alpha_bias = torch.full((b, 1, h, w), float(1.0 - alpha), device=x.device)
            blend_weight = torch.sigmoid(raw_blend_logit + torch.logit(alpha_bias.clamp(1e-3, 1 - 1e-3)))

            alpha_map = torch.full((b, 1, h, w), float(alpha), device=x.device)
            out = self.refine_net(warped_fwd, warped_bwd, blend_weight, alpha_map, x)
            outputs.append(out)

        return softplus_output(torch.cat(outputs, dim=1))  # (B, 3, H, W)


def build(cfg: Dict[str, Any]) -> nn.Module:
    mcfg = cfg["model"].get("flow_refine", {})
    return FlowRefineNet(
        flow_hidden=mcfg.get("flow_hidden_channels", 32),
        refine_hidden=mcfg.get("refine_hidden_channels", 32),
    )
