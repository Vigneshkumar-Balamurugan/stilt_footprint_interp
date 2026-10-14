"""
Baseline architecture: a 3-D U-Net that treats the 4 bracket frames as a small
temporal "depth" axis (B, 1, T=4, H, W) and regresses the 3 interior frames
directly as output channels. No explicit motion modeling -- included as a
black-box baseline to check whether `flow_refine`'s inductive bias actually
earns its keep on your data.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import softplus_output


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, base_channels: int = 32, depth: int = 3):
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.in_conv = ConvBlock3D(1, base_channels)

        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(depth):
            self.downs.append(
                nn.Sequential(
                    nn.MaxPool3d((1, 2, 2)),  # never pool the small T=4 axis
                    ConvBlock3D(ch, ch * 2),
                )
            )
            ch *= 2

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(nn.ConvTranspose3d(ch, ch // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            self.up_convs.append(ConvBlock3D(ch, ch // 2))
            ch //= 2

        # collapse T=4 -> 3 output frames via a 1x1x1 conv over the time axis,
        # implemented as a linear map applied on the temporal dimension.
        self.temporal_project = nn.Conv3d(ch, ch, kernel_size=(4, 1, 1))
        self.out_conv = nn.Conv2d(ch, 3, kernel_size=3, padding=1)

    @staticmethod
    def _match_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return torch.cat([skip, x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, H, W) -> reshaped internally to (B, 1, T=4, H, W)."""
        x = x.unsqueeze(1)  # (B,1,4,H,W)

        skips = []
        h = self.in_conv(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        skips.pop()  # last one is `h` itself, not a skip target
        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            h = up(h)
            h = up_conv(self._match_and_cat(h, skip))

        h = self.temporal_project(h).squeeze(2)  # (B, ch, H, W), T axis collapsed to 1 then squeezed
        out = self.out_conv(h)
        return softplus_output(out)


def build(cfg: Dict[str, Any]) -> nn.Module:
    base_channels = cfg["model"].get("base_channels", 32)
    depth = cfg["model"].get("unet3d", {}).get("depth", 3)
    return UNet3D(base_channels=base_channels, depth=depth)
