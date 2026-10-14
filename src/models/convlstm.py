"""
Baseline architecture: a ConvLSTM encoder-decoder.

The encoder consumes the 4 bracket frames sequentially and carries a hidden
state forward in time; the decoder then unrolls 3 more steps, feeding each
predicted frame back in autoregressively, to produce the 3 interior frames.
Unlike `flow_refine`, motion is entirely implicit in the learned recurrent
dynamics -- included as a second baseline (representing the "sequence model"
family, as opposed to `unet3d`'s "single-shot volumetric" family).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .layers import softplus_output


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding)

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        combined = torch.cat([x, h], dim=1)
        i, f, o, g = torch.split(self.conv(combined), self.hidden_ch, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, b: int, h: int, w: int, device: torch.device):
        z = torch.zeros(b, self.hidden_ch, h, w, device=device)
        return z, z.clone()


class ConvLSTMStack(nn.Module):
    def __init__(self, in_ch: int, hidden_channels: List[int], kernel_size: int = 3):
        super().__init__()
        self.cells = nn.ModuleList()
        cur_in = in_ch
        for h_ch in hidden_channels:
            self.cells.append(ConvLSTMCell(cur_in, h_ch, kernel_size))
            cur_in = h_ch

    def init_states(self, b: int, h: int, w: int, device: torch.device):
        return [cell.init_state(b, h, w, device) for cell in self.cells]

    def step(self, x: torch.Tensor, states):
        new_states = []
        for cell, state in zip(self.cells, states):
            h_state, c_state = cell(x, state)
            new_states.append((h_state, c_state))
            x = h_state
        return x, new_states


class ConvLSTMSeq2Seq(nn.Module):
    def __init__(self, hidden_channels: List[int], kernel_size: int = 3):
        super().__init__()
        self.encoder = ConvLSTMStack(1, hidden_channels, kernel_size)
        self.decoder = ConvLSTMStack(1, hidden_channels, kernel_size)
        self.out_conv = nn.Conv2d(hidden_channels[-1], 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, H, W)."""
        b, t_len, h, w = x.shape
        device = x.device
        frames = [x[:, i : i + 1] for i in range(t_len)]

        states = self.encoder.init_states(b, h, w, device)
        for frame in frames:
            _, states = self.encoder.step(frame, states)

        # decoder starts from the encoder's final state and autoregressively
        # feeds back its own previous prediction (last real frame to start).
        dec_states = states
        prev_frame = frames[-1]
        outputs = []
        for _ in range(3):
            top_h, dec_states = self.decoder.step(prev_frame, dec_states)
            pred = self.out_conv(top_h)
            outputs.append(pred)
            prev_frame = pred

        return softplus_output(torch.cat(outputs, dim=1))


def build(cfg: Dict[str, Any]) -> nn.Module:
    mcfg = cfg["model"].get("convlstm", {})
    return ConvLSTMSeq2Seq(
        hidden_channels=mcfg.get("hidden_channels", [32, 32]),
        kernel_size=mcfg.get("kernel_size", 3),
    )
