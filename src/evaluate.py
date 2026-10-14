"""
Evaluate a trained model on the held-out test split.

Reports both the model's metrics and a naive linear-time-interpolation
baseline (interpolating pixel-wise between the real t and t+1h frames) --
always check that the learned model actually beats this cheap baseline
before trusting it operationally. This mirrors the X-STILT literature's own
practice of comparing a learned/weighted interpolation against a linear one.

Metrics (all computed in de-normalized, i.e. raw ma_foot, units):
  - MAE, RMSE: plain pixel-wise error
  - weighted_RMSE: RMSE weighted by target magnitude (matches the training loss'
    emphasis on the near-field, high-sensitivity region)
  - spatial_corr: per-sample Pearson correlation between predicted and true
    footprint fields, averaged
  - mass_ratio: mean(sum(pred)) / mean(sum(target)) -- 1.0 means no systematic
    bias in total footprint mass
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import StiltFootprintDataset
from src.data.transforms import denormalize
from src.models import get_model
from src.utils import get_device, get_logger, load_config


def linear_time_baseline(inputs_denorm: np.ndarray) -> np.ndarray:
    """inputs_denorm: (B,4,H,W) real frames at t-1h,t,t+1h,t+2h (de-normalized).
    Returns (B,3,H,W) linear-in-time interpolation between the t and t+1h frames."""
    frame_t = inputs_denorm[:, 1]
    frame_t1h = inputs_denorm[:, 2]
    alphas = np.array([0.25, 0.5, 0.75], dtype=np.float32).reshape(1, 3, 1, 1)
    return (1 - alphas) * frame_t[:, None] + alphas * frame_t1h[:, None]


def compute_metrics(pred: np.ndarray, target: np.ndarray, mag_eps: float = 1e-3) -> Dict[str, float]:
    diff = pred - target
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    weights = target + mag_eps
    weights = weights / weights.mean(axis=(1, 2, 3), keepdims=True).clip(min=1e-12)
    weighted_rmse = np.sqrt(np.mean(weights * diff**2))

    # per-sample, per-frame spatial correlation, then averaged
    b, t = pred.shape[0], pred.shape[1]
    corrs = []
    for i in range(b):
        for j in range(t):
            p, y = pred[i, j].ravel(), target[i, j].ravel()
            if p.std() < 1e-12 or y.std() < 1e-12:
                continue
            corrs.append(np.corrcoef(p, y)[0, 1])
    spatial_corr = float(np.mean(corrs)) if corrs else float("nan")

    mass_ratio = float(pred.sum(axis=(2, 3)).mean() / max(target.sum(axis=(2, 3)).mean(), 1e-12))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "weighted_RMSE": float(weighted_rmse),
        "spatial_corr": spatial_corr,
        "mass_ratio": mass_ratio,
    }


def plot_examples(inputs, targets, preds, baseline, out_dir: Path, n: int, logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, inputs.shape[0])
    labels = ["t+15m", "t+30m", "t+45m"]

    for i in range(n):
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        vmax = max(targets[i].max(), preds[i].max(), 1e-12)
        for j in range(3):
            axes[0, j].imshow(targets[i, j], vmin=0, vmax=vmax, cmap="viridis")
            axes[0, j].set_title(f"truth {labels[j]}")
            axes[1, j].imshow(preds[i, j], vmin=0, vmax=vmax, cmap="viridis")
            axes[1, j].set_title(f"model {labels[j]}")
            axes[2, j].imshow(baseline[i, j], vmin=0, vmax=vmax, cmap="viridis")
            axes[2, j].set_title(f"linear baseline {labels[j]}")
        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"example_{i:03d}.png", dpi=120)
        plt.close(fig)
    logger.info("Saved %d example plots to %s", n, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("evaluate")
    device = get_device(cfg["train"]["device"])

    index_df = pd.read_parquet(cfg["data"]["index_path"])
    test_ds = StiltFootprintDataset(index_df, cfg, split="test")
    loader = DataLoader(
        test_ds,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
    )

    model = get_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Loaded checkpoint from epoch %d (val_loss=%.5f)", ckpt["epoch"], ckpt["val_loss"])

    norm_method = cfg["data"]["normalization"]
    scale = cfg["data"]["log1p_scale"]

    all_preds, all_targets, all_baselines = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            preds = model(inputs)

            inputs_dn = denormalize(inputs.cpu().numpy(), norm_method, scale)
            targets_dn = denormalize(targets.cpu().numpy(), norm_method, scale)
            preds_dn = denormalize(preds.cpu().numpy(), norm_method, scale)
            baseline_dn = linear_time_baseline(inputs_dn)

            all_preds.append(preds_dn)
            all_targets.append(targets_dn)
            all_baselines.append(baseline_dn)

    preds_all = np.concatenate(all_preds, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)
    baseline_all = np.concatenate(all_baselines, axis=0)

    model_metrics = compute_metrics(preds_all, targets_all)
    baseline_metrics = compute_metrics(baseline_all, targets_all)

    logger.info("=== Model metrics (test set) ===")
    for k, v in model_metrics.items():
        logger.info("  %s: %.6f", k, v)
    logger.info("=== Linear-interpolation baseline metrics (test set) ===")
    for k, v in baseline_metrics.items():
        logger.info("  %s: %.6f", k, v)

    out_dir = Path(cfg["evaluate"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"source": "model", **model_metrics}, {"source": "linear_baseline", **baseline_metrics}]).to_csv(
        out_dir / "metrics.csv", index=False
    )

    # for plotting, re-grab a small batch of raw (normalized) inputs to pass through again cleanly
    example_batch = next(iter(loader))
    inputs = example_batch["inputs"].to(device)
    targets = example_batch["targets"].to(device)
    with torch.no_grad():
        preds = model(inputs)
    inputs_dn = denormalize(inputs.cpu().numpy(), norm_method, scale)
    targets_dn = denormalize(targets.cpu().numpy(), norm_method, scale)
    preds_dn = denormalize(preds.cpu().numpy(), norm_method, scale)
    baseline_dn = linear_time_baseline(inputs_dn)

    plot_examples(
        inputs_dn, targets_dn, preds_dn, baseline_dn, out_dir / "examples", cfg["evaluate"]["num_example_plots"], logger
    )

    logger.info("Wrote metrics.csv and example plots under %s", out_dir)


if __name__ == "__main__":
    main()
