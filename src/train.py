from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import StiltFootprintDataset
from src.losses import CombinedLoss
from src.models import get_model
from src.utils import (
    count_parameters,
    freeze_config,
    get_device,
    get_logger,
    load_config,
    save_checkpoint,
    set_seed,
)


def build_dataloaders(cfg: Dict[str, Any]):
    index_df = pd.read_parquet(cfg["data"]["index_path"])
    if "split" not in index_df.columns:
        raise RuntimeError(
            "index.parquet has no 'split' column -- run scripts/build_index.py first "
            "(it builds the index AND assigns the train/val/test day split)."
        )

    dl_cfg = cfg["dataloader"]
    loaders = {}
    for split in ("train", "val", "test"):
        ds = StiltFootprintDataset(index_df, cfg, split=split)
        loaders[split] = DataLoader(
            ds,
            batch_size=dl_cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"],
            drop_last=(split == "train"),
        )
    return loaders


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    optimizer=None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
    log_every: int = 50,
    logger=None,
    epoch: int = 0,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0

    for step, batch in enumerate(loader):
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                preds = model(inputs)
                loss_dict = criterion(preds, targets)
                loss = loss_dict["loss"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if is_train and logger is not None and step % log_every == 0:
            logger.info("epoch %d step %d/%d loss=%.5f", epoch, step, len(loader), loss.item())

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None, help="Defaults to a random id under train.out_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = Path(cfg["train"]["out_dir"]) / run_id
    freeze_config(cfg, run_dir)

    logger = get_logger("train", log_file=run_dir / "train.log")
    logger.info("Run directory: %s", run_dir)

    device = get_device(cfg["train"]["device"])
    logger.info("Using device: %s", device)

    loaders = build_dataloaders(cfg)
    logger.info(
        "Samples -- train: %d, val: %d, test: %d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
    )

    model = get_model(cfg).to(device)
    logger.info("Model: %s (%d trainable params)", cfg["model"]["name"], count_parameters(model))

    criterion = CombinedLoss(cfg)

    train_cfg = cfg["train"]
    if train_cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")

    if train_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg["amp"] and device.type == "cuda"))
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()
        train_loss = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=train_cfg["grad_clip_norm"],
            log_every=train_cfg["log_every_n_steps"],
            logger=logger,
            epoch=epoch,
        )
        val_loss = run_epoch(model, loaders["val"], criterion, device)

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            },
            out_dir=run_dir,
            is_best=is_best,
        )

        logger.info(
            "epoch %d done in %.1fs | train_loss=%.5f val_loss=%.5f best=%.5f%s",
            epoch,
            time.time() - t0,
            train_loss,
            val_loss,
            best_val_loss,
            " (new best)" if is_best else "",
        )

        if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
            logger.info("Early stopping at epoch %d (no val improvement for %d epochs).", epoch, epochs_without_improvement)
            break

    writer.close()
    logger.info("Training complete. Best val loss: %.5f. Checkpoints in %s", best_val_loss, run_dir)


if __name__ == "__main__":
    main()
