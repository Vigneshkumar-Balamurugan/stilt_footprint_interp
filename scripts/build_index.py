"""CLI: scan the raw NetCDF folder, build the sample index, assign the day-level split."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.index_builder import build_index, split_by_day
from src.utils import get_logger, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("build_index")

    index_df = build_index(cfg)
    index_df = split_by_day(
        index_df,
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        seed=cfg["seed"],
    )

    out_path = Path(cfg["data"]["index_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(out_path, index=False)
    logger.info("Wrote index with %d samples to %s", len(index_df), out_path)


if __name__ == "__main__":
    main()
