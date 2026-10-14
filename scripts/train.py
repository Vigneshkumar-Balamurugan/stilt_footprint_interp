"""CLI: train a model. Thin wrapper so `python scripts/train.py --config ...` works
from the repo root without needing `python -m src.train`."""
from src.train import main

if __name__ == "__main__":
    main()
