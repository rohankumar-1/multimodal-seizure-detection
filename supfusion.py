"""
SupFusion binary seizure detection (official train/val/test splits).

  python supfusion.py --fusion bil
  python supfusion.py --fusion cat --label-key binary_label
"""

from __future__ import annotations

import argparse

import torch

from lib.cli import resolve_device
from lib.supfusion.binary import BinaryRunConfig, MLP, run_binary  # MLP re-exported for eval_dropout

__all__ = ["MLP", "main"]
from lib.splits import DataConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Train SupFusion on binary_label.")
    p.add_argument("--fusion", choices=("add", "bil", "cat"), default="bil")
    p.add_argument("--label-key", default="binary_label")
    p.add_argument("--out-dir", default="runs/supfusion")
    p.add_argument("--results-json", default=None)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    run_binary(
        BinaryRunConfig(
            data=DataConfig(
                label_key=args.label_key,
                batch_size=args.batch_size,
                keep_classes=None,
                device=device,
            ),
            fusion=args.fusion,
            lr=args.lr,
            max_epochs=args.max_epochs,
            out_dir=args.out_dir,
            results_json=args.results_json,
        )
    )


if __name__ == "__main__":
    main()
