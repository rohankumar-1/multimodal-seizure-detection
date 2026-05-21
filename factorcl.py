"""
FactorCL training CLI. Model and loops live in ``lib.factorcl``; data splits in ``lib.splits``.

  python factorcl.py
  python factorcl.py --label binary_label --mode kfold
  python factorcl.py --label lateralization --mode pooled --seed 0
"""

from __future__ import annotations

import argparse

import torch

from lib.cli import parse_folds, parse_keep_classes, resolve_device  # noqa: F401 — eval scripts
from lib.factorcl import FactorCLSupModel, remap_factorcl_state_dict
from lib.factorcl.config import factorcl_loss_scale
from lib.factorcl.runner import FactorCLRunConfig, run_kfold, run_official, run_pooled
from lib.splits import DataConfig

# Re-exported for eval scripts and checkpoints
TRAIN_NPZ = "data/processed_2s_128Hz/train.npz"
VAL_NPZ = "data/processed_2s_128Hz/val.npz"
TEST_NPZ = "data/processed_2s_128Hz/test.npz"
LABEL_KEY = "lateralization"
LABEL_KEEP_CLASSES: tuple[int, ...] | None = (2, 3)
CRITIC_LAYERS = 1
CRITIC_HIDDEN = 128

_PRESETS: dict[str, dict] = {
    "lateralization": {
        "keep_classes": (2, 3),
        "class_names": ("left", "right"),
        "balance": True,
        "mode": "official",
    },
    "binary_label": {
        "keep_classes": None,
        "class_names": None,
        "balance": False,
        "mode": "kfold",
    },
}


def _preset(label: str) -> dict:
    return _PRESETS.get(label, {"keep_classes": None, "class_names": None, "balance": False, "mode": "official"})


def main() -> None:
    p = argparse.ArgumentParser(description="Train FactorCL on a binary label.")
    p.add_argument("--label", default=LABEL_KEY, help="npz label key (default lateralization)")
    p.add_argument("--mode", choices=("official", "pooled", "kfold"), default=None)
    p.add_argument("--keep-classes", default=None, help="e.g. 2,3 or none")
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bce-weight", type=float, default=5.0)
    p.add_argument("--club-lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--folds", default=None)
    p.add_argument("--split-eeg-channels", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--factorcl-loss-scale", type=float, default=None)
    p.add_argument("--eeg-pair-boost", type=float, default=1.0)
    args = p.parse_args()

    preset = _preset(args.label)
    keep = parse_keep_classes(args.keep_classes) if args.keep_classes is not None else preset["keep_classes"]
    mode = args.mode or preset["mode"]
    dev = resolve_device(args.device)
    nw = args.num_workers if args.num_workers is not None else (4 if dev in ("cuda", "mps") else 0)
    if dev in ("cuda", "mps"):
        torch.set_float32_matmul_precision("high")

    run = FactorCLRunConfig(
        data=DataConfig(
            label_key=args.label,
            keep_classes=keep,
            batch_size=args.batch_size,
            balance_train_sampler=preset["balance"],
            split_eeg_channels=args.split_eeg_channels,
            num_workers=max(0, nw),
            device=dev,
        ),
        max_epochs=args.max_epochs,
        lr=args.lr,
        bce_weight=args.bce_weight,
        club_lambda=args.club_lambda,
        factorcl_loss_scale=args.factorcl_loss_scale,
        eeg_pair_boost=args.eeg_pair_boost,
        patience=max(0, args.patience),
        remapped_class_names=preset.get("class_names"),
    )
    scale = factorcl_loss_scale(3 if args.split_eeg_channels else 2, args.factorcl_loss_scale)
    print(f"[config] label={args.label} mode={mode} device={dev} loss_scale={scale}")

    if mode == "official":
        run_official(run)
    elif mode == "pooled":
        run_pooled(run, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)
    else:
        run_kfold(run, seed=args.seed, n_splits=args.n_splits, folds=parse_folds(args.folds))


if __name__ == "__main__":
    main()
