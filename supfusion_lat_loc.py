"""
SupFusion geo tasks (lateralization / localization).

  python supfusion_lat_loc.py --task lateralization
  python supfusion_lat_loc.py --task lateralization --mode kfold --fusion-type bilinear --split-eeg-channels false
"""

from __future__ import annotations

import argparse

import torch

from lib.cli import parse_folds, resolve_device, str2bool
from lib.supfusion.geo import GeoRunConfig, run_kfold, run_official
from lib.splits import DataConfig

TASK_KEEP_CLASSES: dict[str, tuple[int, ...] | None] = {
    "lateralization": (2, 3),
    "localization": (2, 3, 4, 5, 6, 7, 8),
}
TASK_NUM_CLASSES = {"lateralization": 5, "localization": 9}
REMAPPED_CLASS_NAMES = {"lateralization": ("left", "right"), "localization": None}


def main() -> None:
    p = argparse.ArgumentParser(description="SupFusion lateralization or localization.")
    p.add_argument("--task", choices=tuple(TASK_KEEP_CLASSES), required=True)
    p.add_argument("--mode", choices=("official", "kfold"), default="official")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folds", default=None)
    p.add_argument("--fusion-type", choices=("bilinear", "concat", "lowrank_tensor"), default="concat")
    p.add_argument("--split-eeg-channels", type=str2bool, default=True)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GeoRunConfig(
        task=args.task,
        data=DataConfig(batch_size=args.batch_size, device=device),
        device=device,
        task_keep_classes=TASK_KEEP_CLASSES,
        task_num_classes=TASK_NUM_CLASSES,
        remapped_class_names=REMAPPED_CLASS_NAMES,
        fusion_type=args.fusion_type,
        split_eeg_channels=bool(args.split_eeg_channels),
        max_epochs=args.max_epochs,
        lr=args.lr,
    )
    suffix = f"{args.fusion_type}_{'split' if cfg.split_eeg_channels else 'joint'}"
    print(f"[config] task={args.task} mode={args.mode} fusion={args.fusion_type} eeg_split={cfg.split_eeg_channels}")

    if args.mode == "official":
        run_official(cfg)
    else:
        run_kfold(cfg, n_splits=args.n_splits, seed=args.seed, folds=parse_folds(args.folds), variant_suffix=suffix)


if __name__ == "__main__":
    main()
