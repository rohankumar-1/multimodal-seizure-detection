#!/usr/bin/env python3
"""Batch-mean FactorCL InfoNCE/CLUB critics on held-out test windows (paper Table critic summary).

Usage::

  PYTHONPATH=. python scripts/eval_critics.py \\
    --task binary_label --factorcl-ckpt runs/factorcl_binary_label_kfold_seed0_k5/fold0/factorcl_best_val_auc.pt

  PYTHONPATH=. python scripts/eval_critics.py \\
    --task lateralization --keep-classes 2,3 \\
    --factorcl-ckpt runs/factorcl_lateralization_kfold_seed0_k5_joint/fold0/factorcl_best_val_auc.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import factorcl  # noqa: E402
from factorcl import FactorCLSupModel, LABEL_KEY  # noqa: E402
from lib import FilteredRemappedDataset  # noqa: E402
from preprocess import MultitaskFusionDataset, collate_multitask_fusion  # noqa: E402

BUCKET_NAMES = (
    "infonce_x1x2",
    "club_x1x2",
    "infonce_xy",
    "infonce_x1x2_cond",
    "club_x1x2_cond",
)

STREAM_NAMES = ("eeg", "ecg")


def _parse_keep_classes(s: str | None) -> tuple[int, ...] | None:
    if s is None:
        return None
    norm = s.strip().lower()
    if norm in ("", "none", "null", "all"):
        return None
    return tuple(sorted({int(x.strip()) for x in s.split(",") if x.strip()}))


def _linear_probe_auc(X: np.ndarray, y: np.ndarray, *, seed: int) -> float:
    if np.unique(y).size < 2 or X.shape[0] < 10:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for tr, va in skf.split(X, y):
        if np.unique(y[tr]).size < 2 or np.unique(y[va]).size < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ])
        pipe.fit(X[tr], y[tr])
        pr = pipe.predict_proba(X[va])[:, 1]
        aucs.append(float(roc_auc_score(y[va], pr)))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def _eval_batch_mi(
    model: FactorCLSupModel, batch: dict[str, torch.Tensor], label_key: str
) -> dict[str, float]:
    """Per-batch critic values; keys match training ``info`` dict plus per-stream xy."""
    zs = model._encode(batch)
    all_reps = model._projections(zs)
    y = batch[label_key].view(-1).long().to(zs[0].device)
    y_ohe = model._one_hot(y, model.num_classes)

    out: dict[str, float] = {}
    out["infonce_x1x2"] = float(
        model._pairwise_critic_sum(all_reps, 0, model.infonce_x1x2, conditional=False, y_ohe=None).item()
    )
    out["club_x1x2"] = float(
        model._pairwise_critic_sum(all_reps, 1, model.club_x1x2, conditional=False, y_ohe=None).item()
    )
    xy_sum = zs[0].new_zeros(())
    for si, reps in enumerate(all_reps):
        v = float(model.infonce_x1y(reps[2], y_ohe).item())
        out[f"infonce_xy_{STREAM_NAMES[si]}"] = v
        xy_sum = xy_sum + model.infonce_x1y(reps[2], y_ohe)
    out["infonce_xy"] = float(xy_sum.item())
    out["infonce_x1x2_cond"] = float(
        model._pairwise_critic_sum(all_reps, 3, model.infonce_x1x2_cond, conditional=True, y_ohe=y_ohe).item()
    )
    out["club_x1x2_cond"] = float(
        model._pairwise_critic_sum(all_reps, 4, model.club_x1x2_cond, conditional=True, y_ohe=y_ohe).item()
    )

    # Pairwise critics on the single EEG–ECG pair (bucket index → scalar).
    ai0, aj0 = all_reps[0][0], all_reps[1][0]
    ai1, aj1 = all_reps[0][1], all_reps[1][1]
    ai3 = torch.cat([all_reps[0][3], y_ohe], dim=-1)
    aj3 = torch.cat([all_reps[1][3], y_ohe], dim=-1)
    ai4 = torch.cat([all_reps[0][4], y_ohe], dim=-1)
    aj4 = torch.cat([all_reps[1][4], y_ohe], dim=-1)
    out["pair_infonce_x1x2"] = float(model.infonce_x1x2(ai0, aj0).item())
    out["pair_club_x1x2"] = float(model.club_x1x2(ai1, aj1).item())
    out["pair_infonce_x1x2_cond"] = float(model.infonce_x1x2_cond(ai3, aj3).item())
    out["pair_club_x1x2_cond"] = float(model.club_x1x2_cond(ai4, aj4).item())
    return out


@torch.no_grad()
def run_report(
    model: FactorCLSupModel,
    loader: DataLoader,
    device: str,
    *,
    label_key: str,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    mi_sums: dict[str, float] = {}
    mi_count = 0
    latents: dict[str, list[np.ndarray]] = {}
    ys: list[np.ndarray] = []

    for raw in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
        y = batch[label_key].view(-1).long().cpu().numpy()
        ys.append(y)

        batch_mi = _eval_batch_mi(model, batch, label_key)
        for k, v in batch_mi.items():
            mi_sums[k] = mi_sums.get(k, 0.0) + v
        mi_count += 1

        zs = model._encode(batch)
        all_reps = model._projections(zs)
        latents.setdefault("eeg_backbone", []).append(zs[0].cpu().numpy())
        latents.setdefault("ecg_backbone", []).append(zs[1].cpu().numpy())
        for bi, bname in enumerate(BUCKET_NAMES):
            latents.setdefault(f"eeg_proj_{bname}", []).append(all_reps[0][bi].cpu().numpy())
            latents.setdefault(f"ecg_proj_{bname}", []).append(all_reps[1][bi].cpu().numpy())
        feat = model._concat_bucket_features(all_reps)
        latents.setdefault("joint_concat", []).append(feat.cpu().numpy())

    y_all = np.concatenate(ys).astype(np.int64)
    mi_mean = {k: v / max(mi_count, 1) for k, v in mi_sums.items()}

    probes: dict[str, dict[str, Any]] = {}
    for key, chunks in latents.items():
        X = np.concatenate(chunks, axis=0).astype(np.float32)
        probes[key] = {
            "dim": int(X.shape[1]),
            "linear_probe_auc": _linear_probe_auc(X, y_all, seed=seed),
        }

    return {
        "n_eval": int(y_all.shape[0]),
        "per_class": [int((y_all == c).sum()) for c in sorted(np.unique(y_all))],
        "mi_critic_batch_mean": mi_mean,
        "linear_probes": probes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("binary_label", "lateralization"), required=True)
    p.add_argument("--factorcl-ckpt", required=True)
    p.add_argument("--eval-npz", default="data/processed_2s_128Hz/test.npz")
    p.add_argument("--keep-classes", default=None, help="e.g. 2,3 for lateralization")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/report/critics")
    args = p.parse_args()

    keep = _parse_keep_classes(args.keep_classes)
    if args.task == "lateralization" and keep is None:
        keep = (2, 3)

    device = factorcl.resolve_device(None)
    base = MultitaskFusionDataset(args.eval_npz, label_keys=[args.task])
    ds = base if keep is None else FilteredRemappedDataset(base, args.task, keep)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_multitask_fusion,
    )

    sample = ds[0]
    eeg_ch = int(sample["eeg"].shape[0])
    ecg_ch = int(sample["ecg"].shape[0])

    ckpt = torch.load(args.factorcl_ckpt, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise SystemExit(f"checkpoint missing 'model' key: {args.factorcl_ckpt}")
    num_classes = int(ckpt.get("num_classes", 2))
    embed_dim = int(ckpt.get("embed_dim", 32))
    sd = ckpt["model"]
    critic_hidden = int(ckpt.get("critic_hidden") or sd["infonce_x1x2._f.0.weight"].shape[0])
    critic_layers = int(ckpt.get("critic_layers") or factorcl.CRITIC_LAYERS)
    model = FactorCLSupModel(
        eeg_ch=eeg_ch,
        ecg_ch=ecg_ch,
        embed_dim=embed_dim,
        num_classes=num_classes,
        critic_hidden=critic_hidden,
        critic_layers=critic_layers,
        split_eeg_channels=False,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    label_key = str(ckpt.get("label_key") or args.task)
    report = run_report(model, loader, device, label_key=label_key, seed=args.seed)
    report["task"] = args.task
    report["label_key"] = args.task
    report["keep_classes_original"] = list(keep) if keep else None
    report["eval_npz"] = os.path.abspath(args.eval_npz)
    report["factorcl_ckpt"] = os.path.abspath(args.factorcl_ckpt)
    report["num_streams"] = 2
    report["bucket_roles"] = {
        "infonce_x1x2": "InfoNCE lower bound on shared cross-view information I(X_1;X_2)",
        "club_x1x2": "CLUB upper bound on view-private information (minimized in main loss)",
        "infonce_xy": "InfoNCE lower bound on view–label information I(X;Y), summed over views",
        "infonce_x1x2_cond": "Shared cross-view information conditioned on label",
        "club_x1x2_cond": "Private cross-view information conditioned on label",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"{args.task}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
