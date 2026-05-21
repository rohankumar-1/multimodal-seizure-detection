#!/usr/bin/env python3
"""Re-evaluate a SupFusion binary checkpoint and compare to frozen report JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib import BilinearFusion, ConcatFusion, AdditiveFusion, MultimodalModel
from lib.cli import resolve_device
from lib.encoders import build_modality_encoders
from lib.metrics import evaluate_binary_threshold, find_best_threshold_youden, metric_subset
from lib.splits import DataConfig, build_official_loaders
from lib.supfusion.binary import MLP, _predict_probs, _build_fusion

_FUSION_MAP = {"add": "additive", "bil": "bilinear", "cat": "concat"}
_METRICS = ("auc_score", "accuracy", "f1", "precision", "recall")


def load_model(ckpt_path: str, fusion: str, device: str) -> tuple[MultimodalModel, str]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    eeg_ch = int(ckpt.get("eeg_ch", 2))
    ecg_ch = int(ckpt.get("ecg_ch", 1))
    embed_dim = int(ckpt.get("embed_dim", 32))
    num_classes = int(ckpt.get("num_classes", 2))
    encoders, _ = build_modality_encoders(
        eeg_ch, ecg_ch, embed_dim=embed_dim, split_eeg_channels=False, dropout=0.1
    )
    fusion_mod, head_in = _build_fusion(fusion, embed_dim)
    model = MultimodalModel(
        encoders=dict(encoders.items()),
        fusion=fusion_mod,
        head=MLP(head_in, embed_dim, num_classes),
        fusion_modality_order=["eeg", "ecg"],
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), "binary_label"


def compare(ref: dict, got: dict, *, tol: float) -> list[str]:
    lines: list[str] = []
    for split in ("val", "test"):
        for k in _METRICS:
            a = float(ref[split][k])
            b = float(got[split][k])
            d = abs(a - b)
            ok = d <= tol
            lines.append(f"  {split}/{k}: ref={a:.6f} got={b:.6f} delta={d:.6f} {'OK' if ok else 'FAIL'}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", choices=("add", "bil", "cat"), default="bil")
    p.add_argument("--ckpt", default="runs/supfusion/supfusion_last.pt")
    p.add_argument("--reference", default=None)
    p.add_argument("--tol", type=float, default=1e-5, help="max abs diff for eval-only match")
    p.add_argument("--train", action="store_true", help="run full training first")
    args = p.parse_args()

    fusion = args.fusion
    ref_path = args.reference or os.path.join(
        ROOT, "results", "report", f"supfusion_binary_{_FUSION_MAP[fusion]}.json"
    )
    with open(ref_path) as f:
        ref = json.load(f)

    device = resolve_device("auto")
    out_json = os.path.join(ROOT, "results", "verify", f"supfusion_binary_{fusion}_rerun.json")

    if args.train:
        from lib.supfusion.binary import BinaryRunConfig, run_binary

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        run_binary(
            BinaryRunConfig(
                data=DataConfig(label_key="binary_label", batch_size=32, device=device),
                fusion=fusion,
                max_epochs=20,
                results_json=out_json,
            )
        )
        with open(out_json) as f:
            got = json.load(f)
    else:
        cfg = DataConfig(label_key="binary_label", batch_size=32, device=device)
        _, val_loader, test_loader, _ = build_official_loaders(cfg)
        model, label_key = load_model(args.ckpt, fusion, device)
        predict = lambda b: _predict_probs(model, b, device, label_key)
        thr = find_best_threshold_youden(val_loader, device, label_key=label_key, predict_probs=predict)
        got = {
            "fusion": fusion,
            "threshold_youden": thr,
            "val": metric_subset(
                evaluate_binary_threshold(val_loader, thr, device, label_key=label_key, predict_probs=predict),
                _METRICS,
            ),
            "test": metric_subset(
                evaluate_binary_threshold(test_loader, thr, device, label_key=label_key, predict_probs=predict),
                _METRICS,
            ),
            "checkpoint": args.ckpt,
        }
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(got, f, indent=2)

    print(f"\n=== compare to {ref_path} (tol={args.tol}) ===")
    print("\n".join(compare(ref, got, tol=args.tol)))
    test_auc_ref = ref["test"]["auc_score"]
    test_auc_got = got["test"]["auc_score"]
    print(f"\ntest AUC: ref={test_auc_ref:.6f} got={test_auc_got:.6f} delta={abs(test_auc_ref - test_auc_got):.6f}")
    if args.train:
        print("(training rerun — small deltas vs report are expected due to stochasticity)")


if __name__ == "__main__":
    main()
