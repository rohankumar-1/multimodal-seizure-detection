#!/usr/bin/env python3
"""Modality-dropout robustness: FactorCL vs SupFusion on held-out windows.

Reports classifier AUROC when ECG or EEG is zeroed at test time (no retraining).

Usage::

    PYTHONPATH=. python scripts/eval_dropout.py \\
      --factorcl-ckpt runs/factorcl_binary_label_kfold_seed0_k5/fold0/factorcl_best_val_auc.pt \\
      --supfusion-ckpt runs/supfusion/supfusion_bilinear.pt \\
      --label binary_label
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import factorcl  # noqa: E402
import supfusion_lat_loc as slt  # noqa: E402
from factorcl import FactorCLSupModel  # noqa: E402
from lib import (  # noqa: E402
    BilinearFusion,
    ConcatFusion,
    FilteredRemappedDataset,
    LowRankTensorFusion,
    MultimodalModel,
    SingleGeoHead,
    build_modality_encoders,
)
from supfusion import MLP  # noqa: E402
from preprocess import MultitaskFusionDataset, collate_multitask_fusion  # noqa: E402


# ---------- helpers ----------


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _parse_keep_classes(s: str | None) -> tuple[int, ...] | None:
    if s is None:
        return None
    norm = s.strip().lower()
    if norm in ("", "none", "null", "all"):
        return None
    return tuple(sorted({int(x.strip()) for x in s.split(",") if x.strip()}))


def _bin_auc_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Binary AUC from 2-column logits (softmax[:, 1] = P(class=1)), via numerically stable
    softmax. Returns NaN if y is single-class."""
    if np.unique(y).size < 2:
        return float("nan")
    z = logits - logits.max(axis=1, keepdims=True)
    p1 = np.exp(z[:, 1]) / np.exp(z).sum(axis=1)
    return float(roc_auc_score(y, p1))


# ---------- model loading ----------


@dataclass
class LoadedSupFusion:
    model: MultimodalModel
    modality_order: list[str]
    fusion_type: str
    head_kind: str  # "MLP" or "SingleGeoHead"
    head_hidden_dim: int  # dim of the post-trunk activation we treat as a latent
    fused_dim: int
    embed_dim: int
    eeg_ch: int
    ecg_ch: int
    num_classes: int
    split_eeg_channels: bool
    keep_classes_original: list[int] | None


def _resolve_supfusion_ckpt(ckpt_path: str, device: str) -> dict[str, Any]:
    """Returns a dict that always contains a ``model`` key plus architecture metadata.

    Two on-disk formats are supported:

      1. ``best.pt`` (saved by :class:`multimodal.train.Trainer`):
         has ``model_state_dict`` + training-only metadata, NO architecture info. In that
         case we look for ``last.pt`` in the same directory (saved by
         ``supfusion_lat_loc.save_last_checkpoint``) to fetch ``{task, eeg_ch, ecg_ch,
         embed_dim, modality_order, fusion_type, fused_dim, split_eeg_channels, ...}``.
      2. ``last.pt`` / ``supfusion.py`` checkpoints: have ``model`` directly.

    We always return ``best.pt``'s weights if available (they were selected on val AUC),
    merged onto ``last.pt``'s architecture metadata.
    """
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in payload:
        return payload
    if "model_state_dict" not in payload:
        raise SystemExit(
            f"unknown supfusion checkpoint format at {ckpt_path}; keys={list(payload)}"
        )
    sibling_last = os.path.join(os.path.dirname(ckpt_path), "last.pt")
    if not os.path.isfile(sibling_last):
        raise SystemExit(
            f"{ckpt_path} is a best.pt (training metadata only); cannot infer architecture. "
            f"Expected sibling `last.pt` at {sibling_last}, which is written by "
            "`supfusion_lat_loc.save_last_checkpoint` at end of training."
        )
    last = torch.load(sibling_last, map_location=device, weights_only=False)
    if "model" not in last:
        raise SystemExit(f"sibling {sibling_last} also missing 'model' key; keys={list(last)}")
    merged = dict(last)
    merged["model"] = payload["model_state_dict"]
    merged["_source"] = f"weights={ckpt_path}; arch={sibling_last}"
    return merged


def _build_supfusion_from_ckpt(ckpt_path: str, device: str) -> LoadedSupFusion:
    """Auto-detects old (``supfusion.py``) vs new (``supfusion_lat_loc.py``) checkpoints.

    Old payloads carry only ``{model, eeg_ch, ecg_ch, embed_dim, num_classes}`` and use an
    ``MLP(embed, embed, num_classes)`` head over a 2-modality bilinear fusion.

    New payloads add ``task``, ``modality_order``, ``fusion_type``, ``fused_dim``,
    ``split_eeg_channels`` and use ``SingleGeoHead(fused, 64, num_classes)``. Bilinear
    requires exactly 2 modalities (``split_eeg_channels=False``).
    """
    ckpt = _resolve_supfusion_ckpt(ckpt_path, device)
    eeg_ch = int(ckpt.get("eeg_ch", 2))
    ecg_ch = int(ckpt.get("ecg_ch", 1))
    embed_dim = int(ckpt.get("embed_dim", 32))
    num_classes = int(ckpt.get("num_classes", 2))
    keep = ckpt.get("keep_classes_original")
    if "_source" in ckpt:
        print(f"        (loaded via two-file resolve: {ckpt['_source']})")

    if "task" in ckpt:
        modality_order = list(ckpt.get("modality_order") or ["eeg", "ecg"])
        fusion_type = str(ckpt.get("fusion_type") or "bilinear")
        fused_dim = int(ckpt.get("fused_dim") or embed_dim)
        split_eeg = bool(ckpt.get("split_eeg_channels", False))
        encoders, _, fusion, fused_dim_built = slt.build_encoders_and_fusion(
            eeg_ch=eeg_ch,
            ecg_ch=ecg_ch,
            split_eeg_channels=split_eeg,
            fusion_type=fusion_type,
            embed_dim=embed_dim,
            lrtf_rank=int(ckpt.get("lrtf_rank") or slt.LRTF_RANK or 32),
        )
        assert fused_dim_built == fused_dim, (
            f"fused_dim mismatch: ckpt {fused_dim} vs reconstructed {fused_dim_built}"
        )
        head = SingleGeoHead(
            task_name=str(ckpt["task"]),
            fused_dim=fused_dim,
            hidden_dim=slt.HEAD_HIDDEN,
            num_classes=num_classes,
            dropout=slt.HEAD_DROPOUT,
        )
        model = MultimodalModel(
            encoders=encoders, fusion=fusion, head=head, fusion_modality_order=modality_order
        )
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        return LoadedSupFusion(
            model=model,
            modality_order=modality_order,
            fusion_type=fusion_type,
            head_kind="SingleGeoHead",
            head_hidden_dim=slt.HEAD_HIDDEN,
            fused_dim=fused_dim,
            embed_dim=embed_dim,
            eeg_ch=eeg_ch,
            ecg_ch=ecg_ch,
            num_classes=num_classes,
            split_eeg_channels=split_eeg,
            keep_classes_original=list(keep) if keep is not None else None,
        )

    # Old supfusion.py format: BilinearFusion + MLP head, 2 modalities, joint EEG.
    encoders_dict, _ = build_modality_encoders(
        eeg_ch, ecg_ch, embed_dim=embed_dim, split_eeg_channels=False
    )
    encoders = dict(encoders_dict.items())
    fusion = BilinearFusion(embed_dim, embed_dim, embed_dim)
    head = MLP(embed_dim, embed_dim, num_classes)
    model = MultimodalModel(
        encoders=encoders, fusion=fusion, head=head, fusion_modality_order=["eeg", "ecg"]
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return LoadedSupFusion(
        model=model,
        modality_order=["eeg", "ecg"],
        fusion_type="bilinear",
        head_kind="MLP",
        head_hidden_dim=embed_dim,
        fused_dim=embed_dim,
        embed_dim=embed_dim,
        eeg_ch=eeg_ch,
        ecg_ch=ecg_ch,
        num_classes=num_classes,
        split_eeg_channels=False,
        keep_classes_original=None,
    )


@dataclass
class LoadedFactorCL:
    model: FactorCLSupModel
    embed_dim: int
    num_classes: int
    label_key: str
    keep_classes_original: list[int] | None


def _build_factorcl_from_ckpt(ckpt_path: str, device: str) -> LoadedFactorCL:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise SystemExit(f"factorcl checkpoint missing 'model' key; keys={list(ckpt)}")
    eeg_ch = int(ckpt.get("eeg_ch", 2))
    ecg_ch = int(ckpt.get("ecg_ch", 1))
    embed_dim = int(ckpt.get("embed_dim", ckpt.get("EMBED_DIM", 32)))
    num_classes = int(ckpt.get("num_classes", 2))
    label_key = str(ckpt.get("label_key") or factorcl.LABEL_KEY)
    keep = ckpt.get("keep_classes_original")
    model = FactorCLSupModel(
        eeg_ch=eeg_ch, ecg_ch=ecg_ch, embed_dim=embed_dim, num_classes=num_classes
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return LoadedFactorCL(
        model=model,
        embed_dim=embed_dim,
        num_classes=num_classes,
        label_key=label_key,
        keep_classes_original=list(keep) if keep is not None else None,
    )


# ---------- data ----------


class _SplitEEGOnTheFly(Dataset):
    """Wraps a base dataset and inserts ``eeg_0`` / ``eeg_1`` views derived from ``eeg``.

    Mirrors :class:`supfusion_lat_loc.SplitEEGDataset` but is duplicated here so the
    comparison script can pull only what it needs from the existing modules.
    """

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base[idx]
        eeg = sample["eeg"]
        if eeg.ndim != 2 or eeg.shape[0] < 2:
            raise ValueError(f"split-eeg expects (C>=2, T); got {tuple(eeg.shape)}")
        sample["eeg_0"] = eeg[0:1].clone()
        sample["eeg_1"] = eeg[1:2].clone()
        return sample


def _build_eval_loader(
    *,
    label: str,
    keep_classes: tuple[int, ...] | None,
    npz_path: str,
    batch_size: int,
    needs_split_eeg: bool,
) -> tuple[DataLoader, int]:
    base = MultitaskFusionDataset(npz_path, label_keys=[label])
    n0 = len(base)
    ds: Dataset = base if keep_classes is None else FilteredRemappedDataset(base, label, keep_classes)
    if needs_split_eeg:
        ds = _SplitEEGOnTheFly(ds)
    n = len(ds)  # type: ignore[arg-type]
    print(f"[eval] {npz_path}: kept {n} / {n0} rows (filter={keep_classes}, split_eeg={needs_split_eeg})")
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multitask_fusion),
        n,
    )


def _apply_mask(batch: dict[str, torch.Tensor], mask: str) -> dict[str, torch.Tensor]:
    """Returns a new batch with the requested modality zeroed. Operates on the keys we may
    encounter: ``eeg``, ``eeg_0``, ``eeg_1``, ``ecg``.
    """
    if mask == "none":
        return batch
    out = dict(batch)
    if mask == "mask_eeg":
        for k in ("eeg", "eeg_0", "eeg_1"):
            if k in out:
                out[k] = torch.zeros_like(out[k])
        return out
    if mask == "mask_ecg":
        if "ecg" in out:
            out["ecg"] = torch.zeros_like(out["ecg"])
        return out
    raise ValueError(f"unknown mask {mask!r}; expected one of: none, mask_eeg, mask_ecg")


# ---------- per-model extractors ----------

FACTORCL_BUCKETS = (
    "eeg_backbone",
    "eeg_proj_infonce_x1x2",
    "eeg_proj_club_x1x2",
    "eeg_proj_infonce_xy",
    "eeg_proj_infonce_x1x2_cond",
    "eeg_proj_club_x1x2_cond",
    "ecg_backbone",
    "ecg_proj_infonce_x1x2",
    "ecg_proj_club_x1x2",
    "ecg_proj_infonce_xy",
    "ecg_proj_infonce_x1x2_cond",
    "ecg_proj_club_x1x2_cond",
    "joint_concat",
)

SUPFUSION_BUCKETS = ("eeg_z", "ecg_z", "fused", "head_hidden")


@torch.no_grad()
def _extract_factorcl(
    fact: LoadedFactorCL, loader: DataLoader, device: str, mask: str, label_key: str
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Returns ``(latents_by_bucket, y_true, classifier_logits)``."""
    bufs: dict[str, list[np.ndarray]] = {k: [] for k in FACTORCL_BUCKETS}
    ys: list[np.ndarray] = []
    logits_buf: list[np.ndarray] = []
    m = fact.model
    for raw in loader:
        batch = _apply_mask({k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}, mask)
        y = batch[label_key].view(-1).long().cpu().numpy()
        z1 = m.backbone_x1(batch["eeg"])
        z2 = m.backbone_x2(batch["ecg"])
        x1_reps, x2_reps = m._projections(z1, z2)
        feat = torch.cat(x1_reps + x2_reps, dim=-1)
        logits = m.classifier(feat)

        bufs["eeg_backbone"].append(z1.cpu().numpy())
        bufs["ecg_backbone"].append(z2.cpu().numpy())
        for i, key in enumerate((
            "eeg_proj_infonce_x1x2",
            "eeg_proj_club_x1x2",
            "eeg_proj_infonce_xy",
            "eeg_proj_infonce_x1x2_cond",
            "eeg_proj_club_x1x2_cond",
        )):
            bufs[key].append(x1_reps[i].cpu().numpy())
        for i, key in enumerate((
            "ecg_proj_infonce_x1x2",
            "ecg_proj_club_x1x2",
            "ecg_proj_infonce_xy",
            "ecg_proj_infonce_x1x2_cond",
            "ecg_proj_club_x1x2_cond",
        )):
            bufs[key].append(x2_reps[i].cpu().numpy())
        bufs["joint_concat"].append(feat.cpu().numpy())
        ys.append(y)
        logits_buf.append(logits.cpu().numpy())
    y_all = np.concatenate(ys).astype(np.int64)
    lat = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in bufs.items()}
    return lat, y_all, np.concatenate(logits_buf, axis=0)


@torch.no_grad()
def _extract_supfusion(
    sf: LoadedSupFusion, loader: DataLoader, device: str, mask: str, label_key: str
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    bufs: dict[str, list[np.ndarray]] = {k: [] for k in SUPFUSION_BUCKETS}
    ys: list[np.ndarray] = []
    logits_buf: list[np.ndarray] = []

    model = sf.model
    head = model.head

    def _eeg_z(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return EEG-side feature (concatenated across towers when split, else single)."""
        if sf.split_eeg_channels:
            z0 = model.encoders["eeg_0"](batch["eeg_0"])
            z1 = model.encoders["eeg_1"](batch["eeg_1"])
            return torch.cat([z0, z1], dim=-1)
        return model.encoders["eeg"](batch["eeg"])

    def _fused(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feats_list: list[torch.Tensor] = []
        for k in sf.modality_order:
            feats_list.append(model.encoders[k](batch[k]))
        return model.fusion(feats_list)

    for raw in loader:
        batch = _apply_mask({k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}, mask)
        y = batch[label_key].view(-1).long().cpu().numpy()

        z_eeg = _eeg_z(batch)
        z_ecg = model.encoders["ecg"](batch["ecg"])
        fused = _fused(batch)

        if sf.head_kind == "MLP":
            h_hidden = torch.relu(head.fc1(fused))
            logits = head.fc2(h_hidden)
        elif sf.head_kind == "SingleGeoHead":
            h_hidden = head.trunk(fused)
            logits_dict = head(fused)
            logits = next(iter(logits_dict.values()))
        else:
            raise RuntimeError(f"unknown head_kind {sf.head_kind!r}")

        bufs["eeg_z"].append(z_eeg.cpu().numpy())
        bufs["ecg_z"].append(z_ecg.cpu().numpy())
        bufs["fused"].append(fused.cpu().numpy())
        bufs["head_hidden"].append(h_hidden.cpu().numpy())
        ys.append(y)
        logits_buf.append(logits.cpu().numpy())

    y_all = np.concatenate(ys).astype(np.int64)
    lat = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in bufs.items()}
    return lat, y_all, np.concatenate(logits_buf, axis=0)


# ---------- driver ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--factorcl-ckpt", type=str, required=True)
    p.add_argument("--supfusion-ckpt", type=str, required=True)
    p.add_argument("--label", type=str, required=True, help="Label key in the npz, e.g. lateralization, binary_label.")
    p.add_argument(
        "--keep-classes",
        type=str,
        default=None,
        help="Comma-separated original class ids to keep (e.g. '2,3' for lateralization).",
    )
    p.add_argument(
        "--eval-npz",
        type=str,
        default=factorcl.TEST_NPZ,
        help="Path to npz to evaluate on. Default: official test.npz.",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir. Default: results/report/robustness/",
    )
    return p.parse_args()


def _infer_official_train_only(ckpt_path: str) -> bool:
    """True if the path *suggests* official split training (no pooled / k-fold in dirname)."""
    n = os.path.abspath(ckpt_path).lower()
    return "kfold" not in n and "pooled" not in n


def _format_markdown(report: dict[str, Any]) -> str:
    """Build a concise markdown report from the structured JSON payload."""
    lines: list[str] = []
    ep = report.get("eval_protocol") or {}
    if ep.get("both_on_test_npz_truly_held_out_inferred"):
        lines.append(
            "> **Eval protocol:** Inferred **official** training for both models "
            "(no `pooled` / `kfold` in checkpoint paths). Metrics below use **`test.npz` "
            "held out from training** (train = `train.npz`, val selection on `val.npz`)."
        )
    else:
        lines.append(
            "> **Warning — possible `test.npz` leakage:** At least one checkpoint path "
            "contains `pooled` or `kfold`, meaning training likely pooled the three npz "
            "files. Scoring on all of `test.npz` can then include rows seen during training. "
            "See `eval_protocol` in `comparison.json`."
        )
    lines.append("")
    lines.append(f"# FactorCL vs SupFusion — {report['label']!r} on `{report['eval_npz']}`")
    lines.append("")
    lines.append(
        f"n_eval = **{report['n_eval']}**, per-class = {report['per_class']}, "
        f"seed = {report['seed']}"
    )
    lines.append("")

    # ---- Part B: robustness ----
    lines.append("## Modality-dropout robustness (classifier AUC)")
    lines.append("")
    lines.append("| Setting | FactorCL AUC | Δ vs full | SupFusion AUC | Δ vs full |")
    lines.append("|---|---:|---:|---:|---:|")
    masks = ("none", "mask_ecg", "mask_eeg")
    f_full = report["per_mask"]["none"]["factorcl"]["classifier_auc"]
    s_full = report["per_mask"]["none"]["supfusion"]["classifier_auc"]
    for m in masks:
        f = report["per_mask"][m]["factorcl"]["classifier_auc"]
        s = report["per_mask"][m]["supfusion"]["classifier_auc"]
        if m == "none":
            df = ds = ""
        else:
            df = f"{f - f_full:+.3f}"
            ds = f"{s - s_full:+.3f}"
        lbl = {"none": "full input", "mask_ecg": "EEG only (ECG zeroed)", "mask_eeg": "ECG only (EEG zeroed)"}[m]
        lines.append(f"| {lbl} | {f:.3f} | {df} | {s:.3f} | {ds} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    keep = _parse_keep_classes(args.keep_classes)
    device = args.device

    print(f"[load] factorcl ckpt: {args.factorcl_ckpt}")
    fact = _build_factorcl_from_ckpt(args.factorcl_ckpt, device)
    print(
        f"        embed_dim={fact.embed_dim} num_classes={fact.num_classes} "
        f"label_key={fact.label_key!r} keep={fact.keep_classes_original}"
    )
    print(f"[load] supfusion ckpt: {args.supfusion_ckpt}")
    sf = _build_supfusion_from_ckpt(args.supfusion_ckpt, device)
    print(
        f"        head={sf.head_kind} fusion={sf.fusion_type} split_eeg={sf.split_eeg_channels} "
        f"modality_order={sf.modality_order} fused_dim={sf.fused_dim}"
    )

    loader_fact, n_eval = _build_eval_loader(
        label=args.label,
        keep_classes=keep,
        npz_path=args.eval_npz,
        batch_size=args.batch_size,
        needs_split_eeg=False,
    )
    if sf.split_eeg_channels:
        loader_sf, _ = _build_eval_loader(
            label=args.label,
            keep_classes=keep,
            npz_path=args.eval_npz,
            batch_size=args.batch_size,
            needs_split_eeg=True,
        )
    else:
        loader_sf = loader_fact

    # Compute per-mask metrics for both models.
    masks = ("none", "mask_ecg", "mask_eeg")
    per_mask: dict[str, dict[str, dict[str, Any]]] = {m: {"factorcl": {}, "supfusion": {}} for m in masks}
    y_global: np.ndarray | None = None
    per_class: list[int] = []

    for m in masks:
        # --- FactorCL pass ---
        _, y_f, logits_f = _extract_factorcl(fact, loader_fact, device, m, args.label)
        y_global = y_f if y_global is None else y_global
        if not per_class:
            per_class = np.bincount(y_f).tolist()
        cls_auc_f = _bin_auc_from_logits(logits_f, y_f)
        per_mask[m]["factorcl"] = {"classifier_auc": cls_auc_f}

        # --- SupFusion pass ---
        _, y_s, logits_s = _extract_supfusion(sf, loader_sf, device, m, args.label)
        assert np.array_equal(y_s, y_f), "label order mismatch between factorcl and supfusion loaders"
        cls_auc_s = _bin_auc_from_logits(logits_s, y_s)
        per_mask[m]["supfusion"] = {"classifier_auc": cls_auc_s}

        print(
            f"[mask={m:9s}] classifier AUC -- FactorCL {cls_auc_f:.4f}  |  SupFusion {cls_auc_s:.4f}"
        )

    fc_abs = os.path.abspath(args.factorcl_ckpt)
    sf_abs = os.path.abspath(args.supfusion_ckpt)
    fc_off = _infer_official_train_only(fc_abs)
    sf_off = _infer_official_train_only(sf_abs)
    test_npz_abs = os.path.abspath(args.eval_npz)
    default_test_abs = os.path.abspath(factorcl.TEST_NPZ)
    on_default_test = os.path.normpath(test_npz_abs) == os.path.normpath(default_test_abs)

    report: dict[str, Any] = {
        "label": args.label,
        "keep_classes_original": list(keep) if keep is not None else None,
        "eval_npz": test_npz_abs,
        "n_eval": int(n_eval),
        "per_class": per_class,
        "seed": args.seed,
        "eval_protocol": {
            "eval_is_default_official_test_npz": on_default_test,
            "factorcl_ckpt": fc_abs,
            "factorcl_official_train_only_inferred": fc_off,
            "supfusion_ckpt": sf_abs,
            "supfusion_official_train_only_inferred": sf_off,
            "both_on_test_npz_truly_held_out_inferred": bool(on_default_test and fc_off and sf_off),
        },
        "factorcl": {
            "ckpt": os.path.abspath(args.factorcl_ckpt),
            "embed_dim": fact.embed_dim,
            "num_classes": fact.num_classes,
            "label_key": fact.label_key,
            "keep_classes_original": fact.keep_classes_original,
        },
        "supfusion": {
            "ckpt": os.path.abspath(args.supfusion_ckpt),
            "head_kind": sf.head_kind,
            "fusion_type": sf.fusion_type,
            "modality_order": sf.modality_order,
            "fused_dim": sf.fused_dim,
            "split_eeg_channels": sf.split_eeg_channels,
            "embed_dim": sf.embed_dim,
            "num_classes": sf.num_classes,
            "keep_classes_original": sf.keep_classes_original,
        },
        "per_mask": per_mask,
    }

    out_dir = args.out_dir or os.path.join("results", "report", "robustness")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{args.label}.json")
    md_path = os.path.join(out_dir, f"{args.label}.md")
    with open(json_path, "w") as f:
        json.dump(_jsonable(report), f, indent=2)
    with open(md_path, "w") as f:
        f.write(_format_markdown(report))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
