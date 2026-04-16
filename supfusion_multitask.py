"""
SupFusion-style multi-task training: EEG + ECG encoders, fusion, and multiple classification heads.

This is a direct adaptation of `supfusion.py`, but with multiple `ClassificationTask`s trained
against the same label `y`:
  - `fusion`   : primary head on fused embedding
  - `eeg_aux`  : auxiliary head (trained on the same fused embedding; see note below)
  - `ecg_aux`  : auxiliary head (trained on the same fused embedding; see note below)

Note:
The processed `.npz` files only contain one supervision signal (`binary_label`), and the upstream
`multimodal` package's `MultimodalModel` head is only given the fused embedding. So the auxiliary
heads are trained on the fused embedding as well (they behave like auxiliary losses, not true
unimodal heads). If you want *true* unimodal heads, we can wrap the encoders/fusion ourselves and
compute per-modality embeddings explicitly.

Run:
  python supfusion_multitask.py
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multimodal.fusion import ConcatFusion
from multimodal.model import MultimodalModel
from multimodal.tasks import ClassificationTask
from multimodal.train import Trainer, TrainerConfig, iter_training_parameters

# --- Edit these ---
TRAIN_NPZ = "data/processed_2s_128Hz/train.npz"
VAL_NPZ = "data/processed_2s_128Hz/val.npz"
TEST_NPZ = "data/processed_2s_128Hz/test.npz"
BATCH_SIZE = 32
LR = 1e-4
EMBED_DIM = 32
NUM_CLASSES = 2  # binary seizure label -> CrossEntropy
LABEL_KEY_IN_BATCH = "y"  # must match ClassificationTask label key
FUSION_MODALITY_ORDER: tuple[str, ...] = ("eeg", "ecg")

# Task names are the keys returned by the head.
PRIMARY_TASK = "fusion"
AUX_TASKS: tuple[str, ...] = ("eeg_aux", "ecg_aux")
ALL_TASKS: tuple[str, ...] = (PRIMARY_TASK,) + AUX_TASKS

MAX_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = False
LOG_EVERY = 10

OUT_DIR = "runs/supfusion_multitask"
CKPT_PATH = os.path.join(OUT_DIR, "supfusion_multitask_last.pt")
RESULTS_JSON = os.path.join(OUT_DIR, "supfusion_multitask_eval.json")
# ------------------


class MultiHeadMLP(nn.Module):
    """Multiple classification heads after fusion."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        task_names: tuple[str, ...],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_names = task_names
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, output_dim) for name in task_names})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        return {name: head(h) for name, head in self.heads.items()}


class EEGEncoder(nn.Module):
    """1D CNN on EEG (B, 2, T) -> (B, embed_dim)."""

    def __init__(self, in_ch: int = 2, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class ECGEncoder(nn.Module):
    """1D CNN on ECG (B, 1, T) -> (B, embed_dim)."""

    def __init__(self, in_ch: int = 1, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class SupFusionDictDataset(Dataset):
    """
    Loads processed npz windows. Each sample is a dict with modality tensors and a label,
    suitable for multimodal trainers that expect batch dicts (eeg, ecg, y).
    """

    def __init__(
        self,
        npz_path: str,
        label_key: str = "y",
        source_label_key: str = "binary_label",
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.eeg = torch.tensor(np.asarray(data["eeg"]), dtype=torch.float32)
        self.ecg = torch.tensor(np.asarray(data["ecg"]), dtype=torch.float32)
        labels = np.asarray(data[source_label_key])
        self.labels = torch.tensor(labels, dtype=torch.long)
        if self.labels.ndim > 1:
            self.labels = self.labels.squeeze(-1)
        self.label_key = label_key

        n = min(self.eeg.shape[0], self.ecg.shape[0], self.labels.shape[0])
        self.eeg = self.eeg[:n]
        self.ecg = self.ecg[:n]
        self.labels = self.labels[:n]

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # ty:ignore[invalid-method-override]
        return {
            "eeg": self.eeg[idx],
            "ecg": self.ecg[idx],
            self.label_key: self.labels[idx],
        }


def collate_supfusion(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        return {}
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def _unwrap_predictions(raw: Any) -> dict[str, torch.Tensor]:
    """MultimodalModel.predict returns (preds, embeddings); we only need preds dict."""
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, dict):
        return cast(dict[str, torch.Tensor], raw)
    raise TypeError(f"Expected dict predictions from model, got {type(raw)}")


def forward_predictions(
    model: MultimodalModel,
    batch: dict[str, torch.Tensor],
    device: str,
    label_key: str,
) -> dict[str, torch.Tensor]:
    """Forward on modalities only (strip label key)."""
    x = {k: v.to(device) for k, v in batch.items() if k != label_key}
    return _unwrap_predictions(model.predict(x))


def positive_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """P(class=1) for binary CE head: (B,2) logits -> (B,) probs."""
    if logits.dim() == 2 and logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    return torch.sigmoid(logits.view(-1))


@torch.no_grad()
def find_best_threshold_youden(
    model: MultimodalModel,
    val_loader: DataLoader,
    device: str,
    task_key: str,
) -> float:
    """Youden's J on validation set using P(class=1) from softmax (2-class head)."""
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    for batch in val_loader:
        y = batch[LABEL_KEY_IN_BATCH]
        logits_dict = forward_predictions(model, batch, device, LABEL_KEY_IN_BATCH)
        probs = positive_class_probs(logits_dict[task_key])
        all_probs.append(probs.cpu())
        all_labels.append(y.view(-1).float().cpu())

    all_probs_np = torch.cat(all_probs).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[np.argmax(tpr - fpr)])


@torch.no_grad()
def evaluate_binary_threshold(
    model: MultimodalModel,
    data_loader: DataLoader,
    threshold: float,
    device: str,
    task_key: str,
) -> dict[str, Any]:
    """Metrics on dict batches (eeg, ecg, y); uses P(class=1) vs threshold."""
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[torch.Tensor] = []
    for batch in data_loader:
        y = batch[LABEL_KEY_IN_BATCH]
        logits_dict = forward_predictions(model, batch, device, LABEL_KEY_IN_BATCH)
        probs = positive_class_probs(logits_dict[task_key])
        preds = (probs >= threshold).cpu().numpy()
        all_labels.append(y.view(-1).cpu())
        all_preds.append(preds)
        all_probs.append(probs.cpu())

    all_labels_np = torch.cat(all_labels).numpy().astype(int)
    all_probs_np = torch.cat(all_probs).numpy()
    all_preds_np = np.concatenate(all_preds)

    auc = roc_auc_score(all_labels_np, all_probs_np)
    acc = accuracy_score(all_labels_np, all_preds_np)
    f1 = f1_score(all_labels_np, all_preds_np)
    precision = precision_score(all_labels_np, all_preds_np, zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, zero_division=0)
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    return {
        "auc_score": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def build_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train, val, test — `SupFusionDictDataset` + `collate_supfusion` (dict with eeg, ecg, y)."""
    train_ds = SupFusionDictDataset(TRAIN_NPZ, label_key=LABEL_KEY_IN_BATCH, source_label_key="binary_label")
    val_ds = SupFusionDictDataset(VAL_NPZ, label_key=LABEL_KEY_IN_BATCH, source_label_key="binary_label")
    test_ds = SupFusionDictDataset(TEST_NPZ, label_key=LABEL_KEY_IN_BATCH, source_label_key="binary_label")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_supfusion,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_supfusion,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_supfusion,
    )
    return train_loader, val_loader, test_loader


def save_checkpoint(
    path: str,
    model: nn.Module,
    eeg_ch: int,
    ecg_ch: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "eeg_ch": eeg_ch,
            "ecg_ch": ecg_ch,
            "embed_dim": EMBED_DIM,
            "num_classes": NUM_CLASSES,
            "tasks": list(ALL_TASKS),
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def main() -> None:
    train_loader, val_loader, test_loader = build_loaders()

    sample = train_loader.dataset[0]
    eeg_ch = int(sample["eeg"].shape[0])
    ecg_ch = int(sample["ecg"].shape[0])

    encoders: dict[str, nn.Module] = {
        "eeg": EEGEncoder(in_ch=eeg_ch, embed_dim=EMBED_DIM),
        "ecg": ECGEncoder(in_ch=ecg_ch, embed_dim=EMBED_DIM),
    }

    # Fusion output dimension depends on the fusion module; Concat -> 2*EMBED_DIM.
    fusion = ConcatFusion()
    fusion_out_dim = EMBED_DIM * 2

    model = MultimodalModel(
        encoders=encoders,
        fusion=fusion,
        head=MultiHeadMLP(fusion_out_dim, EMBED_DIM, NUM_CLASSES, task_names=ALL_TASKS),
        fusion_modality_order=list(FUSION_MODALITY_ORDER),
    )

    tasks = [ClassificationTask(task_name, LABEL_KEY_IN_BATCH) for task_name in ALL_TASKS]

    cfg = TrainerConfig(
        max_epochs=MAX_EPOCHS,
        mixed_precision=MIXED_PRECISION,
        device=DEVICE,
        progress_bar=True,
        log_every=LOG_EVERY,
    )

    opt = torch.optim.Adam(iter_training_parameters(model, tasks), lr=LR)
    trainer = Trainer(model, cast(Any, tasks), opt, cfg)
    training_stopped_early = False
    try:
        trainer.train(train_loader, val_loader=val_loader)
    except KeyboardInterrupt:
        training_stopped_early = True
        print("\nKeyboardInterrupt: stopped training; saving checkpoint and running val/test eval.\n")

    model.to(DEVICE)
    if hasattr(model, "eval"):
        model.eval()
    save_checkpoint(CKPT_PATH, model, eeg_ch, ecg_ch)

    thresholds: dict[str, float] = {}
    val_metrics_all: dict[str, dict[str, Any]] = {}
    test_metrics_all: dict[str, dict[str, Any]] = {}
    for task_key in ALL_TASKS:
        thr = find_best_threshold_youden(model, val_loader, DEVICE, task_key=task_key)
        thresholds[task_key] = thr
        val_metrics_all[task_key] = evaluate_binary_threshold(model, val_loader, thr, DEVICE, task_key=task_key)
        test_metrics_all[task_key] = evaluate_binary_threshold(model, test_loader, thr, DEVICE, task_key=task_key)

    print("Thresholds (Youden, P(class=1)):", {k: round(v, 4) for k, v in thresholds.items()})
    print(
        "Val metrics:",
        {k: {m: val_metrics_all[k][m] for m in ("auc_score", "accuracy", "f1", "precision", "recall")} for k in ALL_TASKS},
    )
    print(
        "Test metrics:",
        {k: {m: test_metrics_all[k][m] for m in ("auc_score", "accuracy", "f1", "precision", "recall")} for k in ALL_TASKS},
    )

    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    results_payload = {
        "thresholds_youden": thresholds,
        "training_stopped_early": training_stopped_early,
        "val": {k: {m: val_metrics_all[k][m] for m in ("auc_score", "accuracy", "f1", "precision", "recall")} for k in ALL_TASKS},
        "test": {k: {m: test_metrics_all[k][m] for m in ("auc_score", "accuracy", "f1", "precision", "recall")} for k in ALL_TASKS},
        "checkpoint": CKPT_PATH,
        "train_npz": TRAIN_NPZ,
        "val_npz": VAL_NPZ,
        "test_npz": TEST_NPZ,
        "tasks": list(ALL_TASKS),
        "primary_task": PRIMARY_TASK,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Wrote results to {RESULTS_JSON}")


if __name__ == "__main__":
    main()

