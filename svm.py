"""
SVM on flattened modality windows. Edit constants at the top only.

Modalities are concatenated in the order given (must exist in each .npz).
Run: python svm.py
"""

from __future__ import annotations

import json
import os
from typing import Sequence, Union, cast

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from metrics import evaluate_svm
from preprocess import SupervisedMultimodalDataset

# --- Edit these ---
MODALITIES: tuple[str, ...] = ("eeg", "ecg")
TRAIN_NPZ = "data/processed_2s_128Hz/train.npz"
VAL_NPZ = "data/processed_2s_128Hz/val.npz"
TEST_NPZ = "data/processed_2s_128Hz/test.npz"
OUT_PATH = "runs/svm/svm_model_2s_128Hz_eeg_ecg.joblib"
RESULTS_PATH = "results/svm_2s_128Hz_eeg_ecg.json"
C = 1.0
KERNEL = "rbf"
GAMMA = "scale"
CLASS_WEIGHT = "balanced"  # or None
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SUBSAMPLE_N: int | None = 1000  # e.g. 1000, or None for full train
RNG_SEED = 42
# ------------------


class SVMModel:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", class_weight=None):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, class_weight=class_weight)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def eval(self):
        pass

    def save(self, path: str):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str):
        m = SVMModel()
        m.model = joblib.load(path)
        return m


def _validate_modalities(npz: np.lib.npyio.NpzFile, modalities: Sequence[str]) -> None:
    for m in modalities:
        if m not in npz:
            raise KeyError(f"Modality {m!r} not in npz; available: {sorted(npz.files)}")


def features_from_npz(d: np.lib.npyio.NpzFile, modalities: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten each modality to (N, D_m) and concatenate along features in `modalities` order."""
    _validate_modalities(d, modalities)
    parts = [np.asarray(d[m]).reshape(d[m].shape[0], -1) for m in modalities]
    X = np.concatenate(parts, axis=1)
    y = d["binary_label"].astype(int)
    return X, y


def loader_from_npz(
    d: np.lib.npyio.NpzFile,
    modalities: Sequence[str],
    batch_size: int,
) -> DataLoader:
    """One tensor per modality key, shape (N, D_m), same order as `modalities` for evaluate_svm."""
    _validate_modalities(d, modalities)
    mods: dict[str, torch.Tensor] = {}
    for m in modalities:
        arr = np.asarray(d[m])
        mods[m] = torch.tensor(arr.reshape(arr.shape[0], -1), dtype=torch.float32)
    modalities_for_ds = cast(dict[str, Union[np.ndarray, torch.Tensor]], mods)
    ds = SupervisedMultimodalDataset(modalities_for_ds, torch.tensor(d["binary_label"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def main() -> None:
    modalities = MODALITIES
    if not modalities:
        raise SystemExit("Set MODALITIES to at least one modality name.")

    train = np.load(TRAIN_NPZ, allow_pickle=True)
    val = np.load(VAL_NPZ, allow_pickle=True)
    test = np.load(TEST_NPZ, allow_pickle=True)

    Xtr, ytr = features_from_npz(train, modalities)
    Xva, yva = features_from_npz(val, modalities)

    if TRAIN_SUBSAMPLE_N is not None:
        rng = np.random.default_rng(RNG_SEED)
        n = min(int(TRAIN_SUBSAMPLE_N), len(Xtr))
        idx = rng.permutation(len(Xtr))[:n]
        Xtr, ytr = Xtr[idx], ytr[idx]
        print(f"Training on {len(Xtr)} subsampled rows (of full train).")

    cw = CLASS_WEIGHT if CLASS_WEIGHT else None
    svm = SVMModel(C=C, kernel=KERNEL, gamma=GAMMA, class_weight=cw)
    svm.fit(Xtr, ytr)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    svm.save(OUT_PATH)

    probs_va = svm.predict_proba(Xva)[:, 1]
    fpr, tpr, thresholds = roc_curve(yva, probs_va)
    threshold = float(thresholds[np.argmax(tpr - fpr)]) if len(thresholds) else 0.5
    print(f"Modalities (order): {list(modalities)}")
    print(f"Validation threshold (Youden): {threshold:.4f}")

    val_loader = loader_from_npz(val, modalities, BATCH_SIZE)
    _ = evaluate_svm(svm, val_loader, threshold=threshold, device=DEVICE, modality_keys=modalities)

    test_loader = loader_from_npz(test, modalities, BATCH_SIZE)
    metrics = evaluate_svm(svm, test_loader, threshold=threshold, device=DEVICE, modality_keys=modalities)

    selected_metrics = {k: metrics[k] for k in ("auc_score", "accuracy", "f1", "precision", "recall") if k in metrics}
    print("Test metrics:", selected_metrics)

    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(selected_metrics, f, indent=2)


if __name__ == "__main__":
    main()
