"""Classification metrics for NN, SVM, and matrix-profile baselines."""

from __future__ import annotations

from typing import Literal

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
from torch.utils.data import DataLoader
from tqdm import tqdm

VALIDATION_METRICS = Literal["youdens_j", "accuracy", "f1", "precision", "recall"]


def find_best_threshold(
    model,
    valloader: DataLoader,
    metric: VALIDATION_METRICS = "youdens_j",
    device: str = "cpu",
) -> float:
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch, y in valloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.sigmoid(outputs).view(-1)
            all_probs.append(probs.cpu())
            all_labels.append(y.view(-1).cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    if metric == "youdens_j":
        return float(thresholds[np.argmax(tpr - fpr)])

    best_score, best_threshold = -np.inf, 0.5
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        if metric == "accuracy":
            score = accuracy_score(all_labels, preds)
        elif metric == "f1":
            score = f1_score(all_labels, preds)
        elif metric == "precision":
            score = precision_score(all_labels, preds)
        elif metric == "recall":
            score = recall_score(all_labels, preds)
        else:
            raise ValueError(metric)
        if score > best_score:
            best_score, best_threshold = score, thr
    return float(best_threshold)


def evaluate_nn(model, data_loader: DataLoader, threshold: float = 0.5, device: str = "cpu") -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch, y in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.sigmoid(outputs)
            preds = probs > threshold
            all_labels.append(y)
            all_preds.append(preds)
            all_probs.append(probs)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    return {
        "auc_score": auc,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def evaluate_svm(
    model,
    data_loader: DataLoader,
    threshold: float = 0.5,
    device: str = "cpu",
    modality_keys: tuple[str, ...] = ("ecg", "eeg"),
) -> dict:
    if not modality_keys:
        raise ValueError("modality_keys must name at least one batch key (e.g. ecg, eeg).")
    maybe_eval = getattr(model, "eval", None)
    if callable(maybe_eval):
        maybe_eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch, y in data_loader:
            parts = [batch[k].to(device) for k in modality_keys]
            features = torch.cat(parts, dim=1).cpu().numpy()
            probs = model.predict_proba(features)[:, 1]
            preds = probs > threshold
            all_labels.append(y.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    return {
        "auc_score": auc,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def evaluate_matrixprofile(
    mp,
    preprocess_fn,
    data_paths: list[str] | None = None,
    target: str = "binary_label",
) -> dict:
    if data_paths is None:
        data_paths = []
    all_labels, all_probs = [], []
    for data_path in tqdm(data_paths):
        data = np.load(data_path, allow_pickle=True)
        ecg = preprocess_fn(data["ecg"])
        targets = data[target]
        out = mp.predict(ecg)
        all_labels.append(targets)
        all_probs.append(out["scores"])

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    best_score, best_threshold = -np.inf, 0.5
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        score = f1_score(all_labels, preds)
        if score > best_score:
            best_threshold, best_score = thr, score
    preds = (all_probs >= best_threshold).astype(int)

    return {
        "auc_score": roc_auc_score(all_labels, all_probs),
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds),
        "precision": precision_score(all_labels, preds),
        "recall": recall_score(all_labels, preds),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
