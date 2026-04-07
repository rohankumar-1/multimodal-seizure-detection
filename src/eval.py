import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve

import matplotlib.pyplot as plt

from typing import Literal

from src.models.matrixprofile import MatrixProfile


VALIDATION_METRICS = Literal['youdens_j', 'accuracy', 'f1', 'precision', 'recall']

def find_best_threshold(model, valloader: DataLoader, metric: VALIDATION_METRICS = 'youdens_j', device: str = "cpu") -> float:
    """
    Find the best classification threshold for a neural network model.
    Inputs:
        model: pytorch-based neural network
        valloader: DataLoader for validation data, returns dict of modalities and labels
        metric: metric to use for finding the best threshold
        device: 'cuda' or 'cpu'
    """
    model.eval()
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch, y in valloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.sigmoid(outputs).view(-1)  # flatten in case of shape []
            all_probs.append(probs.cpu())
            all_labels.append(y.view(-1).cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Compute best threshold based on metric
    if metric == 'youdens_j':
        best_threshold = thresholds[np.argmax(tpr - fpr)]
    else:
        best_score = -np.inf
        best_threshold = 0.5
        for thr in thresholds:
            preds = (all_probs >= thr).astype(int)
            if metric == 'accuracy':
                score = accuracy_score(all_labels, preds)
            elif metric == 'f1':
                score = f1_score(all_labels, preds)
            elif metric == 'precision':
                score = precision_score(all_labels, preds)
            elif metric == 'recall':
                score = recall_score(all_labels, preds)
            if score > best_score:
                best_score = score
                best_threshold = thr

    return float(best_threshold)


def evaluate_nn(model, data_loader: DataLoader, threshold=0.5, device: str = "cpu") -> dict:
    """
    Evaluate a neural network model on ECG+EEG data.
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

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



def evaluate_svm(model, data_loader: DataLoader, threshold=0.5, device: str = "cpu") -> dict:
    """
    Evaluate an SVM (or any sklearn classifier) using DataLoader batches.
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for ecg, eeg, labels in data_loader:
            # Assume dataset returns precomputed features instead of raw signals
            features = torch.cat([ecg, eeg], dim=1)   # Example: concat features
            features = features.cpu().numpy()

            probs = model.predict_proba(features)[:, 1]
            preds = probs > threshold

            all_labels.append(labels)
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


def evaluate_matrixprofile(mp: MatrixProfile, preprocess_fn, data_paths: list[str] = [], target='binary_label', plot_run=True) -> dict:
    """
    Evaluate a matrix profile model on ECG data. Runs the unsupervised method on a set of data. 
    """
    all_labels = []
    all_preds = []
    all_probs = []

    for data_path in tqdm(data_paths):
        data = np.load(data_path, allow_pickle=True)
        ecg = data['ecg']
        targets = data[target]

        ecg = preprocess_fn(ecg)

        print(ecg.shape)

        out = mp.predict(ecg)

        all_labels.append(targets)
        all_preds.append(out['events'])
        all_probs.append(out['scores'])

        print(out['events'].shape, out['scores'].shape)


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


def plot_run_results(run_results: dict):
    """ plots run results for multiple models """
    for model_name, results in run_results.items():
        plt.plot(results['fpr'], results['tpr'], label=model_name)
    plt.legend()
    plt.show()


def plot_roc_curve(all_labels: dict[str, np.ndarray], all_probs: dict[str, np.ndarray]):
    """ plots ROC curves for multiple models """
    for model_name, labels in all_labels.items():
        fpr, tpr, thresholds = roc_curve(labels, all_probs[model_name])
        plt.plot(fpr, tpr, label=model_name)
    plt.legend()
    plt.show()



