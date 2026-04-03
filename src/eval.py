import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve

import matplotlib.pyplot as plt



def find_best_threshold(model, valloader: DataLoader, device: str = "cpu") -> float:
    """
    Find the best classificationthreshold for a neural network model. Considers tpr - fpr.
    """
    model.eval()
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch, y in valloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs)
            all_labels.append(y)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    best_threshold = thresholds[np.argmax(tpr - fpr)]  # best threshold is where tpr - fpr is maximized

    return best_threshold


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



def evaluate_svm(model, data_loader: DataLoader, device: str = "cpu") -> dict:
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

            preds = model.predict(features)
            probs = model.predict_proba(features)[:, 1]

            all_labels.append(labels.numpy())
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


def evaluate_matrixprofile(mp, preprocess_fn, data_paths: list[str] = [], target='binary_label', plot_run=True) -> dict:
    all_labels = []
    all_preds = []
    all_probs = []

    for data_path in tqdm(data_paths):
        data = np.load(data_path, allow_pickle=True)
        ecg = data['ecg']
        target = data[target]

        ecg = preprocess_fn(ecg)

        mp = mp.predict(ecg)

        all_labels.append(target)
        all_preds.append(mp['events'])
        all_probs.append(mp['mp'])


    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

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



