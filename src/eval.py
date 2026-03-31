import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def evaluate_nn(model, data_loader: DataLoader, device: str = "cpu") -> dict:
    """
    Evaluate a neural network model on ECG+EEG data.
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for ecg, eeg, labels in data_loader:
            ecg = ecg.to(device)
            eeg = eeg.to(device)
            labels = labels.to(device)

            outputs = model(ecg, eeg)        # shape (B, 2)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class=1
            preds = torch.argmax(outputs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate across batches
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # Compute metrics
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return {
        "auc_score": auc,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
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

    return {
        "auc_score": auc,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

