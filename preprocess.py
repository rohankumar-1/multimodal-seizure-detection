"""Signal preprocessing and PyTorch datasets for EEG/ECG windows."""

from __future__ import annotations

from typing import Union, Callable

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample
from torch.utils.data import Dataset


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal, axis=-1)


def apply_notch(signal: np.ndarray, freq: float, fs: float, Q: float = 35) -> np.ndarray:
    b, a = iirnotch(freq / (0.5 * fs), Q)
    return filtfilt(b, a, signal, axis=-1)


def preprocess_signal_nn(
    train_signal: np.ndarray,
    val_signal: np.ndarray,
    test_signal: np.ndarray,
    desired_fs: float = 256,
    lowcut: float = 0.5,
    highcut: float = 60,
    notch_freq: float = 50,
):
    """
    Bandpass + notch (assumes 256 Hz input), resample, z-score from train, return float32 tensors.
    Shapes: (N, C, T) per split.
    """
    train_signal = apply_bandpass(train_signal, lowcut, highcut, 256.0)
    val_signal = apply_bandpass(val_signal, lowcut, highcut, 256.0)
    test_signal = apply_bandpass(test_signal, lowcut, highcut, 256.0)

    train_signal = apply_notch(train_signal, notch_freq, 256.0)
    val_signal = apply_notch(val_signal, notch_freq, 256.0)
    test_signal = apply_notch(test_signal, notch_freq, 256.0)

    def resample_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
        return resample(signal, target_len, axis=-1)

    train_len = int(train_signal.shape[-1] * desired_fs / 256.0)
    val_len = int(val_signal.shape[-1] * desired_fs / 256.0)
    test_len = int(test_signal.shape[-1] * desired_fs / 256.0)

    train_signal = resample_signal(train_signal, train_len)
    val_signal = resample_signal(val_signal, val_len)
    test_signal = resample_signal(test_signal, test_len)

    mean = train_signal.mean(axis=(0, 2), keepdims=True)
    std = train_signal.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    train_signal = (train_signal - mean) / std
    val_signal = (val_signal - mean) / std
    test_signal = (test_signal - mean) / std

    return (
        torch.tensor(train_signal, dtype=torch.float32),
        torch.tensor(val_signal, dtype=torch.float32),
        torch.tensor(test_signal, dtype=torch.float32),
    )


def preprocess_three_npz(
    train_npz: str,
    val_npz: str,
    test_npz: str,
    out_dir: str,
    desired_fs: float = 128.0,
    lowcut: float = 0.5,
    highcut: float = 60.0,
    notch_freq: float = 50.0,
    train_out: str = "train.npz",
    val_out: str = "val.npz",
    test_out: str = "test.npz",
) -> tuple[str, str, str]:
    """Load train/val/test .npz, preprocess with train stats, save three outputs. Returns paths."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = np.load(train_npz, allow_pickle=True)
    val = np.load(val_npz, allow_pickle=True)
    test = np.load(test_npz, allow_pickle=True)

    eeg_tr, eeg_va, eeg_te = preprocess_signal_nn(
        train["eeg"], val["eeg"], test["eeg"], desired_fs, lowcut, highcut, notch_freq
    )
    ecg_tr, ecg_va, ecg_te = preprocess_signal_nn(
        train["ecg"], val["ecg"], test["ecg"], desired_fs, lowcut, highcut, notch_freq
    )

    p_train = out / train_out
    p_val = out / val_out
    p_test = out / test_out
    np.savez(p_train, eeg=eeg_tr.numpy(), ecg=ecg_tr.numpy(), binary_label=train["binary_label"], lateralization=train["lateralization"], label=train["label"], localization=train["localization"], vigilance=train["vigilance"], seizure_duration_sec=train["seizure_duration_sec"])
    np.savez(p_val, eeg=eeg_va.numpy(), ecg=ecg_va.numpy(), binary_label=val["binary_label"], lateralization=val["lateralization"], label=val["label"], localization=val["localization"], vigilance=val["vigilance"], seizure_duration_sec=val["seizure_duration_sec"])
    np.savez(p_test, eeg=eeg_te.numpy(), ecg=ecg_te.numpy(), binary_label=test["binary_label"], lateralization=test["lateralization"], label=test["label"], localization=test["localization"], vigilance=test["vigilance"], seizure_duration_sec=test["seizure_duration_sec"])
    return str(p_train), str(p_val), str(p_test)


class SupervisedMultimodalDataset(Dataset):
    """Returns (batch_dict, y) with keys eeg, ecg."""

    def __init__(
        self,
        modalities: dict[str, Union[np.ndarray, torch.Tensor]],
        labels: Union[np.ndarray, torch.Tensor],
        transform_dict: dict | None = None,
    ):
        self.modalities = modalities
        self.labels = labels
        self.transform_dict = transform_dict

    def __len__(self):
        return min(self.labels.shape[0], *[d.shape[0] for d in self.modalities.values()])

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        sample: dict[str, torch.Tensor] = {}
        for name, data in self.modalities.items():
            x = data[idx]
            if self.transform_dict and name in self.transform_dict and self.transform_dict[name]:
                x = self.transform_dict[name](x)
            sample[name] = x.float()
        y = self.labels[idx]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        return sample, y


class SequentialMultimodalDataset:
    """Iterate (ecg, eeg, targets) per file for run-level loading."""

    def __init__(self, file_paths: list[str], target: str = "binary_label"):
        self.file_paths = file_paths
        self.ecg, self.eeg, self.targets = [], [], []
        for fp in file_paths:
            data = np.load(fp, allow_pickle=True)
            self.ecg.append(data["ecg"])
            self.eeg.append(data["eeg"])
            self.targets.append(data[target])

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        return iter([(self.ecg[i], self.eeg[i], self.targets[i]) for i in range(len(self))])


def collate_multitask_fusion(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        return {}
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


class MultitaskFusionDataset(Dataset):
    """
    Loads processed npz windows. Each sample is a dict with modality tensors and a label,
    suitable for multimodal trainers that expect batch dicts (eeg, ecg, y).
    """

    def __init__(
        self,
        npz_path: str,
        eeg_transform: Callable | None = None,
        ecg_transform: Callable | None = None,
        label_keys: list[str] = ["y"],
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.eeg = torch.tensor(np.asarray(data["eeg"]), dtype=torch.float32)
        self.ecg = torch.tensor(np.asarray(data["ecg"]), dtype=torch.float32)
        self.label_keys = label_keys
        self.labels = {label_key: torch.tensor(np.asarray(data[label_key]), dtype=torch.long) for label_key in self.label_keys}

        n = min(self.eeg.shape[0], self.ecg.shape[0], *[d.shape[0] for d in self.labels.values()])
        self.eeg = self.eeg[:n]
        self.ecg = self.ecg[:n]
        for label_key in self.label_keys:
            self.labels[label_key] = self.labels[label_key][:n]

        self.eeg_transform = eeg_transform
        self.ecg_transform = ecg_transform

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # ty:ignore[invalid-method-override]

        eeg = self.eeg[idx]
        ecg = self.ecg[idx]
        labels = {label_key: self.labels[label_key][idx] for label_key in self.label_keys}
    
        if self.eeg_transform:
            eeg = self.eeg_transform(eeg)
        if self.ecg_transform:
            ecg = self.ecg_transform(ecg)

        return {
            "eeg": eeg,
            "ecg": ecg,
            **labels,
        }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preprocess train/val/test .npz (normalization from train).")
    p.add_argument("--train_npz", required=True)
    p.add_argument("--val_npz", required=True)
    p.add_argument("--test_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--desired_fs", type=float, default=128.0)
    p.add_argument("--lowcut", type=float, default=0.5)
    p.add_argument("--highcut", type=float, default=60.0)
    p.add_argument("--notch_freq", type=float, default=50.0)
    a = p.parse_args()
    paths = preprocess_three_npz(
        a.train_npz,
        a.val_npz,
        a.test_npz,
        a.out_dir,
        desired_fs=a.desired_fs,
        lowcut=a.lowcut,
        highcut=a.highcut,
        notch_freq=a.notch_freq,
    )
    print("Wrote:", paths)
