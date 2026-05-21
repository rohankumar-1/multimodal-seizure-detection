"""Signal preprocessing and PyTorch datasets for EEG/ECG windows."""

from __future__ import annotations

from typing import Union, Callable, Any, cast

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample
from torch.utils.data import Dataset


DEFAULT_LABEL_MAPS: dict[str, dict[Any, int]] = {
    "binary_label": {
        0: 0,
        1: 1,
    },
    "lateralization": {
        "nonseizure": 0,
        "un": 1,
        "left": 2,
        "right": 3,
        "bi": 4,
    },
    "label": {
        "nonseizure": 0,
        "sz_foc_ia_m_automatisms": 1,
        "sz_foc_ia_nm": 2,
        "sz_foc_ia_m_hyperkinetic": 3,
        "sz_foc_ia_nm_behavior": 4,
        "sz_foc_a_nm_behavior": 5,
        "sz_foc_a_nm": 6,
        "sz_foc_f2b": 7,
        "sz_foc_ia": 8,
        "sz_foc_ia_um": 9,
        "sz_foc_ua_nm_behavior": 10,
        "sz_foc_a_um": 11,
        "sz_foc_ua_um": 12,
        "sz_foc_ua_m_hyperkinetic": 13,
        "sz_foc_a_m_hyperkinetic": 14,
        "sz_foc_ia_m_tonic": 15,
        "sz_foc_ua_nm": 16,
        "sz_uo_nm": 17,
        "sz_foc_a_m_automatisms": 18
    },
    "localization": {
        "nonseizure": 0,
        "un": 1,
        "temp": 2,
        "front": 3,
        "front_temp": 4,
        "occ": 5,
        "temp_par": 6,
        "front_cen_temp": 7,
        "cen_temp": 8,
        "temp_occ": 5,
        "front_cen": 7,
        "cen_temp_par": 8,
    },
    "vigilance": {
        "nonseizure": 0,
        "un": 1,
        "awake": 2,
        "asleep": 3,
    },
}

# Optional: remap already-numeric class ids (after casting / categorical mapping).
DEFAULT_CLASS_GROUPINGS: dict[str, dict[int, int]] = {}

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


def _sanitize_signal_array(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in EEG/ECG so the model does not propagate NaNs."""
    x = np.asarray(x, dtype=np.float32)
    if np.isnan(x).any() or np.isinf(x).any():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def preprocess_labels(
    *,
    raw: dict,
    label_keys: list[str],
    label_maps: dict[str, dict[Any, int]] | None = None,
    class_groupings: dict[str, dict[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Encode labels without changing window↔label alignment.

    - If a label key is present in `label_maps`, we treat it as categorical and map each value
      via `label_maps[label_key][value]`.
    - Otherwise, we treat it as numeric and cast to int64 for downstream (optionally apply class_groupings).
    - For regression label (seizure_duration_sec): cast to float32.
    """
    out: dict[str, np.ndarray] = {}

    for k in label_keys:
        if k not in raw:
            raise KeyError(f"Label key '{k}' missing from npz.")
        y = np.asarray(raw[k])
        if y.ndim > 1 and y.shape[1:] == (1,):
            y = y.reshape(-1)

        if k == "seizure_duration_sec":
            out[k] = y.astype(np.float32, copy=False)
            continue

        if label_maps and k in label_maps:
            mapping_any = label_maps[k]
            mapped = np.empty((y.shape[0],), dtype=np.int64)
            for i, v in enumerate(y.tolist()):
                if v not in mapping_any:
                    raise ValueError(f"Unmapped value for '{k}': {v!r}. Add it to label_maps['{k}'].")
                mapped[i] = int(mapping_any[v])
            y = mapped
        else:
            if np.issubdtype(y.dtype, np.floating):
                y = np.rint(y).astype(np.int64, copy=False)
            else:
                y = y.astype(np.int64, copy=False)

        if class_groupings and k in class_groupings:
            mapping = class_groupings[k]
            y = y.copy()
            for old, new in mapping.items():
                y[y == int(old)] = int(new)

        out[k] = y

    # For compatibility with earlier code, keep return shape but with empty encoders.
    return out


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
    label_maps: dict[str, dict[Any, int]] | None = None,
    class_groupings: dict[str, dict[int, int]] | None = None,
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

    label_keys = ["binary_label", "lateralization", "label", "localization", "vigilance", "seizure_duration_sec"]
    y_train = preprocess_labels(raw=train, label_keys=label_keys, label_maps=label_maps, class_groupings=class_groupings)
    y_val = preprocess_labels(raw=val, label_keys=label_keys, label_maps=label_maps, class_groupings=class_groupings)
    y_test = preprocess_labels(raw=test, label_keys=label_keys, label_maps=label_maps, class_groupings=class_groupings)

    payload_train: dict[str, Any] = {
        "eeg": _sanitize_signal_array(eeg_tr.numpy()),
        "ecg": _sanitize_signal_array(ecg_tr.numpy()),
        **y_train,
    }
    payload_val: dict[str, Any] = {
        "eeg": _sanitize_signal_array(eeg_va.numpy()),
        "ecg": _sanitize_signal_array(ecg_va.numpy()),
        **y_val,
    }
    payload_test: dict[str, Any] = {
        "eeg": _sanitize_signal_array(eeg_te.numpy()),
        "ecg": _sanitize_signal_array(ecg_te.numpy()),
        **y_test,
    }

    np.savez(p_train, **cast(dict[str, Any], payload_train))
    np.savez(p_val, **cast(dict[str, Any], payload_val))
    np.savez(p_test, **cast(dict[str, Any], payload_test))
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
        self.eeg = torch.nan_to_num(
            torch.tensor(np.asarray(data["eeg"]), dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.ecg = torch.nan_to_num(
            torch.tensor(np.asarray(data["ecg"]), dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.label_keys = label_keys
        self.labels = {}
        for label_key in self.label_keys:
            arr = np.asarray(data[label_key])
            if label_key == "seizure_duration_sec":
                t = torch.tensor(arr, dtype=torch.float32)
                # Match typical regression head output shape (B, 1).
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                self.labels[label_key] = t
            else:
                self.labels[label_key] = torch.tensor(arr, dtype=torch.long)

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
        label_maps=DEFAULT_LABEL_MAPS,
        class_groupings=DEFAULT_CLASS_GROUPINGS,
    )
    print("Wrote:", paths)
