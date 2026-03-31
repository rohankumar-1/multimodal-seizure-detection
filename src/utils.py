from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.signal import resample, butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(signal, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal, axis=-1)

def preprocess_signal_nn(train_signal, val_signal, test_signal, 
                      desired_fs=256,
                      lowcut=0.5, highcut=50):
    """
    Preprocess EEG/ECG signals for neural network models:
    - Resample to desired sampling rate
    - Apply bandpass filter (lowcut/highcut)
    - Z-score normalization based on train set
    - Convert to PyTorch tensor
    Parameters:
    - train_signal, val_signal, test_signal: np.arrays of shape (channels, time)
    - desired_fs: target sampling rate
    - lowcut, highcut: bandpass cutoff frequencies
    Returns:
    - train_tensor, val_tensor, test_tensor: torch tensors of same shape
    """
    # Resample all signals
    def resample_signal(signal, target_len):
        return resample(signal, target_len, axis=-1)
    
    train_len = int(train_signal.shape[-1] * desired_fs / desired_fs)  # identity if already same
    val_len = int(val_signal.shape[-1] * desired_fs / desired_fs)
    test_len = int(test_signal.shape[-1] * desired_fs / desired_fs)
    
    train_signal = resample_signal(train_signal, train_len)
    val_signal = resample_signal(val_signal, val_len)
    test_signal = resample_signal(test_signal, test_len)

    # Bandpass filter
    train_signal = apply_bandpass(train_signal, lowcut, highcut, desired_fs)
    val_signal = apply_bandpass(val_signal, lowcut, highcut, desired_fs)
    test_signal = apply_bandpass(test_signal, lowcut, highcut, desired_fs)

    mean = train_signal.mean(axis=(0, 2), keepdims=True)   # shape (1, C, 1)
    std  = train_signal.std(axis=(0, 2), keepdims=True)    # shape (1, C, 1)

    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)

    # Apply normalization
    train_signal = (train_signal - mean) / std
    val_signal   = (val_signal   - mean) / std
    test_signal  = (test_signal  - mean) / std

    # Convert to tensors
    train_tensor = torch.tensor(train_signal, dtype=torch.float32)
    val_tensor = torch.tensor(val_signal, dtype=torch.float32)
    test_tensor = torch.tensor(test_signal, dtype=torch.float32)

    return train_tensor, val_tensor, test_tensor


class SeizureDataset(Dataset):
    """
    Super basic dataset class for the SeizeIT2 dataset.
    """
    def __init__(self, eeg: np.ndarray, ecg: np.ndarray, labels: np.ndarray, eeg_transform=None, ecg_transform=None):
        self.eeg = eeg
        self.ecg = ecg
        self.labels = labels

        self.eeg_transform = eeg_transform
        self.ecg_transform = ecg_transform

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        eeg = self.eeg[idx]
        ecg = self.ecg[idx]
        labels = self.labels[idx]

        if self.eeg_transform:
            eeg = self.eeg_transform(eeg)

        if self.ecg_transform:
            ecg = self.ecg_transform(ecg)

        return ecg, eeg, labels

