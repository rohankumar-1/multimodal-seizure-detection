from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.signal import resample, butter, filtfilt, iirnotch
from typing import Callable, Union


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass(signal, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal, axis=-1)


def apply_notch(signal, freq, fs, Q=35):
    b, a = iirnotch(freq / (0.5*fs), Q)
    return filtfilt(b, a, signal, axis=-1)


def preprocess_signal_nn(train_signal, val_signal, test_signal, 
                      desired_fs=256,
                      lowcut=0.5, 
                      highcut=60,
                      notch_freq=50):
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
    - notch_freq: notch frequency
    Returns:
    - train_tensor, val_tensor, test_tensor: torch tensors of same shape
    """
    # Resample all signals
    def resample_signal(signal, target_len):
        return resample(signal, target_len, axis=-1)
    
    train_len = int(train_signal.shape[-1] * desired_fs / 256.0)  # identity if already same
    val_len = int(val_signal.shape[-1] * desired_fs / 256.0)
    test_len = int(test_signal.shape[-1] * desired_fs / 256.0)
    
    train_signal = resample_signal(train_signal, train_len)
    val_signal = resample_signal(val_signal, val_len)
    test_signal = resample_signal(test_signal, test_len)

    # Bandpass filter
    train_signal = apply_bandpass(train_signal, lowcut, highcut, desired_fs)
    val_signal = apply_bandpass(val_signal, lowcut, highcut, desired_fs)
    test_signal = apply_bandpass(test_signal, lowcut, highcut, desired_fs)

    # Notch filter
    train_signal = apply_notch(train_signal, notch_freq, desired_fs)
    val_signal = apply_notch(val_signal, notch_freq, desired_fs)
    test_signal = apply_notch(test_signal, notch_freq, desired_fs)

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



class SupervisedMultimodalDataset(Dataset):
    """
    A flexible supervised multimodal dataset class.
    modalities: dict[str, np.ndarray | torch.Tensor], each shape (N, C, T)
    labels: np.ndarray | torch.Tensor, shape (N,)
    transform_dict: dict[str, callable or None]
    """
    def __init__(self, modalities: dict,
                 labels,
                 transform_dict: Union[dict, None] = None):

        self.modalities: dict = modalities
        self.labels = labels
        self.transform_dict: Union[dict, None] = transform_dict or None

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        sample: dict[str, torch.Tensor] = {}

        for name, data in self.modalities.items():
            x = data[idx]  # extract sample for this modality

            # apply transform if provided
            if self.transform_dict and (name in self.transform_dict) and (self.transform_dict[name] is not None):
                x = self.transform_dict[name](x)

            sample[name] = x.float()

        return sample, self.labels[idx]
