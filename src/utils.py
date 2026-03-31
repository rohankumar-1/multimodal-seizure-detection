from torch.utils.data import Dataset
import numpy as np



class SeizureDataset(Dataset):

  def __init__(self, data_path, eeg_transform=None, ecg_transform=None):
    data = np.load(file=data_path, allow_pickle=True)

    self.eeg = data['eeg']
    self.ecg = data['ecg']

    self.binary_label=data['binary_label']
    self.lateralization=data['lateralization']
    self.label=data['label']
    self.localization=data['localization']
    self.vigilance=data['vigilance']
    self.seizure_duration_sec=data['seizure_duration_sec']

    self.eeg_transform = eeg_transform
    self.ecg_transform = ecg_transform

  def __len__(self):
    return len(self.eeg)

  def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
    eeg = self.eeg[idx]
    ecg = self.ecg[idx]
    label = self.binary_label[idx]

    if self.eeg_transform:
      eeg = self.eeg_transform(eeg)

    if self.ecg_transform:
      ecg = self.ecg_transform(ecg)

    return ecg, eeg, label



