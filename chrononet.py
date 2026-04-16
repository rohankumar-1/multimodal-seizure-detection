"""
ChronoNet: train on processed .npz, then evaluate on test.
Run: python chrononet.py
Edit paths and hyperparameters in main() or pass via environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import evaluate_nn, find_best_threshold
from preprocess import SupervisedMultimodalDataset


TRAIN_NPZ = "data/processed_2s_40Hz/train.npz"
VAL_NPZ = "data/processed_2s_40Hz/val.npz"
TEST_NPZ = "data/processed_2s_40Hz/test.npz"
OUT_CKPT = "runs/chrononet/best.pt"
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
POS_WEIGHT = 4.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# model hyperparameters
@dataclass
class ChronoNetConfig:
    CH: int = 2
    strided: bool = False
    cnn_drop: float = 0.5
    maxpool: bool = False
    avgpool: bool = False
    batchnorm: bool = True


# --- Model (same architecture as before) ---


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.5, maxpool=False, avgpool=False, batchnorm=True):
        super().__init__()
        self.conv2 = nn.Conv1d(in_ch, out_ch, kernel_size=2, stride=stride, padding="same")
        self.conv4 = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=stride, padding="same")
        self.conv8 = nn.Conv1d(in_ch, out_ch, kernel_size=8, stride=stride, padding="same")
        self.batchnorm = nn.BatchNorm1d(out_ch * 3) if batchnorm else None
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(2, padding=1) if maxpool else None
        self.avgpool = nn.AvgPool1d(2, padding=1) if avgpool else None

    def forward(self, x):
        c0 = F.relu(self.conv2(x))
        c1 = F.relu(self.conv4(x))
        c2 = F.relu(self.conv8(x))
        x = torch.cat([c0, c1, c2], dim=1)
        if self.maxpool:
            x = self.maxpool(x)
        elif self.avgpool:
            x = self.avgpool(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        return self.dropout(x)


class ResidualGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru3 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.gru4 = nn.GRU(hidden_size * 3, hidden_size, batch_first=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        g1_out, _ = self.gru1(x)
        g2_out, _ = self.gru2(g1_out)
        g12 = torch.cat([g1_out, g2_out], dim=-1)
        g3_out, _ = self.gru3(g12)
        g123 = torch.cat([g1_out, g2_out, g3_out], dim=-1)
        g4_out, _ = self.gru4(g123)
        return g4_out[:, -1, :]




class ChronoNet(nn.Module):
    def __init__(self, config: ChronoNetConfig):
        super().__init__()
        filters = 32
        stride = 2 if config.strided else 1
        self.block1 = InceptionBlock(
            config.CH, filters, stride=stride, dropout=config.cnn_drop, maxpool=config.maxpool, avgpool=config.avgpool, batchnorm=config.batchnorm
        )
        self.block2 = InceptionBlock(
            filters * 3, filters, stride=stride, dropout=config.cnn_drop, maxpool=config.maxpool, avgpool=config.avgpool, batchnorm=config.batchnorm
        )
        self.block3 = InceptionBlock(
            filters * 3, filters, stride=stride, dropout=config.cnn_drop, maxpool=config.maxpool, avgpool=config.avgpool, batchnorm=config.batchnorm
        )
        self.gru = ResidualGRU(input_size=filters * 3, hidden_size=32)
        self.fc = nn.Linear(32, 1)

    def forward(self, **kwargs):
        x = kwargs["eeg"]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.fc(self.gru(x))


def train_supervised_nn(
    model,
    trainloader,
    valloader,
    epochs: int,
    pos_weight: float = 4.0,
    lr: float = 1e-4,
    patience: int = 3,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> None:
    if checkpoint_path and not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_val_loss = float("inf")
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        for batch, y in tqdm(trainloader, desc=f"Epoch {ep}"):
            y = y.float().to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(model(**batch).squeeze(), y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for batch, y in valloader:
                y = y.float().to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch).view(-1)
                val_losses.append(loss_fn(out, y).item())
                preds.append(out.cpu())
                trues.append(y.cpu())

        val_loss = sum(val_losses) / (len(val_losses) + 1e-8)
        preds_t = torch.cat(preds)
        trues_t = torch.cat(trues)
        auc = roc_auc_score(trues_t.numpy(), preds_t.numpy())
        print(f"Epoch {ep} | Val AUC: {auc:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break


def main() -> None:
    # --- Edit these ---
    # ------------------

    train = np.load(TRAIN_NPZ, allow_pickle=True)
    val = np.load(VAL_NPZ, allow_pickle=True)
    test = np.load(TEST_NPZ, allow_pickle=True)

    cfg = ChronoNetConfig(CH=int(train["eeg"].shape[1]))
    model = ChronoNet(cfg)

    train_ds = SupervisedMultimodalDataset(
        {"eeg": torch.tensor(train["eeg"]), "ecg": torch.tensor(train["ecg"])},
        torch.tensor(train["binary_label"]),
    )
    val_ds = SupervisedMultimodalDataset(
        {"eeg": torch.tensor(val["eeg"]), "ecg": torch.tensor(val["ecg"])},
        torch.tensor(val["binary_label"]),
    )
    test_ds = SupervisedMultimodalDataset(
        {"eeg": torch.tensor(test["eeg"]), "ecg": torch.tensor(test["ecg"])},
        torch.tensor(test["binary_label"]),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    os.makedirs(os.path.dirname(OUT_CKPT) or ".", exist_ok=True)
    train_supervised_nn(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        pos_weight=POS_WEIGHT,
        lr=LR,
        checkpoint_path=OUT_CKPT,
        device=DEVICE,
    )

    if os.path.isfile(OUT_CKPT):
        model.load_state_dict(torch.load(OUT_CKPT, map_location=DEVICE))
    model.to(DEVICE)
    threshold = find_best_threshold(model, val_loader, metric="youdens_j", device=DEVICE)
    print(f"Validation threshold (Youden): {threshold:.4f}")

    metrics = evaluate_nn(model, test_loader, threshold=threshold, device=DEVICE)
    print("Test metrics:", {k: metrics[k] for k in ("auc_score", "accuracy", "f1", "precision", "recall") if k in metrics})


if __name__ == "__main__":
    main()
