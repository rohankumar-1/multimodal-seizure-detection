import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def train_model_multimodal(model, trainloader, valloader, epochs, pos_weight=4.0, lr=1e-4, device='cuda'):
    """
    Trains a general multimodal neural network.
    """
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    pos_w = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {ep}")

        for m1, m2, y in pbar:
            m1, m2, y = m1.to(device), m2.to(device), y.to(device).float()
            pred = model(m1, m2).squeeze()
            loss = bce(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # ---- Validation AUC ----
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for m1, m2, y in valloader:
                m1, m2 = m1.to(device), m2.to(device)
                pred = model(m1, m2).squeeze().cpu()
                preds.append(pred)
                trues.append(y)

        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()
        auc = roc_auc_score(trues, preds)

        print(f"Epoch {ep} | Val AUC: {auc:.4f}")

    return model




def train_model_unimodal(model, trainloader, valloader, epochs, pos_weight=4.0, lr=1e-4, device='cuda'):
    """
    Trains a general unimodal neural network.
    """
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    pos_w = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {ep}")

        for m, _, y in pbar:
            m = m.to(device)
            y = y.to(device).float()
            pred = model(m).squeeze()
            loss = bce(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # ---- Validation AUC ----
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for m, _, y in valloader:
                m = m.to(device)
                y = y.to(device).float()
                pred = model(m).squeeze().cpu()
                preds.append(pred)
                trues.append(y)

        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()
        auc = roc_auc_score(trues, preds)

        print(f"Epoch {ep} | Val AUC: {auc:.4f}")

    return model