import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def train_supervised_nn(model, trainloader, valloader, epochs, task='classification', pos_weight=4.0, lr=1e-4, device='cpu'):
    """
    Supervised training of a general neural network model. Can handle both unimodal and multimodal data. 
    Inputs:
        model: pytorch-based neural network
        trainloader: DataLoader for training data, returns dict of modalites and labels
        valloader: DataLoader for validation data, returns dict of modalites and labels
        epochs: number of epochs to train for
        task: 'classification', 'regression', or 'multiclass'
        pos_weight: positive weight for the BCE loss
        lr: learning rate
        device: 'cuda' or 'cpu'
    """

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if task == 'classification':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data=[pos_weight], device=device))
    elif task == 'regression':
        loss_fn = nn.MSELoss()
    elif task == 'multiclass':
        loss_fn = loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()

        for batch, y in tqdm(trainloader, desc=f"Epoch {ep}"):
            y = y.float().to(device)

            # Move all modalities to device dynamically
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass (model expects **kwargs)
            outputs = model(**batch)
            loss = loss_fn(outputs, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # ---- Validation ----
        model.eval()
        preds, trues = [], []

        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch, y in valloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(**batch).cpu()

                preds.append(pred)
                trues.append(y)

        preds = torch.cat(preds)
        trues = torch.cat(trues)

        # --- Task-specific metrics ---
        if task == 'classification':  # binary
            auc = roc_auc_score(trues.numpy(), preds.numpy())
            print(f"Epoch {ep} | Val AUC: {auc:.4f}")

        elif task == 'multiclass':
            acc = accuracy_score(trues.numpy(), preds.argmax(dim=1).numpy())
            print(f"Epoch {ep} | Val Acc: {acc:.4f}")

        elif task == 'regression':
            mse = mean_squared_error(trues.numpy(), preds.numpy())
            print(f"Epoch {ep} | Val MSE: {mse:.4f}")

    return model