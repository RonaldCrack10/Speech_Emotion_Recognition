from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import class_weights, labels
from tqdm import tqdm
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "ser.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for x, y in tqdm(loader, desc= "train", leave= False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        preds += logits.argmax(dim=1).tolist()
        targets += y.tolist()
    f1 = f1_score(targets, preds, average="macro")
    return total_loss / len(targets), f1

def load(name):
    d = np.load(EMB / f"emb_{name}.npz")
    return d["X"].astype(np.float32), d["y"]


class WeightedSER(nn.Module):
    def __init__(self, n_layers=13, hidden=768, n_classes=7, dropout=0.3):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers)) # Layer weight...which layer influences most
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(256, n_classes)
            )

    def forward(self, x):                              # x: (B, 13, 768)
        w = torch.softmax(self.layer_w, dim=0)         # 13 Gewichte, Summe = 1
        pooled = (x * w[None, :, None]).sum(dim=1)     # (B, 768) aus allen Schichten eine einzige machen x1*w1 + x2*w2
        return self.head(pooled)

    X_train, y_train = load("train")
    X_val, y_val = load("val")

    mean = X_train.mean(axis=0, keepdims=True)                 # (1, 13, 768)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std

def loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=64, shuffle=shuffle)

if __name__ == "__main__":

    X_train, y_train = load("train")
    X_val, y_val = load("val")

    mean = X_train.mean(axis=0, keepdims=True)                 # (1, 13, 768)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std

    train_loader, val_loader = loader(X_train, y_train, True), loader(X_val, y_val, False)

    model = WeightedSER().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(pd.DataFrame({"label": y_train})).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_f1, wait = 0.0, 0
    for epoch in range(1, 61):
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        val_loss, val_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoche {epoch:2d} | val loss {val_loss:.3f} | val F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            CKPT.parent.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "labels": labels,
                        "mean": mean, "std": std, "val_f1": val_f1}, CKPT)
        else:
            wait += 1
            if wait >= 10:
                print("Early Stopping")
                break

    print("beste Val-F1:", round(best_f1, 3))
    print("Schichtgewichte:", torch.softmax(model.layer_w, 0).detach().cpu().numpy().round(2))