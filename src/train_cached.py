from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import class_weights, labels
from train import evaluate, device

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "ser_cached.pt"

def load(name):
    d = np.load(EMB / f"emb_{name}.npz")
    return d["X"].astype(np.float32), d["y"]


class WeightedHead(nn.Module):
    def __init__(self, n_layers=13, hidden=768, n_classes=7, dropout=0.3):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):                              # x: (B, 13, 768)
        w = torch.softmax(self.layer_w, dim=0)         # 13 Gewichte, Summe = 1
        pooled = (x * w[None, :, None]).sum(dim=1)     # (B, 768)
        return self.head(pooled)

Xtr, ytr = load("train")
Xva, yva = load("val")

mean = Xtr.mean(axis=0, keepdims=True)                 # (1, 13, 768)
std = Xtr.std(axis=0, keepdims=True) + 1e-6
Xtr, Xva = (Xtr - mean) / std, (Xva - mean) / std

def loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=64, shuffle=shuffle)

if __name__ == "__main__":
    train_loader, val_loader = loader(Xtr, ytr, True), loader(Xva, yva, False)

    model = WeightedHead().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(pd.DataFrame({"label": ytr})).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_f1, wait = 0.0, 0
    for epoch in range(1, 61):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        va_loss, va_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoche {epoch:2d} | val loss {va_loss:.3f} | val F1 {va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1, wait = va_f1, 0
            CKPT.parent.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "labels": labels,
                        "mean": mean, "std": std, "val_f1": va_f1}, CKPT)
        else:
            wait += 1
            if wait >= 10:
                print("Early Stopping")
                break

    print("beste Val-F1:", round(best_f1, 3))
    print("Schichtgewichte:", torch.softmax(model.layer_w, 0).detach().cpu().numpy().round(2))