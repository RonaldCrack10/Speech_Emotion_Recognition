"""Train and load the learned-layer-weighting classifier head (WeightedSER)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import class_weights, labels

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "ser.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    """Return (average loss, macro F1) of the model on `loader`."""
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        preds.extend(logits.argmax(dim=1).tolist())
        targets.extend(y.tolist())
    f1 = f1_score(targets, preds, average="macro")
    return total_loss / max(len(targets), 1), float(f1)


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed embeddings and labels for split ('train', 'val', or 'test')."""
    d = np.load(EMB / f"emb_{name}.npz")
    return d["X"].astype(np.float32), d["y"]


def load_split_full(name: str) -> dict[str, np.ndarray]:
    """Load embeddings, labels, speakers, and corpus tags for one split."""
    d = np.load(EMB / f"emb_{name}.npz")
    return {
        "X": d["X"].astype(np.float32),
        "y": d["y"].astype(np.int64),
        "speaker": d["speaker"],
        "corpus": d["corpus"],
    }


class WeightedSER(nn.Module):
    """Learns softmax weights over all WavLM layers, then classifies with a 2-layer MLP."""

    def __init__(self, n_layers: int = 25, hidden: int = 1024, n_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def layer_weights(self) -> torch.Tensor:
        """Return normalized layer importance weights summing to 1."""
        return torch.softmax(self.layer_w, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.layer_weights()
        pooled = (x * w[None, :, None]).sum(dim=1)
        return self.head(pooled)


def load_checkpoint(
    ckpt_path: str | Path = CKPT,
    map_location: str = "cpu",
) -> tuple[WeightedSER, dict]:
    """Load a saved WeightedSER checkpoint without requiring dataset files on disk."""
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    n_layers, hidden = ckpt["mean"].shape[1], ckpt["mean"].shape[2]
    ckpt_labels = ckpt.get("labels", labels)
    model = WeightedSER(n_layers=n_layers, hidden=hidden, n_classes=len(ckpt_labels))
    model.load_state_dict(ckpt["model"])
    model.to(map_location).eval()
    return model, ckpt


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, batch_size: int = 64) -> DataLoader:
    """Wrap arrays in a DataLoader."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_weighted_ser(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 42,
    save_path: Path | None = CKPT,
) -> tuple[WeightedSER, np.ndarray, np.ndarray, float]:
    """Train WeightedSER with standardization fitted strictly on X_train."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    train_loader = make_loader(X_train_n, y_train, shuffle=True)
    val_loader = make_loader(X_val_n, y_val, shuffle=False)

    n_layers, hidden = X_train.shape[1], X_train.shape[2]
    model = WeightedSER(n_layers=n_layers, hidden=hidden, n_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(pd.DataFrame({"label": y_train})).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1, wait, best_state = 0.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        val_loss, val_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:2d} | val loss {val_loss:.3f} | val F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": best_state,
                        "labels": labels,
                        "mean": mean,
                        "std": std,
                        "val_f1": val_f1,
                    },
                    save_path,
                )
        else:
            wait += 1
            if wait >= patience:
                print("Early Stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_f1


if __name__ == "__main__":
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    model, _, _, best_f1 = train_weighted_ser(X_train, y_train, X_val, y_val, save_path=CKPT)
    print("Best Val-F1:", round(best_f1, 3))
    print("Layer weights:", model.layer_weights().detach().cpu().numpy().round(3))
