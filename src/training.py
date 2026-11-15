"""
train.py: train the classification head on pre-computed WavLM embeddings.

What this file does, in short:
    1. Loads the embeddings that extract.py saved to disk (one array of shape
       (25 layers, 1024 values) per audio clip) together with their emotion labels.
    2. Standardizes them with the mean and std of the training set.
    3. Trains a small model (WeightedSER) on top of them. It first learns one weight
       per encoder layer and mixes the 25 layers into a single vector, then a small
       neural network turns that vector into 7 emotion scores.
    4. After every epoch it measures the macro F1 score on the validation set and
       saves the best model as a checkpoint (weights, labels, mean, std).
    5. Stops early when the validation F1 has not improved for 10 epochs and prints
       the learned layer weights, which show which layers carry the most emotion
       information.

Why it is fast: the expensive encoder (WavLM) is not part of the training. It already
ran once in extract.py, so each epoch only processes small vectors and takes seconds.

Input:  data/processed/emb_*.npz
Output: checkpoints/ser.pt (used later by live.py and the evaluation script)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import class_weights, labels
from tqdm import tqdm
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]      # repo root, independent of the working directory
EMB = ROOT / "data" / "processed"               # folder with the saved embeddings
CKPT = ROOT / "checkpoints" / "ser.pt"          # where the best model is saved
device = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if available


@torch.no_grad()   # no gradients needed for evaluation, saves memory and time
def evaluate(model, loader, criterion):
    """Return (average loss, macro F1) of the model on the data in `loader`."""
    model.eval()                                # turn dropout off
    total_loss, preds, targets = 0.0, [], []
    for x, y in tqdm(loader, desc= "train", leave= False):   # progress bar (label says "train", it is evaluation)
        x, y = x.to(device), y.to(device)       # move the batch to the GPU/CPU
        logits = model(x)                       # raw scores, shape (B, 7)
        total_loss += criterion(logits, y).item() * len(y)   # loss times batch size -> correct average at the end
        preds += logits.argmax(dim=1).tolist()  # predicted class = highest score
        targets += y.tolist()                   # true classes
    f1 = f1_score(targets, preds, average="macro")   # F1 per class, averaged with equal weight
    return total_loss / len(targets), f1


def load(name):
    """Load the embeddings and labels of one split ("train", "val" or "test")."""
    # NOTE: extract.py saves emb_large_{name}.npz, but this reads emb_{name}.npz
    d = np.load(EMB / f"emb_{name}.npz")
    return d["X"].astype(np.float32), d["y"]    # X: (N, layers, width) as float32, y: (N,) labels


class WeightedSER(nn.Module):
    """Mixes all encoder layers with learned weights, then classifies the result."""
    def __init__(self, n_layers=25, hidden=1024, n_classes=7, dropout=0.3):
        super().__init__()
        # one learnable number per layer; starts at 0, so all layers begin with equal weight
        self.layer_w = nn.Parameter(torch.zeros(n_layers)) # Layer weight...which layer influences most
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),             # 1024 numbers -> 256
            nn.ReLU(),                          # non-linearity
            nn.Dropout(dropout),                # randomly switches off 30% of values while training (against overfitting)
            nn.Linear(256, n_classes)           # 256 -> 7 emotion scores
            )

    def forward(self, x):                              # x: (B, 25, 1024) for large
        w = torch.softmax(self.layer_w, dim=0)         # turn the weights into positive numbers that sum to 1
        pooled = (x * w[None, :, None]).sum(dim=1)     # weighted sum over the layers: (B, 1024), i.e. x1*w1 + x2*w2 + ...
        return self.head(pooled)                       # (B, 7) scores

    
    X_train, y_train = load("train")
    X_val, y_val = load("val")

    mean = X_train.mean(axis=0, keepdims=True)                 # mean per layer and value: (1, layers, width)
    std = X_train.std(axis=0, keepdims=True) + 1e-6            # std per layer and value (+ tiny number against division by 0)
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std   # standardize; val uses the TRAIN statistics

def loader(X, y, shuffle):
    """Wrap arrays in a DataLoader that delivers batches of 64."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())   # pair every embedding with its label
    return DataLoader(ds, batch_size=64, shuffle=shuffle)   # shuffle only for training


if __name__ == "__main__":    # runs only when this file is started directly

    X_train, y_train = load("train")
    X_val, y_val = load("val")

    mean = X_train.mean(axis=0, keepdims=True)                 # statistics from the training set only
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std   # standardize train and val

    train_loader, val_loader = loader(X_train, y_train, True), loader(X_val, y_val, False)

    model = WeightedSER().to(device)
    # class weights: rarer classes count more in the loss, so every class matters equally
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(pd.DataFrame({"label": y_train})).to(device))
    # AdamW updates the weights; weight_decay keeps them small (against overfitting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_f1, wait = 0.0, 0    # best validation F1 so far, epochs without improvement
    for epoch in range(1, 61):
        model.train()                           # dropout on
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()               # 1. clear old gradients
            criterion(model(x), y).backward()   # 2. compute the loss and the gradients
            optimizer.step()                    # 3. adjust the weights

        val_loss, val_f1 = evaluate(model, val_loader, criterion)   # measure on unseen speakers
        print(f"Epoche {epoch:2d} | val loss {val_loss:.3f} | val F1 {val_f1:.3f}")

        if val_f1 > best_f1:                    # new best model: save it
            best_f1, wait = val_f1, 0
            CKPT.parent.mkdir(exist_ok=True)
            # mean and std are saved too, because live.py needs the same standardization
            torch.save({"model": model.state_dict(), "labels": labels,
                        "mean": mean, "std": std, "val_f1": val_f1}, CKPT)
        else:
            wait += 1                           # no improvement
            if wait >= 10:                      # early stopping after 10 epochs without improvement
                print("Early Stopping")
                break

    print("beste Val-F1:", round(best_f1, 3))
    # learned layer weights (sum = 1): show which layers carry the most emotion information
    print("Schichtgewichte:", torch.softmax(model.layer_w, 0).detach().cpu().numpy().round(2))