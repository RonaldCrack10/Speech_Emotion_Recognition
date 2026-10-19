import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data import (build_metadata, split, EmotionDataset, class_weights,
                  ravdess_path, tess_path, labels)
from model import SER
from pathlib import Path
from tqdm import tqdm

EPOCHS = 15
PATIENCE = 4                       # so viele Epochen ohne Verbesserung, dann Stopp
BATCH_SIZE = 8
CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "ser_best.pt"

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

def train(model, loader, criterion, optimizer):
    model.train()
    if not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.eval()
    total_loss = 0.0
    for x, y in tqdm(loader, desc= "train", leave= "False"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

if __name__ == "__main__":
    print("device:", device)

    df = build_metadata(ravdess_path, tess_path)
    train_df, val_df, test_df = split(df)

    train_loader = DataLoader(EmotionDataset(train_df, augment=True),
                                batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EmotionDataset(val_df), batch_size=BATCH_SIZE)

    model = SER().to(device)
    model.freeze_encoder()
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df).to(device))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

    best_f1, wait = 0.0, 0
    CKPT.parent.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train(model, train_loader, criterion, optimizer)
        va_loss, va_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoche {epoch:2d} | train loss {tr_loss:.3f} | val loss {va_loss:.3f} | val F1 {va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1, wait = va_f1, 0
            torch.save({"model": model.state_dict(), "labels": labels, "val_f1": va_f1}, CKPT)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early Stopping")
                break

    print("beste Val-F1:", round(best_f1, 3))