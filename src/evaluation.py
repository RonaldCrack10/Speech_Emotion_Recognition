# test the models results
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

from data import build_metadata, split, EmotionDataset, ravdess_path, tess_path
from model import SER

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "ser_best.pt"
RESULTS = ROOT / "results"
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, targets = [], []
    for x, y in tqdm(loader):
        logits = model(x.to(device))
        preds += logits.argmax(dim=1).cpu().tolist()
        targets += y.tolist()
    return np.array(targets), np.array(preds)

if __name__ == "__main__":
    ckpt = torch.load(CKPT, map_location=device)
    labels = ckpt["labels"]

    model = SER().to(device)
    model.load_state_dict(ckpt["model"])

    df = build_metadata(ravdess_path, tess_path)
    train_df, val_df, test_df = split(df)
    loader = DataLoader(EmotionDataset(test_df), batch_size=16)

    y_true, y_pred = predict(model, loader)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Accuracy {acc:.3f} | Macro-F1 {f1:.3f}")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test_metrics.json").write_text(json.dumps({"accuracy": acc, "macro_f1": f1}))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_test.png", dpi=150)