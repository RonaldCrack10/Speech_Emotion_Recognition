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
from training import WeightedSER, CKPT, ROOT, load

from data import build_metadata, split, EmotionDataset, ravdess_path, tess_path


ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "ser_cached.pt"
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
    
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]

    Xte, yte = load("test")
    Xte = (Xte - ckpt["mean"]) / ckpt["std"]

    model = WeightedSER()
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(dim=1).numpy()

    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    print(f"Accuracy {acc:.3f} | Macro-F1 {f1:.3f}")
    print(classification_report(yte, pred, target_names=labels, zero_division=0))

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "test_metrics.json").write_text(json.dumps({"accuracy": acc, "macro_f1": f1}))
    cm = confusion_matrix(yte, pred, labels=list(range(len(labels))))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(res / "confusion_matrix.png", dpi=150)