from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMModel

from data import build_metadata, split, EmotionDataset, ravdess_path, tess_path
'''
Der Encoder ist der teure Teil und ändert sich nie, deshalb rechnen wir ihn einmal für alle Clips und speichern das Ergebnis, 
statt es in jeder Epoche zu wiederholen. 
Dabei behalten wir alle 13 Schichten, weil wir vorher nicht wissen, welche für Emotionen am nützlichsten sind. 
Das Training des kleinen Kopfs läuft dadurch in Sekunden statt Stunden, und der Kopf findet selbst heraus, welche Schichten er wie stark nutzt.

'''

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def extract_batch(encoder, x): # Audio dateien gehen durchs WavLM modell und werden von Wellen Form (B, 64_000) zu Vektoren umgewandelt (B, 13, 768)
    out = encoder(x, output_hidden_states=True) # schickt den Batch durch WavLM. Ohne das Argument bekäme man nur die letzte Schicht, mit True zusätzlich out.hidden_states, ein Tupel aus 13 Tensoren (Eingabe-Embeddings plus 12 Transformer-Schichten), jeder (B, T, 768). Dabei ist T die Zahl der Frames (199 bei 4 s).
    h = torch.stack(out.hidden_states, dim=1)      # (B, 13, T, 768)
    return h.mean(dim=2)                           # (B, 13, 768)


def extract_split(encoder, df, name, batch_size=16):
    loader = DataLoader(EmotionDataset(df), batch_size=batch_size, shuffle=False)
    feats = []
    for x, _ in tqdm(loader, desc=name):
        feats.append(extract_batch(encoder, x.to(device)).cpu().numpy())
    X = np.concatenate(feats)                          # (N, 13, 768)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"emb_{name}.npz",
             X=X,
             y=df["label"].to_numpy(),
             speaker=df["speaker"].to_numpy().astype(str),
             corpus=df["corpus"].to_numpy().astype(str))
    print(name, X.shape)

if __name__ == "__main__":
    encoder = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()

    df = build_metadata(ravdess_path, tess_path)
    train_df, val_df, test_df = split(df)

    for name, d in [("val", val_df), ("test", test_df), ("train", train_df)]:
        extract_split(encoder, d, name)