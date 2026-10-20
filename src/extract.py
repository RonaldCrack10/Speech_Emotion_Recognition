from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMModel

from data import build_metadata, split, EmotionDataset, ravdess_path, tess_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def extract_batch(encoder, x):
    out = encoder(x, output_hidden_states=True)
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
    encoder = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    df = build_metadata(ravdess_path, tess_path)
    train_df, val_df, test_df = split(df)

    for name, d in [("val", val_df), ("test", test_df), ("train", train_df)]:
        extract_split(encoder, d, name)