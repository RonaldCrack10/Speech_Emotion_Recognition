"""extract.py: compute frozen WavLM-Large embeddings once and save them to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMModel

from data import EmotionDataset, build_metadata, own_path, ravdess_path, split, tess_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract_batch(encoder: WavLMModel, x: torch.Tensor) -> torch.Tensor:
    """Convert normalized waveforms (B, 64000) into mean-pooled layer embeddings (B, 25, 1024)."""
    out = encoder(x, output_hidden_states=True)
    h = torch.stack(out.hidden_states, dim=1)  # (B, 25, T, 1024) for wavlm-large
    return h.mean(dim=2)  # (B, 25, 1024)


def extract_split(encoder, df, name: str, batch_size: int = 16, augment_copies: int = 0) -> None:
    """Run all clips of one split through WavLM and save embeddings as .npz."""
    passes = [("clean", False)] + [(f"aug_{i + 1}", True) for i in range(augment_copies)]
    feats, ys, speakers, corpora = [], [], [], []

    for tag, aug in passes:
        loader = DataLoader(EmotionDataset(df, augment=aug), batch_size=batch_size, shuffle=False)
        for x, _ in tqdm(loader, desc=f"{name} ({tag})"):
            feats.append(extract_batch(encoder, x.to(device)).cpu().numpy())
        ys.append(df["label"].to_numpy())
        speakers.append(df["speaker"].to_numpy().astype(str))
        corpora.append(df["corpus"].to_numpy().astype(str))

    X = np.concatenate(feats)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"emb_{name}.npz",
        X=X,
        y=np.concatenate(ys),
        speaker=np.concatenate(speakers),
        corpus=np.concatenate(corpora),
    )
    print(name, X.shape)


if __name__ == "__main__":
    encoder = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()
    df = build_metadata(ravdess_path, tess_path, own_path)
    train_df, val_df, test_df = split(df)

    for split_name, split_df, copies in [
        ("val", val_df, 0),
        ("test", test_df, 0),
        ("train", train_df, 0),
    ]:
        extract_split(encoder, split_df, split_name, augment_copies=copies)
