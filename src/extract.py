"""
extract.py: compute embeddings once and save them to disk.

The encoder (WavLM) is the expensive part and never changes because it is frozen.
So every clip goes through it exactly once and the result is stored, instead of
repeating the computation in every training epoch. We keep ALL layers, because we
do not know in advance which ones are most useful for emotion. The small head in
train_cached.py later learns on its own how strongly to weight each layer.

Output: data/processed/{train,val,test}.npz
Shape of X: (N clips, 25 layers, 1024 values). For WavLM-base it would be (N, 13, 768).
"""
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
def extract_batch(encoder, x): # Audio dateien gehen durchs WavLM modell und werden von Wellen Form (B, 64_000) zu Vektoren umgewandelt (B, 13, 768)
    """Waveforms -> one vector per encoder layer.

    Input:   x with shape (B, 64000), i.e. B clips of 4 s at 16 kHz, already normalized
    Output:  shape (B, 25, 1024), one vector per clip for each of the 25 outputs
             (input embeddings plus 24 transformer layers)

    live.py uses this same function, so training and live inference run
    exactly the same computation.
    """
    # output_hidden_states=True: the encoder returns ALL layers, not just the last one
    out = encoder(x, output_hidden_states=True)
    # stack the tuple of 25 tensors (B, T, 1024) into one tensor; T = frames (about 199 for 4 s)
    h = torch.stack(out.hidden_states, dim=1)      # (B, 25, T, 1024) T = 
    # average over the time axis: T frame vectors become one vector per layer
    return h.mean(dim=2)                           # (B, 25, 1024)
                    


def extract_split(encoder, df, name, batch_size=16):
    """Run all clips of one split through the encoder and save them as .npz.

    encoder:    loaded WavLM in eval() mode
    df:         table of the split (columns path, label, speaker, corpus)
    name:       "train", "val" or "test" (used in the file name)
    batch_size: clips per pass; reduce on CUDA out-of-memory
    """
    # shuffle=False is important: the order of the vectors must match the order of
    # the rows in df, otherwise labels and features no longer belong together.
    # EmotionDataset without augment: fixed preprocessing (trim, 4 s, normalize).
    loader = DataLoader(EmotionDataset(df), batch_size=batch_size, shuffle=False)

    feats = []
    for x, _ in tqdm(loader, desc=name):   # the dataset label is not needed here
        # batch to the GPU, compute embeddings, bring the result back to the CPU as NumPy
        feats.append(extract_batch(encoder, x.to(device)).cpu().numpy())
    X = np.concatenate(feats)              # join all batches: (N, 25, 1024)

    OUT.mkdir(parents=True, exist_ok=True)   # create the folder if it does not exist
    # Four arrays in one file. speaker and corpus are stored as str so the file
    # can be loaded later without allow_pickle.
    np.savez(OUT / f"emb_{name}.npz",
             X=X,                                          # embeddings
             y=df["label"].to_numpy(),                     # labels (0 to 6)
             speaker=df["speaker"].to_numpy().astype(str), # for per-speaker evaluation
             corpus=df["corpus"].to_numpy().astype(str))   # ravdess or tess
    print(name, X.shape)                   # sanity check, e.g. train (3528, 25, 1024)


# Runs only when extract.py is started directly, not on import
# (live.py imports extract_batch and must not execute anything).
if __name__ == "__main__":
    # .eval() turns dropout off: the same clip always produces the same vectors
    encoder = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()

    # same table and same speaker-independent split as in training
    df = build_metadata(ravdess_path, tess_path)
    train_df, val_df, test_df = split(df)

    # small splits first: errors show up after seconds instead of minutes
    for name, d in [("val", val_df), ("test", test_df), ("train", train_df)]:
        extract_split(encoder, d, name)