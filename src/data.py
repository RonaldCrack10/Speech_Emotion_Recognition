"""Dataset loading, speaker-independent splitting, and shared audio preprocessing."""

from __future__ import annotations

from pathlib import Path
import random

import librosa
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def _resolve_raw_dir(*candidates: Path) -> Path:
    """Return the first existing dataset path, falling back to the primary layout."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


ravdess_path = _resolve_raw_dir(
    ROOT / "data" / "raw" / "Ravedess dataset",
    ROOT / "data" / "raw" / "RAVDESS",
    ROOT / "src" / "RawData" / "Ravedess dataset",
)
tess_path = _resolve_raw_dir(
    ROOT / "data" / "raw" / "TESS Toronto emotional speech set data",
    ROOT / "data" / "raw" / "TESS",
    ROOT / "src" / "RawData" / "TESS Toronto emotional speech set data",
)
own_path = _resolve_raw_dir(
    ROOT / "data" / "raw",
    ROOT / "src" / "RawData",
)

sample_rate = 16_000
max_len = 4 * sample_rate  # 4 seconds * 16,000 samples/s = 64,000 samples

ravdess_map = {
    "01": "neutral",
    "02": None,  # calm is excluded
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

tess_map = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprise",
}

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_id = {label: idx for idx, label in enumerate(labels)}


def build_metadata(
    ravdess_dir: str | Path = ravdess_path,
    tess_dir: str | Path = tess_path,
    own_dir: str | Path | None = own_path,
) -> pd.DataFrame:
    """Scan RAVDESS, TESS, and optional custom recordings into one metadata table."""
    rows: list[dict] = []

    ravdess_root = Path(ravdess_dir)
    if ravdess_root.exists():
        for p in sorted(ravdess_root.rglob("*.wav")):
            parts = p.stem.split("-")
            if len(parts) != 7:
                continue
            emo = ravdess_map.get(parts[2])
            if emo is None:
                continue
            rows.append(
                {
                    "path": str(p),
                    "label": label_to_id[emo],
                    "speaker": f"rav_{int(parts[6]):02d}",
                    "corpus": "ravdess",
                }
            )

    tess_root = Path(tess_dir)
    if tess_root.exists():
        for p in sorted(tess_root.rglob("*.wav")):
            parts = p.stem.split("_")
            if len(parts) < 2:
                continue
            emo = tess_map.get(parts[-1].lower())
            if emo is None:
                continue
            rows.append(
                {
                    "path": str(p),
                    "label": label_to_id[emo],
                    "speaker": f"tess_{parts[0].lower()}",
                    "corpus": "tess",
                }
            )

    if own_dir is not None and Path(own_dir).exists():
        for p in sorted(Path(own_dir).glob("own_*.wav")):
            parts = p.stem.split("_")
            if len(parts) < 4:
                continue
            speaker_tag = parts[1].lower() or "anon"
            emo = parts[2].lower()
            if emo not in label_to_id:
                continue
            rows.append(
                {
                    "path": str(p),
                    "label": label_to_id[emo],
                    "speaker": f"own_{speaker_tag}",
                    "corpus": "own",
                }
            )

    return pd.DataFrame(rows, columns=["path", "label", "speaker", "corpus"])


def split(
    df: pd.DataFrame,
    val_actors: set[int] | None = None,
    test_actors: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata by speaker so validation and test actors never appear in training."""
    if val_actors is None:
        val_actors = {15, 16, 17, 18, 19}
    if test_actors is None:
        test_actors = {20, 21, 22, 23, 24}

    val_ids = {f"rav_{int(a):02d}" for a in val_actors}
    test_ids = {f"rav_{int(b):02d}" for b in test_actors}

    val = df[df["speaker"].isin(val_ids)]
    test = df[df["speaker"].isin(test_ids)]
    train = df[~df["speaker"].isin(val_ids | test_ids)]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def augment_data(y: np.ndarray) -> np.ndarray:
    """Apply lightweight waveform perturbations (gain, shift, noise) for training."""
    y = y.astype(np.float32, copy=True)
    if len(y) == 0:
        return y

    if random.random() < 0.5:
        y = y * random.uniform(0.7, 1.3)

    if random.random() < 0.5 and len(y) > 1600:
        shift = random.randint(-int(0.15 * sample_rate), int(0.15 * sample_rate))
        y = np.roll(y, shift)

    if random.random() < 0.5:
        noise_scale = float(y.std()) * random.uniform(0.002, 0.05)
        y = y + np.random.randn(len(y)).astype(np.float32) * noise_scale

    return y


def to_mono_float32(y: np.ndarray, sr: int = sample_rate) -> np.ndarray:
    """Convert arbitrary integer/float mono or stereo audio to 16 kHz mono float32."""
    arr = np.asarray(y)
    if arr.ndim > 1:
        # Handle both (samples, channels) and (channels, samples)
        channel_axis = 0 if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] else -1
        arr = arr.mean(axis=channel_axis)

    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32)

    if sr != sample_rate and len(arr) > 0:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=sample_rate).astype(np.float32)
    return arr


def preprocess_audio(
    y: np.ndarray,
    sr: int = sample_rate,
    target_len: int = max_len,
    trim_db: float = 30.0,
    augment: bool = False,
) -> np.ndarray:
    """Normalize sample rate, trim silence, crop/pad to 4 s, and z-score normalize."""
    mono = to_mono_float32(y, sr=sr)
    if len(mono) > 0:
        mono, _ = librosa.effects.trim(y=mono, top_db=trim_db)

    if augment:
        mono = augment_data(mono)

    if len(mono) > target_len:
        if augment:
            start = random.randint(0, len(mono) - target_len)
        else:
            start = (len(mono) - target_len) // 2
        mono = mono[start : start + target_len]

    if len(mono) < target_len:
        mono = np.pad(mono, (0, target_len - len(mono)))

    std = float(mono.std())
    return ((mono - float(mono.mean())) / (std + 1e-7)).astype(np.float32)


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced emotion counts."""
    counts = np.bincount(df["label"], minlength=len(labels))
    weights = counts.sum() / (len(labels) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


class EmotionDataset:
    """PyTorch-compatible dataset returning (normalized_waveform, label_id)."""

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        y, _ = librosa.load(path=row["path"], sr=sample_rate, mono=True)
        processed = preprocess_audio(y, sr=sample_rate, augment=self.augment)
        return torch.tensor(processed, dtype=torch.float32), int(row["label"])


if __name__ == "__main__":
    df = build_metadata(ravdess_dir=ravdess_path, tess_dir=tess_path)
    train_df, val_df, test_df = split(df=df)
    print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Corpora:", df["corpus"].value_counts().to_dict())
