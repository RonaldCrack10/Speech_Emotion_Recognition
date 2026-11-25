"""Live inference pipeline, microphone recording, and sliding-window timeline."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WavLMModel

from data import labels as default_labels
from data import max_len, preprocess_audio, sample_rate, to_mono_float32
from extract import extract_batch
from training import CKPT, load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_rms(y: np.ndarray, sr: int = sample_rate) -> float:
    """Return root-mean-square loudness of mono-converted float32 audio."""
    mono = to_mono_float32(y, sr=sr)
    if len(mono) == 0:
        return 0.0
    return float(np.sqrt(np.mean(mono**2)))


def preprocess(y: np.ndarray, sr: int = sample_rate) -> np.ndarray:
    """Preprocess live audio identically to training/evaluation clips."""
    return preprocess_audio(y, sr=sr, target_len=max_len, trim_db=30.0, augment=False)


def record(second: float = 4.0) -> np.ndarray:
    """Record mono 16 kHz audio from the default input device."""
    import sounddevice as sd

    audio = sd.rec(
        int(second * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio[:, 0]


def show(audio: dict[str, float]) -> None:
    """Print sorted emotion probabilities to stdout."""
    for k, v in sorted(audio.items(), key=lambda kv: -kv[1]):
        print(f"{k:9s} {'.' * int(v * 30):30s} {v:.2f}")


class Prediction:
    """End-to-end predictor wrapping frozen WavLM-Large and a trained WeightedSER head."""

    def __init__(
        self,
        ckpt_path: str | Path = CKPT,
        encoder: torch.nn.Module | None = None,
        device_name: str = device,
    ):
        self.device = device_name
        self.encoder = (
            encoder.to(self.device).eval()
            if encoder is not None
            else WavLMModel.from_pretrained("microsoft/wavlm-large").to(self.device).eval()
        )
        self.head, ckpt = load_checkpoint(ckpt_path=ckpt_path, map_location=self.device)
        self.mean = torch.from_numpy(ckpt["mean"]).to(self.device)
        self.std = torch.from_numpy(ckpt["std"]).to(self.device)
        self.labels = list(ckpt.get("labels", default_labels))
        self.temperature = float(ckpt.get("temperature", 1.0))

    def layer_weights(self) -> np.ndarray:
        """Return the learned softmax weights across all WavLM layers."""
        return self.head.layer_weights().detach().cpu().numpy()

    @torch.no_grad()
    def _logits_from_waveforms(self, batch: torch.Tensor) -> torch.Tensor:
        emb = extract_batch(self.encoder, batch.to(self.device))
        emb = (emb - self.mean) / self.std
        return self.head(emb)

    @torch.no_grad()
    def predict(
        self,
        y: np.ndarray,
        sr: int = sample_rate,
        temperature: float | None = None,
    ) -> dict[str, float]:
        """Predict emotion probability distribution for a single audio clip."""
        temp = self.temperature if temperature is None else float(temperature)
        x = torch.from_numpy(preprocess(y, sr=sr)).float().unsqueeze(0)
        logits = self._logits_from_waveforms(x) / max(temp, 1e-4)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return {label: float(p) for label, p in zip(self.labels, probs)}

    @torch.no_grad()
    def predict_timeline(
        self,
        y: np.ndarray,
        sr: int = sample_rate,
        window_s: float = 2.0,
        hop_s: float = 0.5,
        ema_alpha: float = 0.65,
        temperature: float | None = None,
    ) -> list[dict]:
        """Compute smoothed emotion trajectories over overlapping sliding windows."""
        mono = to_mono_float32(y, sr=sr)
        if len(mono) == 0:
            return []

        trimmed, _ = librosa.effects.trim(mono, top_db=35)
        signal = trimmed if len(trimmed) > int(0.4 * sample_rate) else mono

        win_samples = max(int(window_s * sample_rate), int(0.5 * sample_rate))
        hop_samples = max(int(hop_s * sample_rate), int(0.25 * sample_rate))

        if len(signal) <= win_samples:
            probs = self.predict(signal, sr=sample_rate, temperature=temperature)
            return [{"t": round(len(signal) / sample_rate / 2, 2), "probs": probs}]

        windows = []
        times = []
        for start in range(0, len(signal) - win_samples + 1, hop_samples):
            chunk = signal[start : start + win_samples]
            chunk_norm = (chunk - float(chunk.mean())) / (float(chunk.std()) + 1e-7)
            padded = np.pad(chunk_norm, (0, max(0, max_len - len(chunk_norm))))[:max_len]
            windows.append(padded.astype(np.float32))
            times.append(round((start + win_samples / 2) / sample_rate, 2))

        if not windows:
            probs = self.predict(signal, sr=sample_rate, temperature=temperature)
            return [{"t": round(len(signal) / sample_rate / 2, 2), "probs": probs}]

        temp = self.temperature if temperature is None else float(temperature)
        batch = torch.from_numpy(np.stack(windows)).float()
        logits = self._logits_from_waveforms(batch) / max(temp, 1e-4)
        raw_probs = torch.softmax(logits, dim=1).cpu().numpy()

        timeline = []
        smoothed = raw_probs[0].copy()
        for idx, t in enumerate(times):
            if idx > 0:
                smoothed = ema_alpha * raw_probs[idx] + (1.0 - ema_alpha) * smoothed
            probs_dict = {label: float(p) for label, p in zip(self.labels, smoothed)}
            timeline.append({"t": t, "probs": probs_dict})
        return timeline


if __name__ == "__main__":
    pred = Prediction()
    while True:
        input("\nPress Enter and Speak for 4s (Ctrl+C to End) ")
        y = record()
        print(f"Level (RMS): {compute_rms(y):.4f}")
        show(pred.predict(y))
        response = input("Do you want to test once more? (y/n): ").strip().lower()
        if response != "y":
            break
