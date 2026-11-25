"""Unit tests for data splitting, preprocessing, WeightedSER, live pipeline, and app helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import app  # noqa: E402
import data  # noqa: E402
import live  # noqa: E402
import training  # noqa: E402


class DummyEncoder(nn.Module):
    """Lightweight stand-in for WavLMModel returning deterministic (B, T, hidden) hidden states."""

    def __init__(self, n_layers: int = 25, hidden: int = 1024):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = hidden

    def forward(self, x: torch.Tensor, output_hidden_states: bool = True):
        bsz = x.shape[0]
        energy = x.abs().mean(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        base = torch.ones((bsz, 8, self.hidden), device=x.device, dtype=x.dtype) * energy
        states = tuple(base * (1.0 + 0.05 * i) for i in range(self.n_layers))
        return SimpleNamespace(hidden_states=states)


def test_weighted_ser_import_and_forward():
    model = training.WeightedSER(n_layers=25, hidden=1024, n_classes=7)
    x = torch.randn(4, 25, 1024)
    logits = model(x)
    assert logits.shape == (4, 7)
    weights = model.layer_weights()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_build_metadata_and_speaker_independent_split(tmp_path: Path):
    rav = tmp_path / "ravdess"
    tess = tmp_path / "tess"
    own = tmp_path / "own"
    rav.mkdir()
    tess.mkdir()
    own.mkdir()

    # Include single-digit actor (03), val actor (15), test actor (20), and calm (02 -> skipped)
    for fname in [
        "03-01-05-01-01-01-03.wav",  # angry, rav_03
        "03-01-02-01-01-01-03.wav",  # calm -> excluded
        "03-01-03-01-01-01-15.wav",  # happy, rav_15 (val)
        "03-01-04-01-01-01-20.wav",  # sad, rav_20 (test)
    ]:
        (rav / fname).write_bytes(b"")

    (tess / "OAF_back_ps.wav").write_bytes(b"")
    (own / "own_ronald_fear_s0_01.wav").write_bytes(b"")

    df = data.build_metadata(ravdess_dir=rav, tess_dir=tess, own_dir=own)
    assert len(df) == 5
    assert set(df["corpus"]) == {"ravdess", "tess", "own"}

    train_df, val_df, test_df = data.split(df, val_actors={3, 15}, test_actors={20})
    assert set(val_df["speaker"]) == {"rav_03", "rav_15"}
    assert set(test_df["speaker"]) == {"rav_20"}
    assert set(train_df["speaker"]) == {"tess_oaf", "own_ronald"}
    assert set(train_df["speaker"]).isdisjoint(set(val_df["speaker"]))
    assert set(train_df["speaker"]).isdisjoint(set(test_df["speaker"]))


def test_preprocess_audio_handles_stereo_resampling_and_normalization():
    sr = 48_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([mono, mono], axis=1)  # (N, 2)

    out = data.preprocess_audio(stereo, sr=sr)
    assert out.shape == (data.max_len,)
    assert out.dtype == np.float32
    assert abs(float(out.mean())) < 1e-4
    assert abs(float(out.std()) - 1.0) < 1e-3


def test_prediction_and_timeline_with_mock_checkpoint(tmp_path: Path):
    ckpt_path = tmp_path / "mock_ser.pt"
    model = training.WeightedSER(n_layers=25, hidden=1024, n_classes=7)
    torch.save(
        {
            "model": model.state_dict(),
            "labels": data.labels,
            "mean": np.zeros((1, 25, 1024), dtype=np.float32),
            "std": np.ones((1, 25, 1024), dtype=np.float32),
            "val_f1": 0.80,
        },
        ckpt_path,
    )

    pred = live.Prediction(ckpt_path=ckpt_path, encoder=DummyEncoder(), device_name="cpu")
    t = np.linspace(0, 3.5, int(data.sample_rate * 3.5), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    probs = pred.predict(wave, sr=data.sample_rate)
    assert set(probs.keys()) == set(data.labels)
    assert abs(sum(probs.values()) - 1.0) < 1e-5

    timeline = pred.predict_timeline(wave, sr=data.sample_rate, window_s=1.5, hop_s=0.5)
    assert len(timeline) >= 3
    assert all("t" in pt and set(pt["probs"].keys()) == set(data.labels) for pt in timeline)


def test_app_quiet_rejection_and_result_html(tmp_path: Path):
    ckpt_path = tmp_path / "mock_ser.pt"
    model = training.WeightedSER(n_layers=25, hidden=1024, n_classes=7)
    torch.save(
        {
            "model": model.state_dict(),
            "labels": data.labels,
            "mean": np.zeros((1, 25, 1024), dtype=np.float32),
            "std": np.ones((1, 25, 1024), dtype=np.float32),
        },
        ckpt_path,
    )
    pred = live.Prediction(ckpt_path=ckpt_path, encoder=DummyEncoder(), device_name="cpu")

    silent = np.zeros(data.sample_rate * 2, dtype=np.float32)
    probs, timeline, msg = app.analyze_audio(silent, sr=data.sample_rate, predictor=pred)
    assert probs is None
    assert timeline == []
    assert "Too quiet" in msg

    low_conf_probs = {lbl: 1.0 / len(data.labels) for lbl in data.labels}
    html = app.result_html(low_conf_probs, threshold=0.5, ground_truth="happy")
    assert "Not sure" in html
    assert "Ground truth" in html
