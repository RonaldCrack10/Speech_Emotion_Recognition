# Wie sich das Modell verhält wenn ganz neue Daten zum Vorhersagen einfliessen

import numpy as np
import torch
import librosa
from transformers import WavLMModel
from data import build_metadata, split, EmotionDataset, ravdess_path, tess_path

from data import labels, sample_rate, max_len
from training import WeightedSER as Head, CKPT
from extract import extract_batch
import sounddevice as sd

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess(y):
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) > max_len:
        s = (len(y) - max_len) // 2
        y = y[s:s + max_len]
    y = np.pad(y, (0, max_len - len(y)))
    return (y - y.mean()) / (y.std() + 1e-7)

def record(second = 4):
    audio = sd.rec(
        int(second * sample_rate),
        samplerate = sample_rate,
        channels = 1,
        dtype = "float32"
    )

    sd.wait()
    return audio[:, 0]

def show(audio):
    for k, v in sorted(audio.items(), key= lambda kv: -kv[1]):
        print(f"{k:9s} {'.' * int(v * 30):30s} {v:.2f}")

class Prediction:
    def __init__(self):
        self.encoder = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
        ckpt = torch.load(CKPT, map_location= device, weights_only= False
                          )
        self.mean = torch.from_numpy(ckpt["mean"]).to(device)     # (1, 13, 768)
        self.std = torch.from_numpy(ckpt["std"]).to(device)
        self.head = Head().to(device)
        self.head.load_state_dict(ckpt["model"])
        self.head.eval()
        self.labels = ckpt["labels"]

    @torch.no_grad()
    def predict(self, y):
        x = torch.from_numpy(preprocess(y)).float()[None].to(device)   # (1, 64000)
        e = extract_batch(self.encoder, x)                             # (1, 13, 768)
        e = (e - self.mean) / self.std
        p = torch.softmax(self.head(e), dim=1)[0].cpu().numpy()
        return dict(zip(self.labels, p))



if __name__ == "__main__":
    pred = Prediction()
    while True:
        input("\nPress Enter and Speak for 4s (Strg+C to End) ")
        y = record()
        print(f"Pegel (RMS): {np.sqrt((y ** 2).mean()):.4f}")
        show(pred.predict(y))
        responce = input('Do you want to test once more ? (y/n): ')
        if responce == "y":
            continue
        else:
            break