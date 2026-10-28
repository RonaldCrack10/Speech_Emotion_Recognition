from pathlib import Path

import numpy as np
import soundfile as sf

from live import record
from data import sample_rate, labels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"

SPEAKER = ""          # pro Person ändern, z. B. "anna"
REPS = 3                   # Wiederholungen pro Emotion und Satz
SENTENCES = [
    "Kids are talking by the door",
    "Dogs are sitting by the door",
    "The car is in front of the house",
    "I have just receive the message",
]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for emo in labels:
        for s_idx, sentence in enumerate(SENTENCES):
            for rep in range(1, REPS + 1):
                path = OUT / f"own_{SPEAKER}_{emo}_s{s_idx}_{rep:02d}.wav"
                if path.exists():
                    continue
                while True:
                    input(f'\n[{emo.upper()}] "{sentence}"   Enter, dann sprechen ')
                    y = record(4)
                    rms = float(np.sqrt((y ** 2).mean()))
                    if rms < 0.01:
                        print(f"zu leise (Pegel {rms:.4f}), nochmal")
                        continue
                    if input("Behalten? Enter = ja, n = nochmal ").strip().lower() != "n":
                        break
                sf.write(path, y, sample_rate)
                print("gespeichert:", path.name)


if __name__ == "__main__":
    run()