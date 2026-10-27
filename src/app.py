import gradio as gr
import numpy as np
import librosa
from live import Prediction, record
from data import sample_rate

predictor = Prediction() 



def analyze(y):                       
    rms = float(np.sqrt((y ** 2).mean()))
    if rms < 0.005:
        return None, f"Your voice is to low (Pegel {rms:.4f}), Please come closer to the microphone"
    probs = predictor.predict(y)
    return {k: float(v) for k, v in probs.items()}, f"Pegel {rms:.3f}"


def classify(audio):                 
    if audio is None:
        return None, "No record"
    sr, y = audio
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    return analyze(y)                 # ersetzt die alten Zeilen ab rms


def record_and_classify():            # Knopf-Aufnahme über sounddevice
    yield None, "Recording is on. Say some thing for 4s..."
    y = record(4)
    yield analyze(y)

with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown("# Speech Emotion Recognition\nPress and start speaking for 4s.")
    btn = gr.Button("Record for 4 seconds")
    label = gr.Label(num_top_classes=7, label="Emotion")
    status = gr.Textbox(label="Status")
    btn.click(record_and_classify, outputs=[label, status])

if __name__ == "__main__":
    demo.launch()