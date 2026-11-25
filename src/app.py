"""Conference-ready Gradio web demo for Speech Emotion Recognition."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import build_metadata, labels, ravdess_path, sample_rate, split, to_mono_float32
from live import Prediction, compute_rms, record

COLOR = {
    "angry": "#ef4444",
    "disgust": "#84cc16",
    "fear": "#a855f7",
    "happy": "#facc15",
    "neutral": "#94a3b8",
    "sad": "#3b82f6",
    "surprise": "#f97316",
}
DEFAULT_THRESHOLD = 0.5
MIN_RMS = 0.005


@lru_cache(maxsize=1)
def get_predictor() -> Prediction:
    """Lazy-load WavLM-Large and the trained WeightedSER head once."""
    return Prediction()


def discover_sample_clips(max_clips: int = 7) -> list[dict]:
    """Find one clean sample clip per emotion from unseen RAVDESS test actors (20-24)."""
    if not ravdess_path.exists():
        return []
    df = build_metadata(ravdess_dir=ravdess_path, tess_dir=Path("__nonexistent__"), own_dir=None)
    if df.empty:
        return []
    _, _, test_df = split(df)
    if test_df.empty:
        return []

    samples = []
    for emo_id, emo_name in enumerate(labels):
        subset = test_df[test_df["label"] == emo_id]
        if subset.empty:
            continue
        row = subset.iloc[len(subset) // 2]
        samples.append(
            {
                "label": emo_name,
                "speaker": row["speaker"],
                "path": row["path"],
                "button_text": f"{emo_name.capitalize()} ({row['speaker']})",
            }
        )
        if len(samples) >= max_clips:
            break
    return samples


def analyze_audio(
    y: np.ndarray,
    sr: int = sample_rate,
    predictor: Prediction | None = None,
    temperature: float = 1.0,
) -> tuple[dict[str, float] | None, list[dict], str]:
    """Validate loudness, classify clip, and compute sliding-window timeline."""
    mono = to_mono_float32(y, sr=sr)
    rms = compute_rms(mono, sr=sample_rate)
    if rms < MIN_RMS:
        return (
            None,
            [],
            f"Too quiet (RMS {rms:.4f}). Move closer to the microphone or raise input gain.",
        )

    pred = predictor or get_predictor()
    probs = pred.predict(mono, sr=sample_rate, temperature=temperature)
    timeline = pred.predict_timeline(mono, sr=sample_rate, temperature=temperature)
    return probs, timeline, f"Signal OK · RMS {rms:.3f} · Duration {len(mono) / sample_rate:.1f}s"


def status_html(text: str, kind: str = "idle") -> str:
    return f'<div class="pill {kind}"><span class="dot"></span>{text}</div>'


EMPTY = (
    '<div class="card result"><div class="orb" style="--c:#64748b"><span>&ndash;</span></div>'
    '<div class="title" style="color:#c7c7d9">Ready when you are</div>'
    '<div class="sub">Record live, upload audio, or click an unseen test-actor sample</div></div>'
)


def result_html(
    probs: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    ground_truth: str | None = None,
) -> str:
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    top, p = items[0]
    if p < threshold:
        inner, title, color = "?", "Not sure", "#94a3b8"
        sub = f"best guess: {top} ({p:.0%}) &middot; below {threshold:.0%} threshold"
    else:
        inner, title, color = f"{p:.0%}", top.capitalize(), COLOR[top]
        sub = "confidence"

    gt_badge = ""
    if ground_truth:
        match = top == ground_truth
        badge_cls = "gt-ok" if match else "gt-miss"
        verdict = "Match" if match else "Mismatch"
        gt_badge = (
            f'<div class="gt-badge {badge_cls}">Ground truth: <b>{ground_truth}</b> &middot; {verdict}</div>'
        )

    bars = "".join(
        f'<div class="row"><span class="nm"><i class="sw" style="background:{COLOR[k]}"></i>{k}</span>'
        f'<div class="track"><div class="fill" style="width:{v * 100:.1f}%;background:{COLOR[k]}"></div></div>'
        f'<span class="pc">{v:.0%}</span></div>'
        for k, v in items
    )
    return (
        f'<div class="card result"><div class="orb" style="--c:{color}"><span>{inner}</span></div>'
        f'<div class="title" style="color:{color}">{title}</div>'
        f'<div class="sub">{sub}</div>{gt_badge}<div class="bars">{bars}</div></div>'
    )


def history_html(history: list[tuple[str, float]]) -> str:
    if not history:
        return '<div class="card hist"><div class="hlabel">Recent attempts</div><div class="sub">none yet</div></div>'
    chips = "".join(
        f'<span class="chip"><i class="sw" style="background:{COLOR[e]}"></i>{e} {p:.0%}</span>'
        for e, p in reversed(history)
    )
    return f'<div class="card hist"><div class="hlabel">Recent attempts</div>{chips}</div>'


def _dark_fig(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.patch.set_facecolor("#0f1020")
    ax.set_facecolor("#0f1020")
    ax.tick_params(colors="#cbd5e1", labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#334155")
    return fig, ax


def build_audio_figure(y: np.ndarray, sr: int = sample_rate):
    """Plot waveform and mel-spectrogram of the analyzed audio."""
    mono = to_mono_float32(y, sr=sr)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 4.0), dpi=130, sharex=True)
    fig.patch.set_facecolor("#0f1020")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1020")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#334155")

    t = np.linspace(0, len(mono) / sample_rate, num=max(len(mono), 1))
    ax1.plot(t, mono, color="#818cf8", linewidth=0.9)
    ax1.fill_between(t, mono, color="#818cf8", alpha=0.18)
    ax1.set_ylabel("Amplitude", color="#cbd5e1", fontsize=8.5)
    ax1.set_title("Waveform & Mel-Spectrogram (16 kHz Mono)", color="#f8fafc", fontsize=10, fontweight="bold")

    if len(mono) >= 512:
        S = librosa.feature.melspectrogram(y=mono, sr=sample_rate, n_mels=64, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(
            S_db, sr=sample_rate, x_axis="time", y_axis="mel", fmax=8000, cmap="magma", ax=ax2
        )
    ax2.set_ylabel("Hz", color="#cbd5e1", fontsize=8.5)
    ax2.set_xlabel("Time (s)", color="#cbd5e1", fontsize=8.5)

    fig.tight_layout()
    return fig


def build_timeline_figure(timeline: list[dict]):
    """Plot emotion probability curves over overlapping sliding windows."""
    fig, ax = _dark_fig((6.8, 3.3))
    if not timeline:
        ax.text(0.5, 0.5, "No timeline available", color="#94a3b8", ha="center", va="center")
        return fig

    times = [pt["t"] for pt in timeline]
    avg_scores = {emo: np.mean([pt["probs"][emo] for pt in timeline]) for emo in labels}
    top_emotions = [k for k, _ in sorted(avg_scores.items(), key=lambda kv: -kv[1])[:4]]

    for emo in top_emotions:
        vals = [pt["probs"][emo] * 100.0 for pt in timeline]
        marker = "o" if len(times) <= 12 else None
        ax.plot(times, vals, label=emo.capitalize(), color=COLOR[emo], linewidth=2.2, marker=marker, markersize=4)

    ax.set_ylim(0, 102)
    ax.set_xlabel("Window Center Time (s)", color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("Probability (%)", color="#cbd5e1", fontsize=8.5)
    ax.set_title("Emotion Trajectory Over Time (2.0 s Sliding Window)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.16, color="#64748b")
    ax.legend(loc="upper right", frameon=False, labelcolor="#e2e8f0", fontsize=8.5, ncol=2)
    fig.tight_layout()
    return fig


def build_layer_weights_figure(weights: np.ndarray | None = None):
    """Visualize the learned WavLM-Large layer weights inside WeightedSER."""
    if weights is None:
        weights = get_predictor().layer_weights()

    fig, ax = _dark_fig((6.8, 3.0))
    layers = np.arange(len(weights))
    peak = int(np.argmax(weights))
    bar_colors = [
        "#ec4899" if i == peak else ("#818cf8" if w >= np.median(weights) else "#475569")
        for i, w in enumerate(weights)
    ]
    ax.bar(layers, weights * 100.0, color=bar_colors, width=0.72)
    ax.axhline(100.0 / len(weights), color="#94a3b8", linestyle="--", linewidth=0.9, label="Uniform (4%)")
    ax.set_xticks(layers[::2])
    ax.set_xticklabels([f"L{i}" for i in layers[::2]], fontsize=8)
    ax.set_ylabel("Weight (%)", color="#cbd5e1", fontsize=8.5)
    ax.set_xlabel("WavLM-Large Layer (L0 = acoustic, L1–L24 = transformer)", color="#cbd5e1", fontsize=8.5)
    ax.set_title("Inside the Model: Learned Layer Importance", color="#f8fafc", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.15, color="#64748b")
    ax.legend(frameon=False, labelcolor="#e2e8f0", fontsize=8.5)
    fig.tight_layout()
    return fig


def _run_pipeline(
    y: np.ndarray,
    sr: int,
    history: list[tuple[str, float]],
    threshold: float,
    ground_truth: str | None = None,
):
    probs, timeline, msg = analyze_audio(y, sr=sr)
    if probs is None:
        return (
            status_html(msg, "warn"),
            EMPTY,
            history_html(history),
            history,
            build_audio_figure(y, sr=sr),
            None,
        )

    top = max(probs, key=probs.get)
    new_history = (history + [(top, probs[top])])[-6:]
    return (
        status_html(msg, "ok"),
        result_html(probs, threshold=threshold, ground_truth=ground_truth),
        history_html(new_history),
        new_history,
        build_audio_figure(y, sr=sr),
        build_timeline_figure(timeline),
    )


def run_local_mic(history, threshold):
    yield (
        gr.update(interactive=False, value="Recording 4s..."),
        status_html("Speak now (4 seconds)", "rec"),
        EMPTY,
        history_html(history),
        history,
        None,
        None,
        None,
    )
    y = record(4)
    st, res, hist_h, new_hist, aud_fig, tl_fig = _run_pipeline(y, sample_rate, history, threshold)
    yield (
        gr.update(interactive=True, value="Record 4s"),
        st,
        res,
        hist_h,
        new_hist,
        (sample_rate, y),
        aud_fig,
        tl_fig,
    )


def run_uploaded_or_browser(audio_input, history, threshold):
    if audio_input is None:
        return (
            status_html("Record with the browser mic or upload a WAV/MP3 clip first.", "warn"),
            EMPTY,
            history_html(history),
            history,
            None,
            None,
        )
    sr, y = audio_input
    return _run_pipeline(y, int(sr), history, threshold)


def run_sample_clip(sample_path: str, ground_truth: str, history, threshold):
    y, sr = librosa.load(sample_path, sr=sample_rate, mono=True)
    st, res, hist_h, new_hist, aud_fig, tl_fig = _run_pipeline(
        y, sr, history, threshold, ground_truth=ground_truth
    )
    return st, res, hist_h, new_hist, (sr, y), aud_fig, tl_fig


CSS = """
.gradio-container {
  background: radial-gradient(1100px 600px at 8% -10%, #312e81 0%, transparent 60%),
              radial-gradient(900px 500px at 100% 0%, #831843 0%, transparent 55%), #0b0b14 !important;
}
footer {display: none !important;}
.hero {text-align: center; padding: 24px 10px 6px;}
.hero h1 {font-size: 2.5rem; font-weight: 800; margin: 0; color: transparent;
  background: linear-gradient(90deg, #818cf8, #f472b6, #fbbf24); -webkit-background-clip: text; background-clip: text;}
.hero p {color: #a1a1b5; margin: 8px 0 0; font-size: 1.02rem;}
.card {background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10);
  border-radius: 20px; padding: 20px; color: #e5e7f0; backdrop-filter: blur(8px);}
.result {text-align: center;}
.orb {width: 140px; height: 140px; margin: 6px auto 12px; border-radius: 50%;
  border: 5px solid var(--c); display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 50px -6px var(--c), inset 0 0 30px -10px var(--c);
  background: radial-gradient(circle, rgba(255,255,255,.07), transparent 70%);}
.orb span {font-size: 2.2rem; font-weight: 800; color: #fff;}
.title {font-size: 1.9rem; font-weight: 800; margin-top: 4px;}
.sub {color: #9a9ab0; margin-bottom: 10px;}
.gt-badge {display: inline-block; padding: 4px 12px; border-radius: 99px; font-size: .85rem; margin-bottom: 10px;}
.gt-ok {background: rgba(34,197,94,.16); color: #86efac; border: 1px solid rgba(34,197,94,.4);}
.gt-miss {background: rgba(245,158,11,.16); color: #fcd34d; border: 1px solid rgba(245,158,11,.4);}
.bars {margin-top: 12px; text-align: left;}
.row {display: grid; grid-template-columns: 110px 1fr 48px; align-items: center; gap: 10px; margin: 7px 0;}
.nm {color: #cfd0e0; font-size: .92rem; display: flex; align-items: center; gap: 8px;}
.sw {display: inline-block; width: 10px; height: 10px; border-radius: 50%;}
.pc {color: #9a9ab0; text-align: right; font-variant-numeric: tabular-nums;}
.track {height: 11px; background: rgba(255,255,255,.08); border-radius: 99px; overflow: hidden;}
.fill {height: 100%; border-radius: 99px; transition: width .6s ease;}
.rec-btn {background: linear-gradient(90deg, #6366f1, #ec4899) !important; color: #fff !important; border: none !important;
  font-size: 1.05rem !important; font-weight: 700 !important; padding: 14px !important; border-radius: 14px !important;
  box-shadow: 0 8px 30px rgba(236,72,153,.3);}
.pill {display: inline-flex; align-items: center; gap: 10px; padding: 9px 15px; border-radius: 99px;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); color: #d6d7e6; margin-top: 10px;}
.dot {width: 10px; height: 10px; border-radius: 50%; background: #64748b;}
.pill.ok .dot {background: #22c55e;}
.pill.warn .dot {background: #f59e0b;}
.pill.rec .dot {background: #ef4444; animation: pulse 1s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(239,68,68,.7);} 100% {box-shadow: 0 0 0 14px rgba(239,68,68,0);}}
.hist {margin-top: 12px;}
.hlabel {color: #9a9ab0; font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px;}
.chip {display: inline-flex; align-items: center; gap: 8px; margin: 4px 6px 4px 0; padding: 5px 11px; border-radius: 99px;
  border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04); color: #e5e7f0; font-size: .88rem;}
.how {color: #b8b9cc; line-height: 1.55;}
.how b {color: #fff;}
.foot {text-align: center; color: #6b6b85; font-size: .85rem; padding: 14px 0 6px;}
"""

HERO = (
    '<div class="hero"><h1>Speech Emotion Recognition</h1>'
    "<p>WavLM-Large with Learned Layer Weighting &middot; Real-Time Acoustic Emotion Explorer</p></div>"
)
HOWTO = (
    '<div class="card how"><b>Try it 3 ways</b><br>'
    "1. Click <b>Record 4s</b> on the demo laptop<br>"
    "2. Record in your browser or upload any <code>.wav/.mp3</code> clip<br>"
    "3. Click an <b>unseen test-actor sample</b> below to verify live inference</div>"
)
FOOT = (
    '<div class="foot">WavLM-Large (frozen) &middot; Learned Layer Weighting (SUPERB-style) &middot; '
    "Speaker-Independent RAVDESS &amp; TESS</div>"
)


def create_demo() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    sample_clips = discover_sample_clips()

    with gr.Blocks(title="Speech Emotion Recognition") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(HERO)
        hist = gr.State([])

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.HTML(HOWTO)
                local_btn = gr.Button("Record 4s", elem_classes="rec-btn")
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Browser Microphone / Upload Clip",
                )
                analyze_btn = gr.Button("Analyze Audio Input", variant="secondary")
                threshold = gr.Slider(
                    minimum=0.2,
                    maximum=0.85,
                    value=DEFAULT_THRESHOLD,
                    step=0.05,
                    label='Confidence threshold ("Not sure" below)',
                )
                status = gr.HTML(status_html("Ready", "idle"))
                history_box = gr.HTML(history_html([]))

                if sample_clips:
                    gr.Markdown("#### Unseen Test-Actor Presets (Actors 20–24)")
                    with gr.Row():
                        sample_buttons = [
                            (gr.Button(s["button_text"], size="sm"), s) for s in sample_clips
                        ]
                else:
                    sample_buttons = []

            with gr.Column(scale=7):
                result = gr.HTML(EMPTY)
                with gr.Tabs():
                    with gr.Tab("Emotion Timeline"):
                        timeline_plot = gr.Plot(label="Sliding-Window Emotion Trajectory")
                    with gr.Tab("Waveform & Spectrogram"):
                        audio_plot = gr.Plot(label="Waveform & Mel-Spectrogram")
                    with gr.Tab("Inside the Model"):
                        layer_plot = gr.Plot(label="Learned WavLM Layer Weights")
                        layer_btn = gr.Button("Load Model Layer Weights", size="sm")

        gr.HTML(FOOT)

        local_btn.click(
            run_local_mic,
            inputs=[hist, threshold],
            outputs=[local_btn, status, result, history_box, hist, audio_in, audio_plot, timeline_plot],
        )
        analyze_btn.click(
            run_uploaded_or_browser,
            inputs=[audio_in, hist, threshold],
            outputs=[status, result, history_box, hist, audio_plot, timeline_plot],
        )
        layer_btn.click(
            lambda: build_layer_weights_figure(),
            inputs=None,
            outputs=layer_plot,
        )

        for btn, sample in sample_buttons:
            btn.click(
                lambda h, th, p=sample["path"], gt=sample["label"]: run_sample_clip(p, gt, h, th),
                inputs=[hist, threshold],
                outputs=[status, result, history_box, hist, audio_in, audio_plot, timeline_plot],
            )

        demo.load(
            lambda: build_layer_weights_figure(),
            inputs=None,
            outputs=layer_plot,
        )

    return demo


demo = create_demo()

if __name__ == "__main__":
    get_predictor()
    demo.launch()
