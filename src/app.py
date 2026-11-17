import numpy as np
import gradio as gr

from live import Prediction, record

predictor = Prediction()

COLOR = {"angry": "#ef4444", "disgust": "#84cc16", "fear": "#a855f7", "happy": "#facc15",
         "neutral": "#94a3b8", "sad": "#3b82f6", "surprise": "#f97316"}
THRESHOLD = 0.5          # below this confidence the app shows "Not sure"


# ---------- analysis ----------
def analyze(y):
    """Check the loudness (RMS) of the recording, then classify it.
    Returns ({emotion: probability} or None, status message)."""
    rms = float(np.sqrt((y ** 2).mean()))
    if rms < 0.005:
        return None, f"Too quiet (level {rms:.4f}). Please move closer to the microphone."
    probs = predictor.predict(y)
    return {k: float(v) for k, v in probs.items()}, f"Level {rms:.3f}"


# ---------- HTML builders ----------
def status_html(text, kind="idle"):
    return f'<div class="pill {kind}"><span class="dot"></span>{text}</div>'


EMPTY = ('<div class="card result"><div class="orb" style="--c:#64748b"><span>&ndash;</span></div>'
         '<div class="title" style="color:#c7c7d9">Ready when you are</div>'
         '<div class="sub">Press the button and say a sentence with emotion</div></div>')


def result_html(probs):
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    top, p = items[0]
    if p < THRESHOLD:
        inner, title, color = "?", "Not sure", "#94a3b8"
        sub = f"best guess: {top} ({p:.0%})"
    else:
        inner, title, color = f"{p:.0%}", top.capitalize(), COLOR[top]
        sub = "confidence"
    bars = "".join(
        f'<div class="row"><span class="nm"><i class="sw" style="background:{COLOR[k]}"></i>{k}</span>'
        f'<div class="track"><div class="fill" style="width:{v * 100:.1f}%;background:{COLOR[k]}"></div></div>'
        f'<span class="pc">{v:.0%}</span></div>'
        for k, v in items)
    return (f'<div class="card result"><div class="orb" style="--c:{color}"><span>{inner}</span></div>'
            f'<div class="title" style="color:{color}">{title}</div>'
            f'<div class="sub">{sub}</div><div class="bars">{bars}</div></div>')


def history_html(history):
    if not history:
        return '<div class="card hist"><div class="hlabel">Recent attempts</div><div class="sub">none yet</div></div>'
    chips = "".join(
        f'<span class="chip"><i class="sw" style="background:{COLOR[e]}"></i>{e} {p:.0%}</span>'
        for e, p in reversed(history))
    return f'<div class="card hist"><div class="hlabel">Recent attempts</div>{chips}</div>'


# ---------- button callback ----------
def run(history):
    yield (gr.update(interactive=False, value="Recording ..."),
           status_html("Speak now (4 seconds)", "rec"), EMPTY, history_html(history), history)
    y = record(4)
    probs, msg = analyze(y)
    if probs is None:
        yield (gr.update(interactive=True, value="Try again"),
               status_html(msg, "warn"), EMPTY, history_html(history), history)
        return
    top = max(probs, key=probs.get)
    history = (history + [(top, probs[top])])[-6:]
    yield (gr.update(interactive=True, value="Record again"),
           status_html(msg, "ok"), result_html(probs), history_html(history), history)


# ---------- styling ----------
CSS = """
.gradio-container {
  background: radial-gradient(1100px 600px at 8% -10%, #312e81 0%, transparent 60%),
              radial-gradient(900px 500px at 100% 0%, #831843 0%, transparent 55%), #0b0b14 !important;
}
footer {display: none !important;}
.hero {text-align: center; padding: 30px 10px 6px;}
.hero h1 {font-size: 2.7rem; font-weight: 800; margin: 0; color: transparent;
  background: linear-gradient(90deg, #818cf8, #f472b6, #fbbf24); -webkit-background-clip: text; background-clip: text;}
.hero p {color: #a1a1b5; margin: 10px 0 0; font-size: 1.05rem;}
.card {background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10);
  border-radius: 20px; padding: 22px; color: #e5e7f0; backdrop-filter: blur(8px);}
.result {text-align: center;}
.orb {width: 150px; height: 150px; margin: 6px auto 14px; border-radius: 50%;
  border: 5px solid var(--c); display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 50px -6px var(--c), inset 0 0 30px -10px var(--c);
  background: radial-gradient(circle, rgba(255,255,255,.07), transparent 70%);}
.orb span {font-size: 2.4rem; font-weight: 800; color: #fff;}
.title {font-size: 2rem; font-weight: 800; margin-top: 4px;}
.sub {color: #9a9ab0; margin-bottom: 10px;}
.bars {margin-top: 14px; text-align: left;}
.row {display: grid; grid-template-columns: 120px 1fr 48px; align-items: center; gap: 10px; margin: 8px 0;}
.nm {color: #cfd0e0; font-size: .95rem; display: flex; align-items: center; gap: 8px;}
.sw {display: inline-block; width: 10px; height: 10px; border-radius: 50%;}
.pc {color: #9a9ab0; text-align: right; font-variant-numeric: tabular-nums;}
.track {height: 12px; background: rgba(255,255,255,.08); border-radius: 99px; overflow: hidden;}
.fill {height: 100%; border-radius: 99px; transition: width .6s ease;}
.rec-btn {background: linear-gradient(90deg, #6366f1, #ec4899) !important; color: #fff !important; border: none !important;
  font-size: 1.15rem !important; font-weight: 700 !important; padding: 18px !important; border-radius: 16px !important;
  box-shadow: 0 8px 30px rgba(236,72,153,.35);}
.rec-btn:hover {filter: brightness(1.1);}
.rec-btn:disabled {opacity: .6;}
.pill {display: inline-flex; align-items: center; gap: 10px; padding: 10px 16px; border-radius: 99px;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); color: #d6d7e6; margin-top: 12px;}
.dot {width: 10px; height: 10px; border-radius: 50%; background: #64748b;}
.pill.ok .dot {background: #22c55e;}
.pill.warn .dot {background: #f59e0b;}
.pill.rec .dot {background: #ef4444; animation: pulse 1s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(239,68,68,.7);} 100% {box-shadow: 0 0 0 14px rgba(239,68,68,0);}}
.hist {margin-top: 14px;}
.hlabel {color: #9a9ab0; font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px;}
.chip {display: inline-flex; align-items: center; gap: 8px; margin: 4px 6px 4px 0; padding: 6px 12px; border-radius: 99px;
  border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04); color: #e5e7f0; font-size: .9rem;}
.how {color: #b8b9cc; line-height: 1.6;}
.how b {color: #fff;}
.foot {text-align: center; color: #6b6b85; font-size: .85rem; padding: 18px 0 6px;}
"""

HERO = ('<div class="hero"><h1>Speech Emotion Recognition</h1>'
        '<p>Act out an emotion. The model tries to guess it from your voice.</p></div>')
HOWTO = ('<div class="card how"><b>How it works</b><br>1. Press the button<br>'
         '2. Say a sentence with <b>clear emotion</b> within 4 seconds, e.g. <i>"Kids are talking by the door"</i><br>'
         '3. See what the model hears</div>')
FOOT = '<div class="foot">WavLM-large &middot; learned layer weighting &middot; trained on RAVDESS &amp; TESS</div>'


# ---------- layout ----------
with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.HTML(f"<style>{CSS}</style>")
    gr.HTML(HERO)
    hist = gr.State([])
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            gr.HTML(HOWTO)
            btn = gr.Button("Record 4 seconds", elem_classes="rec-btn")
            status = gr.HTML(status_html("Ready", "idle"))
            history_box = gr.HTML(history_html([]))
        with gr.Column(scale=6):
            result = gr.HTML(EMPTY)
    gr.HTML(FOOT)

    btn.click(run, inputs=hist, outputs=[btn, status, result, history_box, hist])

if __name__ == "__main__":
    demo.launch()