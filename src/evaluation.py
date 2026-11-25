"""Evaluate the trained WeightedSER checkpoint, calibrate confidence, and export plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from training import CKPT, load, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over top-class confidences."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accuracies = (preds == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(targets)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def fit_temperature(val_logits: np.ndarray, val_targets: np.ndarray) -> float:
    """Fit scalar temperature on validation logits to minimize negative log-likelihood."""
    logits_t = torch.from_numpy(val_logits).float()
    targets_t = torch.from_numpy(val_targets).long()
    log_temp = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=0.1, max=10.0)
        loss = criterion(logits_t / temp, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).clamp(min=0.1, max=10.0).item())


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Test Confusion Matrix",
) -> None:
    """Save a publication-grade confusion matrix with row percentages and counts."""
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=180)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall per Emotion", fontsize=10)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Emotion", fontsize=11, labelpad=8)
    ax.set_ylabel("True Emotion", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100.0
            count = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.52 else "#0f172a"
            ax.text(
                j,
                i,
                f"{pct:.0f}%\n({count})",
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
                fontweight="bold" if i == j else "normal",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_layer_weights(weights: np.ndarray, out_path: Path) -> None:
    """Save a bar chart showing learned importance across all WavLM-Large layers."""
    layers = np.arange(len(weights))
    peak_idx = int(np.argmax(weights))
    colors = ["#ec4899" if i == peak_idx else ("#6366f1" if w >= np.median(weights) else "#94a3b8") for i, w in enumerate(weights)]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=180)
    ax.bar(layers, weights * 100.0, color=colors, edgecolor="none", width=0.72)
    ax.axhline(100.0 / len(weights), color="#475569", linestyle="--", linewidth=1.0, label="Uniform weight (4.0%)")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=8)
    ax.set_ylabel("Learned Weight (%)", fontsize=10.5)
    ax.set_xlabel("WavLM-Large Layer (L0 = CNN input, L1–L24 = Transformer)", fontsize=10.5)
    ax.set_title("Learned Layer Importance Across Frozen WavLM-Large", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_evaluation(ckpt_path: Path = CKPT, out_dir: Path = RESULTS) -> dict:
    """Evaluate checkpoint on val/test embeddings, fit calibration, and save metrics/plots."""
    model, ckpt = load_checkpoint(ckpt_path=ckpt_path, map_location="cpu")
    ckpt_labels = list(ckpt["labels"])

    X_val, y_val = load("val")
    X_test, y_test = load("test")
    X_val_n = (X_val - ckpt["mean"]) / ckpt["std"]
    X_test_n = (X_test - ckpt["mean"]) / ckpt["std"]

    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_val_n)).numpy()
        test_logits = model(torch.from_numpy(X_test_n)).numpy()

    temperature = fit_temperature(val_logits, y_val)
    raw_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    cal_probs = torch.softmax(torch.from_numpy(test_logits / temperature), dim=1).numpy()
    pred = test_logits.argmax(axis=1)

    acc = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, pred, average="weighted"))
    raw_ece = expected_calibration_error(raw_probs, y_test)
    cal_ece = expected_calibration_error(cal_probs, y_test)
    report_dict = classification_report(
        y_test, pred, target_names=ckpt_labels, output_dict=True, zero_division=0
    )

    print(f"Accuracy {acc:.3f} | Macro-F1 {macro_f1:.3f} | Weighted-F1 {weighted_f1:.3f}")
    print(f"Calibration: T={temperature:.3f} | Raw ECE={raw_ece:.4f} -> Calibrated ECE={cal_ece:.4f}")
    print(classification_report(y_test, pred, target_names=ckpt_labels, zero_division=0))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "val_f1_checkpoint": float(ckpt.get("val_f1", 0.0)),
        "temperature": temperature,
        "raw_ece": raw_ece,
        "calibrated_ece": cal_ece,
        "per_class": {
            name: {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }
            for name in ckpt_labels
        },
        "layer_weights": model.layer_weights().detach().cpu().numpy().round(5).tolist(),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, pred, labels=list(range(len(ckpt_labels))))
    plot_normalized_confusion_matrix(cm, ckpt_labels, out_dir / "confusion_matrix.png")
    plot_layer_weights(model.layer_weights().detach().cpu().numpy(), out_dir / "layer_weights.png")
    return metrics


if __name__ == "__main__":
    run_evaluation()
