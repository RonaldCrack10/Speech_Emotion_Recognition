"""Conference experiments: openSMILE eGeMAPS baseline, WavLM ablations, fusion, and 5-fold speaker CV."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opensmile
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from data import build_metadata, class_weights, labels, own_path, ravdess_path, split, tess_path
from training import EMB, WeightedSER, make_loader, train_weighted_ser

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_egemaps_split(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract and cache 88-dim eGeMAPSv02 acoustic functionals for a dataset split."""
    cache_path = EMB / f"egemaps_{name}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return d["X"].astype(np.float32), d["y"].astype(np.int64)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feats = []
    for path in tqdm(df["path"], desc=f"eGeMAPS ({name})"):
        vec = smile.process_file(path).to_numpy(dtype=np.float32).squeeze(0)
        feats.append(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0))

    X = np.stack(feats).astype(np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    EMB.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        X=X,
        y=y,
        speaker=df["speaker"].to_numpy().astype(str),
        corpus=df["corpus"].to_numpy().astype(str),
    )
    return X, y


def evaluate_egemaps_baselines() -> dict[str, dict]:
    """Evaluate classical acoustic baselines (eGeMAPS + SVM / Logistic Regression)."""
    df = build_metadata(ravdess_path, tess_path, own_path)
    train_df, _, test_df = split(df)
    X_train, y_train = extract_egemaps_split(train_df, "train")
    X_test, y_test = extract_egemaps_split(test_df, "test")

    results = {}
    models = {
        "eGeMAPS + LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=42),
        ),
        "eGeMAPS + RBF SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=2.0, class_weight="balanced", random_state=42),
        ),
    }
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        }
    return results


def _train_fixed_pool_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    """Train a 2-layer MLP on fixed 2D embeddings (B, D) with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_tr, X_va, X_te = (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std

    train_loader = make_loader(X_tr, y_train, shuffle=True)
    val_loader = make_loader(X_va, y_val, shuffle=False)

    head = nn.Sequential(
        nn.Linear(X_train.shape[1], 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(labels)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(pd.DataFrame({"label": y_train})).to(device)
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

    best_f1, wait, best_state = 0.0, 0, None
    for _ in range(60):
        head.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(head(xb), yb).backward()
            optimizer.step()

        head.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                vp.extend(head(xb.to(device)).argmax(dim=1).cpu().tolist())
                vt.extend(yb.tolist())
        vf1 = f1_score(vt, vp, average="macro")
        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
            if wait >= 10:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(torch.from_numpy(X_te).to(device)).argmax(dim=1).cpu().numpy()
    return float(accuracy_score(y_test, pred)), float(f1_score(y_test, pred, average="macro"))


class FusionSER(nn.Module):
    """Jointly learns WavLM layer weights and fuses them with standardized eGeMAPS features."""

    def __init__(self, n_layers: int = 25, hidden: int = 1024, acoustic_dim: int = 88, n_classes: int = 7):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.acoustic_proj = nn.Sequential(nn.Linear(acoustic_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x_ssl: torch.Tensor, x_ac: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.layer_w, dim=0)
        pooled = (x_ssl * w[None, :, None]).sum(dim=1)
        ac = self.acoustic_proj(x_ac)
        return self.head(torch.cat([pooled, ac], dim=1))


def evaluate_fusion(
    X_tr_ssl: np.ndarray,
    X_tr_ac: np.ndarray,
    y_train: np.ndarray,
    X_va_ssl: np.ndarray,
    X_va_ac: np.ndarray,
    y_val: np.ndarray,
    X_te_ssl: np.ndarray,
    X_te_ac: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    """Train and evaluate WavLM + eGeMAPS fusion on the speaker-independent split."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    m_s, s_s = X_tr_ssl.mean(0, keepdims=True), X_tr_ssl.std(0, keepdims=True) + 1e-6
    m_a, s_a = X_tr_ac.mean(0, keepdims=True), X_tr_ac.std(0, keepdims=True) + 1e-6

    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy((X_tr_ssl - m_s) / s_s),
        torch.from_numpy((X_tr_ac - m_a) / s_a),
        torch.from_numpy(y_train).long(),
    )
    va_ds = torch.utils.data.TensorDataset(
        torch.from_numpy((X_va_ssl - m_s) / s_s),
        torch.from_numpy((X_va_ac - m_a) / s_a),
        torch.from_numpy(y_val).long(),
    )
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=64, shuffle=False)

    model = FusionSER(
        n_layers=X_tr_ssl.shape[1], hidden=X_tr_ssl.shape[2], acoustic_dim=X_tr_ac.shape[1]
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(pd.DataFrame({"label": y_train})).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_f1, wait, best_state = 0.0, 0, None
    for _ in range(60):
        model.train()
        for xs, xa, yb in tr_loader:
            xs, xa, yb = xs.to(device), xa.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xs, xa), yb).backward()
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xs, xa, yb in va_loader:
                vp.extend(model(xs.to(device), xa.to(device)).argmax(dim=1).cpu().tolist())
                vt.extend(yb.tolist())
        vf1 = f1_score(vt, vp, average="macro")
        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = (
            model(
                torch.from_numpy((X_te_ssl - m_s) / s_s).to(device),
                torch.from_numpy((X_te_ac - m_a) / s_a).to(device),
            )
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    return float(accuracy_score(y_test, pred)), float(f1_score(y_test, pred, average="macro"))


def run_5fold_speaker_cv(seeds_per_fold: int = 1) -> dict:
    """Run 5-fold speaker-independent CV across all 24 RAVDESS actors (+ TESS in train)."""
    parts = [np.load(EMB / f"emb_{s}.npz") for s in ("train", "val", "test")]
    X_all = np.concatenate([p["X"].astype(np.float32) for p in parts])
    y_all = np.concatenate([p["y"].astype(np.int64) for p in parts])
    spk_all = np.concatenate([p["speaker"] for p in parts])
    corp_all = np.concatenate([p["corpus"] for p in parts])

    rav_speakers = sorted({s for s, c in zip(spk_all, corp_all) if c == "ravdess"})
    folds = np.array_split(rav_speakers, 5)
    tess_mask = corp_all == "tess"

    fold_metrics = []
    for fold_idx in range(5):
        test_spk = set(folds[fold_idx])
        val_spk = set(folds[(fold_idx + 1) % 5])

        test_mask = np.isin(spk_all, list(test_spk))
        val_mask = np.isin(spk_all, list(val_spk))
        train_mask = tess_mask | ((corp_all == "ravdess") & ~test_mask & ~val_mask)

        accs, f1s = [], []
        for s in range(seeds_per_fold):
            model, mean, std, _ = train_weighted_ser(
                X_all[train_mask],
                y_all[train_mask],
                X_all[val_mask],
                y_all[val_mask],
                seed=100 + fold_idx * 10 + s,
                save_path=None,
            )
            model.eval()
            X_te_n = (X_all[test_mask] - mean) / std
            with torch.no_grad():
                pred = model(torch.from_numpy(X_te_n).to(device)).argmax(dim=1).cpu().numpy()
            accs.append(float(accuracy_score(y_all[test_mask], pred)))
            f1s.append(float(f1_score(y_all[test_mask], pred, average="macro")))

        fold_metrics.append(
            {
                "fold": fold_idx + 1,
                "test_speakers": sorted(test_spk),
                "n_test": int(test_mask.sum()),
                "accuracy": float(np.mean(accs)),
                "macro_f1": float(np.mean(f1s)),
            }
        )

    all_acc = [m["accuracy"] for m in fold_metrics]
    all_f1 = [m["macro_f1"] for m in fold_metrics]
    return {
        "folds": fold_metrics,
        "mean_accuracy": float(np.mean(all_acc)),
        "std_accuracy": float(np.std(all_acc)),
        "mean_macro_f1": float(np.mean(all_f1)),
        "std_macro_f1": float(np.std(all_f1)),
    }


def plot_ablation_comparison(ablation: dict, out_path: Path) -> None:
    """Save a conference-ready horizontal bar chart comparing baselines and ablations."""
    names = list(ablation.keys())
    means = [ablation[k]["macro_f1_mean"] * 100.0 for k in names]
    stds = [ablation[k].get("macro_f1_std", 0.0) * 100.0 for k in names]

    palette = []
    for k in names:
        if "WeightedSER" in k and "Fusion" not in k:
            palette.append("#6366f1")
        elif "Fusion" in k:
            palette.append("#ec4899")
        elif "WavLM" in k:
            palette.append("#3b82f6")
        else:
            palette.append("#64748b")

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=180)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, means, xerr=stds, color=palette, height=0.58, capsize=4, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test Macro-F1 (%)", fontsize=10.5)
    ax.set_title("Model Ablation & Baseline Comparison (Speaker-Independent Test)", fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    for bar, m, s in zip(bars, means, stds):
        label = f"{m:.1f}%" if s == 0 else f"{m:.1f}% ± {s:.1f}%"
        ax.text(m + s + 1.5, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_all_experiments(seeds: tuple[int, ...] = (42, 43, 44)) -> dict:
    """Run eGeMAPS baselines, WavLM ablations, fusion, and 5-fold CV, saving results & plots."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. Classical eGeMAPS baselines
    egemaps_res = evaluate_egemaps_baselines()
    ablation = {
        k: {
            "accuracy_mean": v["accuracy"],
            "accuracy_std": 0.0,
            "macro_f1_mean": v["macro_f1"],
            "macro_f1_std": 0.0,
        }
        for k, v in egemaps_res.items()
    }

    # 2. Cached WavLM embeddings + eGeMAPS embeddings
    tr = np.load(EMB / "emb_train.npz")
    va = np.load(EMB / "emb_val.npz")
    te = np.load(EMB / "emb_test.npz")
    X_tr, y_tr = tr["X"].astype(np.float32), tr["y"].astype(np.int64)
    X_va, y_va = va["X"].astype(np.float32), va["y"].astype(np.int64)
    X_te, y_te = te["X"].astype(np.float32), te["y"].astype(np.int64)

    eg_tr = np.load(EMB / "egemaps_train.npz")["X"].astype(np.float32)
    eg_va = np.load(EMB / "egemaps_val.npz")["X"].astype(np.float32) if (EMB / "egemaps_val.npz").exists() else None
    if eg_va is None:
        _, val_df, _ = split(build_metadata(ravdess_path, tess_path, own_path))
        eg_va, _ = extract_egemaps_split(val_df, "val")
    eg_te = np.load(EMB / "egemaps_test.npz")["X"].astype(np.float32)

    # 2a. WavLM Last Layer Only
    last_accs, last_f1s = [], []
    for s in seeds:
        acc, f1 = _train_fixed_pool_mlp(X_tr[:, -1, :], y_tr, X_va[:, -1, :], y_va, X_te[:, -1, :], y_te, seed=s)
        last_accs.append(acc)
        last_f1s.append(f1)
    ablation["WavLM-Large (Last Layer)"] = {
        "accuracy_mean": float(np.mean(last_accs)),
        "accuracy_std": float(np.std(last_accs)),
        "macro_f1_mean": float(np.mean(last_f1s)),
        "macro_f1_std": float(np.std(last_f1s)),
    }

    # 2b. WavLM Uniform Layer Average
    uni_accs, uni_f1s = [], []
    for s in seeds:
        acc, f1 = _train_fixed_pool_mlp(X_tr.mean(1), y_tr, X_va.mean(1), y_va, X_te.mean(1), y_te, seed=s)
        uni_accs.append(acc)
        uni_f1s.append(f1)
    ablation["WavLM-Large (Uniform Avg)"] = {
        "accuracy_mean": float(np.mean(uni_accs)),
        "accuracy_std": float(np.std(uni_accs)),
        "macro_f1_mean": float(np.mean(uni_f1s)),
        "macro_f1_std": float(np.std(uni_f1s)),
    }

    # 2c. WeightedSER (Learned Layer Weights)
    w_accs, w_f1s = [], []
    for s in seeds:
        model, mean, std, _ = train_weighted_ser(X_tr, y_tr, X_va, y_va, seed=s, save_path=None)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy((X_te - mean) / std).to(device)).argmax(dim=1).cpu().numpy()
        w_accs.append(float(accuracy_score(y_te, pred)))
        w_f1s.append(float(f1_score(y_te, pred, average="macro")))
    ablation["WavLM-Large (WeightedSER)"] = {
        "accuracy_mean": float(np.mean(w_accs)),
        "accuracy_std": float(np.std(w_accs)),
        "macro_f1_mean": float(np.mean(w_f1s)),
        "macro_f1_std": float(np.std(w_f1s)),
    }

    # 2d. Multimodal Fusion (WeightedSER + eGeMAPS)
    f_accs, f_f1s = [], []
    for s in seeds:
        acc, f1 = evaluate_fusion(X_tr, eg_tr, y_tr, X_va, eg_va, y_va, X_te, eg_te, y_te, seed=s)
        f_accs.append(acc)
        f_f1s.append(f1)
    ablation["WavLM WeightedSER + eGeMAPS"] = {
        "accuracy_mean": float(np.mean(f_accs)),
        "accuracy_std": float(np.std(f_accs)),
        "macro_f1_mean": float(np.mean(f_f1s)),
        "macro_f1_std": float(np.std(f_f1s)),
    }

    (RESULTS / "ablation_results.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    plot_ablation_comparison(ablation, RESULTS / "ablation_comparison.png")

    # 3. 5-fold speaker-independent CV
    cv_res = run_5fold_speaker_cv(seeds_per_fold=1)
    (RESULTS / "cv_5fold_results.json").write_text(json.dumps(cv_res, indent=2), encoding="utf-8")

    return {"ablation": ablation, "cv_5fold": cv_res}


if __name__ == "__main__":
    summary = run_all_experiments()
    print("\n=== Ablation Summary (Test Set) ===")
    for k, v in summary["ablation"].items():
        print(
            f"{k:32s} | Acc: {v['accuracy_mean']:.3f} ± {v['accuracy_std']:.3f} | "
            f"Macro-F1: {v['macro_f1_mean']:.3f} ± {v['macro_f1_std']:.3f}"
        )
    cv = summary["cv_5fold"]
    print(
        f"\n=== 5-Fold Speaker-Independent CV (24 RAVDESS Actors) ===\n"
        f"Mean Accuracy: {cv['mean_accuracy']:.3f} ± {cv['std_accuracy']:.3f} | "
        f"Mean Macro-F1: {cv['mean_macro_f1']:.3f} ± {cv['std_macro_f1']:.3f}"
    )
