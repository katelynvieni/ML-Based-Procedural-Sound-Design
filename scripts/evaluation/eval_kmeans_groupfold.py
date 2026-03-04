import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
VALID_CATEGORIES = {"chemical", "electrical", "fire", "space"}
TARGET_LABELS = ["ground", "shock", "roar"]

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "data" / "audio"
INDEX_PATH = REPO_ROOT / "data" / "metadata" / "audio_dataset_index.json"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"

KMEANS_DIR = REPO_ROOT / "data" / "kmeans_results"
CV_PRED_PATH = KMEANS_DIR / "cv_fold_predictions_groupkfold.json"

OUT_DIR = KMEANS_DIR / "eval_min"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Helpers (match your training script behavior)
# =========================
def load_index():
    with open(INDEX_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, dict):
        vals = list(data.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals
    if isinstance(data, list) and (not data or all(isinstance(x, dict) for x in data)):
        return data
    raise ValueError(f"Unrecognized index JSON format: {type(data)}")

def normalize_vectors(X):
    return normalize(X, norm="l2", axis=1)

def parse_segment_pairs(seg_data):
    if not seg_data:
        return []
    pairs = []
    if isinstance(seg_data, list) and seg_data and isinstance(seg_data[0], dict):
        for obj in seg_data:
            s, e = obj.get("start"), obj.get("end")
            if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
                pairs.append((float(s), float(e)))
        return pairs
    if isinstance(seg_data, list) and len(seg_data) == 2 and all(isinstance(x, (int, float)) for x in seg_data):
        s, e = seg_data
        return [(float(s), float(e))] if e > s else []
    if isinstance(seg_data, list):
        for it in seg_data:
            if isinstance(it, list) and len(it) == 2:
                s, e = it
                if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
                    pairs.append((float(s), float(e)))
    return pairs

def get_window_label(ws, we, segments_dict):
    center = (ws + we) / 2.0
    for lab in TARGET_LABELS:
        for s, e in parse_segment_pairs(segments_dict.get(lab)):
            if s <= center <= e:
                return lab
    return None

def get_audio_relpath(item):
    rel = item.get("relative_path") or item.get("relpath") or ""
    if rel:
        return rel
    p = (item.get("path") or "").replace("\\", "/")
    if p and "data/audio/" in p:
        return p.split("data/audio/")[-1]
    return p

def find_embedding_file_from_item(item):
    fn = item.get("filename", "")
    if not fn:
        return None
    wav_name = Path(fn).name
    stem = Path(wav_name).stem
    candidates = [f"{stem}.npy", f"{stem}_emb.npy", f"{wav_name}.npy", f"{wav_name}_emb.npy"]
    for cand in candidates:
        hits = list(EMBEDDINGS_DIR.rglob(cand))
        if hits:
            return hits[0]
    return None

def load_embedding_with_bounds(embedding_path: Path, item: dict):
    emb = np.load(embedding_path).astype(np.float32)
    T = len(emb)
    if T == 0:
        return emb, None

    stem = embedding_path.stem
    candidates = [
        embedding_path.with_name(f"{stem}_bounds.npy"),
        embedding_path.with_name(f"{stem.replace('_emb','')}_bounds.npy"),
    ]
    bounds_path = next((p for p in candidates if p.exists()), None)

    if bounds_path is not None:
        bounds = np.load(bounds_path)
        if bounds.ndim == 2 and bounds.shape[1] == 2 and len(bounds) == T:
            return emb, [(float(s), float(e)) for s, e in bounds]

    # fallback: uniform across duration
    duration = item.get("duration", None)
    if not isinstance(duration, (int, float)) or duration <= 0:
        try:
            rel = get_audio_relpath(item)
            info = sf.info(str(AUDIO_DIR / rel))
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            duration = None
    if not isinstance(duration, (int, float)) or duration <= 0:
        return emb, None

    step = float(duration) / float(T)
    bounds = [(i * step, (i + 1) * step) for i in range(T)]
    return emb, bounds

def rebuild_labeled_windows_X_meta():
    """Rebuild embeddings for labeled windows so we can PCA the *same* windows used in CV."""
    index = load_index()
    X_list, meta_list = [], []
    for item in index:
        if not item.get("labeled", False):
            continue
        segments = item.get("segments")
        if not segments:
            continue
        cat = item.get("category", "")
        if cat not in VALID_CATEGORIES:
            continue
        rel = get_audio_relpath(item)
        if not rel:
            continue
        emb_path = find_embedding_file_from_item(item)
        if emb_path is None:
            continue
        emb, bounds = load_embedding_with_bounds(emb_path, item)
        if bounds is None or len(bounds) != len(emb):
            continue
        for vec, (ws, we) in zip(emb, bounds):
            lab = get_window_label(ws, we, segments)
            if lab in TARGET_LABELS:
                X_list.append(vec)
                meta_list.append({"rel": rel, "window_start": float(ws), "window_end": float(we)})
    if not X_list:
        raise RuntimeError("Could not rebuild labeled windows for PCA scatter.")
    X = np.asarray(X_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)
    meta["key"] = meta["rel"].astype(str) + "||" + meta["window_start"].astype(str) + "||" + meta["window_end"].astype(str)
    return X, meta

# =========================
# Plots
# =========================
def plot_avg_confusion_row_norm(df, out_path):
    """Average confusion across folds (using summed counts), row-normalized."""
    folds = sorted(df["fold"].unique())
    cm_sum = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=float)
    for fold in folds:
        d = df[df["fold"] == fold]
        cm = confusion_matrix(d["true_label"], d["pred_label"], labels=TARGET_LABELS)
        cm_sum += cm

    row_sums = cm_sum.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_sum, row_sums, out=np.zeros_like(cm_sum), where=row_sums != 0)

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm_norm, interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_title("Average Confusion Matrix (Row-Normalized, summed across folds)")
    ax.set_xticks(np.arange(len(TARGET_LABELS)))
    ax.set_yticks(np.arange(len(TARGET_LABELS)))
    ax.set_xticklabels(TARGET_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(TARGET_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_f1_bar_mean_std(df, out_path):
    """Per-class F1 mean ± std across folds."""
    folds = sorted(df["fold"].unique())
    per_fold = []
    for fold in folds:
        d = df[df["fold"] == fold]
        y_true = d["true_label"].to_numpy()
        y_pred = d["pred_label"].to_numpy()
        f1_each = f1_score(y_true, y_pred, labels=TARGET_LABELS, average=None, zero_division=0)
        per_fold.append(f1_each)

    per_fold = np.vstack(per_fold)  # (k, 3)
    mean = per_fold.mean(axis=0)
    std = per_fold.std(axis=0, ddof=1) if len(folds) > 1 else np.zeros_like(mean)

    fig = plt.figure()
    ax = plt.gca()
    x = np.arange(len(TARGET_LABELS))
    ax.bar(x, mean, yerr=std, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS)
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class F1 across folds (mean ± std)")
    ax.set_ylabel("F1 score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_one_scatter_pca(df, out_path):
    """
    One PCA scatter of pooled TEST windows:
    - point color = true_label
    - marker shape = pred_label
    """
    # Rebuild X/meta so we can locate the embeddings for the *test* windows
    X_all, meta_all = rebuild_labeled_windows_X_meta()
    X_all = normalize_vectors(X_all)

    pca = PCA(n_components=2, random_state=0)
    X2_all = pca.fit_transform(X_all)

    key_to_idx = {k: i for i, k in enumerate(meta_all["key"].tolist())}

    d = df.copy()
    d["key"] = (
        d["rel"].astype(str)
        + "||"
        + d["window_start"].astype(str)
        + "||"
        + d["window_end"].astype(str)
    )

    idxs = [key_to_idx.get(k, None) for k in d["key"].tolist()]
    keep = [i for i, ix in enumerate(idxs) if ix is not None]
    if not keep:
        raise RuntimeError("Could not join CV windows to rebuilt embeddings for PCA scatter.")

    d2 = d.iloc[keep].reset_index(drop=True)
    X2 = np.asarray([X2_all[idxs[i]] for i in keep], dtype=np.float32)

    # --- FIX: stable mappings for BOTH legends ---
    color_map = {
        "ground": "#1f77b4",  # blue
        "shock":  "#ff7f0e",  # orange
        "roar":   "#9467bd",  # purple
    }
    marker_map = {
        "ground": "o",
        "shock":  "^",
        "roar":   "s",
    }

    # Map true labels -> actual colors used in scatter
    true_colors = d2["true_label"].map(color_map).to_numpy()

    pred_uniq = TARGET_LABELS[:]  # stable order for marker legend

    fig = plt.figure()
    ax = plt.gca()

    # Plot by predicted label so marker legend is clean/stable
    for pred in pred_uniq:
        mask = (d2["pred_label"].to_numpy() == pred)
        if mask.sum() == 0:
            continue
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            c=true_colors[mask],
            s=18,
            marker=marker_map[pred],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.25,
        )

    ax.set_title("PCA scatter of pooled test windows\nColor=true label, Marker=predicted label")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # --- Legends that actually match the plot ---

    # Marker legend (predicted)
    marker_handles = [
        plt.Line2D([], [], marker=marker_map[p], linestyle="", color="black", label=f"pred={p}")
        for p in pred_uniq
    ]
    leg1 = ax.legend(
        handles=marker_handles,
        title="Predicted label (marker)",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )

    # Color legend (true)
    color_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color_map[t], label=f"true={t}")
        for t in TARGET_LABELS
    ]
    ax.add_artist(leg1)
    ax.legend(
        handles=color_handles,
        title="True label (color)",
        loc="upper right",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# =========================
# MAIN
# =========================
def main():
    if not CV_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing CV predictions: {CV_PRED_PATH}")

    with open(CV_PRED_PATH, "r") as f:
        df = pd.DataFrame(json.load(f))

    needed = {"fold", "true_label", "pred_label", "rel", "window_start", "window_end"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CV predictions: {missing}")

    df["fold"] = df["fold"].astype(int)

    # Save fold-level metrics
    folds = sorted(df["fold"].unique())
    metrics = []
    for fold in folds:
        d = df[df["fold"] == fold]
        y_true = d["true_label"].to_numpy()
        y_pred = d["pred_label"].to_numpy()
        metrics.append({
            "fold": fold,
            "n_test": int(len(d)),
            "accuracy": float((y_true == y_pred).mean()),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=TARGET_LABELS, zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=TARGET_LABELS, zero_division=0)),
        })
    pd.DataFrame(metrics).to_csv(OUT_DIR / "metrics_by_fold.csv", index=False)

    # ---- THE THREE PLOTS ----
    plot_avg_confusion_row_norm(df, OUT_DIR / "avg_confusion_row_norm.png")
    plot_f1_bar_mean_std(df, OUT_DIR / "f1_per_class_mean_std.png")
    plot_one_scatter_pca(df, OUT_DIR / "pca_scatter_truecolor_predmarker.png")

    print(f"Saved minimal plots + metrics to: {OUT_DIR}")

if __name__ == "__main__":
    main()
