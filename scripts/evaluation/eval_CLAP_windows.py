#!/usr/bin/env python3
"""
Evaluate ONLY 512-D CLAP embeddings (4 equal windows per original audio file).

Inputs:
- data/metadata/audio_dataset_index.json
- data/embeddings/<category>/<stem>_emb.npy      shape: (4, D) where D=512 typically
- data/embeddings/<category>/<stem>_meta.json    list of 4 dicts with {"start","end"} in seconds

Labels:
- Uses item["segments"] with ground/shock/roar time intervals (seconds).
- Each window is assigned the label with max overlap; if overlap is tiny, midpoint fallback.
- Optionally drop windows that overlap no labeled segment.

Evaluation:
- 1-Nearest-Neighbor (cosine) leave-one-out accuracy
- Confusion matrix
- 2D projection (PCA->2D) for visualization ONLY
- Nearest-neighbor cosine distance distributions

Outputs (default):
- data/eval_clap512_windows/metrics.json
- data/eval_clap512_windows/samples.jsonl
- data/eval_clap512_windows/*.png
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


VALID_LABELS = ("ground", "shock", "roar")
LABEL_TO_ID = {lab: i for i, lab in enumerate(VALID_LABELS)}


# -----------------------------
# Index + path helpers
# -----------------------------
def load_index_items(index_path: Path) -> List[Dict[str, Any]]:
    with index_path.open("r") as f:
        root = json.load(f)

    if isinstance(root, dict):
        items = root.get("data", root.get("items", []))
    elif isinstance(root, list):
        items = root
    else:
        items = []

    if not isinstance(items, list):
        items = []

    return [it for it in items if isinstance(it, dict)]


def resolve_audio_path(repo_root: Path, assets_dir: Path, item: Dict[str, Any]) -> Optional[Path]:
    candidates: List[str] = []
    for key in ("path", "relative_path", "filepath", "file", "audio_path", "filename"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    for p in candidates:
        pp = Path(p)
        if pp.is_absolute() and pp.exists():
            return pp

        pr = (repo_root / pp).resolve()
        if pr.exists():
            return pr

        pa = (assets_dir / pp).resolve()
        if pa.exists():
            return pa

    return None


def get_category(item: Dict[str, Any], audio_path: Optional[Path], assets_dir: Path) -> str:
    c = item.get("category")
    if isinstance(c, str) and c.strip():
        return c.strip().lower()

    if audio_path is not None:
        try:
            rel = audio_path.resolve().relative_to(assets_dir.resolve())
            if rel.parts:
                return rel.parts[0].lower()
        except Exception:
            pass

    return "other"


# -----------------------------
# Embedding loading
# -----------------------------
def load_embedding_pair(emb_root: Path, category: str, stem: str) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    emb_path = emb_root / category / f"{stem}_emb.npy"
    meta_path = emb_root / category / f"{stem}_meta.json"
    if not emb_path.exists() or not meta_path.exists():
        return None, None

    emb = np.load(str(emb_path))  # expected (4, D)
    with meta_path.open("r") as f:
        meta = json.load(f)

    if not isinstance(meta, list) or len(meta) != emb.shape[0]:
        return None, None

    # Ensure emb is 2D (N,D)
    if emb.ndim == 1:
        emb = emb[None, :]
    if emb.ndim != 2:
        return None, None

    return emb.astype(np.float32), meta


# -----------------------------
# Label assignment
# -----------------------------
def normalize_segments(segments_obj: Any) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {k: [] for k in VALID_LABELS}
    if not isinstance(segments_obj, dict):
        return out

    for lab in VALID_LABELS:
        segs = segments_obj.get(lab, [])
        if not isinstance(segs, list):
            continue
        for s in segs:
            if not isinstance(s, dict):
                continue
            st = s.get("start")
            en = s.get("end")
            if isinstance(st, (int, float)) and isinstance(en, (int, float)) and en > st:
                out[lab].append((float(st), float(en)))
    return out


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0.0, right - left)


def total_overlap_any_label(w0: float, w1: float, segs: Dict[str, List[Tuple[float, float]]]) -> float:
    tot = 0.0
    for lab in VALID_LABELS:
        for s0, s1 in segs.get(lab, []):
            tot += interval_overlap(w0, w1, s0, s1)
    return tot


def assign_window_label(
    w0: float,
    w1: float,
    segs: Dict[str, List[Tuple[float, float]]],
    min_overlap_ratio: float,
) -> Optional[str]:
    """
    Assign label by max overlap; if overlap ratio is too small, use midpoint fallback.
    """
    wlen = max(1e-9, w1 - w0)

    best_lab = None
    best_ov = 0.0
    for lab, intervals in segs.items():
        for s0, s1 in intervals:
            ov = interval_overlap(w0, w1, s0, s1)
            if ov > best_ov:
                best_ov = ov
                best_lab = lab

    if best_lab is not None and (best_ov / wlen) >= min_overlap_ratio:
        return best_lab

    mid = 0.5 * (w0 + w1)
    for lab, intervals in segs.items():
        for s0, s1 in intervals:
            if s0 <= mid <= s1:
                return lab

    return None


# -----------------------------
# Build dataset of labeled window embeddings
# -----------------------------
def build_labeled_windows(
    repo_root: Path,
    index_path: Path,
    assets_dir: Path,
    emb_root: Path,
    strict_coverage: bool,
    min_overlap_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    items = load_index_items(index_path)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    rows: List[Dict[str, Any]] = []

    dropped_no_segments = 0
    dropped_unlabeled_window = 0
    missing_embedding_pairs = 0

    for item in items:
        if not bool(item.get("labeled", False)):
            continue

        segments = item.get("segments")
        if not isinstance(segments, dict):
            dropped_no_segments += 1
            continue

        audio_path = resolve_audio_path(repo_root, assets_dir, item)
        if audio_path is None:
            continue

        cat = get_category(item, audio_path, assets_dir)
        stem = audio_path.stem

        emb, meta = load_embedding_pair(emb_root, cat, stem)
        if emb is None or meta is None:
            missing_embedding_pairs += 1
            continue

        segs = normalize_segments(segments)

        for wi in range(emb.shape[0]):
            w0 = float(meta[wi].get("start", math.nan))
            w1 = float(meta[wi].get("end", math.nan))
            if not (np.isfinite(w0) and np.isfinite(w1) and w1 > w0):
                dropped_unlabeled_window += 1
                continue

            if strict_coverage and total_overlap_any_label(w0, w1, segs) <= 0.0:
                dropped_unlabeled_window += 1
                continue

            lab = assign_window_label(w0, w1, segs, min_overlap_ratio=min_overlap_ratio)
            if lab is None:
                dropped_unlabeled_window += 1
                continue

            X_list.append(emb[wi])
            y_list.append(LABEL_TO_ID[lab])
            rows.append(
                {
                    "id": item.get("id"),
                    "stem": stem,
                    "category": cat,
                    "window_index": wi,
                    "window_start_s": w0,
                    "window_end_s": w1,
                    "y_true": lab,
                }
            )

    if not X_list:
        raise SystemExit(
            "No labeled window samples built.\n"
            "Check that your index items have: labeled=true AND segments{...}\n"
            "and embeddings exist at: data/embeddings/<category>/<stem>_emb.npy + _meta.json"
        )

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Built N={len(y)} window samples, D={X.shape[1]}")
    print(f"Missing embedding/meta pairs: {missing_embedding_pairs}")
    print(f"Dropped files missing segments: {dropped_no_segments}")
    print(f"Dropped unlabeled windows: {dropped_unlabeled_window}")
    print("Label counts:", {lab: int((y == i).sum()) for i, lab in enumerate(VALID_LABELS)})

    return X, y, rows


# -----------------------------
# 1-NN cosine LOO
# -----------------------------
def nn_leave_one_out_cosine(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X)

    dists, idxs = nn.kneighbors(X, return_distance=True)
    nn_dist = dists[:, 1]
    nn_idx = idxs[:, 1]

    y_pred = y[nn_idx]
    correct = (y_pred == y)

    per_label = {}
    for lab_id, lab in enumerate(VALID_LABELS):
        m = (y == lab_id)
        per_label[lab] = float((y_pred[m] == y[m]).mean()) if m.any() else None

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": float(correct.mean()),
        "per_label_accuracy": per_label,
        "confusion_matrix": cm.tolist(),
        "nn_distance": nn_dist,
        "y_pred": y_pred,
        "correct": correct,
    }


# -----------------------------
# Plots
# -----------------------------
def plot_2d(out_path: Path, coords: np.ndarray, y: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 6))
    for lab_id, lab in enumerate(VALID_LABELS):
        m = (y == lab_id)
        if m.any():
            plt.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.8, label=lab)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist(out_path: Path, data: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=40)
    plt.title(title)
    plt.xlabel("cosine distance to 1-NN")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist_groups(out_path: Path, data: np.ndarray, groups: List[Tuple[str, np.ndarray]], title: str) -> None:
    plt.figure(figsize=(8, 5))
    for name, mask in groups:
        if mask.any():
            plt.hist(data[mask], bins=40, alpha=0.6, label=name)
    plt.title(title)
    plt.xlabel("cosine distance to 1-NN")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(out_path: Path, cm: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks([0, 1, 2], VALID_LABELS)
    plt.yticks([0, 1, 2], VALID_LABELS)
    plt.xlabel("predicted")
    plt.ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_accuracy_summary(
    out_path: Path,
    overall_acc: float,
    per_label_acc: Dict[str, Optional[float]],
    title: str,
) -> None:
    labels = ["overall"] + list(VALID_LABELS)
    values = [overall_acc] + [per_label_acc.get(lab) for lab in VALID_LABELS]
    values = [float(v) if v is not None else 0.0 for v in values]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("accuracy")
    plt.title(title)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=None, help="Repo root (optional).")
    ap.add_argument("--index", type=str, default="data/metadata/audio_dataset_index.json")
    ap.add_argument("--assets_dir", type=str, default="data/audio")
    ap.add_argument("--emb_root", type=str, default="data/embeddings")
    ap.add_argument("--out_dir", type=str, default="data/eval_clap512_windows")

    ap.add_argument("--strict_coverage", action="store_true",
                    help="Drop windows that overlap no labeled segment intervals (recommended).")
    ap.add_argument("--min_overlap_ratio", type=float, default=0.10,
                    help="Min overlap fraction before midpoint fallback (seconds-based).")
    ap.add_argument("--make_tsne", action="store_true",
                    help="Also create a t-SNE plot (slow, optional).")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else script_path.parents[1]

    index_path = (repo_root / args.index).resolve()
    assets_dir = (repo_root / args.assets_dir).resolve()
    emb_root = (repo_root / args.emb_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Repo root:", repo_root)
    print("Index:", index_path)
    print("Embeddings root:", emb_root)
    print("Output:", out_dir)

    X, y, rows = build_labeled_windows(
        repo_root=repo_root,
        index_path=index_path,
        assets_dir=assets_dir,
        emb_root=emb_root,
        strict_coverage=args.strict_coverage,
        min_overlap_ratio=args.min_overlap_ratio,
    )

    # L2 normalize so cosine distance is meaningful
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    res = nn_leave_one_out_cosine(X, y)
    print("\n1-NN cosine LOO accuracy:", res["accuracy"])
    print("Per-label:", res["per_label_accuracy"])
    cm = np.array(res["confusion_matrix"], dtype=int)
    print("Confusion matrix (true rows, pred cols):\n", cm)

    # Save metrics.json
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "n_samples": int(X.shape[0]),
                "dim": int(X.shape[1]),
                "accuracy_1nn_cosine": res["accuracy"],
                "per_label_accuracy": res["per_label_accuracy"],
                "confusion_matrix": res["confusion_matrix"],
                "strict_coverage": bool(args.strict_coverage),
                "min_overlap_ratio": float(args.min_overlap_ratio),
            },
            f,
            indent=2,
        )
    print("Wrote:", metrics_path)

    # Save samples.jsonl
    samples_path = out_dir / "samples.jsonl"
    y_pred = res["y_pred"]
    nn_dist = res["nn_distance"]
    correct = res["correct"]
    with samples_path.open("w") as f:
        for i, r in enumerate(rows):
            rec = dict(r)
            rec["y_pred"] = VALID_LABELS[int(y_pred[i])]
            rec["nn_cosine_distance"] = float(nn_dist[i])
            rec["correct"] = bool(correct[i])
            f.write(json.dumps(rec) + "\n")
    print("Wrote:", samples_path)

    # Distance plots
    plot_hist(out_dir / "nn_distance_all.png", nn_dist, "1-NN cosine distance (all)")
    plot_hist_groups(
        out_dir / "nn_distance_correct_vs_wrong.png",
        nn_dist,
        groups=[("correct", correct), ("incorrect", ~correct)],
        title="1-NN cosine distance: correct vs incorrect",
    )
    for lab_id, lab in enumerate(VALID_LABELS):
        m = (y == lab_id)
        if m.any():
            plot_hist(out_dir / f"nn_distance_{lab}.png", nn_dist[m], f"1-NN cosine distance ({lab})")

    # Confusion matrix plot
    plot_confusion(out_dir / "confusion_matrix.png", cm, "Confusion matrix (1-NN cosine)")

    # Accuracy summary plot
    plot_accuracy_summary(
        out_dir / "accuracy_summary.png",
        overall_acc=res["accuracy"],
        per_label_acc=res["per_label_accuracy"],
        title="1-NN Cosine LOO Accuracy",
    )

    # 2D visualization (PCA only, not PCA embeddings)
    coords_pca = PCA(n_components=2, random_state=0).fit_transform(X)
    plot_2d(out_dir / "proj_pca2.png", coords_pca, y, "PCA 2D (CLAP512 window embeddings)")

    if args.make_tsne:
        # t-SNE is just visualization; reduce first for speed/stability
        from sklearn.manifold import TSNE
        X50 = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
        coords_tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(X50)
        plot_2d(out_dir / "proj_tsne.png", coords_tsne, y, "t-SNE 2D (CLAP512 window embeddings)")

    print("\nSaved outputs to:", out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
