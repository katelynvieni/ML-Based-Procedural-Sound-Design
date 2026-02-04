"""
K-Means segmentation for ground/shock/roar using CLAP embeddings.

- L2-normalizes vectors (cosine-style KMeans via Euclidean on L2-normalized vectors)
- GroupKFold train/test on LABELED windows (grouped by file path; no leakage)
- Saves fold test predictions (no metrics here; evaluate separately)
- Trains final model on all labeled windows
- Predicts unlabeled files
- Exports segmented audio .wav files for playback
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

SR = 48000
VALID_CATEGORIES = {"chemical", "electrical", "fire", "space"}
TARGET_LABELS = ["ground", "shock", "roar"]

REPO_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIR = REPO_ROOT / "data" / "audio"
INDEX_PATH = REPO_ROOT / "data" / "metadata" / "audio_dataset_index.json"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"

OUTPUT_DIR = REPO_ROOT / "data" / "kmeans_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTS_AUDIO_DIR = OUTPUT_DIR / "segmented_audio"
SEGMENTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# KMeans params
N_CLUSTERS = 3
RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 500

# Cross-validation (file-grouped)
K_FOLDS = 5 

# ============================================================================
# HELPERS
# ============================================================================

def load_index():
    with open(INDEX_PATH, "r") as f:
        data = json.load(f)

    # Your format: {"data": [ ... ]}
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]

    # Back-compat formats
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]

    if isinstance(data, dict):
        vals = list(data.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals

    if isinstance(data, list) and (not data or all(isinstance(x, dict) for x in data)):
        return data

    raise ValueError(f"Unrecognized index JSON format: {type(data)}")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors."""
    return normalize(vectors, norm="l2", axis=1)


def parse_segment_pairs(seg_data):
    """
    Supports:
      - [{"start": s, "end": e}, ...]   (YOUR JSON)
      - [[s,e], [s,e], ...]
      - [s,e]
    Returns list[(s,e)] as floats.
    """
    if not seg_data:
        return []

    pairs = []

    # list of dict segments
    if isinstance(seg_data, list) and seg_data and isinstance(seg_data[0], dict):
        for obj in seg_data:
            s = obj.get("start")
            e = obj.get("end")
            if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
                pairs.append((float(s), float(e)))
        return pairs

    # [s, e]
    if isinstance(seg_data, list) and len(seg_data) == 2 and all(isinstance(x, (int, float)) for x in seg_data):
        s, e = seg_data
        return [(float(s), float(e))] if e > s else []

    # [[s,e], ...]
    if isinstance(seg_data, list):
        for item in seg_data:
            if isinstance(item, list) and len(item) == 2:
                s, e = item
                if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
                    pairs.append((float(s), float(e)))
        return pairs

    return []


def get_window_label(window_start, window_end, segments_dict):
    """Label by whether window center falls inside a labeled segment."""
    center = (window_start + window_end) / 2.0
    for label in TARGET_LABELS:
        pairs = parse_segment_pairs(segments_dict.get(label))
        for s, e in pairs:
            if s <= center <= e:
                return label
    return None


def get_audio_relpath(item: dict) -> str:
    """
    Prefer relative_path like: chemical/labeled/foo.wav
    Fall back to path like: data/audio/chemical/labeled/foo.wav
    """
    rel = item.get("relative_path") or item.get("relpath") or ""
    if rel:
        return rel

    p = item.get("path") or ""
    if p:
        p = p.replace("\\", "/")
        if "data/audio/" in p:
            return p.split("data/audio/")[-1]
        return p  # may already be relative-ish

    return ""


def find_audio_path_from_item(item: dict) -> Path:
    """
    Uses relative_path or path from JSON to find the wav under repo.
    """
    rel = get_audio_relpath(item)
    if rel:
        candidate = AUDIO_DIR / rel
        if candidate.exists():
            return candidate

    p = item.get("path")
    if p:
        p2 = REPO_ROOT / p
        if p2.exists():
            return p2

    # last resort: try by filename anywhere under data/audio
    fn = item.get("filename", "")
    if fn:
        hits = list(AUDIO_DIR.rglob(fn))
        if hits:
            return hits[0]

    raise FileNotFoundError(f"Audio file not found for item: {item.get('filename')}")


def find_embedding_file_from_item(item: dict) -> Path | None:
    fn = item.get("filename", "")
    if not fn:
        return None

    wav_name = Path(fn).name
    stem = Path(wav_name).stem

    candidates = [
        f"{stem}.npy",
        f"{stem}_emb.npy",
        f"{wav_name}.npy",      
        f"{wav_name}_emb.npy",
    ]

    for cand in candidates:
        matches = list(EMBEDDINGS_DIR.rglob(cand))
        if matches:
            return matches[0]

    return None


def load_embedding_with_bounds(embedding_path: Path, item: dict):
    """
    Returns (emb, bounds) where bounds is a list of (start,end) per embedding row.
    If *_bounds.npy exists, use it.
    Otherwise infer bounds uniformly across audio duration.
    """
    emb = np.load(embedding_path).astype(np.float32)
    T = len(emb)
    if T == 0:
        return emb, None

    # 1) Try explicit bounds files first (if you ever generate them later)
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

    # 2) Infer bounds from duration (uniform windows across file)
    duration = item.get("duration", None)

    if not isinstance(duration, (int, float)) or duration <= 0:
        try:
            audio_path = find_audio_path_from_item(item)
            info = sf.info(str(audio_path))
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            duration = None

    if not isinstance(duration, (int, float)) or duration <= 0:
        return emb, None

    duration = float(duration)
    step = duration / float(T)
    bounds = [(i * step, (i + 1) * step) for i in range(T)]
    return emb, bounds


def resample_if_needed(audio: np.ndarray, sr_in: int, sr_out: int) -> tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return audio, sr_in

    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g

    if audio.ndim == 1:
        res = resample_poly(audio, up, down)
    else:
        res = np.stack([resample_poly(audio[:, ch], up, down) for ch in range(audio.shape[1])], axis=1)

    return res.astype(audio.dtype), sr_out


def group_windows_into_segments(predicted_labels: list[str], bounds: list[tuple[float, float]]):
    segs = {lab: [] for lab in TARGET_LABELS}
    if not predicted_labels:
        return segs

    current_label = predicted_labels[0]
    current_start = bounds[0][0]

    for i in range(1, len(predicted_labels)):
        if predicted_labels[i] != current_label:
            segs[current_label].append([current_start, bounds[i - 1][1]])
            current_label = predicted_labels[i]
            current_start = bounds[i][0]

    segs[current_label].append([current_start, bounds[-1][1]])
    return segs


def write_segments_to_wav(audio_path: Path, segments_pred: dict, out_base_dir: Path):
    audio, sr_in = sf.read(audio_path, always_2d=False)
    audio, sr = resample_if_needed(audio, sr_in, SR)

    duration_sec = (len(audio) / sr) if audio.ndim == 1 else (audio.shape[0] / sr)

    for label, seg_list in segments_pred.items():
        label_dir = out_base_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for idx, (s_sec, e_sec) in enumerate(seg_list, start=1):
            s_sec = max(0.0, min(float(s_sec), duration_sec))
            e_sec = max(0.0, min(float(e_sec), duration_sec))
            if e_sec <= s_sec:
                continue

            s = int(round(s_sec * sr))
            e = int(round(e_sec * sr))
            chunk = audio[s:e] if audio.ndim == 1 else audio[s:e, :]

            out_path = label_dir / f"{label}_{idx:03d}.wav"
            sf.write(out_path, chunk, sr)


def fit_kmeans(X_train_norm: np.ndarray) -> KMeans:
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    kmeans.fit(X_train_norm)
    return kmeans


def build_cluster_to_label_mapping(y_true: np.ndarray, cluster_ids: np.ndarray) -> dict[int, str]:
    """Hungarian match between cluster IDs and true labels."""
    label_to_idx = {lab: i for i, lab in enumerate(TARGET_LABELS)}
    confusion = np.zeros((len(TARGET_LABELS), N_CLUSTERS), dtype=float)

    for lab, c in zip(y_true, cluster_ids):
        confusion[label_to_idx[str(lab)], int(c)] += 1.0

    row_ind, col_ind = linear_sum_assignment(-confusion)
    return {int(c): TARGET_LABELS[r] for r, c in zip(row_ind, col_ind)}


# ============================================================================
# LOAD LABELED WINDOWS (FOR CV + TRAINING)
# ============================================================================

def load_labeled_windows_with_meta():
    """
    Returns:
      X (n, d), y (n,), meta list[dict], groups (n,) where group is file path
    """
    index = load_index()

    X_list, y_list, meta_list, groups_list = [], [], [], []
    counts = defaultdict(int)
    labeled_files_seen = set()

    for item in index:
        if not item.get("labeled", False):
            continue

        segments = item.get("segments")
        if not segments:
            continue

        category = item.get("category", "")
        if category not in VALID_CATEGORIES:
            continue

        rel = get_audio_relpath(item)
        if not rel:
            continue

        emb_path = find_embedding_file_from_item(item)
        if emb_path is None:
            print("NO EMBEDDING FOUND FOR:", item.get("filename"))
            continue

        emb, bounds = load_embedding_with_bounds(emb_path, item)
        if bounds is None or len(bounds) != len(emb):
            print("NO/INVALID BOUNDS FOR:", emb_path.name)
            continue

        labeled_files_seen.add(rel)

        for vec, (ws, we) in zip(emb, bounds):
            lab = get_window_label(ws, we, segments)
            if lab in TARGET_LABELS:
                X_list.append(vec)
                y_list.append(lab)
                meta_list.append(
                    {
                        "rel": rel,
                        "category": category,
                        "stem": Path(rel).stem,
                        "window_start": float(ws),
                        "window_end": float(we),
                        "filename": item.get("filename", ""),
                        "embedding_file": emb_path.name,
                    }
                )
                groups_list.append(rel)  # group by file
                counts[lab] += 1

    print(f"Loaded labeled windows from {len(labeled_files_seen)} labeled files")
    print(f"Label counts (windows): {dict(counts)}")

    if not X_list:
        raise RuntimeError("No labeled windows found. Check bounds files exist and match embedding rows.")

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list),
        meta_list,
        np.asarray(groups_list),
        labeled_files_seen,
    )


# ============================================================================
# GROUP K-FOLD TRAIN/TEST 
# ============================================================================

def group_kfold_train_test(X: np.ndarray, y: np.ndarray, meta: list[dict], groups: np.ndarray, labeled_files: set):
    n_groups = len(labeled_files)
    if K_FOLDS > n_groups:
        raise ValueError(f"K_FOLDS={K_FOLDS} but you only have {n_groups} labeled files. Reduce K_FOLDS.")

    print("\n" + "=" * 60)
    print(f"GROUP K-FOLD TRAIN/TEST (k={K_FOLDS})  [group = file path]")
    print("=" * 60)

    Xn = normalize_vectors(X)
    gkf = GroupKFold(n_splits=K_FOLDS)

    folds_out = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(Xn, y, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        print(f"\nFold {fold_idx}/{K_FOLDS}")
        print(f"  train files: {len(train_groups)} | test files: {len(test_groups)}")
        print(f"  train windows: {len(train_idx)} | test windows: {len(test_idx)}")

        X_train, X_test = Xn[train_idx], Xn[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        kmeans = fit_kmeans(X_train)

        train_cluster_ids = kmeans.predict(X_train)
        cluster_to_label = build_cluster_to_label_mapping(y_train, train_cluster_ids)

        test_cluster_ids = kmeans.predict(X_test)
        y_pred = [cluster_to_label[int(c)] for c in test_cluster_ids]

        for i, idx in enumerate(test_idx):
            folds_out.append(
                {
                    "fold": fold_idx,
                    "true_label": str(y_test[i]),
                    "pred_label": str(y_pred[i]),
                    "cluster_id": int(test_cluster_ids[i]),
                    **meta[idx],
                }
            )

    out_path = OUTPUT_DIR / "cv_fold_predictions_groupkfold.json"
    with open(out_path, "w") as f:
        json.dump(folds_out, f, indent=2)
    print(f"\nSaved group-kfold test predictions to: {out_path}")

    return folds_out


# ============================================================================
# FINAL TRAIN ON ALL LABELED + PREDICT UNLABELED + EXPORT AUDIO
# ============================================================================

def train_final_and_export_unlabeled(X: np.ndarray, y: np.ndarray):
    print("\n" + "=" * 60)
    print("TRAIN FINAL MODEL (ALL LABELED) + PREDICT UNLABELED + EXPORT WAV")
    print("=" * 60)

    Xn = normalize_vectors(X)
    kmeans = fit_kmeans(Xn)

    all_cluster_ids = kmeans.predict(Xn)
    cluster_to_label = build_cluster_to_label_mapping(y, all_cluster_ids)
    print(f"Final cluster->label mapping: {cluster_to_label}")

    model_path = OUTPUT_DIR / "kmeans_model.npz"
    np.savez(
        model_path,
        centroids=kmeans.cluster_centers_.astype(np.float32),
        cluster_to_label=json.dumps(cluster_to_label),
    )
    print(f"Saved final model to: {model_path}")

    index = load_index()
    predictions = {}
    processed = 0

    for item in index:
        if item.get("labeled", False):
            continue  # unlabeled only

        category = item.get("category", "")
        if category not in VALID_CATEGORIES:
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

        X_un = normalize_vectors(emb)
        cluster_ids = kmeans.predict(X_un)
        labels = [cluster_to_label[int(c)] for c in cluster_ids]

        segments_pred = group_windows_into_segments(labels, bounds)
        predictions[rel] = segments_pred
        processed += 1

        try:
            audio_path = find_audio_path_from_item(item)
            out_dir = SEGMENTS_AUDIO_DIR / category / Path(rel).stem
            out_dir.mkdir(parents=True, exist_ok=True)
            write_segments_to_wav(audio_path, segments_pred, out_dir)
        except Exception as e:
            print(f"[WARN] Could not export audio for {item.get('filename')}: {e}")

    pred_path = OUTPUT_DIR / "predicted_segments.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved unlabeled predictions to: {pred_path}")
    print(f"Exported segmented audio for {processed} unlabeled files to: {SEGMENTS_AUDIO_DIR}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("KMEANS: GROUP-KFOLD TRAIN/TEST + FINAL TRAIN + EXPORT")
    print("=" * 60)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Index: {INDEX_PATH}")
    print(f"Embeddings: {EMBEDDINGS_DIR}")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    X, y, meta, groups, labeled_files = load_labeled_windows_with_meta()
    group_kfold_train_test(X, y, meta, groups, labeled_files)
    train_final_and_export_unlabeled(X, y)

    print("\nDONE.")


if __name__ == "__main__":
    main()