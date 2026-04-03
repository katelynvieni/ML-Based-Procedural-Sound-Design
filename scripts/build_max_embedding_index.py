"""
Build segment-level CLAP embeddings + PCA16 JSON for Max/MSP.

What this does
--------------
1. Reads labeled explosion segments from:
   - data/audio/<category>/labeled/*.wav
   - data/labels/<category>/*.txt

2. Matches label txt files to wav files, including txt names with trailing _1234.

3. Ignores macOS AppleDouble junk files like:
   - ._something.wav

4. Extracts segment-level CLAP embeddings (512-D) for each labeled segment.

5. Fits PCA -> 16-D, z-scores the PCA dimensions, and writes:
   - .npy CLAP embeddings
   - .npy PCA16 embeddings
   - JSON index containing:
       * item metadata
       * embedded pca16 vectors
       * learned intensity/distance axes
       * axis stats for JS retrieval

Outputs
-------
- data/embeddings/segments_clap512/<category>/*.npy
- data/embeddings/segments_pca16/<category>/*.npy
- data/metadata/segments_index_max.json
- data/segments_index_max.json
- data/metadata/build_segments_report.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import soundfile as sf
import librosa

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["chemical", "electrical", "fire", "space"]

AUDIO_ROOT = ROOT / "data" / "audio"
LABELS_ROOT = ROOT / "data" / "labels"

OUT_CLAP512 = ROOT / "data" / "embeddings" / "segments_clap512"
OUT_PCA16 = ROOT / "data" / "embeddings" / "segments_pca16"

OUT_JSON_META = ROOT / "data" / "metadata" / "segments_index_max.json"
OUT_JSON_COPY = ROOT / "data" / "segments_index_max.json"
OUT_REPORT = ROOT / "data" / "metadata" / "build_segments_report.json"

TARGET_SR = 44100
MIN_SEGMENT_SECONDS = 0.0
N_PCA = 16

ALLOWED = {"ground", "shock", "roar"}
ENV_FOLDED_INTO = "roar"

REQUIRE_EXACT_WAV_COUNT = False
EXPECTED_WAVS_PER_CATEGORY = 30

# ============================================================
# CLAP CONFIG
# ============================================================

ENABLE_FUSION = False

# Your checkpoint
CLAP_CKPT_PATH = Path("/Users/katelynvieni/Downloads/630k-audioset-best.pt")

# Try these model backbones with YOUR checkpoint until one loads.
CLAP_AMODEL_CANDIDATES = [
    "HTSAT-tiny",
    "HTSAT-base",
    "PANN-14",
]


# ============================================================
# HELPERS
# ============================================================

def norm(s):
    return (s or "").strip().lower()


def rel_posix(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_stem_for_match(stem: str) -> str:
    """
    Normalize stems so:
    - txt files with trailing _1234 still match wav
    - whitespace / underscores / hyphens are ignored
    - accidental (1)/(2) is ignored
    """
    s = Path(stem).stem
    s = norm(s)
    s = s.replace("(1)", "").replace("(2)", "")
    s = re.sub(r"_\d{4}$", "", s)
    s = re.sub(r"[\s\-_]+", "", s)
    return s


def list_real_wavs(audio_dir: Path):
    """
    Ignore macOS AppleDouble sidecar files like:
      ._whatever.wav
    because they are not real audio.
    """
    return [
        p for p in audio_dir.glob("*.wav")
        if not p.name.startswith("._")
    ]


def parse_label_line(line: str):
    """
    Expected row:
      <idx> <start_s> <label> <col4> <end_s> <col6>

    Example:
      1 0.00000000000000 Ground 1 0.18679659645623 0
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        start_s = float(parts[1])
        raw_label = parts[2]
        end_s = float(parts[4])
    except ValueError:
        return None

    if end_s < start_s:
        start_s, end_s = end_s, start_s

    if end_s <= start_s:
        return None

    return start_s, end_s, raw_label


def to_segment_label(raw_label: str):
    r = norm(raw_label)

    if r.startswith("debris"):
        return None

    if r in {"env", "environment", "ambience", "ambient", "bg", "background"}:
        return ENV_FOLDED_INTO

    if r in ALLOWED:
        return r

    return None


def find_audio_for_label(category: str, label_path: Path):
    audio_dir = AUDIO_ROOT / category / "labeled"
    if not audio_dir.exists():
        return None

    direct = audio_dir / f"{label_path.stem}.wav"
    if direct.exists() and not direct.name.startswith("._"):
        return direct

    target = normalize_stem_for_match(label_path.stem)
    matches = [
        p for p in list_real_wavs(audio_dir)
        if normalize_stem_for_match(p.stem) == target
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(f"[WARN] Multiple wav matches for {label_path.name}: {[m.name for m in matches]}")
        return sorted(matches)[0]

    return None


def load_audio_resampled(audio_path: Path, target_sr: int):
    if audio_path.name.startswith("._"):
        raise ValueError(f"Refusing to read macOS sidecar file: {audio_path}")

    x, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        x = x.mean(axis=1)

    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return np.asarray(x, dtype=np.float32), int(sr)


def safe_segment(x: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(np.floor(start_s * sr)))
    b = min(len(x), int(np.ceil(end_s * sr)))
    seg = x[a:b]

    if seg.size == 0:
        seg = np.zeros(16, dtype=np.float32)

    return seg.astype(np.float32)


def save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, arr)


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


def zscore_cols(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - np.mean(v)) / (np.std(v) + 1e-12)


def minmax01(v: np.ndarray) -> np.ndarray:
    lo = np.min(v)
    hi = np.max(v)
    if hi <= lo + 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


# ============================================================
# AUDIO FEATURES FOR LEARNED AXES
# ============================================================

def _safe_rms(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    x = x.astype(np.float64, copy=False)
    if len(x) < 16:
        return 0.0
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    mag = np.abs(X) + 1e-12
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return float(np.sum(freqs * mag) / np.sum(mag))


def _hf_ratio(x: np.ndarray, sr: int, split_hz: float = 4000.0) -> float:
    x = x.astype(np.float64, copy=False)
    if len(x) < 16:
        return 0.0
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    mag2 = (np.abs(X) ** 2) + 1e-12
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    hi = mag2[freqs >= split_hz].sum()
    tot = mag2.sum()
    return float(hi / tot)


def _tail_ratio(x: np.ndarray) -> float:
    n = len(x)
    if n < 16:
        return 0.0
    k = max(1, int(0.30 * n))
    head = x[:k]
    tail = x[-k:]
    return float(_safe_rms(tail) / (_safe_rms(head) + 1e-12))


def extract_features_for_segment(audio_path: Path, start_s: float, end_s: float, target_sr: int):
    x, sr = load_audio_resampled(audio_path, target_sr)
    seg = safe_segment(x, sr, start_s, end_s)

    rms = _safe_rms(seg)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    centroid_hz = _spectral_centroid(seg, sr)
    hf_ratio = _hf_ratio(seg, sr, split_hz=4000.0)
    tail_ratio = _tail_ratio(seg)

    return {
        "rms_db": float(rms_db),
        "centroid_hz": float(centroid_hz),
        "hf_ratio": float(hf_ratio),
        "tail_ratio": float(tail_ratio),
    }


# ============================================================
# CLAP
# ============================================================

def load_clap_model():
    """
    Uses YOUR checkpoint.
    Tries multiple amodel backbones until one matches the checkpoint.
    """
    import torch

    try:
        import laion_clap
    except ImportError:
        raise SystemExit(
            "ERROR: laion_clap not installed.\n"
            "Activate your venv and run:\n"
            "  pip install laion-clap\n"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    if CLAP_CKPT_PATH is None:
        raise RuntimeError("CLAP_CKPT_PATH is None.")

    if not Path(CLAP_CKPT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CLAP_CKPT_PATH}")

    print(f"[INFO] FORCING USER CHECKPOINT: {CLAP_CKPT_PATH}")

    last_error = None

    for amodel in CLAP_AMODEL_CANDIDATES:
        print(f"[INFO] Trying amodel={amodel} WITH USER CHECKPOINT")

        try:
            model = laion_clap.CLAP_Module(
                enable_fusion=ENABLE_FUSION,
                amodel=amodel
            )

            model.load_ckpt(ckpt=str(CLAP_CKPT_PATH))
            model.to(device)
            model.eval()

            print(f"[OK] Loaded USER checkpoint successfully with amodel={amodel}")
            return model, device, amodel

        except Exception as e:
            last_error = e
            print(f"[WARN] USER checkpoint failed with amodel={amodel}")
            print(f"       {type(e).__name__}: {e}")

    raise RuntimeError(
        "Your checkpoint could not be loaded with any candidate amodel.\n"
        f"Checkpoint: {CLAP_CKPT_PATH}\n"
        f"Tried amodels: {CLAP_AMODEL_CANDIDATES}\n"
        f"Last error: {last_error}"
    )


def clap_embed_segment(model, wave: np.ndarray) -> np.ndarray:
    import torch

    if wave.ndim != 1:
        wave = wave.reshape(-1)

    if wave.size < 1024:
        wave = np.pad(wave, (0, 1024 - wave.size))

    with torch.no_grad():
        emb = model.get_audio_embedding_from_data(x=[wave], use_tensor=False)

    emb = np.asarray(emb[0], dtype=np.float32).reshape(-1)

    if emb.shape[0] != 512:
        raise ValueError(f"Expected 512-D CLAP embedding, got shape {emb.shape}")

    return emb


# ============================================================
# PASS 1: BUILD SEGMENT ITEMS + CLAP512 FILES
# ============================================================

def build_segment_items_and_clap():
    model, _device, selected_amodel = load_clap_model()

    items = []
    uid = 0

    report = {
        "version": "1.1",
        "target_sr": TARGET_SR,
        "expected_wavs_per_category": EXPECTED_WAVS_PER_CATEGORY,
        "clap": {
            "enable_fusion": ENABLE_FUSION,
            "checkpoint_path": str(CLAP_CKPT_PATH),
            "selected_amodel": selected_amodel,
        },
        "processed_wavs": [],
        "skipped": {
            "no_audio_match_for_label": [],
            "segment_too_short": [],
            "bad_label_rows": [],
        },
        "counts": {}
    }

    per_key_counter = defaultdict(int)

    for category in CATEGORIES:
        label_dir = LABELS_ROOT / category
        audio_dir = AUDIO_ROOT / category / "labeled"

        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label dir: {label_dir}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"Missing audio dir: {audio_dir}")

        ensure_dir(OUT_CLAP512 / category)
        ensure_dir(OUT_PCA16 / category)

        wavs_found = len(list_real_wavs(audio_dir))
        labels_found = len(list(label_dir.glob("*.txt")))

        report["counts"][category] = {
            "wavs_found": wavs_found,
            "labels_found": labels_found,
            "wavs_used": 0,
            "segments_total": 0,
            "segments_by_label": {"ground": 0, "shock": 0, "roar": 0},
        }

        used_wav_stems = set()

        for label_path in sorted(label_dir.glob("*.txt")):
            audio_path = find_audio_for_label(category, label_path)
            if audio_path is None:
                report["skipped"]["no_audio_match_for_label"].append(rel_posix(label_path))
                continue

            x, sr = load_audio_resampled(audio_path, TARGET_SR)
            found_any_segment = False

            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_label_line(line)
                if parsed is None:
                    if line.strip():
                        report["skipped"]["bad_label_rows"].append({
                            "label_file": rel_posix(label_path),
                            "line": line.strip()
                        })
                    continue

                start_s, end_s, raw_label = parsed
                segment_label = to_segment_label(raw_label)
                if segment_label is None:
                    continue

                duration_s = float(end_s - start_s)
                if duration_s < MIN_SEGMENT_SECONDS:
                    report["skipped"]["segment_too_short"].append({
                        "label_file": rel_posix(label_path),
                        "audio_file": rel_posix(audio_path),
                        "segment_label": segment_label,
                        "start_s": start_s,
                        "end_s": end_s,
                        "duration_s": duration_s,
                        "raw_line": line.strip(),
                    })
                    continue

                seg_i = per_key_counter[(category, audio_path.stem, segment_label)]
                per_key_counter[(category, audio_path.stem, segment_label)] += 1

                seg = safe_segment(x, sr, start_s, end_s)
                emb512 = clap_embed_segment(model, seg)

                emb512_path = OUT_CLAP512 / category / f"{audio_path.stem}__{segment_label}__{seg_i:02d}.npy"
                save_npy(emb512_path, emb512)

                pca16_rel = (
                    Path("data") / "embeddings" / "segments_pca16" / category /
                    f"{audio_path.stem}__{segment_label}__{seg_i:02d}__pca16.npy"
                )

                item = {
                    "uid": uid,
                    "category": category,
                    "file_stem": audio_path.stem,
                    "audio_path": rel_posix(audio_path),
                    "label_path": rel_posix(label_path),
                    "segment_label": segment_label,
                    "segment_index": int(seg_i),
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "duration_s": float(duration_s),
                    "sr": int(TARGET_SR),
                    "embedding_clap512_path": rel_posix(emb512_path),
                    "embedding_pca16_path": str(pca16_rel).replace("\\", "/"),
                    "raw_label": raw_label,
                    "raw_line": line.strip(),
                }

                items.append(item)
                uid += 1
                found_any_segment = True

                report["counts"][category]["segments_total"] += 1
                report["counts"][category]["segments_by_label"][segment_label] += 1

            if found_any_segment:
                used_wav_stems.add(audio_path.stem)
                report["processed_wavs"].append(rel_posix(audio_path))

        report["counts"][category]["wavs_used"] = len(used_wav_stems)

        print(
            f"[INFO] {category}: "
            f"wavs_found={report['counts'][category]['wavs_found']}, "
            f"labels_found={report['counts'][category]['labels_found']}, "
            f"wavs_used={report['counts'][category]['wavs_used']}, "
            f"segments_total={report['counts'][category]['segments_total']}"
        )

        if REQUIRE_EXACT_WAV_COUNT and report["counts"][category]["wavs_used"] != EXPECTED_WAVS_PER_CATEGORY:
            raise RuntimeError(
                f"{category} used {report['counts'][category]['wavs_used']} wavs, "
                f"expected {EXPECTED_WAVS_PER_CATEGORY}"
            )

    ensure_dir(OUT_REPORT.parent)
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote report: {rel_posix(OUT_REPORT)}")

    if len(items) == 0:
        raise RuntimeError("No segment items were built.")

    return items, report


# ============================================================
# PASS 2: PCA16
# ============================================================

def compute_and_write_pca16(items):
    X512 = []

    for it in items:
        p = ROOT / it["embedding_clap512_path"]
        if not p.exists():
            raise FileNotFoundError(f"Missing CLAP file for PCA: {p}")
        X512.append(np.load(p).astype(np.float32).reshape(-1))

    X512 = np.stack(X512, axis=0)
    if X512.shape[1] != 512:
        raise ValueError(f"Expected X512 shape (*, 512), got {X512.shape}")

    X512_l2 = l2_normalize_rows(X512)

    pca = PCA(n_components=N_PCA, random_state=0)
    X16 = pca.fit_transform(X512_l2).astype(np.float32)

    X16z, z_mu, z_sd = zscore_cols(X16)
    X16z = X16z.astype(np.float32)

    for it, v16 in zip(items, X16z):
        outp = ROOT / it["embedding_pca16_path"]
        save_npy(outp, v16)
        it["pca16"] = v16.astype(float).tolist()

    pca_meta = {
        "n_components": int(N_PCA),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "input_l2_normalized": True,
        "output_zscored": True,
        "zscore_mean": z_mu.astype(float).tolist(),
        "zscore_std": z_sd.astype(float).tolist(),
    }

    return items, pca_meta


# ============================================================
# PASS 3: LEARN PARAM AXES
# ============================================================

def learn_param_axes(items):
    X = []
    feat_rows = []

    for it in items:
        if "pca16" not in it:
            continue

        audio_path = ROOT / it["audio_path"]
        if not audio_path.exists():
            continue

        feats = extract_features_for_segment(
            audio_path=audio_path,
            start_s=float(it["start_s"]),
            end_s=float(it["end_s"]),
            target_sr=TARGET_SR,
        )

        X.append(np.asarray(it["pca16"], dtype=np.float64))
        feat_rows.append(feats)

    if len(X) < 20:
        raise RuntimeError(f"Not enough segments with PCA/audio to learn axes. Got {len(X)}")

    X = np.stack(X, axis=0)

    rms_db = np.array([r["rms_db"] for r in feat_rows], dtype=np.float64)
    cent = np.array([r["centroid_hz"] for r in feat_rows], dtype=np.float64)
    hf = np.array([r["hf_ratio"] for r in feat_rows], dtype=np.float64)
    tail = np.array([r["tail_ratio"] for r in feat_rows], dtype=np.float64)

    intensity_raw = (
        1.00 * zscore(rms_db) +
        0.40 * zscore(cent) +
        0.30 * zscore(hf) -
        0.20 * zscore(tail)
    )
    intensity_t = minmax01(intensity_raw)

    distance_raw = (
        0.70 * zscore(-cent) +
        0.50 * zscore(-hf) +
        0.35 * zscore(tail) +
        0.10 * zscore(-rms_db)
    )
    distance_t = minmax01(distance_raw)

    model_int = Ridge(alpha=1.0, fit_intercept=True)
    model_dst = Ridge(alpha=1.0, fit_intercept=True)

    model_int.fit(X, intensity_t)
    model_dst.fit(X, distance_t)

    pred_int = model_int.predict(X)
    pred_dst = model_dst.predict(X)

    param_axes = {
        "intensity": {
            "w": model_int.coef_.astype(float).tolist(),
            "bias": float(model_int.intercept_),
            "target_description": "0..1 proxy: louder/brighter/more transient = higher intensity",
        },
        "distance": {
            "w": model_dst.coef_.astype(float).tolist(),
            "bias": float(model_dst.intercept_),
            "target_description": "0..1 proxy: duller/more tail/softer = farther distance",
        }
    }

    diagnostics = {
        "n_used": int(X.shape[0]),
        "intensity_r2": float(r2_score(intensity_t, pred_int)),
        "distance_r2": float(r2_score(distance_t, pred_dst)),
        "feature_notes": {
            "intensity": "proxy from rms_db + spectral centroid + HF ratio - tail ratio",
            "distance": "proxy from -centroid - HF ratio + tail ratio - rms_db",
        }
    }

    return param_axes, diagnostics


# ============================================================
# PASS 4: AXIS STATS BY POOL
# ============================================================

def _project_axis(v16: np.ndarray, axis_w: np.ndarray, axis_b: float) -> float:
    return float(np.dot(v16, axis_w) + axis_b)


def build_axis_stats(items, param_axes):
    axis_w_int = np.asarray(param_axes["intensity"]["w"], dtype=np.float64)
    axis_b_int = float(param_axes["intensity"]["bias"])

    axis_w_dst = np.asarray(param_axes["distance"]["w"], dtype=np.float64)
    axis_b_dst = float(param_axes["distance"]["bias"])

    pools = defaultdict(list)
    for it in items:
        key = (it["category"], it["segment_label"])
        pools[key].append(np.asarray(it["pca16"], dtype=np.float64))

    out = {}
    for (category, label), vecs in sorted(pools.items()):
        vecs = np.stack(vecs, axis=0)

        proj_i = np.array([_project_axis(v, axis_w_int, axis_b_int) for v in vecs], dtype=np.float64)
        proj_d = np.array([_project_axis(v, axis_w_dst, axis_b_dst) for v in vecs], dtype=np.float64)

        k = f"{category}:{label}"
        out[k] = {
            "category": category,
            "segment_label": label,
            "n": int(len(vecs)),
            "intensity": {
                "min": float(np.min(proj_i)),
                "max": float(np.max(proj_i)),
                "mean": float(np.mean(proj_i)),
                "std": float(np.std(proj_i) + 1e-12),
            },
            "distance": {
                "min": float(np.min(proj_d)),
                "max": float(np.max(proj_d)),
                "mean": float(np.mean(proj_d)),
                "std": float(np.std(proj_d) + 1e-12),
            },
        }

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    items, report = build_segment_items_and_clap()
    items, pca_meta = compute_and_write_pca16(items)
    param_axes, diagnostics = learn_param_axes(items)
    axis_stats = build_axis_stats(items, param_axes)

    out = {
        "version": "1.3",
        "target_sr": TARGET_SR,
        "label_rules": {
            "allowed": ["ground", "shock", "roar"],
            "env_folded_into": ENV_FOLDED_INTO,
            "debris_ignored": True,
        },
        "pca16": pca_meta,
        "param_axes": param_axes,
        "param_axes_diagnostics": diagnostics,
        "axis_stats": axis_stats,
        "axisStats": axis_stats,
        "items": items,
    }

    ensure_dir(OUT_JSON_META.parent)
    OUT_JSON_META.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {rel_posix(OUT_JSON_META)} with {len(items)} segments")

    ensure_dir(OUT_JSON_COPY.parent)
    OUT_JSON_COPY.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {rel_posix(OUT_JSON_COPY)} with {len(items)} segments")

    print(f"[OK] PCA16 variance explained (sum): {pca_meta['explained_variance_ratio_sum']:.4f}")

    print("\nAxis learning diagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")

    c = Counter((it["category"], it["segment_label"]) for it in items)
    print("\nCounts by (category, segment_label):")
    for (cat, lbl), n in sorted(c.items()):
        print(f"  {cat:12s} {lbl:6s} {n}")

    print("\nCounts by category:")
    for cat in CATEGORIES:
        row = report["counts"].get(cat, {})
        print(
            f"  {cat:12s} "
            f"wavs_found={row.get('wavs_found', 0)} "
            f"labels_found={row.get('labels_found', 0)} "
            f"wavs_used={row.get('wavs_used', 0)} "
            f"segments_total={row.get('segments_total', 0)}"
        )


if __name__ == "__main__":
    main()