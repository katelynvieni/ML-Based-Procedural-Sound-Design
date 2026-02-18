import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

# ================= CONFIG =================
CATEGORIES = ["chemical", "electrical", "fire", "space"]

AUDIO_ROOT = Path("data/audio")
LABELS_ROOT = Path("data/labels")

OUT_JSON = Path("data/segments_labeled.json")
OUT_EMB_512 = Path("data/embeddings/segments_clap512")
OUT_EMB_PCA16 = Path("data/embeddings/segments_pca16")
OUT_REPORT = Path("data/build_segments_report.json")

TARGET_SR = 48000
MIN_SEGMENT_SECONDS = 0.10

# Map label tokens -> canonical labels
# Debris labels are intentionally ignored (not mapped)
LABEL_MAP = {
    "ground": "ground",
    "shock": "shock",
    "roar": "roar",
    "env": "roar",   # fold Env into Roar
}

# ================= HELPERS =================
def _norm_stem(stem: str) -> str:
    """
    Normalize filenames so wav/txt matching is more forgiving.
    Handles common suffix garbage like (2), whitespace differences, etc.
    """
    s = stem.lower()
    s = s.replace("(2)", "").replace("(1)", "")
    s = re.sub(r"\s+", "", s)
    return s


def find_label_file(wav_path: Path, category: str) -> Optional[Path]:
    """
    Find matching label txt for a wav.
    Primary: exact stem match
    Fallback: normalized stem match (case-insensitive + remove (2), spaces)
    """
    label_dir = LABELS_ROOT / category
    direct = label_dir / f"{wav_path.stem}.txt"
    if direct.exists():
        return direct

    target = _norm_stem(wav_path.stem)
    for p in label_dir.glob("*.txt"):
        if _norm_stem(p.stem) == target:
            return p
    return None


def parse_label_file(path: Path) -> List[Dict]:
    """
    Parse your label file row format (space-separated):

      <idx> <start_s> <label> <col4> <end_s> <col6>

    Example:
      1 0.29470770876834 Shock 1 0.47391697015578 0

    We parse:
      start_s = parts[1]
      label   = parts[2]
      end_s   = parts[4]

    Rules:
      - Accept Ground/Shock/Roar
      - Env folds into Roar
      - Ignore Debris / Debris2 entirely
    """
    segments = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        raw_label = parts[2].strip()
        raw_label_l = raw_label.lower()

        # ignore debris labels completely
        if raw_label_l.startswith("debris"):
            continue

        mapped = LABEL_MAP.get(raw_label_l)
        if mapped is None:
            continue

        try:
            start_s = float(parts[1])
            end_s = float(parts[4])
        except ValueError:
            continue

        start_s, end_s = min(start_s, end_s), max(start_s, end_s)
        dur = end_s - start_s
        if dur < MIN_SEGMENT_SECONDS:
            continue

        segments.append({
            "label": mapped,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": dur,
            "raw_label": raw_label,
            "raw_line": line,
        })

    segments.sort(key=lambda x: x["start_s"])
    return segments


def load_audio_mono_resampled(wav_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(wav_path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr


# ================= MAIN =================
def main():
    import torch
    import laion_clap
    from sklearn.decomposition import PCA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.to(device)
    model.eval()

    def clap_embed(wave: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(x=[wave], use_tensor=False)
        return np.asarray(emb[0], dtype=np.float32)

    # Output dirs
    OUT_EMB_512.mkdir(parents=True, exist_ok=True)
    OUT_EMB_PCA16.mkdir(parents=True, exist_ok=True)

    items: List[Dict] = []
    uid = 0

    report = {
        "processed_wavs": [],
        "skipped": {
            "no_label_file": [],
            "no_segments_parsed": [],
            "segment_too_short": [],
        },
        "counts": {}
    }

    for cat in CATEGORIES:
        audio_dir = AUDIO_ROOT / cat / "labeled"
        label_dir = LABELS_ROOT / cat

        if not audio_dir.exists():
            raise FileNotFoundError(f"Missing audio dir: {audio_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label dir: {label_dir}")

        (OUT_EMB_512 / cat).mkdir(parents=True, exist_ok=True)
        (OUT_EMB_PCA16 / cat).mkdir(parents=True, exist_ok=True)

        wavs = sorted(audio_dir.glob("*.wav"))
        report["counts"][cat] = {"wavs_found": len(wavs), "wavs_used": 0, "segments_total": 0}

        for wav_path in wavs:
            label_path = find_label_file(wav_path, cat)
            if not label_path:
                report["skipped"]["no_label_file"].append(str(wav_path))
                continue

            segments = parse_label_file(label_path)
            if not segments:
                report["skipped"]["no_segments_parsed"].append(str(label_path))
                continue

            y, sr = load_audio_mono_resampled(wav_path, TARGET_SR)

            # per-file indexing per label (ground_00, shock_00, roar_00, etc.)
            per_label_count = {"ground": 0, "shock": 0, "roar": 0}
            seg_added = 0

            for seg in segments:
                lab = seg["label"]
                idx = per_label_count[lab]
                per_label_count[lab] += 1

                a = int(seg["start_s"] * sr)
                b = int(seg["end_s"] * sr)
                clip = y[a:b]

                if len(clip) < int(MIN_SEGMENT_SECONDS * sr):
                    report["skipped"]["segment_too_short"].append({
                        "wav": str(wav_path),
                        "label": lab,
                        "start_s": seg["start_s"],
                        "end_s": seg["end_s"],
                        "raw_line": seg.get("raw_line")
                    })
                    continue

                emb512 = clap_embed(clip)

                emb512_name = f"{wav_path.stem}__{lab}__{idx:02d}.npy"
                emb512_path = OUT_EMB_512 / cat / emb512_name
                np.save(emb512_path, emb512)

                items.append({
                    "uid": uid,
                    "category": cat,
                    "file_stem": wav_path.stem,
                    "audio_path": str(wav_path.as_posix()),
                    "label_path": str(label_path.as_posix()),
                    "segment_label": lab,
                    "segment_index": idx,
                    "start_s": float(seg["start_s"]),
                    "end_s": float(seg["end_s"]),
                    "duration_s": float(seg["duration_s"]),
                    "sr": int(sr),
                    "embedding_clap512_path": str(emb512_path.as_posix()),
                    "embedding_pca16_path": None,
                    "raw_label": seg.get("raw_label"),
                    "raw_line": seg.get("raw_line"),
                })
                uid += 1
                seg_added += 1

            if seg_added > 0:
                report["processed_wavs"].append(str(wav_path.as_posix()))
                report["counts"][cat]["wavs_used"] += 1
                report["counts"][cat]["segments_total"] += seg_added

        print(f"[INFO] {cat}: wavs_found={report['counts'][cat]['wavs_found']}, "
              f"wavs_used={report['counts'][cat]['wavs_used']}, "
              f"segments_total={report['counts'][cat]['segments_total']}")

    # Write report early so failures are diagnosable
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[INFO] Wrote report: {OUT_REPORT.as_posix()}")

    if len(items) == 0:
        raise RuntimeError(
            "No segments were embedded (items=0).\n"
            "Open data/build_segments_report.json and check:\n"
            "  - skipped.no_label_file (wav/txt stem mismatch)\n"
            "  - skipped.no_segments_parsed (parser didn't match your rows)\n"
        )

    # ---- PCA to 16 dims across all segments ----
    X = np.stack([np.load(it["embedding_clap512_path"]) for it in items], axis=0)  # (N, 512)
    pca = PCA(n_components=16, random_state=0)
    X16 = pca.fit_transform(X).astype(np.float32)

    for it, v16 in zip(items, X16):
        cat = it["category"]
        base = Path(it["embedding_clap512_path"]).stem
        outp = OUT_EMB_PCA16 / cat / f"{base}__pca16.npy"
        np.save(outp, v16)
        it["embedding_pca16_path"] = str(outp.as_posix())

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "version": "1.2",
        "target_sr": TARGET_SR,
        "label_rules": {
            "allowed": ["ground", "shock", "roar"],
            "env_folded_into": "roar",
            "debris_ignored": True
        },
        "pca16": {
            "n_components": 16,
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
        "items": items
    }, indent=2))

    print(f"[OK] Wrote {OUT_JSON.as_posix()} with {len(items)} segments")
    print(f"[OK] PCA16 variance explained (sum): {float(np.sum(pca.explained_variance_ratio_)):.4f}")


if __name__ == "__main__":
    main()

