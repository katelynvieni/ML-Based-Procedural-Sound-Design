"""
CLAP Audio Embedding Generator
Generates CLAP embeddings with 4-window segmentation for every audio item in the repo index.

Inputs:
- data/metadata/audio_dataset_index.json
- data/audio/...

Outputs (created):
- data/embeddings/<category>/<stem>_emb.npy   (shape: 4, D)
- data/embeddings/<category>/<stem>_meta.json
"""
print(">>> TOP OF FILE: CLAP_embed.py is executing")


import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import librosa

try:
    import laion_clap
except ImportError:
    raise SystemExit(
        "ERROR: laion_clap not installed.\n"
        "Activate your venv, then run:\n"
        "  pip install laion-clap\n"
    )

# ============================================================================
# CONFIGURATION (RELATIVE TO REPO ROOT)
# ============================================================================

SR = 48000
VALID_CATEGORIES = {"chemical", "electrical", "fire", "space"}

# Auto-detect repo root as the parent of the scripts/ folder
REPO_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = REPO_ROOT / "data" / "audio"
INDEX_PATH = REPO_ROOT / "data" / "metadata" / "audio_dataset_index.json"
CHECKPOINT_PATH = (
    REPO_ROOT / "data" / "checkpoints" / "music_audioset_epoch_15_esc_90.14.pt"
)

OUT_ROOT = REPO_ROOT / "data" / "embeddings"


# ============================================================================
# AUDIO QUANTIZATION HELPERS
# ============================================================================

def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


# ============================================================================
# SETUP
# ============================================================================

def setup_output_directories() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for cat in VALID_CATEGORIES:
        (OUT_ROOT / cat).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "other").mkdir(parents=True, exist_ok=True)
    print(f"✓ Output root: {OUT_ROOT}")

def load_clap_model():
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(
            f"ERROR: Checkpoint not found:\n  {CHECKPOINT_PATH}\n"
            "Put the checkpoint file in data/checkpoints/ or update CHECKPOINT_PATH."
        )
    print("Loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(CHECKPOINT_PATH))
    model.eval()
    print("✓ CLAP model loaded")
    return model

def load_index() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"ERROR: Index not found:\n  {INDEX_PATH}\n"
            "Run scripts/audio_dataset.py first (or generate the index)."
        )
    with INDEX_PATH.open("r") as f:
        root = json.load(f)

    if isinstance(root, dict):
        items = root.get("data", root.get("items", []))
    elif isinstance(root, list):
        items = root
    else:
        items = []

    if not isinstance(items, list):
        items = []

    print(f"✓ Loaded {len(items)} items from index")
    return items


# ============================================================================
# WINDOWING + PATH HELPERS
# ============================================================================

def four_equal_windows(y: np.ndarray, sr: int) -> List[Tuple[float, float, np.ndarray]]:
    n = len(y)
    if n <= 0:
        return []

    cuts = np.linspace(0, n, 5, dtype=int)

    wins_raw: List[Tuple[int, int, np.ndarray]] = []
    for i in range(4):
        a, b = cuts[i], cuts[i + 1]
        if b > a:
            wins_raw.append((a, b, y[a:b]))

    if not wins_raw:
        return []

    max_len = max(len(w[2]) for w in wins_raw)

    wins_timed: List[Tuple[float, float, np.ndarray]] = []
    for a, b, w in wins_raw:
        if len(w) < max_len:
            w = np.pad(w, (0, max_len - len(w)), mode="constant")
        wins_timed.append((a / sr, b / sr, w))

    # Ensure exactly 4 windows
    while len(wins_timed) < 4:
        last_start, last_end, last_audio = wins_timed[-1]
        approx_dur = max(last_end - last_start, 0.0)
        wins_timed.append((last_end, last_end + approx_dur, last_audio.copy()))

    return wins_timed[:4]

def get_category_from_audio_path(audio_path: Path) -> str:
    """
    Category is assumed to be the first folder under data/audio, e.g. data/audio/chemical/...
    """
    try:
        rel = audio_path.resolve().relative_to(ASSETS_DIR.resolve())
        parts = rel.parts
        if parts:
            cat = parts[0].lower()
            if cat in VALID_CATEGORIES:
                return cat
    except Exception:
        pass
    return "other"

def resolve_audio_path(item: Dict[str, Any]) -> Optional[Path]:
    """
    Supports multiple index formats:
    - item["path"] absolute
    - item["path"] relative to repo root
    - item["relative_path"] relative to data/audio
    - item["filepath"] or similar (best effort)
    """
    candidates: List[str] = []

    for key in ("path", "relative_path", "filepath", "file", "audio_path"):
        if key in item and isinstance(item[key], str) and item[key].strip():
            candidates.append(item[key].strip())

    if not candidates:
        return None

    for p in candidates:
        pp = Path(p)

        # If absolute and exists
        if pp.is_absolute() and pp.exists():
            return pp

        # Relative to repo root
        pr = (REPO_ROOT / pp).resolve()
        if pr.exists():
            return pr

        # Relative to assets dir (common when index stores relative_path)
        pa = (ASSETS_DIR / pp).resolve()
        if pa.exists():
            return pa

    return None


# ============================================================================
# PROCESSING + SAVING
# ============================================================================

def process_audio_file(audio_path: Path, model):
    try:
        audio_data, _ = librosa.load(str(audio_path), sr=SR, mono=True)

        if audio_data.size == 0:
            return None, None, None, "empty_audio"

        wins = four_equal_windows(audio_data, SR)
        if len(wins) != 4:
            return None, None, None, "bad_windowing"

        wavs = np.stack([w[2] for w in wins], axis=0)
        wavs_quantized = int16_to_float32(float32_to_int16(wavs))
        audio_tensor = torch.from_numpy(wavs_quantized).float()

        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
            emb = emb.detach().cpu().numpy()

        meta = [
            {"path": str(audio_path), "start": float(w[0]), "end": float(w[1])}
            for w in wins
        ]
        category = get_category_from_audio_path(audio_path)
        return emb, meta, category, None

    except Exception as e:
        return None, None, None, f"exception: {e}"

def save_embeddings(embeddings: np.ndarray, metadata: List[Dict[str, Any]], audio_path: Path, category: str) -> None:
    out_dir = OUT_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    np.save(str(out_dir / f"{stem}_emb.npy"), embeddings)

    with (out_dir / f"{stem}_meta.json").open("w") as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print(">>> ENTERED main()")
    print("=" * 70)
    print("CLAP Audio Embedding Generator (repo-relative)")
    print("=" * 70)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Assets:    {ASSETS_DIR}")
    print(f"Index:     {INDEX_PATH}")
    print(f"Out:       {OUT_ROOT}")
    print("-" * 70)

    setup_output_directories()
    model = load_clap_model()
    items = load_index()

    saved = 0
    skipped = 0
    reasons: Dict[str, int] = {}

    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            skipped += 1
            reasons["bad_item"] = reasons.get("bad_item", 0) + 1
            continue

        audio_path = resolve_audio_path(item)
        if audio_path is None:
            skipped += 1
            reasons["path_missing_or_not_found"] = reasons.get("path_missing_or_not_found", 0) + 1
            continue

        emb, meta, cat, err = process_audio_file(audio_path, model)
        if emb is None:
            skipped += 1
            key = err or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
            continue

        save_embeddings(emb, meta, audio_path, cat)
        saved += 1

        if saved % 10 == 0:
            print(f"✓ Saved {saved} embeddings (processed {i}/{len(items)})")

    print("-" * 70)
    print("✓ Done")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    if reasons:
        print("\nSkip reasons:")
        for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()