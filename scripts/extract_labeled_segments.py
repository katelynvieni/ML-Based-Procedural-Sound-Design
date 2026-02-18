import json
from pathlib import Path
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "data" / "metadata" / "audio_dataset_index.json"
AUDIO_DIR = REPO_ROOT / "data" / "audio"
OUT_DIR = REPO_ROOT / "data" / "labeled_segments_for_max"

TARGET_LABELS = ["ground", "shock", "roar"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_index():
    with open(INDEX_PATH, "r") as f:
        return json.load(f)["data"]

def get_audio_path(item):
    rel = item.get("relative_path", "")
    return AUDIO_DIR / rel

def extract_segments():
    index = load_index()
    count = 0

    for item in index:
        if not item.get("labeled", False):
            continue

        segments = item.get("segments", {})
        audio_path = get_audio_path(item)
        if not audio_path.exists():
            continue

        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr

        for label in TARGET_LABELS:
            for i, seg in enumerate(segments.get(label, []), start=1):
                s = max(0.0, min(seg["start"], duration))
                e = max(0.0, min(seg["end"], duration))
                if e <= s:
                    continue

                s_idx = int(round(s * sr))
                e_idx = int(round(e * sr))
                chunk = audio[s_idx:e_idx]

                out_dir = OUT_DIR / item["category"] / label
                out_dir.mkdir(parents=True, exist_ok=True)

                out_name = f"{Path(item['filename']).stem}_{label}_{i:02d}.wav"
                sf.write(out_dir / out_name, chunk, sr)
                count += 1

    print(f"Extracted {count} labeled segments.")

if __name__ == "__main__":
    extract_segments()
