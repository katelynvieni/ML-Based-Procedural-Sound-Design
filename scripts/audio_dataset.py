"""
Builds JSON index of the explosion audio dataset for analysis and clustering.

This script scans the dataset folders, collects audio metadata, and parses matching .txt
annotation files for labeled audio. It stores file-level information for both labeled and
unlabeled files, and includes segment time ranges for ground, shock, and roar when labels
are available. The output JSON is used for dataset organization and downstream workflows.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import librosa

class AudioDatasetIndexer:
    def __init__(self, repo_root: Path, assets_rel: Path, labels_rel: Path, output_rel: Path):
        self.repo_root = repo_root.resolve()

        self.assets_path = (self.repo_root / assets_rel).resolve()
        self.labels_path = (self.repo_root / labels_rel).resolve()
        self.output_path = (self.repo_root / output_rel).resolve()

        self.assets_rel = assets_rel.as_posix()
        self.labels_rel = labels_rel.as_posix()
        self.output_rel = output_rel.as_posix()

        self.audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}

    def parse_label_file(self, label_path: Path) -> Optional[Dict]:
        """
        Expected line format:
            1 0.0000000000000000 Ground 1 0.18679659645623 0

        Column mapping:
            start = parts[1]
            label = parts[2]
            end   = parts[4]
        """
        if not label_path.exists():
            return None

        segments = {"ground": [], "shock": [], "roar": [], "env": []}

        try:
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    try:
                        start_time = float(parts[1])
                        category = parts[2].strip().lower()
                        end_time = float(parts[4])

                        if end_time <= start_time:
                            continue

                        if category in segments:
                            segments[category].append({"start": start_time, "end": end_time})
                    except (ValueError, IndexError):
                        continue

            # Fold env into roar
            segments["roar"].extend(segments["env"])
            del segments["env"]

            segments = {k: v for k, v in segments.items() if v}
            return segments if segments else None

        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
            return None

    def get_audio_info(self, audio_path: Path) -> Dict:
        """Returns duration + sample rate (and file size as a fallback)."""
        try:
            duration = librosa.get_duration(path=str(audio_path))
            sr = librosa.get_samplerate(str(audio_path))
            return {"duration": duration, "sample_rate": sr}
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            file_size = audio_path.stat().st_size if audio_path.exists() else None
            return {"duration": None, "sample_rate": None, "file_size": file_size}

    def process_audio_file(self, audio_path: Path, category: str, subcategory: str, is_labeled: bool) -> Dict:
        """Build one index entry for a single audio file."""
        relative_to_assets = audio_path.relative_to(self.assets_path)  
        filename = audio_path.name
        audio_info = self.get_audio_info(audio_path)

        # Store a repo-relative audio path in the JSON 
        repo_rel_audio_path = (Path(self.assets_rel) / relative_to_assets).as_posix()

        entry = {
            "id": str(relative_to_assets).replace("/", "_").replace("\\", "_"),
            "filename": filename,
            "path": repo_rel_audio_path,
            "relative_path": str(relative_to_assets),
            "category": category,
            "subcategory": subcategory,
            "labeled": is_labeled,
            "duration": audio_info.get("duration"),
            "sample_rate": audio_info.get("sample_rate"),
        }

        if audio_info.get("file_size") is not None:
            entry["file_size"] = audio_info["file_size"]

        if is_labeled:
            label_filename = audio_path.stem + ".txt"
            label_path = self.labels_path / category / label_filename

            segments = self.parse_label_file(label_path)
            if segments:
                entry["segments"] = segments

                if audio_info.get("duration"):
                    entry["coverage"] = {}
                    for seg_type, seg_list in segments.items():
                        total_duration = sum(s["end"] - s["start"] for s in seg_list)
                        entry["coverage"][seg_type] = total_duration / audio_info["duration"]

        return entry

    def scan_directory(self, directory: Path, category: str, subcategory: str, is_labeled: bool) -> List[Dict]:
        entries: List[Dict] = []

        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return entries

        for audio_file in directory.iterdir():
            if audio_file.name.startswith("._") or audio_file.name == ".DS_Store":
                continue

            if audio_file.is_file() and audio_file.suffix.lower() in self.audio_extensions:
                try:
                    entries.append(self.process_audio_file(audio_file, category, subcategory, is_labeled))
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")

        return entries

    def generate_index(self) -> Dict:
        index = {
            "metadata": {
                "version": "1.0",
                "created_date": "2026-01-28",
                "description": "Audio dataset index for explosion sound effect classification",
                "base_path": self.assets_rel,
                "labels_path": self.labels_rel,
                "categories": ["chemical", "electrical", "fire", "space"],
                "segment_types": ["ground", "shock", "roar"],
                "total_files": 0,
                "labeled_files": 0,
                "unlabeled_files": 0,
            },
            "data": [],
        }

        categories = ["chemical", "electrical", "fire", "space"]

        for category in categories:
            category_path = self.assets_path / category
            if not category_path.exists():
                print(f"Category path does not exist: {category_path}")
                continue

            labeled_path = category_path / "labeled"
            labeled_entries = self.scan_directory(labeled_path, category, "labeled", True)
            index["data"].extend(labeled_entries)
            index["metadata"]["labeled_files"] += len(labeled_entries)
            print(f"Processed {len(labeled_entries)} labeled files in {category}")

            unlabeled_path = category_path / "unlabeled"
            unlabeled_entries = self.scan_directory(unlabeled_path, category, "unlabeled", False)
            index["data"].extend(unlabeled_entries)
            index["metadata"]["unlabeled_files"] += len(unlabeled_entries)
            print(f"Processed {len(unlabeled_entries)} unlabeled files in {category}")

        index["metadata"]["total_files"] = len(index["data"])
        return index

    def save_index(self, index: Dict):
        self.output_path.mkdir(parents=True, exist_ok=True)
        output_file = self.output_path / "audio_dataset_index.json"

        with open(output_file, "w") as f:
            json.dump(index, f, indent=2)

        rel_out = (Path(self.output_rel) / "audio_dataset_index.json").as_posix()
        print(f"\nIndex saved to: {rel_out}")
        print(f"Total files indexed: {index['metadata']['total_files']}")
        print(f"Labeled: {index['metadata']['labeled_files']}")
        print(f"Unlabeled: {index['metadata']['unlabeled_files']}")


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent
    if repo_root.name.lower() in {"scripts", "script"}:
        repo_root = repo_root.parent

    parser = argparse.ArgumentParser(description="Generate audio dataset index JSON (repo-relative paths only).")
    parser.add_argument("--assets", type=str, default="data/audio")
    parser.add_argument("--labels", type=str, default="data/labels")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    assets_rel = Path(args.assets)
    labels_rel = Path(args.labels)
    output_rel = Path(args.output)

    indexer = AudioDatasetIndexer(repo_root=repo_root, assets_rel=assets_rel, labels_rel=labels_rel, output_rel=output_rel)

    if not indexer.assets_path.exists():
        raise FileNotFoundError(
            f"Assets folder not found: {assets_rel.as_posix()}\n"
            f"Expected: {assets_rel.as_posix()}/<chemical|electrical|fire|space>/labeled|unlabeled/"
        )
    if not indexer.labels_path.exists():
        raise FileNotFoundError(
            f"Labels folder not found: {labels_rel.as_posix()}\n"
            f"Expected: {labels_rel.as_posix()}/<chemical|electrical|fire|space>/*.txt"
        )

    print("Starting audio dataset indexing...")
    print(f"Repo root: {repo_root}")
    print(f"Assets: {assets_rel.as_posix()}")
    print(f"Labels: {labels_rel.as_posix()}")
    print(f"Output: {output_rel.as_posix()}\n")

    index = indexer.generate_index()
    indexer.save_index(index)


if __name__ == "__main__":
    main()
