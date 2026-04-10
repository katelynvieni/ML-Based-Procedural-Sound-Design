# ML-Based-Procedural-Sound-Design

## Overview

This repository contains the code, metadata, evaluation scripts, and Max/MSP project files for a hybrid procedural explosion sound design system using CLAP-based audio embeddings, K-means clustering, and real-time retrieval in Max/MSP.

## Requirements

This project requires:

- Python 3.10+ and the packages listed in `requirements.txt`
- Max/MSP 9.1 or later for running the interactive synthesis patch
- external project files downloaded separately from Hugging Face

## External Files Required

The following files are not included in this GitHub repository and must be downloaded separately from Hugging Face for reproduction:

- CLAP checkpoint: place locally at `data/checkpoints/630k-audioset-best.pt`
- Explosion dataset: place locally in `data/audio/`
- Precomputed Embeddings and K-means outputs: place in the appropriate `data/` subfolders 

Download all external project files here: [https://huggingface.co/datasets/krvieni/ML-Based-Procedural-Sound-Design/tree/main]

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
  ```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
4. Install the packages in `requirements.txt`.
5. Install Max/MSP 9.1 or later if you want to run the interactive synthesis patch.
6. Download the CLAP checkpoint and dataset from Hugging Face
7. Place the CLAP checkpoint at `data/checkpoints/630k-audioset-best.pt`.
8. Place the explosion dataset in `data/audio/`.
9. If you want to use the provided precomputed results, download the embeddings and K-means outputs from Hugging Face and place them in the expected `data/` subfolders.
10. Otherwise, generate those files yourself by running the Python scripts below.
```bash
python scripts/audio_dataset.py
python scripts/CLAP_embed.py
python scripts/kmeans.py
python scripts/build_max_embedding_index.py
python scripts/evaluation/eval_CLAP_windows.py
python scripts/evaluation/eval_kmeans_groupfold.py
```
11. Any generated audio from the Max/MSP patch should be placed in `data/eval_audio/` and evaluated running the script
```bash
python scripts/evaluation/plot_max_eval.py
```
