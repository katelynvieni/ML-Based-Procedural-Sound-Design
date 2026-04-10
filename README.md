# ML-Based-Procedural-Sound-Design

## Requirements

This project requires:

- Python 3.10+ and the packages listed in `requirements.txt`
- Max/MSP 9.1 or later for running the interactive synthesis patch
- external project files downloaded separately from Hugging Face

## Python Environment

Create and activate a local virtual environment before installing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## External Files Required

The following files are not included in this GitHub repository and must be downloaded separately from Hugging Face for reproduction:

- CLAP checkpoint: place locally at `data/checkpoints/630k-audioset-best.pt`
- Explosion dataset: place locally in `data/audio/`
- Precomputed Embeddings and K-means outputs: place in the appropriate `data/` subfolders 

Download all external project files here: [https://huggingface.co/datasets/krvieni/ML-Based-Procedural-Sound-Design/tree/main]

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the packages in `requirements.txt`.
4. Install Max/MSP 9.1 or later if you want to run the interactive synthesis patch.
5. Download the CLAP checkpoint and dataset from Hugging Face
6. Place the CLAP checkpoint at `data/checkpoints/630k-audioset-best.pt`.
7. Place the explosion dataset in `data/audio/`.
8. If you want to use the provided precomputed results, download the embeddings and K-means outputs from Hugging Face and place them in the expected `data/` subfolders.
9. Otherwise, generate those files yourself by running the Python scripts below.
