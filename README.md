# ML-Based-Procedural-Sound-Design

## Overview

This project contains a hybrid procedural explosion sound design system using CLAP-based audio embeddings, K-means clustering, and real-time retrieval in Max/MSP.

Because of file size limits, some project assets are hosted separately on Hugging Face rather than included directly in this GitHub repository.

## Choose a Setup Option

There are two ways to use this project:

### Option 1: Run the Max Patch Only

Choose this option if you only want to use the interactive synthesis system in Max/MSP.

You do **not** need to clone this repository or run the Python pipeline for this option.

Download the prepared Max patch package from Hugging Face and open the main patch in Max/MSP.

Requirements:
- Max/MSP 9.1 or later

### Option 2: Reproduce the Full Pipeline

Choose this option if you want to reproduce the dataset indexing, embedding generation, K-means analysis, JSON generation, and evaluation pipeline.

For this option, you will need:
- Python 3.10 or later
- the packages listed in `requirements.txt`
- Max/MSP 9.1 or later if you also want to run the interactive patch
- external project files downloaded separately from Hugging Face

## External Files

The following files are hosted separately on Hugging Face:

- CLAP checkpoint: place locally at `data/checkpoints/630k-audioset-best.pt`
- Explosion dataset: place locally in `data/audio/`

Download them here:  
[Hugging Face Dataset](https://huggingface.co/datasets/krvieni/ML-Based-Procedural-Sound-Design/tree/main)

## Full Pipeline Setup

Clone this repository and install the Python dependencies:

```bash
git clone https://github.com/katelynvieni/ML-Based-Procedural-Sound-Design.git
cd ML-Based-Procedural-Sound-Design
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the required external files in the expected locations before running the pipeline.

## Main Pipeline Scripts

The main scripts in this repository are:

- `scripts/audio_dataset.py`
- `scripts/CLAP_embed.py`
- `scripts/kmeans.py`
- `scripts/build_max_embedding_index.py`

Evaluation scripts are located in:

- `scripts/evaluation/eval_CLAP_windows.py`
- `scripts/evaluation/eval_kmeans_groupfold.py`
- `scripts/evaluation/plot_max_eval.py` (any generated audio from the Max/MSP patch should be placed in `data/eval_audio/` before running the evaluation script

## Running the Max/MSP System

To run the interactive system, open the main Max/MSP patch.

If you downloaded the prepared Max/MSP project, you can use it directly without cloning this repository or running the Python scripts.
