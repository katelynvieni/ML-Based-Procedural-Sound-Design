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
- `scripts/evaluation/plot_max_eval.py`

The script `scripts/evaluation/plot_max_eval.py` generates waveform, envelope, and spectrogram figures from recorded Max/MSP outputs. The default version is set up for the evaluation examples used in this project, but the input paths, filenames, and comparison groups can be modified to match other recordings or custom evaluation cases.

### Max/MSP Project Files

- The Max/MSP project contains the main interactive patch used for real-time explosion generation, playback, recording, and saving.
- The JSON file `segments_index_max.json` acts as a segment index and stores metadata for each labeled segment, including category, label, audio path, start and end times, duration, sample rate, and PCA-reduced embedding values.
- The JavaScript file `json_to_segments.js` loads and parses the JSON data, maps the user controls to retrieval behavior, and selects ground, shock, and roar segments for playback in Max/MSP.

## Running the Max/MSP System

If you downloaded the prepared Max/MSP project, you can use it directly without cloning this repository or running the Python scripts.

1. Open the Max project folder and launch the main patch in Max/MSP.
2. Make sure the required audio files (120 total), JSON index file `segments_index_max.json`, and JS script `json_to_segments.js` are inside the project folder.
3. Turn on DSP/audio in Max if it is not already enabled.
5. Select an explosion category from the dropdown menu.
6. Adjust the Intensity, Distance, Duration, and Cohesion controls as desired.
7. Press Generate to create an explosion.
8. If the first trigger does not fully play, press Generate again. This can occasionally happen due to a brief loading or buffer latency when parameters are updated.
9. Use the record controls if you want to capture and save the generated output.


