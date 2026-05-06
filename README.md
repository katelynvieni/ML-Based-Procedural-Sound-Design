# ML-Based-Procedural-Sound-Design

## Overview

This GitHub repository documents and supports the development of a hybrid procedural explosion sound design system that combines machine learning-based audio analysis with real-time generation in Max/MSP. It contains the code, project files, and documentation needed to reproduce the main stages of the workflow, including dataset indexing, CLAP embedding generation, K-means analysis, JSON construction for retrieval, and the final interactive Max patch. The importance of this research lies in its attempt to bridge the gap between realism and control in sound design, especially for complex effects like explosions that are difficult to recreate convincingly with traditional procedural methods alone. By using audio embeddings to organize and retrieve meaningful sound components, this project explores a practical way to combine recorded effects with real-time variation and user control, making it relevant to modern interactive media applications.

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

## Expected File Structure

Place external files in the following locations:

- `data/checkpoints/630k-audioset-best.pt`
- `data/audio/`

Download them here:  
[Hugging Face Dataset](https://huggingface.co/datasets/krvieni/ML-Based-Procedural-Sound-Design/tree/main)

Generated files will be written to project subfolders such as:

- `data/metadata/`
- `data/embeddings/`
- `data/kmeans_results/`

## Running the Pipeline

Place the required files in the expected locations before running the pipeline then clone this repository and install the Python dependencies:

```bash
git clone https://github.com/katelynvieni/ML-Based-Procedural-Sound-Design.git
cd ML-Based-Procedural-Sound-Design
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then, run the main scripts in the following order: 

1. `scripts/audio_dataset.py`
2. `scripts/CLAP_embed.py`
3. `scripts/kmeans.py`
4. `scripts/build_max_embedding_index.py`

These scripts prepare the dataset metadata, generate CLAP embeddings, run K-means analysis, and build the JSON index used by the Max/MSP patch.

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
2. Make sure the required audio files, JSON index file `segments_index_max.json`, and JS script `json_to_segments.js` are inside the project folder.
3. Turn on DSP/audio in Max if it is not already enabled.
4. Select an explosion category from the dropdown menu.
5. Adjust the Intensity, Distance, Duration, and Cohesion controls as desired.
6. Press Generate to create an explosion
7. If the first trigger does not fully play, press Generate again. This can occasionally happen due to a brief loading or buffer latency when parameters are updated).
8. Use the record controls if you want to capture and save the generated output.

## Demo Video

A short demo of the Max/MSP procedural explosion generator is available here: [Watch Demo]((https://youtube.com/shorts/-OhRycIzDUo?feature=share)).
