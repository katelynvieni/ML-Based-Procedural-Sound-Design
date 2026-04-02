# ML-Based-Procedural-Sound-Design

## Dataset
The explosion audio files are hosted separately on Hugging Face due to file size limitations on GitHub. The dataset can be downloaded from: https://huggingface.co/datasets/krvieni/ML-Based-Procedural-Sound-Design/resolve/main/explosion_dataset.zip

After downloading, place the audio files inside `data/audio/` so the project structure is:

```
data/
├── audio/
├── labels/
└── metadata/
```

The labels and metadata are already included in this repository. Audio filenames should remain unchanged so they continue to correspond correctly with the provided labels.

## CLAP Checkpoint
Audio embeddings in this project were generated using the LAION-CLAP checkpoint music_audioset_epoch_15_esc_90.14.pt. This is a public checkpoint used consistently across the embedding pipeline for all experiments.

For reproduibility, place the file in:
data/checkpoints/music_audioset_epoch_15_esc_90.14.pt

## CLAP Embeddings

Due to file size constraints, the CLAP audio embeddings generated for this project are hosted externally on Hugging Face, hosted at this link: 

https://huggingface.co/datasets/krvieni/ML-Based-Procedural-Sound-Design/resolve/main/embeddings.zip

The embeddings preserve the original folder structure used during processing. 

## K Means Predictions 

The predicted segments and output audio are hosted at this link: 

https://huggingface.co/datasets/krvieni/ML-Based-Procedural-Sound-Design/resolve/main/kmeans_results.zip


