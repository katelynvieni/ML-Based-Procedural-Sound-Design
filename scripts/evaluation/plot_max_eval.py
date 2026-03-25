from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

AUDIO_DIR = Path("eval_audio")
OUT_DIR = Path("eval_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 48000
N_FFT = 2048
HOP_LENGTH = 512

FILES = {
    "intensity": [
        AUDIO_DIR / "intensity_01.wav",
        AUDIO_DIR / "intensity_05.wav",
        AUDIO_DIR / "intensity_09.wav",
    ],
    "distance": [
        AUDIO_DIR / "distance_01.wav",
        AUDIO_DIR / "distance_05.wav",
        AUDIO_DIR / "distance_09.wav",
    ],
    "duration": [
        AUDIO_DIR / "duration_01.wav",
        AUDIO_DIR / "duration_05.wav",
        AUDIO_DIR / "duration_09.wav",
    ],
    "baseline": AUDIO_DIR / "baseline_reference.wav",
    "generated": AUDIO_DIR / "generated_reference.wav",
}

LABELS_3 = ["0.1", "0.5", "0.9"]


def load_audio(path, sr=TARGET_SR):
    y, file_sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y, sr


def normalize_for_plot(y):
    peak = np.max(np.abs(y))
    return y if peak == 0 else y / peak


def amplitude_envelope(y, frame_length=1024, hop_length=256):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=TARGET_SR, hop_length=hop_length)
    return times, rms


def align_to_onset(y, sr=TARGET_SR, threshold_ratio=0.12, frame_length=1024, hop_length=256, pre_roll_ms=5):
    """
    Trim leading silence / offset so the first strong onset begins near time 0.
    Keeps a tiny pre-roll so the transient does not look chopped.
    """
    if len(y) == 0:
        return y

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return y

    peak = np.max(rms)
    if peak <= 0:
        return y

    threshold = peak * threshold_ratio
    onset_frames = np.where(rms >= threshold)[0]

    if len(onset_frames) == 0:
        return y

    onset_sample = librosa.frames_to_samples(onset_frames[0], hop_length=hop_length)
    pre_roll = int(sr * (pre_roll_ms / 1000.0))
    start_sample = max(0, onset_sample - pre_roll)

    return y[start_sample:]


def save_waveform_envelope_figure(file_list, title, outpath):
    signals = []
    for f in file_list:
        y, sr = load_audio(f)
        y = align_to_onset(y, sr)
        signals.append(normalize_for_plot(y))

    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 7), sharex=False)

    if len(signals) == 1:
        axes = [axes]

    for ax, y, label in zip(axes, signals, LABELS_3):
        t = np.arange(len(y)) / TARGET_SR
        ax.plot(t, y, linewidth=0.8, label="Waveform")
        env_t, env = amplitude_envelope(y)
        env = env / np.max(env) if np.max(env) > 0 else env
        ax.plot(env_t, env, linewidth=1.2, label="Envelope")
        ax.set_title(label)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_spectrogram_figure(file_list, title, outpath):
    signals = []
    for f in file_list:
        y, sr = load_audio(f)
        y = align_to_onset(y, sr)
        signals.append(y)

    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 8), sharex=False, sharey=True)

    if len(signals) == 1:
        axes = [axes]

    for ax, y, label in zip(axes, signals, LABELS_3):
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        img = librosa.display.specshow(
            S_db,
            sr=TARGET_SR,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=ax
        )
        ax.set_title(label)
        ax.set_ylabel("Frequency (Hz)")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB")
    cbar.set_label("Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_baseline_generated_spectrogram(baseline_file, generated_file, outpath):
    baseline, _ = load_audio(baseline_file)
    generated, _ = load_audio(generated_file)

    baseline = align_to_onset(baseline)
    generated = align_to_onset(generated)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False, sharey=True)

    for ax, y, title in zip(axes, [baseline, generated], ["Baseline", "Generated"]):
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = librosa.display.specshow(
            S_db,
            sr=TARGET_SR,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=ax
        )
        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Baseline vs Generated Spectrogram")
    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB")
    cbar.set_label("Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_baseline_generated_waveform_envelope(baseline_file, generated_file, outpath):
    baseline, _ = load_audio(baseline_file)
    generated, _ = load_audio(generated_file)

    baseline = align_to_onset(baseline)
    generated = align_to_onset(generated)

    baseline = normalize_for_plot(baseline)
    generated = normalize_for_plot(generated)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    for ax, y, title in zip(axes, [baseline, generated], ["Baseline", "Generated"]):
        t = np.arange(len(y)) / TARGET_SR
        ax.plot(t, y, linewidth=0.8, label="Waveform")
        env_t, env = amplitude_envelope(y)
        env = env / np.max(env) if np.max(env) > 0 else env
        ax.plot(env_t, env, linewidth=1.2, label="Envelope")
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Baseline vs Generated Waveform / Envelope")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    save_waveform_envelope_figure(
        FILES["intensity"],
        "Intensity Sweep: Waveform and Envelope",
        OUT_DIR / "fig_intensity_waveform_envelope.png"
    )

    save_spectrogram_figure(
        FILES["distance"],
        "Distance Sweep: Spectrograms",
        OUT_DIR / "fig_distance_spectrogram.png"
    )

    save_waveform_envelope_figure(
        FILES["duration"],
        "Duration Sweep: Waveform and Envelope",
        OUT_DIR / "fig_duration_waveform_envelope.png"
    )

    save_baseline_generated_spectrogram(
        FILES["baseline"],
        FILES["generated"],
        OUT_DIR / "fig_baseline_generated_spectrogram.png"
    )

    save_baseline_generated_waveform_envelope(
        FILES["baseline"],
        FILES["generated"],
        OUT_DIR / "fig_baseline_generated_waveform_envelope.png"
    )

    print("Done. Figures saved in the eval_figures folder.")


if __name__ == "__main__":
    main()