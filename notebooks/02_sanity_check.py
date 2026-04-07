import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # force saving instead of displaying
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_waveform_and_spectrogram(ax_wave, ax_spec, audio, sr, title):
    """Plot waveform and mel spectrogram for a single clip"""
    librosa.display.waveshow(audio, sr=sr, ax=ax_wave)
    ax_wave.set_title(f"{title} - Waveform")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time',
                                    y_axis='mel', ax=ax_spec)
    ax_spec.set_title(f"{title} - Mel Spectrogram")
    plt.colorbar(img, ax=ax_spec, format='%+2.0f dB')

def run_sanity_check(csv_path, sr=16000):
    df = pd.read_csv(csv_path)

    for machine in ['bearing', 'gearbox']:
        print(f"\nChecking {machine}...")
        machine_df = df[df['machine_type'] == machine]

        normal = machine_df[
            (machine_df['label'] == 0) &
            (machine_df['split'] == 'train')
        ].iloc[0]

        anomaly = machine_df[
            (machine_df['label'] == 1) &
            (machine_df['split'] == 'train')
        ].iloc[0]

        normal_audio, _ = librosa.load(normal['filepath'], sr=sr)
        anomaly_audio, _ = librosa.load(anomaly['filepath'], sr=sr)

        print(f"  Normal segment:   {Path(normal['filepath']).name}")
        print(f"  Anomaly segment:  {Path(anomaly['filepath']).name}")
        print(f"  Normal duration:  {len(normal_audio)/sr:.2f}s")
        print(f"  Anomaly duration: {len(anomaly_audio)/sr:.2f}s")

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"{machine.upper()} - Normal vs Anomalous", fontsize=14)

        plot_waveform_and_spectrogram(
            axes[0][0], axes[0][1], normal_audio, sr, "Normal"
        )
        plot_waveform_and_spectrogram(
            axes[1][0], axes[1][1], anomaly_audio, sr, "Anomalous"
        )

        plt.tight_layout()
        output_path = f"outputs/{machine}_sanity_check.png"
        plt.savefig(output_path, dpi=150)
        print(f"  Plot saved to {output_path}")
        plt.close()

    print("\nSanity check complete!")

if __name__ == "__main__":
    run_sanity_check('data/segments_split.csv')