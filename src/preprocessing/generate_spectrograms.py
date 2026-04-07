import numpy as np
import pandas as pd
import librosa
from pathlib import Path


def wav_to_mel_db(
    wav_path,
    sr=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
):
    """Convert a WAV segment to a log-scaled mel spectrogram."""
    audio, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def to_spectrogram_path(segment_path):
    """Mirror data/segments/... path to data/spectrograms/... and switch extension to .npy."""
    segment_path = Path(segment_path)
    parts = list(segment_path.parts)
    if "segments" not in parts:
        raise ValueError(f"'segments' folder not found in path: {segment_path}")
    parts[parts.index("segments")] = "spectrograms"
    mirrored = Path(*parts)
    return mirrored.with_suffix(".npy")


def generate_spectrograms(
    input_csv="data/segments_split.csv",
    output_csv="data/spectrograms_split.csv",
):
    df = pd.read_csv(input_csv)
    total = len(df)
    records = []
    shapes = {}

    for idx, row in df.iterrows():
        wav_path = Path(row["filepath"])
        spec_path = to_spectrogram_path(wav_path)
        spec_path.parent.mkdir(parents=True, exist_ok=True)

        mel_db = wav_to_mel_db(wav_path)
        np.save(spec_path, mel_db)

        shape = tuple(mel_db.shape)
        shapes[shape] = shapes.get(shape, 0) + 1

        records.append(
            {
                "spectrogram_path": str(spec_path),
                "label": int(row["label"]),
                "split": row["split"],
                "machine_type": row["machine_type"],
            }
        )

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"Processed {idx + 1}/{total}")

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)

    print("\nSpectrogram generation complete")
    print(f"Total spectrograms: {len(out_df)}")
    print(f"Saved CSV: {output_csv}")
    print("\nOutput shape distribution:")
    for shape, count in sorted(shapes.items()):
        print(f"  {shape}: {count}")

    print("\nBy split:")
    print(out_df["split"].value_counts())
    print("\nBy label:")
    print(out_df["label"].value_counts())
    print("\nBy machine_type:")
    print(out_df["machine_type"].value_counts())


if __name__ == "__main__":
    generate_spectrograms()
