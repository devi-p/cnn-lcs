import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def _pick_sample(df, machine, label, split="train"):
    return df[
        (df["machine_type"] == machine)
        & (df["label"] == label)
        & (df["split"] == split)
    ].iloc[0]


def run_spectrogram_sanity_check(csv_path="data/spectrograms_split.csv"):
    df = pd.read_csv(csv_path)

    print("Sanity check: spectrogram outputs")
    print(f"Total rows: {len(df)}")
    print("Unique shapes across all .npy files:")

    shapes = {}
    for path in df["spectrogram_path"]:
        shape = tuple(np.load(path).shape)
        shapes[shape] = shapes.get(shape, 0) + 1
    for shape, count in sorted(shapes.items()):
        print(f"  {shape}: {count}")

    for machine in ["bearing", "gearbox"]:
        normal = _pick_sample(df, machine=machine, label=0, split="train")
        anomaly = _pick_sample(df, machine=machine, label=1, split="train")

        normal_spec = np.load(normal["spectrogram_path"])
        anomaly_spec = np.load(anomaly["spectrogram_path"])

        print(f"\n{machine.upper()}")
        print(f"  Normal sample:  {Path(normal['spectrogram_path']).name}")
        print(f"  Anomaly sample: {Path(anomaly['spectrogram_path']).name}")
        print(f"  Normal shape:   {normal_spec.shape}")
        print(f"  Anomaly shape:  {anomaly_spec.shape}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{machine.upper()} - Normal vs Anomalous (Mel dB)")

        im0 = axes[0].imshow(normal_spec, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title("Normal")
        axes[0].set_xlabel("Time Frames")
        axes[0].set_ylabel("Mel Bins")
        plt.colorbar(im0, ax=axes[0], format="%+2.0f dB")

        im1 = axes[1].imshow(anomaly_spec, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title("Anomalous")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Mel Bins")
        plt.colorbar(im1, ax=axes[1], format="%+2.0f dB")

        plt.tight_layout()
        out_path = f"outputs/{machine}_spectrogram_sanity_check.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {out_path}")


if __name__ == "__main__":
    run_spectrogram_sanity_check()
