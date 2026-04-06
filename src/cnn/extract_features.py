import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.cnn.dataset import SpectrogramDataset
from src.cnn.model import get_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _colab_drive_root() -> Path | None:
    drive_path = Path("/content/drive/MyDrive/cnn-lcs")
    return drive_path if drive_path.exists() else None


def _default_checkpoint_path() -> Path:
    colab_root = _colab_drive_root()
    if colab_root:
        return colab_root / "checkpoints" / "best_model.pth"
    return PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"


def _default_features_dir() -> Path:
    colab_root = _colab_drive_root()
    if colab_root:
        return colab_root / "features"
    return PROJECT_ROOT / "outputs" / "features"


def extract_features(
    csv_path: str = "data/spectrograms_split.csv",
    checkpoint_path: Path | str = _default_checkpoint_path(),
    output_dir: Path | str = _default_features_dir(),
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, str]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    for param in feature_extractor.parameters():
        param.requires_grad = False

    print("Feature extractor loaded")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "csv_path": csv_path,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    for split in ["train", "test"]:
        dataset = SpectrogramDataset(csv_path, split=split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for specs, labels in loader:
                specs = specs.to(device)
                features = feature_extractor(specs)
                features = features.squeeze(-1).squeeze(-1)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())

        split_features = np.concatenate(all_features, axis=0)
        split_labels = np.array(all_labels)

        np.save(output_dir / f"{split}_features.npy", split_features)
        np.save(output_dir / f"{split}_labels.npy", split_labels)

        print(
            f"{split}: "
            f"features={split_features.shape}, "
            f"labels={split_labels.shape}, "
            f"normal={(split_labels == 0).sum()}, "
            f"anomalous={(split_labels == 1).sum()}"
        )

    with open(output_dir / "feature_extraction_run.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    return {
        "output_dir": str(output_dir),
        "train_features": str(output_dir / "train_features.npy"),
        "train_labels": str(output_dir / "train_labels.npy"),
        "test_features": str(output_dir / "test_features.npy"),
        "test_labels": str(output_dir / "test_labels.npy"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen CNN embeddings for LCS training.")
    parser.add_argument("--csv-path", type=str, default="data/spectrograms_split.csv")
    parser.add_argument("--checkpoint-path", type=Path, default=_default_checkpoint_path())
    parser.add_argument("--output-dir", type=Path, default=_default_features_dir())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_features(
        csv_path=args.csv_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
