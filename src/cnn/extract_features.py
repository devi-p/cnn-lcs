import torch
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.cnn.dataset import SpectrogramDataset
from src.cnn.model import get_model
from torch.utils.data import DataLoader
import torch.nn as nn

def extract_features(
    csv_path='data/spectrograms_split.csv',
    checkpoint_path='/content/drive/MyDrive/cnn-lcs/checkpoints/best_model.pth',
    output_dir='/content/drive/MyDrive/cnn-lcs/features'
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # remove classification head
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # freeze weights
    for param in feature_extractor.parameters():
        param.requires_grad = False

    print("Model loaded and classification head removed")

    for split in ['train', 'test']:
        print(f"\nExtracting features for {split} set...")
        dataset = SpectrogramDataset(csv_path, split=split)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (specs, labels) in enumerate(loader):
                specs = specs.to(device)
                features = feature_extractor(specs)
                features = features.squeeze(-1).squeeze(-1)  # flatten to 1D vector
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())

                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(loader)}")

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)

        np.save(f'{output_dir}/{split}_features.npy', all_features)
        np.save(f'{output_dir}/{split}_labels.npy', all_labels)

        print(f"  Features shape: {all_features.shape}")
        print(f"  Labels shape:   {all_labels.shape}")
        print(f"  Normal:    {(all_labels == 0).sum()}")
        print(f"  Anomalous: {(all_labels == 1).sum()}")
        print(f"  Saved to {output_dir}/{split}_features.npy")

    print("\nFeature extraction complete!")

if __name__ == "__main__":
    extract_features()