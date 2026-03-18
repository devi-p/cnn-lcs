import sys
sys.path.append('.')  # ensures src/ is findable from repo root

import torch
from torch.utils.data import DataLoader, Subset
from src.cnn.dataset import SpectrogramDataset
from src.cnn.model import get_model

ds = SpectrogramDataset('data/spectrograms_split.csv', split='train')
subset = Subset(ds, list(range(50)))
loader = DataLoader(subset, batch_size=8)
model = get_model()

for specs, labels in loader:
    out = model(specs)
    print(f"Input shape:  {specs.shape}")   # expect torch.Size([8, 3, 128, 32])
    print(f"Output shape: {out.shape}")     # expect torch.Size([8, 2])
    print(f"Labels:       {labels}")
    break

print("Sanity check passed ✓")
