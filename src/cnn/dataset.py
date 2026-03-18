import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        spec = np.load(row['spectrogram_path'])
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        spec = spec.repeat(3, 1 ,1)

        label = int(row['label'])
        if self.transform:
            spec = self.transform(spec)
        return spec, label