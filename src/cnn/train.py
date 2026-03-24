import sys
sys.path.insert(0, '.')  # ensure src/ is findable from repo root

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.cnn.dataset import SpectrogramDataset
from src.cnn.model import get_model
import os

CSV_PATH = 'data/spectrograms_split.csv'
CHECKPOINT_DIR = 'outputs/checkpoints'
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    print("[1/5] Loading train dataset...")
    train_dataset = SpectrogramDataset(CSV_PATH, split='train')
    print(f"[2/5] Loading test dataset...")
    test_dataset = SpectrogramDataset(CSV_PATH, split='test')
    print(f"[3/5] Creating train loader (num_workers=0)...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"[4/5] Creating test loader (num_workers=0)...")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_weights = torch.tensor([1.0, 11.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"[5/5] Creating model...")
    model = get_model(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Setup complete! Starting training...\n")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        print(f"  Epoch {epoch+1}/{EPOCHS} starting...")
        for batch_idx, (specs, labels) in enumerate(train_loader):
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx+1}/{len(train_loader)}...")

        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/best_model.pth')
            print(f"  Saved best model (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()