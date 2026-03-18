import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpectrogramDataset
from model import get_model
import os

CSV_PATH = 'data/spectrograms_split.csv'
CHECKPOINT_DIR = 'outputs/checkpoints'
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = SpectrogramDataset(CSV_PATH, split='train')
test_dataset = SpectrogramDataset(CSV_PATH, split='test')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_weights = torch.tensor([1.0, 11.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model = get_model(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0,0,0
    for specs, labels in train_loader:
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

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/best_model.pth')
        print(f" Saved best model (acc={best_acc:.4f})")