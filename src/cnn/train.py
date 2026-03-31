import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score as sk_f1, roc_auc_score
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.cnn.dataset import SpectrogramDataset
from src.cnn.model import get_model
import numpy as np

CSV_PATH = 'data/spectrograms_split.csv'
CHECKPOINT_DIR = '/content/drive/MyDrive/cnn-lcs/checkpoints'
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 5  # early stopping patience

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# --- Data Loading ---
print("\nLoading datasets...")
full_train = SpectrogramDataset(CSV_PATH, split='train')
test_dataset = SpectrogramDataset(CSV_PATH, split='test')

# 90/10 train/val split
val_size = int(0.1 * len(full_train))
train_size = len(full_train) - val_size
train_dataset, val_dataset = random_split(
    full_train,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# --- Class Weights ---
# count classes in training set
train_labels = [full_train[i][1] for i in train_dataset.indices]
n_normal = sum(1 for l in train_labels if l == 0)
n_anomalous = sum(1 for l in train_labels if l == 1)
weight_normal = 1.0
weight_anomalous = n_normal / n_anomalous  # dynamic weight based on actual ratio
print(f"Class weights — Normal: {weight_normal:.2f}, Anomalous: {weight_anomalous:.2f}")

class_weights = torch.tensor([weight_normal, weight_anomalous]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# --- Model ---
model = get_model(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# learning rate scheduler — reduce LR when val F1 stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # maximise F1
    factor=0.5,      # halve LR on plateau
    patience=3
)

# --- Training Loop ---
best_val_f1 = 0.0
epochs_no_improve = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    # training phase
    model.train()
    total_loss, correct, total = 0, 0, 0

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

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    # validation phase
    model.eval()
    val_preds, val_labels, val_probs = [], [], []

    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            outputs = model(specs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())
            val_probs.extend(probs.cpu().numpy())

    val_f1 = sk_f1(val_labels, val_preds, zero_division=0)
    try:
        val_auc = roc_auc_score(val_labels, val_probs)
    except:
        val_auc = 0.0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val F1:     {val_f1:.4f} | Val AUC:  {val_auc:.4f}")

    # update learning rate scheduler
    scheduler.step(val_f1)

    # save best model by val F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/best_model.pth')
        print(f"  Saved best model (val_f1={best_val_f1:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")

    # early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} — no improvement for {PATIENCE} epochs")
        break

    print()

# --- Final Test Evaluation ---
print("\n" + "="*50)
print("FINAL TEST SET EVALUATION")
print("="*50)

model.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/best_model.pth', map_location=device))
model.eval()

test_preds, test_labels, test_probs = [], [], []

with torch.no_grad():
    for specs, labels in test_loader:
        specs = specs.to(device)
        outputs = model(specs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.numpy())
        test_probs.extend(probs.cpu().numpy())

from sklearn.metrics import precision_score, recall_score, confusion_matrix

print(f"Accuracy:  {sum(p==l for p,l in zip(test_preds,test_labels))/len(test_labels):.4f}")
print(f"F1 Score:  {sk_f1(test_labels, test_preds):.4f}")
print(f"Precision: {precision_score(test_labels, test_preds):.4f}")
print(f"Recall:    {recall_score(test_labels, test_preds):.4f}")
print(f"AUC:       {roc_auc_score(test_labels, test_probs):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(test_labels, test_preds))
print("(rows=actual, cols=predicted | 0=normal, 1=anomalous)")