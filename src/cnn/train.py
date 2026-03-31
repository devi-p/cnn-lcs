import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import f1_score as sk_f1, roc_auc_score
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.cnn.dataset import SpectrogramDataset
from src.cnn.model import get_model

CSV_PATH = "data/spectrograms_split.csv"
CHECKPOINT_DIR = "/content/drive/MyDrive/cnn-lcs/checkpoints"
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

full_train = SpectrogramDataset(CSV_PATH, split="train")
test_dataset = SpectrogramDataset(CSV_PATH, split="test")

val_size = int(0.1 * len(full_train))
train_size = len(full_train) - val_size
train_dataset, val_dataset = random_split(
    full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

train_labels = [full_train[i][1] for i in train_dataset.indices]
n_normal = sum(1 for label in train_labels if label == 0)
n_anomalous = sum(1 for label in train_labels if label == 1)
weight_normal = 1.0
weight_anomalous = n_normal / n_anomalous
print(f"Class weights: normal={weight_normal:.2f}, anomalous={weight_anomalous:.2f}")

class_weights = torch.tensor([weight_normal, weight_anomalous]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model = get_model(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

best_val_f1 = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

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

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

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
    except Exception:
        val_auc = 0.0

    print(
        f"Epoch {epoch + 1}/{EPOCHS}: "
        f"train_loss={avg_loss:.4f}, "
        f"train_acc={train_acc:.4f}, "
        f"val_f1={val_f1:.4f}, "
        f"val_auc={val_auc:.4f}"
    )

    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
        print(f"Saved best model: val_f1={best_val_f1:.4f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement: {epochs_no_improve}/{PATIENCE}")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}: no improvement for {PATIENCE} epochs")
        break

print("Final test evaluation")

model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=device))
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

from sklearn.metrics import confusion_matrix, precision_score, recall_score

print(f"Accuracy:  {sum(p == l for p, l in zip(test_preds, test_labels)) / len(test_labels):.4f}")
print(f"F1 Score:  {sk_f1(test_labels, test_preds):.4f}")
print(f"Precision: {precision_score(test_labels, test_preds):.4f}")
print(f"Recall:    {recall_score(test_labels, test_preds):.4f}")
print(f"AUC:       {roc_auc_score(test_labels, test_probs):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds))
print("(rows=actual, cols=predicted | 0=normal, 1=anomalous)")
