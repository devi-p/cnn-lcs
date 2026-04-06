import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score as sk_f1, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.cnn.dataset import SpectrogramDataset
from src.cnn.eval_utils import collect_binary_outputs, find_best_f1_threshold, metrics_at_threshold
from src.cnn.model import get_model

CSV_PATH = "data/spectrograms_split.csv"
DEFAULT_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/cnn-lcs/checkpoints"
    if os.path.exists("/content/drive/MyDrive")
    else "outputs/checkpoints"
)
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 5
NUM_WORKERS = 0 if os.name == "nt" else 2

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

train_labels = [full_train[i][1] for i in train_dataset.indices]
n_normal = sum(1 for label in train_labels if label == 0)
n_anomalous = sum(1 for label in train_labels if label == 1)
weight_normal = 1.0
weight_anomalous = (n_normal / n_anomalous) if n_normal > 0 and n_anomalous > 0 else 1.0

# Balanced sampling for train batches: each sample weight is inverse to its class frequency.
class_counts = {0: max(1, n_normal), 1: max(1, n_anomalous)}
sample_weights = [1.0 / class_counts[label] for label in train_labels]
train_sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
print(f"Training split class counts: normal={n_normal}, anomalous={n_anomalous}")
print(f"Class weights: normal={weight_normal:.2f}, anomalous={weight_anomalous:.2f}")

class_weights = torch.tensor([weight_normal, weight_anomalous]).to(device)
# We intentionally keep class-weighted loss with balanced sampling to favor anomaly recall.
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

y_true, y_score = collect_binary_outputs(model, test_loader, device)
metrics_05 = metrics_at_threshold(y_true, y_score, threshold=0.5)
best_metrics = find_best_f1_threshold(y_true, y_score, step=0.01)

try:
    test_auc = roc_auc_score(y_true, y_score)
except Exception:
    test_auc = 0.0

print(f"Accuracy:  {metrics_05['accuracy']:.4f}")
print(f"F1 Score:  {metrics_05['f1']:.4f}")
print(f"Precision: {metrics_05['precision']:.4f}")
print(f"Recall:    {metrics_05['recall']:.4f}")
print(f"AUC:       {test_auc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, metrics_05["y_pred"]))
print("(rows=actual, cols=predicted | 0=normal, 1=anomalous)")

print("Best F1 threshold search:")
print(f"  best_threshold = {best_metrics['threshold']:.2f}")
print(f"  precision = {best_metrics['precision']:.4f}")
print(f"  recall    = {best_metrics['recall']:.4f}")
print(f"  F1        = {best_metrics['f1']:.4f}")
print("Threshold 0.50 comparison:")
print(f"  precision = {metrics_05['precision']:.4f}")
print(f"  recall    = {metrics_05['recall']:.4f}")
print(f"  F1        = {metrics_05['f1']:.4f}")
