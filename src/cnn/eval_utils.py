import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def collect_binary_outputs(model, data_loader, device):
    """Collect binary labels and anomaly scores from a model and data loader."""
    y_true = []
    y_score = []

    model.eval()
    with torch.no_grad():
        for specs, labels in data_loader:
            specs = specs.to(device)
            logits = model(specs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_score)


def metrics_at_threshold(y_true, y_score, threshold):
    """Compute binary metrics at a fixed anomaly threshold."""
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    accuracy = float((y_pred == y_true).mean())
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": accuracy,
        "y_pred": y_pred,
    }


def find_best_f1_threshold(y_true, y_score, step=0.01):
    """Sweep thresholds in [0, 1] and return the one with best anomaly-class F1."""
    thresholds = np.arange(0.0, 1.0 + step, step)
    best = None

    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, y_score, threshold)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics

    return best
