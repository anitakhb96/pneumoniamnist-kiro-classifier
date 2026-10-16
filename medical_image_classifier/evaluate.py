"""
evaluate.py — Evaluate the trained PneumoniaMNIST classifier on the held-out test set.

Metrics computed:
  - Accuracy
  - AUROC  (from predicted probabilities — not binary predictions)
  - Precision
  - Recall / Sensitivity  (TP / (TP + FN))
  - Specificity           (TN / (TN + FP))
  - F1-score

Outputs saved to outputs/:
  - metrics.json
  - roc_curve.png
  - confusion_matrix.png

IMPORTANT: This script must only be run AFTER train.py has saved best_model.pth.
The test set is evaluated once, here, and must not be used during development.

Educational/research use only. Not for clinical diagnosis.
"""

import json
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)

import config
from dataset import get_dataloaders
from model import get_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model: torch.nn.Module, loader, device: torch.device) -> tuple:
    """
    Run the model over a DataLoader and collect predictions.

    Returns
    -------
    all_probs  : np.ndarray  shape (N,)  — sigmoid probabilities for class 1
    all_preds  : np.ndarray  shape (N,)  — binary predictions (threshold 0.5)
    all_labels : np.ndarray  shape (N,)  — ground-truth labels
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)                    # (B, 1)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits_cat = torch.cat(all_logits).squeeze(1)   # (N,)
    labels_cat = torch.cat(all_labels).squeeze(1)   # (N,)

    probs = torch.sigmoid(logits_cat).numpy()        # probabilities in [0, 1]
    preds = (probs >= 0.5).astype(int)               # binary predictions
    labels_np = labels_cat.long().numpy()

    return probs, preds, labels_np


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute all classification metrics.

    AUROC is computed from predicted probabilities, not binary predictions.
    Specificity is computed manually from the confusion matrix.
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy    = accuracy_score(labels, preds)
    auroc       = roc_auc_score(labels, probs)            # probabilities, not preds
    precision   = precision_score(labels, preds, zero_division=0)
    recall      = recall_score(labels, preds, zero_division=0)    # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = f1_score(labels, preds, zero_division=0)

    return {
        "accuracy":    float(accuracy),
        "auroc":       float(auroc),
        "precision":   float(precision),
        "recall":      float(recall),       # sensitivity
        "specificity": float(specificity),
        "f1_score":    float(f1),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_roc_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    auroc: float,
    path: str = str(config.ROC_CURVE_PATH),
) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — PneumoniaMNIST Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved       → {path}")


def save_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    path: str = str(config.CONFUSION_MATRIX_PATH),
) -> None:
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=config.CLASS_NAMES,   # ["Normal", "Pneumonia"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — PneumoniaMNIST Test Set")
    # Axis labels: rows = true, columns = predicted
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    # --- Guard: model must exist ---
    if not config.MODEL_PATH.exists():
        print(f"ERROR: Model checkpoint not found at {config.MODEL_PATH}")
        print("       Please run train.py first.")
        sys.exit(1)

    print("=" * 60)
    print("PneumoniaMNIST CNN — Test Set Evaluation")
    print("=" * 60)
    print(f"Model : {config.MODEL_PATH}")
    print(f"Device: {config.DEVICE}\n")

    # --- Load model ---
    model = get_model(seed=config.SEED, device=config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    print("Checkpoint loaded.\n")

    # --- Load test data ---
    _, _, test_loader = get_dataloaders(batch_size=config.BATCH_SIZE)
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # --- Run inference ---
    probs, preds, labels = run_inference(model, test_loader, config.DEVICE)

    # --- Compute metrics ---
    metrics = compute_metrics(probs, preds, labels)

    # --- Validate all metric values are in [0, 1] ---
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
    for name, value in scalar_metrics.items():
        assert 0.0 <= value <= 1.0, f"Metric out of range: {name}={value}"

    # --- Print formatted table ---
    print(f"{'Metric':<18} {'Value':>8}")
    print("-" * 28)
    for name, value in scalar_metrics.items():
        print(f"{name:<18} {value:>8.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\nConfusion matrix (test set):")
    print(f"  TN={cm['tn']}  FP={cm['fp']}")
    print(f"  FN={cm['fn']}  TP={cm['tp']}")

    # --- Save metrics JSON ---
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved          → {config.METRICS_PATH}")

    # --- Save plots ---
    save_roc_curve(probs, labels, metrics["auroc"])
    save_confusion_matrix(preds, labels)

    # --- Verify all output files exist ---
    for path in [config.METRICS_PATH, config.ROC_CURVE_PATH, config.CONFUSION_MATRIX_PATH]:
        assert path.exists(), f"Expected output file not found: {path}"

    print("\nevaluate.py OK")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
