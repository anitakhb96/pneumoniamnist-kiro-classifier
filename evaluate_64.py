"""
evaluate_64.py — Evaluate the 64×64 enhanced PneumoniaMNIST pipeline.

Two-stage evaluation:
  Stage 1 — Threshold selection (validation set only):
    Run inference on the val split, compute Youden's J (= sensitivity + specificity - 1)
    across all ROC thresholds, filter out any nonfinite values, and select τ* that
    maximises J.  The threshold is NEVER derived from the test set.
    Saved to outputs_64/threshold.json along with val sensitivity and specificity.

  Stage 2 — Test set evaluation:
    Run inference ONCE on the held-out test set.
    Report all eight metrics at both the default 0.5 threshold and τ*.

Metrics: AUROC, AUPRC, accuracy, balanced accuracy, sensitivity,
         specificity, precision, F1-score.

Plots saved to outputs_64/:
  roc_curve.png             — ROC with AUROC; ★ marker at τ* operating point
  pr_curve.png              — PR curve with AUPRC; ★ marker at τ* operating point
  confusion_matrix_05.png   — confusion matrix at threshold=0.5
  confusion_matrix_opt.png  — confusion matrix at threshold=τ* (value in title)

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
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

import config_64
from dataset_64 import get_dataloaders_64
from model_64 import get_model_64


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    Collect sigmoid probabilities and ground-truth labels over a DataLoader.

    Returns
    -------
    probs  : np.ndarray  shape (N,)  — Pneumonia probability in [0, 1]
    labels : np.ndarray  shape (N,)  — ground-truth binary labels
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)            # (B, 1)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits_cat = torch.cat(all_logits).squeeze(1)   # (N,)
    labels_cat = torch.cat(all_labels).squeeze(1)   # (N,)

    probs  = torch.sigmoid(logits_cat).numpy()
    labels = labels_cat.long().numpy()

    return probs, labels


# ---------------------------------------------------------------------------
# Threshold selection (validation set only)
# ---------------------------------------------------------------------------

def select_threshold_youden(val_probs: np.ndarray, val_labels: np.ndarray) -> dict:
    """
    Select the classification threshold that maximises Youden's J on the
    validation set.

    Nonfinite threshold values are excluded before computing J to guard against
    edge cases in roc_curve output.

    Returns a dict with keys: tau_star, val_sensitivity, val_specificity.
    """
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)

    # roc_curve may prepend a sentinel threshold > 1; exclude nonfinite values
    finite_mask = np.isfinite(thresholds)
    fpr_f        = fpr[finite_mask]
    tpr_f        = tpr[finite_mask]
    thresholds_f = thresholds[finite_mask]

    if len(thresholds_f) == 0:
        # Fallback: no finite thresholds found
        print("WARNING: No finite thresholds found — defaulting to τ*=0.5")
        tau_star         = 0.5
        val_sensitivity  = float(recall_score(val_labels, (val_probs >= 0.5).astype(int), zero_division=0))
        val_specificity  = float(np.sum((val_probs < 0.5) & (val_labels == 0)) /
                                  max(np.sum(val_labels == 0), 1))
    else:
        J           = tpr_f - fpr_f                  # Youden's J = sensitivity + specificity - 1
        best_idx    = int(np.argmax(J))
        tau_star    = float(thresholds_f[best_idx])

        # Compute val sensitivity and specificity at τ*
        val_preds       = (val_probs >= tau_star).astype(int)
        tn, fp, fn, tp  = confusion_matrix(val_labels, val_preds).ravel()
        val_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        val_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    result = {
        "tau_star":         tau_star,
        "val_sensitivity":  val_sensitivity,
        "val_specificity":  val_specificity,
    }
    return result


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    probs:     np.ndarray,
    labels:    np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute all eight classification metrics at a given probability threshold.

    AUROC and AUPRC are computed from raw probabilities (not binary predictions).
    Balanced accuracy = (sensitivity + specificity) / 2.
    """
    preds = (probs >= threshold).astype(int)

    # Handle the edge case where confusion matrix is not 2×2
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity  = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_accuracy = (sensitivity + specificity) / 2.0

    return {
        "threshold":          float(threshold),
        "accuracy":           float(accuracy_score(labels, preds)),
        "balanced_accuracy":  float(bal_accuracy),
        "auroc":              float(roc_auc_score(labels, probs)),
        "auprc":              float(average_precision_score(labels, probs)),
        "precision":          float(precision_score(labels, preds, zero_division=0)),
        "sensitivity":        float(sensitivity),      # recall
        "specificity":        float(specificity),
        "f1_score":           float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix":   {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_roc_curve(
    probs:       np.ndarray,
    labels:      np.ndarray,
    auroc:       float,
    tau_star:    float,
    path:        str = str(config_64.ROC_CURVE_PATH),
) -> None:
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # Find the index closest to τ* for the operating-point marker
    finite_mask = np.isfinite(thresholds)
    if finite_mask.any():
        idx_arr = np.where(finite_mask)[0]
        diffs   = np.abs(thresholds[finite_mask] - tau_star)
        op_idx  = idx_arr[np.argmin(diffs)]
    else:
        op_idx = None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    if op_idx is not None:
        ax.plot(
            fpr[op_idx], tpr[op_idx],
            marker="*", markersize=14, color="red",
            label=f"τ* = {tau_star:.3f} (Youden J)",
        )
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — 64×64 Enhanced Pipeline (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved          → {path}")


def save_pr_curve(
    probs:    np.ndarray,
    labels:   np.ndarray,
    auprc:    float,
    tau_star: float,
    path:     str = str(config_64.PR_CURVE_PATH),
) -> None:
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(labels, probs)

    # Find the operating point for τ*
    finite_mask = np.isfinite(pr_thresholds)
    op_precision, op_recall = None, None
    if finite_mask.any():
        idx_arr    = np.where(finite_mask)[0]
        diffs      = np.abs(pr_thresholds[finite_mask] - tau_star)
        op_i       = idx_arr[np.argmin(diffs)]
        op_precision = precision_arr[op_i]
        op_recall    = recall_arr[op_i]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall_arr, precision_arr, lw=2, label=f"PR curve (AUPRC = {auprc:.3f})")
    if op_precision is not None:
        ax.plot(
            op_recall, op_precision,
            marker="*", markersize=14, color="red",
            label=f"τ* = {tau_star:.3f} (Youden J)",
        )
    # Baseline (random classifier)
    baseline = float(labels.mean())
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Random classifier (precision = {baseline:.2f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — 64×64 Enhanced Pipeline (Test Set)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved           → {path}")


def save_confusion_matrix_plot(
    preds:     np.ndarray,
    labels:    np.ndarray,
    threshold: float,
    path:      str,
    title_suffix: str = "",
) -> None:
    cm   = confusion_matrix(labels, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=config_64.CLASS_NAMES,
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    threshold_str = f" (threshold = {threshold:.3f})"
    ax.set_title(
        f"Confusion Matrix — 64×64 Enhanced Pipeline{threshold_str}\n{title_suffix}"
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved   → {path}")


# ---------------------------------------------------------------------------
# Metric table printer
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict, header: str) -> None:
    scalar_keys = [
        "threshold", "accuracy", "balanced_accuracy", "auroc", "auprc",
        "precision", "sensitivity", "specificity", "f1_score",
    ]
    print(f"\n{header}")
    print(f"{'Metric':<22} {'Value':>8}")
    print("-" * 32)
    for key in scalar_keys:
        if key in metrics:
            print(f"{key:<22} {metrics[key]:>8.4f}")
    cm = metrics.get("confusion_matrix", {})
    if cm:
        print(f"\n  Confusion matrix:")
        print(f"    TN={cm['tn']}  FP={cm['fp']}")
        print(f"    FN={cm['fn']}  TP={cm['tp']}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    # Guard: model must exist
    if not config_64.MODEL_PATH.exists():
        print(f"ERROR: Model checkpoint not found at {config_64.MODEL_PATH}")
        print("       Please run train_64.py first.")
        sys.exit(1)

    print("=" * 65)
    print("PneumoniaMNIST+ (64×64) — Enhanced Pipeline Evaluation")
    print("=" * 65)
    print(f"Model  : {config_64.MODEL_PATH}")
    print(f"Device : {config_64.DEVICE}\n")

    # --- Load model ---
    model = get_model_64(seed=config_64.SEED, device=config_64.DEVICE)
    state = torch.load(config_64.MODEL_PATH, map_location=config_64.DEVICE)
    model.load_state_dict(state)
    print("Checkpoint loaded.\n")

    # --- Load data ---
    _, val_loader, test_loader = get_dataloaders_64(batch_size=config_64.BATCH_SIZE)
    print(f"Val samples  : {len(val_loader.dataset)}")
    print(f"Test samples : {len(test_loader.dataset)}\n")

    # -----------------------------------------------------------------------
    # Stage 1: Threshold selection using validation set only
    # -----------------------------------------------------------------------
    print("Stage 1: Selecting threshold on validation set (Youden's J) ...")
    val_probs, val_labels = run_inference(model, val_loader, config_64.DEVICE)
    threshold_info = select_threshold_youden(val_probs, val_labels)
    tau_star = threshold_info["tau_star"]

    print(f"  τ* = {tau_star:.4f}")
    print(f"  Val sensitivity at τ* : {threshold_info['val_sensitivity']:.4f}")
    print(f"  Val specificity  at τ* : {threshold_info['val_specificity']:.4f}")

    # Save threshold info to disk
    with open(config_64.THRESHOLD_PATH, "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"  Threshold saved → {config_64.THRESHOLD_PATH}\n")

    # -----------------------------------------------------------------------
    # Stage 2: Test set evaluation (single inference pass)
    # -----------------------------------------------------------------------
    print("Stage 2: Evaluating on test set ...")
    test_probs, test_labels = run_inference(model, test_loader, config_64.DEVICE)

    metrics_05  = compute_metrics(test_probs, test_labels, threshold=0.5)
    metrics_opt = compute_metrics(test_probs, test_labels, threshold=tau_star)

    # Validate all metric floats are in [0, 1]
    for label, m in [("threshold_05", metrics_05), ("threshold_opt", metrics_opt)]:
        for key, val in m.items():
            if isinstance(val, float):
                assert 0.0 <= val <= 1.0, f"{label}.{key} out of range: {val}"

    # Print tables
    print_metrics_table(metrics_05,  "--- Metrics at threshold = 0.5 ---")
    print_metrics_table(metrics_opt, f"--- Metrics at τ* = {tau_star:.4f} (Youden J, val set) ---")

    # Build full metrics dict
    full_metrics = {
        "threshold_05":      metrics_05,
        "threshold_opt":     metrics_opt,
        "tau_star":          tau_star,
        "val_sensitivity":   threshold_info["val_sensitivity"],
        "val_specificity":   threshold_info["val_specificity"],
    }

    # Save metrics JSON
    with open(config_64.METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\nMetrics saved            → {config_64.METRICS_PATH}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    auroc = metrics_05["auroc"]
    auprc = metrics_05["auprc"]

    save_roc_curve(test_probs, test_labels, auroc, tau_star)
    save_pr_curve(test_probs,  test_labels, auprc, tau_star)

    preds_05  = (test_probs >= 0.5).astype(int)
    preds_opt = (test_probs >= tau_star).astype(int)

    save_confusion_matrix_plot(
        preds_05, test_labels, threshold=0.5,
        path=str(config_64.CONFUSION_MATRIX_05_PATH),
        title_suffix="Test Set",
    )
    save_confusion_matrix_plot(
        preds_opt, test_labels, threshold=tau_star,
        path=str(config_64.CONFUSION_MATRIX_OPT_PATH),
        title_suffix="Test Set  (τ* from val set, Youden J)",
    )

    # Verify all expected output files exist
    expected = [
        config_64.METRICS_PATH,
        config_64.THRESHOLD_PATH,
        config_64.ROC_CURVE_PATH,
        config_64.PR_CURVE_PATH,
        config_64.CONFUSION_MATRIX_05_PATH,
        config_64.CONFUSION_MATRIX_OPT_PATH,
    ]
    for p in expected:
        assert p.exists(), f"Expected output not found: {p}"

    print("\nevaluate_64.py OK")
    return full_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
