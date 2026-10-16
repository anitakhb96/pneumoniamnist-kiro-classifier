"""
compare.py — Side-by-side comparison of the 28×28 baseline and the 64×64
             enhanced pipeline.

IMPORTANT: Performance differences between the two pipelines reflect changes
to IMAGE RESOLUTION, CNN ARCHITECTURE, REGULARISATION (dropout), and
TRAINING PROCEDURE (early stopping) combined.  Do not attribute differences
solely to image resolution.

What this script does:
  1. Load outputs/metrics.json  (6-metric baseline at threshold=0.5).
  2. Attempt to recompute the two missing baseline metrics (AUPRC,
     balanced_accuracy) by re-running inference with the saved 28×28 checkpoint
     and norm_stats.  If loading fails for any reason, those fields are set
     to null and shown as "N/A" in the report — the script never crashes.
  3. Load outputs_64/metrics.json (8 metrics at both thresholds from evaluate_64.py).
  4. Print a formatted comparison table.
  5. Save outputs_64/comparison.json  (machine-readable).
  6. Save outputs_64/comparison.png   (grouped bar chart).

Run:
    python compare.py

Requires:
  - outputs/best_model.pth  and  outputs/norm_stats.json   (28×28 baseline)
  - outputs_64/metrics.json                                 (64×64 pipeline)

Educational/research use only. Not for clinical diagnosis.
"""

import json
import pathlib
import sys
import textwrap
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_64

BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Paths to 28×28 baseline artifacts (never modified)
BASELINE_METRICS_PATH    = BASE_DIR / "outputs" / "metrics.json"
BASELINE_MODEL_PATH      = BASE_DIR / "outputs" / "best_model.pth"
BASELINE_NORM_STATS_PATH = BASE_DIR / "outputs" / "norm_stats.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _recompute_baseline_missing(
    baseline_metrics: dict,
) -> dict:
    """
    Attempt to compute AUPRC and balanced_accuracy for the 28×28 baseline
    by re-running inference with the saved checkpoint.

    On any failure, returns the fields as None so the caller can show "N/A".
    Never raises an exception.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from medmnist import PneumoniaMNIST
        from sklearn.metrics import average_precision_score, balanced_accuracy_score

        # Import baseline model — must not affect 64×64 namespace
        import importlib.util, pathlib as _pl
        spec = importlib.util.spec_from_file_location(
            "model_baseline",
            str(BASE_DIR / "model.py"),
        )
        model_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_mod)
        PneumoniaCNN = model_mod.PneumoniaCNN

        # Load norm stats
        with open(BASELINE_NORM_STATS_PATH) as f:
            ns = json.load(f)
        mean, std = float(ns["mean"]), float(ns["std"])

        # Build 28×28 test loader (no size kwarg — uses the default 28×28 images)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])
        test_dataset = PneumoniaMNIST(
            split="test", transform=transform, download=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        # Load 28×28 checkpoint
        device = torch.device("cpu")
        torch.manual_seed(42)
        model = PneumoniaCNN().to(device)
        state = torch.load(BASELINE_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # Inference
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images.to(device))
                probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.squeeze(1).long().numpy().tolist())

        probs_arr  = np.array(all_probs)
        labels_arr = np.array(all_labels)
        preds_arr  = (probs_arr >= 0.5).astype(int)

        auprc        = float(average_precision_score(labels_arr, probs_arr))
        bal_accuracy = float(balanced_accuracy_score(labels_arr, preds_arr))

        print(f"  Recomputed baseline AUPRC             : {auprc:.4f}")
        print(f"  Recomputed baseline balanced_accuracy : {bal_accuracy:.4f}")
        return {"auprc": auprc, "balanced_accuracy": bal_accuracy}

    except Exception as exc:
        print(f"  WARNING: Could not recompute baseline missing metrics: {exc}")
        print("           AUPRC and balanced_accuracy will be shown as N/A.")
        return {"auprc": None, "balanced_accuracy": None}


def _fmt(value) -> str:
    """Format a metric value or None as a string."""
    if value is None:
        return "  N/A  "
    return f"{value:.4f}"


def _delta_str(v64: Optional[float], v28: Optional[float]) -> str:
    if v64 is None or v28 is None:
        return "  N/A "
    d = v64 - v28
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


# ---------------------------------------------------------------------------
# Main comparison routine
# ---------------------------------------------------------------------------

def compare() -> dict:
    print("=" * 70)
    print("PneumoniaMNIST — Pipeline Comparison (28×28 vs 64×64)")
    print("=" * 70)
    print()
    print("NOTE: Differences reflect changes to IMAGE RESOLUTION, CNN")
    print("      ARCHITECTURE, REGULARISATION, and TRAINING PROCEDURE")
    print("      combined.  Do not attribute results solely to resolution.")
    print()

    # ------------------------------------------------------------------
    # 1.  Load baseline metrics
    # ------------------------------------------------------------------
    if not BASELINE_METRICS_PATH.exists():
        print(f"ERROR: Baseline metrics not found at {BASELINE_METRICS_PATH}")
        sys.exit(1)

    baseline_raw = _load_json(BASELINE_METRICS_PATH)

    # Build a flat dict for the six stored baseline metrics
    baseline = {
        "accuracy":    baseline_raw.get("accuracy"),
        "auroc":       baseline_raw.get("auroc"),
        "precision":   baseline_raw.get("precision"),
        "sensitivity": baseline_raw.get("recall"),     # key is "recall" in old file
        "specificity": baseline_raw.get("specificity"),
        "f1_score":    baseline_raw.get("f1_score"),
        "auprc":           None,
        "balanced_accuracy": None,
    }

    # ------------------------------------------------------------------
    # 2.  Recompute missing baseline metrics
    # ------------------------------------------------------------------
    print("Attempting to recompute missing baseline metrics (AUPRC, balanced accuracy) ...")
    if BASELINE_MODEL_PATH.exists() and BASELINE_NORM_STATS_PATH.exists():
        extra = _recompute_baseline_missing(baseline_raw)
        baseline["auprc"]             = extra["auprc"]
        baseline["balanced_accuracy"] = extra["balanced_accuracy"]
    else:
        print("  Baseline checkpoint or norm_stats not found — skipping recomputation.")
    print()

    # ------------------------------------------------------------------
    # 3.  Load 64×64 metrics
    # ------------------------------------------------------------------
    if not config_64.METRICS_PATH.exists():
        print(f"ERROR: 64×64 metrics not found at {config_64.METRICS_PATH}")
        print("       Please run evaluate_64.py first.")
        sys.exit(1)

    m64_raw  = _load_json(config_64.METRICS_PATH)
    m64_05   = m64_raw["threshold_05"]
    m64_opt  = m64_raw["threshold_opt"]
    tau_star = m64_raw["tau_star"]

    m64_05_flat = {
        "accuracy":           m64_05["accuracy"],
        "balanced_accuracy":  m64_05["balanced_accuracy"],
        "auroc":              m64_05["auroc"],
        "auprc":              m64_05["auprc"],
        "precision":          m64_05["precision"],
        "sensitivity":        m64_05["sensitivity"],
        "specificity":        m64_05["specificity"],
        "f1_score":           m64_05["f1_score"],
    }
    m64_opt_flat = {
        "accuracy":           m64_opt["accuracy"],
        "balanced_accuracy":  m64_opt["balanced_accuracy"],
        "auroc":              m64_opt["auroc"],
        "auprc":              m64_opt["auprc"],
        "precision":          m64_opt["precision"],
        "sensitivity":        m64_opt["sensitivity"],
        "specificity":        m64_opt["specificity"],
        "f1_score":           m64_opt["f1_score"],
    }

    # ------------------------------------------------------------------
    # 4.  Print comparison table
    # ------------------------------------------------------------------
    metrics_order = [
        "accuracy", "balanced_accuracy", "auroc", "auprc",
        "sensitivity", "specificity", "precision", "f1_score",
    ]

    col_w = 14
    print(f"{'Metric':<22}  {'28×28 @0.5':>{col_w}}  {'64×64 @0.5':>{col_w}}  "
          f"{'64×64 @τ*':>{col_w}}  {'Δ (64@0.5−28)':>{col_w}}")
    print("-" * (22 + (col_w + 2) * 4 + 2))

    for key in metrics_order:
        v28   = baseline.get(key)
        v64   = m64_05_flat.get(key)
        v64op = m64_opt_flat.get(key)
        delta = _delta_str(v64, v28)
        print(f"{key:<22}  {_fmt(v28):>{col_w}}  {_fmt(v64):>{col_w}}  "
              f"{_fmt(v64op):>{col_w}}  {delta:>{col_w}}")

    print(f"\nτ* (Youden J, val set) = {tau_star:.4f}")
    print(f"Val sensitivity at τ*  = {m64_raw.get('val_sensitivity', 'N/A')}")
    print(f"Val specificity  at τ*  = {m64_raw.get('val_specificity', 'N/A')}")

    # ------------------------------------------------------------------
    # 5.  Build plain-English summary
    # ------------------------------------------------------------------
    def _improve(new, old, higher_is_better=True):
        if new is None or old is None:
            return "unknown"
        return "improved" if (new > old) == higher_is_better else "decreased"

    auroc_28 = baseline["auroc"]
    auroc_64 = m64_05_flat["auroc"]
    auprc_28 = baseline["auprc"]
    auprc_64 = m64_05_flat["auprc"]
    sens_28  = baseline["sensitivity"]
    sens_64  = m64_05_flat["sensitivity"]
    spec_28  = baseline["specificity"]
    spec_64  = m64_05_flat["specificity"]
    acc_28   = baseline["accuracy"]
    acc_64   = m64_05_flat["accuracy"]

    auroc_d  = f"{auroc_64 - auroc_28:+.4f}" if (auroc_64 and auroc_28) else "N/A"
    spec_d   = f"{spec_64  - spec_28:+.4f}"  if (spec_64  and spec_28)  else "N/A"
    sens_d   = f"{sens_64  - sens_28:+.4f}"  if (sens_64  and sens_28)  else "N/A"

    summary = textwrap.dedent(f"""
        Comparison summary (28×28 baseline vs 64×64 enhanced pipeline):

        The two pipelines differ in image resolution (28×28 vs 64×64), CNN
        architecture (2 blocks vs 3 blocks with AdaptiveAvgPool2d), dropout
        (0.3 vs 0.4), and training procedure (fixed 15 epochs vs early stopping
        with patience=5).  Any performance differences reflect ALL of these
        changes combined.

        Discrimination (AUROC): {_fmt(auroc_28)} → {_fmt(auroc_64)}  (Δ {auroc_d})
          — {_improve(auroc_64, auroc_28)} compared to the 28×28 baseline.

        Precision-Recall (AUPRC): {_fmt(auprc_28)} → {_fmt(auprc_64)}

        Sensitivity: {_fmt(sens_28)} → {_fmt(sens_64)}  (Δ {sens_d})
        Specificity: {_fmt(spec_28)} → {_fmt(spec_64)}  (Δ {spec_d})
        Accuracy:    {_fmt(acc_28)} → {_fmt(acc_64)}

        At the Youden-optimal threshold (τ* = {tau_star:.4f}, selected on the
        validation set), the 64×64 pipeline achieves:
          sensitivity = {_fmt(m64_opt_flat['sensitivity'])}
          specificity = {_fmt(m64_opt_flat['specificity'])}
          balanced accuracy = {_fmt(m64_opt_flat['balanced_accuracy'])}

        Treat this comparison as exploratory: the test set was already examined
        for the baseline, so it does not provide an unbiased model-selection
        estimate.
    """).strip()

    print("\n" + summary)

    # ------------------------------------------------------------------
    # 6.  Build comparison dict and save JSON
    # ------------------------------------------------------------------
    comparison = {
        "caveat": (
            "Differences reflect changes to image resolution, CNN architecture, "
            "regularisation, and training procedure combined."
        ),
        "baseline_28x28_threshold_05": {k: baseline[k] for k in metrics_order},
        "enhanced_64x64_threshold_05": m64_05_flat,
        "enhanced_64x64_threshold_opt": m64_opt_flat,
        "tau_star": tau_star,
        "delta_64x64_05_minus_28x28": {
            k: (round(m64_05_flat[k] - baseline[k], 6)
                if m64_05_flat.get(k) is not None and baseline.get(k) is not None
                else None)
            for k in metrics_order
        },
        "summary": summary,
    }

    with open(config_64.COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison JSON saved  → {config_64.COMPARISON_PATH}")

    # ------------------------------------------------------------------
    # 7.  Grouped bar chart
    # ------------------------------------------------------------------
    _save_comparison_plot(comparison, metrics_order, tau_star)

    return comparison


def _save_comparison_plot(comparison: dict, metrics_order: list, tau_star: float) -> None:
    b28   = comparison["baseline_28x28_threshold_05"]
    e64   = comparison["enhanced_64x64_threshold_05"]
    e64op = comparison["enhanced_64x64_threshold_opt"]

    # Substitute None with NaN so matplotlib skips those bars gracefully
    def _vals(d):
        return [d[k] if d[k] is not None else float("nan") for k in metrics_order]

    vals_28   = np.array(_vals(b28))
    vals_64   = np.array(_vals(e64))
    vals_64op = np.array(_vals(e64op))

    n     = len(metrics_order)
    x     = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_28   = ax.bar(x - width, vals_28,   width, label="28×28 @ 0.5",      color="#4C72B0", alpha=0.85)
    bars_64   = ax.bar(x,         vals_64,   width, label="64×64 @ 0.5",      color="#55A868", alpha=0.85)
    bars_64op = ax.bar(x + width, vals_64op, width, label=f"64×64 @ τ*={tau_star:.3f}", color="#C44E52", alpha=0.85)

    # Annotate AUROC and AUPRC bars with absolute values
    auroc_idx = metrics_order.index("auroc")
    auprc_idx = metrics_order.index("auprc")
    for bars, vals in [(bars_28, vals_28), (bars_64, vals_64), (bars_64op, vals_64op)]:
        for idx in [auroc_idx, auprc_idx]:
            bar = bars[idx]
            v   = vals[idx]
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metrics_order],
        fontsize=9,
    )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(
        "Pipeline Comparison — 28×28 Baseline vs 64×64 Enhanced Pipeline\n"
        "(Differences reflect resolution, architecture, regularisation, and training procedure combined)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(config_64.COMPARISON_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved  → {config_64.COMPARISON_PLOT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    compare()
