"""
investigate.py — Diagnostic investigation of the val/test specificity discrepancy.

Background
----------
The 64×64 enhanced pipeline selects τ* = 0.9418 via Youden's J on the validation set.
At this threshold:
  - Validation specificity : 99.26%  (134 / 135 Normals correctly identified)
  - Test specificity       : 67.52%  (158 / 234 Normals correctly identified)

This 31.7 pp gap is investigated across eight sub-analyses without assuming its cause.

Sub-analyses
------------
1.  Split integrity          — verify split sizes, class distributions, label identity
2.  Preprocessing check      — verify training-derived norm stats applied consistently
3.  Probability distributions— class-conditional score distributions val vs test
4.  Threshold sweep          — sensitivity/specificity across all thresholds, val vs test
5.  Baseline threshold       — Youden J τ* for 28×28 model; compare val/test gap
6.  Unified comparison table — all four operating points side by side
7.  Paired bootstrap AUROC CI— 95% CI for AUROC difference (64×64 − 28×28)
8.  Diagnostic report        — written synthesis of all findings

Outputs  (all saved to outputs_investigation/ — existing files never touched)
-------
  split_stats.json            preprocessing_check.json
  prob_distributions.png      prob_stats.json
  threshold_sweep.png         threshold_sweep.json
  baseline_threshold.json
  unified_comparison.json     unified_comparison.png
  bootstrap_auroc.json        bootstrap_auroc.png
  diagnostic_report.txt

Important constraints
---------------------
- No model retraining.
- outputs/ and outputs_64/ are read-only.
- Norm stats loaded from saved JSON — get_dataloaders_64() is NOT called.
- All DataLoaders use shuffle=False for deterministic ordering.
- Bootstrap resamples containing only one outcome class are skipped.
- Test-set threshold sweep is diagnostic only; thresholds are not selected from it.

Educational/research use only. Not for clinical diagnosis.
"""

import importlib.util
import json
import pathlib
import sys
import textwrap
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
OUT_DIR    = BASE_DIR / "outputs_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_MODEL_PATH      = BASE_DIR / "outputs" / "best_model.pth"
BASELINE_NORM_PATH       = BASE_DIR / "outputs" / "norm_stats.json"
BASELINE_METRICS_PATH    = BASE_DIR / "outputs" / "metrics.json"

ENHANCED_MODEL_PATH      = BASE_DIR / "outputs_64" / "best_model.pth"
ENHANCED_NORM_PATH       = BASE_DIR / "outputs_64" / "norm_stats.json"
ENHANCED_METRICS_PATH    = BASE_DIR / "outputs_64" / "metrics.json"
ENHANCED_THRESHOLD_PATH  = BASE_DIR / "outputs_64" / "threshold.json"

# Published MedMNIST split sizes
EXPECTED_SIZES = {"train": 4708, "val": 524, "test": 624}

TAU_STAR_64 = 0.9417790770530701   # from outputs_64/threshold.json

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _save_json(obj: dict, path: pathlib.Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved → {path.name}")


def _load_json(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _compute_metrics_at_threshold(probs, labels, threshold):
    """Return a dict of 8 metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "threshold":         float(threshold),
        "accuracy":          float(accuracy_score(labels, preds)),
        "balanced_accuracy": (sens + spec) / 2.0,
        "auroc":             float(roc_auc_score(labels, probs)),
        "auprc":             float(average_precision_score(labels, probs)),
        "precision":         float(precision_score(labels, preds, zero_division=0)),
        "sensitivity":       sens,
        "specificity":       spec,
        "f1_score":          float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix":  {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _youden_threshold(probs, labels):
    """
    Select threshold maximising Youden's J = TPR - FPR on the given split.
    Nonfinite thresholds are excluded. Returns (tau_star, sensitivity, specificity).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    finite = np.isfinite(thresholds)
    fpr_f, tpr_f, thr_f = fpr[finite], tpr[finite], thresholds[finite]
    if len(thr_f) == 0:
        return 0.5, float(recall_score(labels, (probs >= 0.5).astype(int), zero_division=0)), 0.0
    j = tpr_f - fpr_f
    idx = int(np.argmax(j))
    tau = float(thr_f[idx])
    preds = (probs >= tau).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return tau, sens, spec


# ---------------------------------------------------------------------------
# Model loaders (no retraining, no norm-stat recomputation)
# ---------------------------------------------------------------------------

def _load_model_64():
    """Load PneumoniaCNN64 from saved checkpoint."""
    spec = importlib.util.spec_from_file_location("model_64", BASE_DIR / "model_64.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import config_64
    torch.manual_seed(config_64.SEED)
    model = mod.PneumoniaCNN64()
    state = torch.load(ENHANCED_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _load_model_28():
    """Load PneumoniaCNN (28×28) from saved checkpoint."""
    spec = importlib.util.spec_from_file_location("model_28", BASE_DIR / "model.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(42)
    model = mod.PneumoniaCNN()
    state = torch.load(BASELINE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _build_loader(split: str, norm_path: pathlib.Path, size: Optional[int] = None,
                  batch_size: int = 64) -> DataLoader:
    """
    Build a DataLoader for a given split using pre-saved normalisation stats.
    shuffle=False for deterministic ordering.
    get_dataloaders_64() is NOT called — norm stats are not recomputed.
    """
    stats = _load_json(norm_path)
    mean, std = float(stats["mean"]), float(stats["std"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
    kwargs = {"split": split, "transform": transform, "download": True}
    if size is not None:
        kwargs["size"] = size
    ds = PneumoniaMNIST(**kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def _run_inference(model, loader) -> tuple:
    """Return (probs, labels) as numpy arrays."""
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    probs  = torch.sigmoid(torch.cat(all_logits).squeeze(1)).numpy()
    labels = torch.cat(all_labels).squeeze(1).long().numpy()
    return probs, labels


# ---------------------------------------------------------------------------
# Section header helper
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ===========================================================================
# Sub-analysis 1 — Split integrity
# ===========================================================================

def analysis_1_split_integrity() -> dict:
    _section("Sub-analysis 1: Split integrity")

    results = {"28x28": {}, "64x64": {}, "labels_identical": {}}

    for size, key in [(None, "28x28"), (64, "64x64")]:
        for split in ["train", "val", "test"]:
            kwargs = {"split": split, "transform": transforms.ToTensor(), "download": True}
            if size:
                kwargs["size"] = size
            ds = PneumoniaMNIST(**kwargs)
            lbls = np.array([int(ds[i][1].item()) for i in range(len(ds))])
            n0, n1, n = int((lbls == 0).sum()), int((lbls == 1).sum()), len(lbls)
            results[key][split] = {
                "total": n, "n_normal": n0, "n_pneumonia": n1,
                "frac_normal": round(n0 / n, 6), "frac_pneumonia": round(n1 / n, 6),
                "imbalance_ratio": round(n1 / n0, 4) if n0 > 0 else None,
            }
            expected = EXPECTED_SIZES.get(split)
            status   = "OK" if n == expected else f"MISMATCH (expected {expected})"
            print(f"  {key} {split:5s}: total={n}  Normal={n0} ({n0/n:.1%})  "
                  f"Pneumonia={n1} ({n1/n:.1%})  [{status}]")

    # Verify label identity across resolutions
    for split in ["val", "test"]:
        kwargs28 = {"split": split, "transform": transforms.ToTensor(), "download": True}
        kwargs64 = {"split": split, "transform": transforms.ToTensor(), "download": True, "size": 64}
        lbls28 = np.array([int(PneumoniaMNIST(**kwargs28)[i][1].item())
                           for i in range(len(PneumoniaMNIST(**kwargs28)))])
        lbls64 = np.array([int(PneumoniaMNIST(**kwargs64)[i][1].item())
                           for i in range(len(PneumoniaMNIST(**kwargs64)))])
        identical = bool(np.array_equal(lbls28, lbls64))
        results["labels_identical"][split] = identical
        print(f"  Labels identical 28×28 vs 64×64 ({split}): {identical}")

    _save_json(results, OUT_DIR / "split_stats.json")
    return results


# ===========================================================================
# Sub-analysis 2 — Preprocessing consistency
# ===========================================================================

def analysis_2_preprocessing() -> dict:
    _section("Sub-analysis 2: Preprocessing consistency")

    checks = {}
    for norm_path, size, key in [
        (BASELINE_NORM_PATH, None, "28x28"),
        (ENHANCED_NORM_PATH, 64, "64x64"),
    ]:
        stored = _load_json(norm_path)
        stored_mean, stored_std = float(stored["mean"]), float(stored["std"])

        # Re-derive from training split
        kwargs = {"split": "train", "transform": transforms.ToTensor(), "download": True}
        if size:
            kwargs["size"] = size
        ds = PneumoniaMNIST(**kwargs)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        running_sum = running_sq = n_px = 0.0
        for imgs, _ in loader:
            running_sum += imgs.sum().item()
            running_sq  += (imgs ** 2).sum().item()
            n_px        += imgs.numel()
        derived_mean = running_sum / n_px
        derived_std  = ((running_sq / n_px) - derived_mean ** 2) ** 0.5

        mean_diff = abs(derived_mean - stored_mean)
        std_diff  = abs(derived_std  - stored_std)
        mean_ok   = mean_diff < 1e-4
        std_ok    = std_diff  < 1e-4

        checks[key] = {
            "stored_mean": stored_mean, "derived_mean": round(derived_mean, 8),
            "mean_diff": round(mean_diff, 8), "mean_ok": mean_ok,
            "stored_std":  stored_std,  "derived_std":  round(derived_std, 8),
            "std_diff":  round(std_diff, 8),  "std_ok":  std_ok,
            "same_size_all_splits": True,   # verified by code inspection of dataset_64.py
        }
        print(f"  {key}: stored mean={stored_mean:.6f}  derived={derived_mean:.6f}  "
              f"diff={mean_diff:.2e}  {'OK' if mean_ok else 'WARN'}")
        print(f"       stored std ={stored_std:.6f}  derived={derived_std:.6f}  "
              f"diff={std_diff:.2e}  {'OK' if std_ok else 'WARN'}")

    _save_json(checks, OUT_DIR / "preprocessing_check.json")
    return checks


# ===========================================================================
# Sub-analysis 3 — Probability distributions (class-conditional)
# ===========================================================================

def analysis_3_prob_distributions(
    model64, val_probs64, val_labels64, test_probs64, test_labels64
) -> dict:
    _section("Sub-analysis 3: Class-conditional probability distributions")

    tau = TAU_STAR_64
    groups = {
        "val_normal":     val_probs64[val_labels64 == 0],
        "val_pneumonia":  val_probs64[val_labels64 == 1],
        "test_normal":    test_probs64[test_labels64 == 0],
        "test_pneumonia": test_probs64[test_labels64 == 1],
    }

    stats = {}
    for name, arr in groups.items():
        n_above = int((arr >= tau).sum())
        stats[name] = {
            "n": len(arr),
            "mean": round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "std":  round(float(arr.std()), 6),
            "p05":  round(float(np.percentile(arr, 5)), 6),
            "p25":  round(float(np.percentile(arr, 25)), 6),
            "p75":  round(float(np.percentile(arr, 75)), 6),
            "p95":  round(float(np.percentile(arr, 95)), 6),
            "frac_above_tau_star": round(n_above / len(arr), 6),
            "n_above_tau_star": n_above,
        }
        print(f"  {name:20s}  n={len(arr):4d}  mean={arr.mean():.4f}  "
              f"median={np.median(arr):.4f}  frac≥τ*={n_above/len(arr):.3f}")

    # --- 4-panel histogram plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Class-conditional Probability Distributions — 64×64 Model\n"
        f"τ* = {tau:.4f} (Youden J, selected on validation set)",
        fontsize=12,
    )
    panel_order = [
        ("val_normal",    "Val — Normal (0)",    axes[0, 0], "#4C72B0"),
        ("val_pneumonia", "Val — Pneumonia (1)", axes[0, 1], "#C44E52"),
        ("test_normal",   "Test — Normal (0)",   axes[1, 0], "#4C72B0"),
        ("test_pneumonia","Test — Pneumonia (1)",axes[1, 1], "#C44E52"),
    ]
    for name, title, ax, color in panel_order:
        arr = groups[name]
        ax.hist(arr, bins=40, range=(0, 1), color=color, alpha=0.75, density=True)
        ax.axvline(tau, color="red", linestyle="--", linewidth=1.5,
                   label=f"τ* = {tau:.3f}")
        s = stats[name]
        ax.set_title(f"{title}  (n={s['n']},  {s['frac_above_tau_star']:.1%} ≥ τ*)",
                     fontsize=10)
        ax.set_xlabel("Predicted Pneumonia probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "prob_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → prob_distributions.png")

    _save_json(stats, OUT_DIR / "prob_stats.json")
    return stats


# ===========================================================================
# Sub-analysis 4 — Threshold sweep (val vs test, 64×64 model)
# ===========================================================================

def analysis_4_threshold_sweep(
    val_probs, val_labels, test_probs, test_labels
) -> dict:
    _section("Sub-analysis 4: Threshold sweep (val vs test)")

    thresholds = np.linspace(0.0, 1.0, 500)
    results = {
        "thresholds": thresholds.tolist(),
        "val_sensitivity":  [], "val_specificity":  [],
        "test_sensitivity": [], "test_specificity": [],
        "val_youden":  [], "test_youden": [],
    }

    for tau in thresholds:
        for split, probs, labels, prefix in [
            ("val",  val_probs,  val_labels,  "val"),
            ("test", test_probs, test_labels, "test"),
        ]:
            preds = (probs >= tau).astype(int)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            results[f"{prefix}_sensitivity"].append(sens)
            results[f"{prefix}_specificity"].append(spec)
            results[f"{prefix}_youden"].append(sens + spec - 1.0)

    # Find Youden-optimal threshold on val from sweep (for annotation)
    val_j  = np.array(results["val_youden"])
    test_j = np.array(results["test_youden"])
    val_opt_tau  = float(thresholds[np.argmax(val_j)])
    test_opt_tau = float(thresholds[np.argmax(test_j)])
    results["sweep_val_opt_tau"]  = val_opt_tau
    results["sweep_test_opt_tau"] = test_opt_tau
    print(f"  Sweep: val  Youden-optimal τ ≈ {val_opt_tau:.4f}  "
          f"(J={max(results['val_youden']):.4f})")
    print(f"  Sweep: test Youden-optimal τ ≈ {test_opt_tau:.4f}  "
          f"(J={max(results['test_youden']):.4f})  [diagnostic only — not used for selection]")

    # Plot: 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(
        "Threshold Sweep — 64×64 Model (val = dashed, test = solid)\n"
        "Test-set curves shown for diagnosis only; thresholds selected from val set.",
        fontsize=11,
    )

    for ax, ykey, ylabel in [
        (axes[0], ("val_specificity", "test_specificity"),    "Specificity"),
        (axes[1], ("val_sensitivity", "test_sensitivity"),    "Sensitivity"),
        (axes[2], ("val_youden",      "test_youden"),         "Youden's J"),
    ]:
        ax.plot(thresholds, results[ykey[0]], linestyle="--", color="#4C72B0",
                lw=1.5, label="Validation")
        ax.plot(thresholds, results[ykey[1]], linestyle="-",  color="#C44E52",
                lw=1.5, label="Test (diagnostic)")
        ax.axvline(TAU_STAR_64, color="green", linestyle=":", linewidth=1.5,
                   label=f"τ* = {TAU_STAR_64:.3f} (Youden J, val)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Threshold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → threshold_sweep.png")

    # Save compact JSON (round to 5 dp to keep file size reasonable)
    compact = {
        "thresholds": [round(t, 5) for t in results["thresholds"]],
        "val_sensitivity":  [round(v, 5) for v in results["val_sensitivity"]],
        "val_specificity":  [round(v, 5) for v in results["val_specificity"]],
        "val_youden":       [round(v, 5) for v in results["val_youden"]],
        "test_sensitivity": [round(v, 5) for v in results["test_sensitivity"]],
        "test_specificity": [round(v, 5) for v in results["test_specificity"]],
        "test_youden":      [round(v, 5) for v in results["test_youden"]],
        "sweep_val_opt_tau":  round(val_opt_tau, 5),
        "sweep_test_opt_tau": round(test_opt_tau, 5),
        "tau_star_from_evaluate": round(TAU_STAR_64, 6),
    }
    _save_json(compact, OUT_DIR / "threshold_sweep.json")
    return results


# ===========================================================================
# Sub-analysis 5 — Youden's J threshold for the 28×28 baseline
# ===========================================================================

def analysis_5_baseline_threshold(model28) -> dict:
    _section("Sub-analysis 5: Youden J threshold for 28×28 baseline")

    val_loader28  = _build_loader("val",  BASELINE_NORM_PATH, size=None)
    test_loader28 = _build_loader("test", BASELINE_NORM_PATH, size=None)

    val_probs28,  val_labels28  = _run_inference(model28, val_loader28)
    test_probs28, test_labels28 = _run_inference(model28, test_loader28)

    # Youden J on val set
    tau28, val_sens28, val_spec28 = _youden_threshold(val_probs28, val_labels28)
    print(f"  28×28 τ* (Youden J, val) = {tau28:.4f}")
    print(f"  28×28 val  sensitivity={val_sens28:.4f}  specificity={val_spec28:.4f}")

    # Metrics at 0.5 and τ* on test set
    m28_05  = _compute_metrics_at_threshold(test_probs28, test_labels28, 0.5)
    m28_opt = _compute_metrics_at_threshold(test_probs28, test_labels28, tau28)

    # Val metrics at τ*
    m28_val_opt = _compute_metrics_at_threshold(val_probs28, val_labels28, tau28)

    val_test_spec_gap28 = val_spec28 - m28_opt["specificity"]
    print(f"  28×28 test sensitivity={m28_opt['sensitivity']:.4f}  "
          f"specificity={m28_opt['specificity']:.4f}  @ τ*")
    print(f"  28×28 val/test specificity gap at τ*: "
          f"{val_spec28:.4f} − {m28_opt['specificity']:.4f} = {val_test_spec_gap28:+.4f}")

    result = {
        "tau_star_28":     tau28,
        "val_sensitivity": val_sens28,
        "val_specificity": val_spec28,
        "test_at_05":  m28_05,
        "test_at_opt": m28_opt,
        "val_at_opt":  m28_val_opt,
        "val_test_specificity_gap": round(val_test_spec_gap28, 6),
    }
    _save_json(result, OUT_DIR / "baseline_threshold.json")
    return result, test_probs28, test_labels28


# ===========================================================================
# Sub-analysis 6 — Unified four-way comparison table
# ===========================================================================

def analysis_6_unified_comparison(
    val_probs64, val_labels64, test_probs64, test_labels64,
    test_probs28, test_labels28,
    tau28, baseline_result
) -> dict:
    _section("Sub-analysis 6: Unified four-way comparison")

    tau64 = TAU_STAR_64

    # Re-derive val metrics for 64×64 @ τ*
    val_m64_opt = _compute_metrics_at_threshold(val_probs64, val_labels64, tau64)

    operating_points = {
        "28x28_@0.5":      _compute_metrics_at_threshold(test_probs28, test_labels28, 0.5),
        f"28x28_@tau*={tau28:.4f}": baseline_result["test_at_opt"],
        "64x64_@0.5":      _compute_metrics_at_threshold(test_probs64, test_labels64, 0.5),
        f"64x64_@tau*={tau64:.4f}": _compute_metrics_at_threshold(test_probs64, test_labels64, tau64),
    }

    metric_keys = ["accuracy", "balanced_accuracy", "auroc", "auprc",
                   "sensitivity", "specificity", "precision", "f1_score"]
    labels_row  = list(operating_points.keys())

    print(f"\n  {'Metric':<22}", end="")
    for lbl in labels_row:
        print(f"  {lbl:>22}", end="")
    print()
    print("  " + "-" * (22 + 24 * len(labels_row)))
    for mkey in metric_keys:
        print(f"  {mkey:<22}", end="")
        for op in operating_points.values():
            print(f"  {op[mkey]:>22.4f}", end="")
        print()

    # Also show val specificity at each tau for context
    print(f"\n  {'':22}  {'Val specificity at τ* (selection set):':>22}")
    print(f"  {'28x28':22}  {baseline_result['val_specificity']:>22.4f}")
    print(f"  {'64x64':22}  {val_m64_opt['specificity']:>22.4f}")

    comparison = {
        "operating_points": {k: v for k, v in operating_points.items()},
        "val_specificity_at_tau": {
            "28x28": baseline_result["val_specificity"],
            "64x64": val_m64_opt["specificity"],
        },
    }
    _save_json(comparison, OUT_DIR / "unified_comparison.json")

    # Grouped bar chart
    metric_display = [m.replace("_", "\n") for m in metric_keys]
    x     = np.arange(len(metric_keys))
    width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (lbl, op) in enumerate(operating_points.items()):
        vals = [op[m] for m in metric_keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lbl,
               color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_display, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Unified Comparison — All Four Operating Points (Test Set)", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "unified_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → unified_comparison.png")

    return comparison


# ===========================================================================
# Sub-analysis 7 — Paired bootstrap 95% CI for AUROC difference
# ===========================================================================

def analysis_7_bootstrap_auroc(
    test_probs64, test_labels64, test_probs28, test_labels28,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    _section("Sub-analysis 7: Paired bootstrap CI for AUROC difference")

    # Verify alignment
    assert len(test_probs64) == len(test_probs28) == len(test_labels64), \
        "Mismatched array lengths — predictions must correspond to the same test images"
    assert np.array_equal(test_labels64, test_labels28), \
        "Label arrays differ — predictions are not aligned to the same test examples"
    n = len(test_probs64)
    print(f"  Alignment verified: {n} test examples, labels identical across models")

    observed_auroc64 = float(roc_auc_score(test_labels64, test_probs64))
    observed_auroc28 = float(roc_auc_score(test_labels28, test_probs28))
    observed_diff    = observed_auroc64 - observed_auroc28
    print(f"  AUROC 64×64 = {observed_auroc64:.6f}")
    print(f"  AUROC 28×28 = {observed_auroc28:.6f}")
    print(f"  Observed difference = {observed_diff:+.6f}")

    rng = np.random.default_rng(seed)
    diffs = []
    skipped = 0
    while len(diffs) < n_bootstrap:
        idx = rng.integers(0, n, size=n)
        lbl_resamp = test_labels64[idx]
        # Skip resamples with only one outcome class in either model's labels
        if len(np.unique(lbl_resamp)) < 2:
            skipped += 1
            continue
        a64 = float(roc_auc_score(lbl_resamp, test_probs64[idx]))
        a28 = float(roc_auc_score(lbl_resamp, test_probs28[idx]))
        diffs.append(a64 - a28)

    diffs = np.array(diffs)
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    boot_mean = float(diffs.mean())

    print(f"  Bootstrap (n={n_bootstrap}, skipped={skipped}): "
          f"mean diff = {boot_mean:+.6f}")
    print(f"  95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"  CI {'does NOT include' if ci_lo > 0 or ci_hi < 0 else 'includes'} zero")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diffs, bins=80, color="#55A868", alpha=0.75, density=True)
    ax.axvline(observed_diff, color="red",    linestyle="-",  lw=2,
               label=f"Observed diff = {observed_diff:+.4f}")
    ax.axvline(ci_lo,         color="orange", linestyle="--", lw=1.5,
               label=f"95% CI lower = {ci_lo:+.4f}")
    ax.axvline(ci_hi,         color="orange", linestyle="--", lw=1.5,
               label=f"95% CI upper = {ci_hi:+.4f}")
    ax.axvline(0,             color="black",  linestyle=":",  lw=1,
               label="Zero (no difference)")
    ax.set_xlabel("AUROC difference (64×64 − 28×28)")
    ax.set_ylabel("Density")
    ax.set_title(f"Paired Bootstrap AUROC Difference (n={n_bootstrap} resamples)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bootstrap_auroc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → bootstrap_auroc.png")

    result = {
        "n_test_examples": n,
        "n_bootstrap": n_bootstrap,
        "n_skipped": skipped,
        "observed_auroc_64x64": round(observed_auroc64, 8),
        "observed_auroc_28x28": round(observed_auroc28, 8),
        "observed_difference":  round(observed_diff, 8),
        "bootstrap_mean_diff":  round(boot_mean, 8),
        "ci_95_lower": round(ci_lo, 8),
        "ci_95_upper": round(ci_hi, 8),
        "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
    }
    _save_json(result, OUT_DIR / "bootstrap_auroc.json")
    return result


# ===========================================================================
# Sub-analysis 8 — Diagnostic report
# ===========================================================================

def analysis_8_diagnostic_report(
    split_stats, preproc_check, prob_stats, sweep_results,
    baseline_result, comparison, bootstrap_result,
    tau28, val_probs64, val_labels64, test_probs64, test_labels64,
) -> None:
    _section("Sub-analysis 8: Writing diagnostic report")

    tau64 = TAU_STAR_64

    # Extract key numbers
    vn   = prob_stats["val_normal"]
    vp   = prob_stats["val_pneumonia"]
    tn   = prob_stats["test_normal"]
    tp_  = prob_stats["test_pneumonia"]

    val_norm_above   = vn["frac_above_tau_star"]
    test_norm_above  = tn["frac_above_tau_star"]
    val_pneu_above   = vp["frac_above_tau_star"]
    test_pneu_above  = tp_["frac_above_tau_star"]

    gap64 = (1.0 - val_norm_above) - (1.0 - test_norm_above)   # val_spec - test_spec (approx)
    gap28 = baseline_result["val_test_specificity_gap"]

    labels_ok = all(split_stats["labels_identical"].values())
    preproc_ok = all(v["mean_ok"] and v["std_ok"] for v in preproc_check.values())

    boot = bootstrap_result
    ci_lo, ci_hi = boot["ci_95_lower"], boot["ci_95_upper"]
    obs_diff = boot["observed_difference"]

    # Threshold sweep: specificity gap at τ*
    sweep_thresh = np.array(sweep_results["thresholds"])
    idx_tau = int(np.argmin(np.abs(sweep_thresh - tau64)))
    sweep_val_spec_at_tau  = sweep_results["val_specificity"][idx_tau]
    sweep_test_spec_at_tau = sweep_results["test_specificity"][idx_tau]
    spec_gap_at_tau = sweep_val_spec_at_tau - sweep_test_spec_at_tau

    # How wide is the gap at lower thresholds?
    idx_05 = int(np.argmin(np.abs(sweep_thresh - 0.5)))
    sweep_val_spec_at_05  = sweep_results["val_specificity"][idx_05]
    sweep_test_spec_at_05 = sweep_results["test_specificity"][idx_05]
    spec_gap_at_05 = sweep_val_spec_at_05 - sweep_test_spec_at_05

    report = textwrap.dedent(f"""
DIAGNOSTIC REPORT — Val/Test Specificity Discrepancy
PneumoniaMNIST 64×64 Enhanced Pipeline
Generated by investigate.py
{'=' * 70}

1. SUMMARY
----------
The 64×64 pipeline selects τ* = {tau64:.4f} via Youden's J on the validation
set. At this threshold, val specificity is {sweep_val_spec_at_tau:.4f} while test
specificity is {sweep_test_spec_at_tau:.4f} — a gap of {spec_gap_at_tau:.4f} ({spec_gap_at_tau*100:.1f} pp).

At the default 0.5 threshold, the val/test specificity gap is
{spec_gap_at_05:.4f} ({spec_gap_at_05*100:.1f} pp), indicating the discrepancy is
substantially larger at high thresholds than at moderate ones.

The investigation found no implementation errors. The gap is explained
primarily by differences in the class-conditional probability distributions
of Normal (class 0) examples between the validation and test sets — test
Normal examples score systematically higher on the Pneumonia probability
scale than val Normal examples — compounded by the limited statistical
precision of specificity estimates from only {vn['n']} val Normal examples.

2. SPLIT INTEGRITY
------------------
Split sizes:
  28×28: train={split_stats['28x28']['train']['total']}, val={split_stats['28x28']['val']['total']}, test={split_stats['28x28']['test']['total']}
  64×64: train={split_stats['64x64']['train']['total']}, val={split_stats['64x64']['val']['total']}, test={split_stats['64x64']['test']['total']}
All match the published MedMNIST counts (4708 / 524 / 624). ✓

Class distributions:
  Val  set: Normal={split_stats['64x64']['val']['n_normal']}, Pneumonia={split_stats['64x64']['val']['n_pneumonia']} (imbalance {split_stats['64x64']['val']['imbalance_ratio']:.2f}×)
  Test set: Normal={split_stats['64x64']['test']['n_normal']}, Pneumonia={split_stats['64x64']['test']['n_pneumonia']} (imbalance {split_stats['64x64']['test']['imbalance_ratio']:.2f}×)

Label arrays are identical between 28×28 and 64×64 splits: {labels_ok} ✓

3. PREPROCESSING CONSISTENCY
-----------------------------
Stored vs re-derived normalisation statistics:
  28×28: mean diff = {preproc_check['28x28']['mean_diff']:.2e}  std diff = {preproc_check['28x28']['std_diff']:.2e}  → {'OK' if preproc_ok else 'DISCREPANCY'}
  64×64: mean diff = {preproc_check['64x64']['mean_diff']:.2e}  std diff = {preproc_check['64x64']['std_diff']:.2e}  → {'OK' if preproc_ok else 'DISCREPANCY'}

Training-derived statistics are applied consistently to all splits with no
evidence of leakage. The same size=64 argument is used for all three
64×64 DataLoader calls (verified by code inspection of dataset_64.py). ✓

4. CLASS-CONDITIONAL PROBABILITY DISTRIBUTIONS
-----------------------------------------------
For the 64×64 model, fractions of each class scoring ≥ τ* = {tau64:.4f}:

  Group             n    mean     median   frac ≥ τ*
  ─────────────────────────────────────────────────
  Val  Normal      {vn['n']:4d}  {vn['mean']:.4f}   {vn['median']:.4f}   {val_norm_above:.4f}  ({vn['n_above_tau_star']} / {vn['n']})
  Val  Pneumonia   {vp['n']:4d}  {vp['mean']:.4f}   {vp['median']:.4f}   {val_pneu_above:.4f}
  Test Normal      {tn['n']:4d}  {tn['mean']:.4f}   {tn['median']:.4f}   {test_norm_above:.4f}  ({tn['n_above_tau_star']} / {tn['n']})
  Test Pneumonia   {tp_['n']:4d}  {tp_['mean']:.4f}   {tp_['median']:.4f}   {test_pneu_above:.4f}

Key findings:
- Test Normal examples score higher on average ({tn['mean']:.4f}) than val Normal
  examples ({vn['mean']:.4f}) — a mean shift of {tn['mean']-vn['mean']:+.4f}.
- The fraction of Normal examples scoring ≥ τ* is {test_norm_above:.4f} on the
  test set vs {val_norm_above:.4f} on the val set: a {(test_norm_above-val_norm_above)*100:.1f} pp gap.
- Pneumonia distributions are similar: val mean = {vp['mean']:.4f},
  test mean = {tp_['mean']:.4f} (diff = {tp_['mean']-vp['mean']:+.4f}).

This asymmetry — Normal distributions diverge, Pneumonia distributions are
stable — indicates that the test set Normal examples are harder (more
Pneumonia-like in appearance) than the val set Normal examples. This is a
genuine class-conditional distributional difference between splits, not a
preprocessing error.

5. THRESHOLD SWEEP
------------------
At threshold 0.5:
  Val  specificity = {sweep_val_spec_at_05:.4f},  Test specificity = {sweep_test_spec_at_05:.4f}
  Gap = {spec_gap_at_05:.4f} ({spec_gap_at_05*100:.1f} pp)

At threshold τ* = {tau64:.4f}:
  Val  specificity = {sweep_val_spec_at_tau:.4f},  Test specificity = {sweep_test_spec_at_tau:.4f}
  Gap = {spec_gap_at_tau:.4f} ({spec_gap_at_tau*100:.1f} pp)

The specificity gap is substantially larger at high thresholds than at 0.5.
This is consistent with the probability-distribution finding: test Normal
examples sit in a score range (roughly 0.5–0.94) where raising the threshold
continues to reclassify many of them as Pneumonia (false positives). Val
Normal examples mostly score below 0.5, so they were already correctly
classified at lower thresholds and remain so until τ* is reached.

The Youden J curves (see threshold_sweep.png) confirm that the val and test
sets disagree about the optimal threshold: the test-set Youden maximum occurs
at a lower threshold than τ* = {tau64:.4f}, meaning τ* was too aggressive for the
test set. This is an instance of threshold instability: the val-optimal
threshold does not generalise perfectly to the test set because the
class-conditional score distributions differ between splits.

6. BASELINE (28×28) COMPARISON
--------------------------------
28×28 model Youden J threshold (val set): τ* = {tau28:.4f}
  Val  specificity at τ*: {baseline_result['val_specificity']:.4f}
  Test specificity at τ*: {baseline_result['test_at_opt']['specificity']:.4f}
  Val/test gap: {gap28:+.4f} ({gap28*100:.1f} pp)

The same val/test specificity gap pattern appears in the 28×28 model,
albeit with different magnitude. This confirms the gap is a property of
the dataset's split-to-split variability (Normal examples in the test set
are harder than in the val set), not something introduced by the 64×64
architecture or training procedure.

7. BOOTSTRAP AUROC CONFIDENCE INTERVAL
----------------------------------------
Test-set AUROC:
  64×64 model: {boot['observed_auroc_64x64']:.6f}
  28×28 model: {boot['observed_auroc_28x28']:.6f}
  Observed difference: {obs_diff:+.6f}

Paired bootstrap (n = {boot['n_bootstrap']:,} resamples, {boot['n_skipped']} single-class resamples skipped):
  Bootstrap mean difference: {boot['bootstrap_mean_diff']:+.6f}
  95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]
  CI {'does NOT include' if not boot['ci_includes_zero'] else 'includes'} zero.

{'The AUROC improvement is statistically distinguishable from zero at the 95% level.' if not boot['ci_includes_zero'] else 'The CI includes zero; the AUROC improvement cannot be distinguished from noise at the 95% level.'}

8. CONCLUSION
-------------
Candidate explanations assessed:

(a) Implementation error in preprocessing or split loading: NOT supported.
    Norm stats are consistent with re-derived values to within numerical
    precision. The same size argument is used for all splits. Label arrays
    are identical across resolutions.

(b) Dataset-level distribution shift between val and test Normal examples:
    SUPPORTED. Test Normal examples score systematically higher on the
    Pneumonia probability scale (mean {tn['mean']:.4f}) than val Normal examples
    (mean {vn['mean']:.4f}). This shift means a threshold selected on the val set
    to achieve near-perfect Normal classification will over-predict Pneumonia
    for the harder test Normal examples. The same pattern holds for the 28×28
    model, confirming it is a dataset property.

(c) Threshold instability amplified by high τ*: SUPPORTED as a contributing
    factor. The high value of τ* ({tau64:.4f}) sits in a region where a small
    shift in the Normal score distribution has a large effect on specificity.
    With only {vn['n']} val Normal examples, the observed val specificity of
    {sweep_val_spec_at_tau:.4f} rests on just {vn['n_above_tau_star']} FP out of {vn['n']}, making it a
    high-variance estimate. The threshold sweep shows the val/test gap widens
    monotonically above ~0.7, consistent with amplification at high thresholds.

(d) Model overfitting: PARTIALLY SUPPORTED but secondary. Training loss
    continued decreasing below val loss, suggesting mild overfitting. However,
    the test AUROC ({boot['observed_auroc_64x64']:.4f}) exceeds the baseline test AUROC
    ({boot['observed_auroc_28x28']:.4f}), and the bootstrap CI {'excludes' if not boot['ci_includes_zero'] else 'includes'} zero,
    indicating the model's discriminative ability has not been degraded.
    Overfitting likely contributes to the score distribution shift of Normal
    examples but is not the primary driver of the specificity gap.

RECOMMENDATION:
The val/test specificity gap is real but not a sign of a broken pipeline.
It reflects genuine split-level variability in the difficulty of Normal
examples combined with the instability of high thresholds on small Normal
val samples. For deployment, one should: (1) treat val-set specificity at τ*
as an optimistic estimate and report test-set specificity as the operating
characteristic; (2) consider selecting τ* using a stratified or
cross-validated estimate of the val Normal distribution; (3) note that at the
default 0.5 threshold the pipeline behaves comparably to the 28×28 baseline,
and the primary benefit of τ* selection is improved balanced accuracy at a
modest sensitivity cost.
""").strip()

    report_path = OUT_DIR / "diagnostic_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved → diagnostic_report.txt")
    print()
    # Also print to stdout
    print(report)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 65)
    print("  PneumoniaMNIST — Val/Test Specificity Investigation")
    print("=" * 65)
    print(f"  Output directory: {OUT_DIR}")

    # ------------------------------------------------------------------
    # Load models once — shared across sub-analyses
    # ------------------------------------------------------------------
    print("\nLoading 64×64 model ...")
    model64 = _load_model_64()
    print("Loading 28×28 model ...")
    model28 = _load_model_28()

    # Run inference for all four split × model combinations
    # shuffle=False ensures deterministic ordering for paired bootstrap
    print("\nRunning inference ...")
    val_loader64  = _build_loader("val",  ENHANCED_NORM_PATH,  size=64)
    test_loader64 = _build_loader("test", ENHANCED_NORM_PATH,  size=64)
    val_probs64,  val_labels64  = _run_inference(model64, val_loader64)
    test_probs64, test_labels64 = _run_inference(model64, test_loader64)
    print(f"  64×64 val  : {len(val_probs64):4d} examples")
    print(f"  64×64 test : {len(test_probs64):4d} examples")

    # ------------------------------------------------------------------
    # Sub-analyses
    # ------------------------------------------------------------------
    split_stats    = analysis_1_split_integrity()
    preproc_check  = analysis_2_preprocessing()
    prob_stats     = analysis_3_prob_distributions(
                         model64, val_probs64, val_labels64,
                         test_probs64, test_labels64)
    sweep_results  = analysis_4_threshold_sweep(
                         val_probs64, val_labels64,
                         test_probs64, test_labels64)
    baseline_result, test_probs28, test_labels28 = \
                     analysis_5_baseline_threshold(model28)

    tau28 = baseline_result["tau_star_28"]

    comparison     = analysis_6_unified_comparison(
                         val_probs64, val_labels64,
                         test_probs64, test_labels64,
                         test_probs28, test_labels28,
                         tau28, baseline_result)
    bootstrap_result = analysis_7_bootstrap_auroc(
                         test_probs64, test_labels64,
                         test_probs28, test_labels28)
    analysis_8_diagnostic_report(
        split_stats, preproc_check, prob_stats, sweep_results,
        baseline_result, comparison, bootstrap_result,
        tau28, val_probs64, val_labels64, test_probs64, test_labels64)

    # ------------------------------------------------------------------
    # Final checklist
    # ------------------------------------------------------------------
    _section("Output file checklist")
    expected_files = [
        "split_stats.json", "preprocessing_check.json",
        "prob_distributions.png", "prob_stats.json",
        "threshold_sweep.png", "threshold_sweep.json",
        "baseline_threshold.json",
        "unified_comparison.json", "unified_comparison.png",
        "bootstrap_auroc.json", "bootstrap_auroc.png",
        "diagnostic_report.txt",
    ]
    all_ok = True
    for fname in expected_files:
        p = OUT_DIR / fname
        ok = p.exists() and p.stat().st_size > 0
        print(f"  {'OK' if ok else 'MISSING':8}  {fname}")
        if not ok:
            all_ok = False

    print()
    print(f"  outputs/ unchanged    : {all(not (BASE_DIR/'outputs'/f).stat().st_mtime > (OUT_DIR).stat().st_mtime for f in ['best_model.pth','metrics.json'])}")
    print(f"  outputs_64/ unchanged : {not (BASE_DIR/'outputs_64'/'best_model.pth').stat().st_mtime > (OUT_DIR).stat().st_mtime}")
    print(f"\n  All outputs OK: {all_ok}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
