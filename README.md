# PneumoniaMNIST Diagnostic Classifier

A pair of lightweight CNNs trained on [PneumoniaMNIST](https://medmnist.com/) to
classify chest X-ray images as **Normal** or **Pneumonia**.

This repository contains two end-to-end pipelines:

| Pipeline | Images | Architecture | Training |
|----------|--------|-------------|---------|
| **28×28 Baseline** | 28×28 grayscale | 2 conv blocks | 15 epochs, fixed |
| **64×64 Enhanced** | 64×64 grayscale (MedMNIST+) | 3 conv blocks + AdaptiveAvgPool | Up to 30 epochs, early stopping |

Built for an AWS training workshop as an educational demonstration of end-to-end
medical image classification with PyTorch.

> ⚠️ **Disclaimer:** This project is for **educational and research demonstration
> only**. It must **not** be used for clinical diagnosis or any medical
> decision-making. Always consult a qualified healthcare professional for medical
> advice.

---

## Dataset

**PneumoniaMNIST** is derived from the Guangzhou Women and Children's Medical
Center chest X-ray dataset. Images are available at 28×28 (standard) and 64×64
(MedMNIST+ native resolution). Splits are identical across both resolutions.

| Split | Size |
|-------|-----:|
| Train | 4708 |
| Val   |  524 |
| Test  |  624 |

- **Class 0 — Normal**
- **Class 1 — Pneumonia**

Datasets are downloaded automatically on first run via the `medmnist` Python package.

---

## Project Structure

```
medical_image_classifier_64/
│
├── README.md
├── requirements.txt
│
├── ── 28×28 BASELINE (original scripts — do not modify) ──
├── config.py           # Hyperparameters, paths (28×28)
├── dataset.py          # Data loading, normalization (28×28)
├── model.py            # PneumoniaCNN — 2-block CNN
├── train.py            # Training loop (28×28, 15 epochs)
├── evaluate.py         # Test set evaluation (28×28)
│
├── ── 64×64 ENHANCED PIPELINE ──
├── config_64.py        # Hyperparameters, paths (64×64)
├── dataset_64.py       # Data loading, normalization (64×64, size=64)
├── model_64.py         # PneumoniaCNN64 — 3-block CNN + AdaptiveAvgPool
├── train_64.py         # Training loop with early stopping
├── evaluate_64.py      # Evaluation: dual-threshold, 8 metrics, 4 plots
├── compare.py          # Side-by-side comparison of both pipelines
│
├── app.py              # Streamlit UI — model selector (28×28 / 64×64)
│
├── outputs/            # 28×28 baseline artifacts (never overwritten)
│   ├── best_model.pth
│   ├── norm_stats.json
│   ├── metrics.json
│   ├── loss_curve.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── sample_grid.png
│
└── outputs_64/         # 64×64 pipeline artifacts
    ├── best_model.pth
    ├── norm_stats.json
    ├── threshold.json          # τ* (Youden J, val set) + val sens/spec
    ├── metrics.json            # 8 metrics at both thresholds
    ├── loss_curve.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── confusion_matrix_05.png
    ├── confusion_matrix_opt.png
    ├── sample_grid.png
    ├── comparison.json
    └── comparison.png
```

---

## Setup

Requires Python 3.9+.

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the 28×28 Baseline Pipeline

### 1. Train

```bash
python train.py
```

- Downloads PneumoniaMNIST (28×28) on first run
- Computes training-set normalisation → `outputs/norm_stats.json`
- Trains for 15 epochs, saves best checkpoint → `outputs/best_model.pth`
- Saves loss curve → `outputs/loss_curve.png`

### 2. Evaluate

```bash
python evaluate.py
```

- Loads `outputs/best_model.pth`
- Evaluates on the held-out test set
- Saves metrics → `outputs/metrics.json`
- Saves ROC curve → `outputs/roc_curve.png`
- Saves confusion matrix → `outputs/confusion_matrix.png`

---

## Running the 64×64 Enhanced Pipeline

### 1. Train

```bash
python train_64.py
```

- Downloads PneumoniaMNIST+ at native 64×64 resolution (`size=64`)
- Computes training-set normalisation → `outputs_64/norm_stats.json`
- Trains up to 30 epochs with **early stopping** (patience = 5)
- Saves best checkpoint → `outputs_64/best_model.pth`
- Saves annotated loss curve → `outputs_64/loss_curve.png`

### 2. Evaluate

```bash
python evaluate_64.py
```

- **Stage 1 — Threshold selection (validation set only):**
  Computes Youden's J across all ROC thresholds on the validation split.
  Selects τ* that maximises J (nonfinite thresholds excluded).
  Saves τ* with validation sensitivity and specificity →
  `outputs_64/threshold.json`.
- **Stage 2 — Test set evaluation:**
  Reports all eight metrics at both threshold = 0.5 and τ*.

Metrics: AUROC · AUPRC · Accuracy · Balanced Accuracy ·
         Sensitivity · Specificity · Precision · F1-score

Saves:
- `outputs_64/metrics.json`
- `outputs_64/roc_curve.png`        (★ marker at τ* operating point)
- `outputs_64/pr_curve.png`         (★ marker at τ* operating point)
- `outputs_64/confusion_matrix_05.png`
- `outputs_64/confusion_matrix_opt.png`

### 3. Generate comparison report

```bash
python compare.py
```

- Loads `outputs/metrics.json` (baseline) and `outputs_64/metrics.json`
- Recomputes missing baseline metrics (AUPRC, balanced accuracy) from the
  saved 28×28 checkpoint; marks them N/A if the checkpoint is unavailable
- Prints a formatted comparison table
- Saves `outputs_64/comparison.json` and `outputs_64/comparison.png`

> **Note:** Differences between pipelines reflect changes to **image
> resolution, CNN architecture, regularisation, and training procedure
> combined**. Do not attribute results solely to image resolution.

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Streamlit Application

The app supports two models via a **sidebar selector**:

- **28×28 Baseline** — shows prediction at threshold = 0.5
- **64×64 Enhanced Pipeline** — shows predictions at both threshold = 0.5
  and the Youden-optimal τ* (from the validation set)

Two input modes:

- **Select test example** — samples a random image from the test set at the
  selected model's resolution (true label shown)
- **Upload image** — upload any PNG/JPG chest X-ray

The sidebar info card shows the architecture, input size, test AUROC, and
optimal threshold for the selected model.

If a model's artifacts are missing, the app displays a clear error message
instead of crashing.

---

## Model Architectures

### PneumoniaCNN (28×28 Baseline)

```
Input: (B, 1, 28, 28)

Block 1: Conv2d(1→16,  3×3) → BN → ReLU → MaxPool(2×2)   → 14×14
Block 2: Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2×2)   →  7×7

FC head: Flatten → Linear(1568→64) → ReLU → Dropout(0.3) → Linear(64→1)
```

| Property | Value |
|----------|-------|
| Trainable parameters | 105,329 |
| Input size | 28×28 grayscale |
| Dropout | 0.3 |
| Loss function | BCEWithLogitsLoss |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| Training epochs | 15 (fixed) |
| Model selection | Best val loss |

### PneumoniaCNN64 (64×64 Enhanced Pipeline)

```
Input: (B, 1, 64, 64)

Block 1: Conv2d(1→16,  3×3) → BN → ReLU → MaxPool(2×2)        → 32×32
Block 2: Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2×2)        → 16×16
Block 3: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
         AdaptiveAvgPool2d((4,4))                               →  4×4

FC head: Flatten → Linear(1024→128) → ReLU → Dropout(0.4) → Linear(128→1)
```

| Property | Value |
|----------|-------|
| Trainable parameters | ~140,000 |
| Input size | 64×64 grayscale |
| Dropout | 0.4 |
| Loss function | BCEWithLogitsLoss |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| Training epochs | Up to 30 (early stopping, patience=5) |
| Model selection | Best val loss |

`AdaptiveAvgPool2d((4, 4))` decouples the FC input dimension from the spatial
resolution, making the architecture robust to future resolution changes.

---

## Threshold Selection

The 64×64 pipeline selects an operating threshold τ* from the **validation set
only** (test set never used for threshold selection).

Method: Youden's J = Sensitivity + Specificity − 1, maximised over all
finite ROC thresholds. τ* is saved with the validation sensitivity and
specificity at that point.

Results are reported at both threshold = 0.5 and τ* to allow direct comparison
with the baseline.

---

## Results

### 28×28 Baseline (test set, threshold = 0.5)

| Metric | Value |
|--------|------:|
| AUROC | 0.9489 |
| Accuracy | 0.8253 |
| Sensitivity | 0.9974 |
| Specificity | 0.5385 |
| Precision | 0.7827 |
| F1-score | 0.8771 |

Confusion matrix: TN=126 FP=108 FN=1 TP=389

### 64×64 Enhanced Pipeline (test set)

τ* = 0.9418 (Youden J, selected on validation set only)

| Metric | @ 0.5 | @ τ* = 0.9418 |
|--------|------:|--------------:|
| AUROC  | 0.9532 | 0.9532 |
| AUPRC  | 0.9641 | 0.9641 |
| Accuracy | 0.8173 | 0.8702 |
| Balanced Accuracy | 0.7581 | 0.8312 |
| Sensitivity | 0.9949 | 0.9872 |
| Specificity | 0.5214 | 0.6752 |
| Precision | 0.7760 | 0.8351 |
| F1-score | 0.8719 | 0.9048 |

Confusion matrix @ 0.5: TN=122 FP=112 FN=2 TP=388
Confusion matrix @ τ*:  TN=158 FP=76  FN=5 TP=385

Training: 18 epochs, early stopping triggered (patience=5, best epoch 13, val_loss=0.0581)

### Comparison note

Performance differences between the two pipelines reflect changes to **image
resolution, CNN architecture (depth and regularisation), and training
procedure (early stopping)** combined. This comparison is exploratory: the
test set was already examined for the baseline, so it does not provide an
unbiased final model-selection estimate.

---

## Outputs

### outputs/ (28×28 baseline — read-only after training)

| File | Description |
|------|-------------|
| `best_model.pth` | Trained model weights (best val loss) |
| `norm_stats.json` | Training-set mean and std |
| `metrics.json` | 6 test-set metrics |
| `loss_curve.png` | Train/val loss per epoch |
| `roc_curve.png` | ROC curve with AUROC |
| `confusion_matrix.png` | Confusion matrix |
| `sample_grid.png` | Sample training images |

### outputs_64/ (64×64 enhanced pipeline)

| File | Description |
|------|-------------|
| `best_model.pth` | Trained model weights (best val loss) |
| `norm_stats.json` | Training-set mean and std (64×64) |
| `threshold.json` | τ*, val sensitivity and specificity at τ* |
| `metrics.json` | 8 metrics at threshold=0.5 and τ* |
| `loss_curve.png` | Train/val loss with best-epoch marker |
| `roc_curve.png` | ROC curve with AUROC and τ* operating point |
| `pr_curve.png` | Precision-Recall curve with AUPRC and τ* |
| `confusion_matrix_05.png` | Confusion matrix at threshold=0.5 |
| `confusion_matrix_opt.png` | Confusion matrix at τ* |
| `sample_grid.png` | Sample 64×64 training images |
| `comparison.json` | Full comparison table and summary |
| `comparison.png` | Grouped bar chart of all metrics |

---

## Investigation: Val/Test Specificity Discrepancy

After the 64×64 pipeline was evaluated, a concerning discrepancy was observed:
at the Youden-optimal threshold τ* = 0.9418 (selected on the validation set),
validation specificity was 99.3% while test specificity was only 67.5% — a
gap of 31.7 pp.

A diagnostic investigation was run to determine the cause before drawing any
conclusions about model quality.

```bash
python investigate.py
```

All results are saved under `outputs_investigation/`. No existing models,
checkpoints, or evaluation outputs are modified.

### Key findings

| Finding | Result |
|---------|--------|
| Implementation error (preprocessing / split loading) | Not supported |
| Label integrity (28×28 vs 64×64 splits identical) | Confirmed ✓ |
| Norm stats re-derivation agrees with stored values | Confirmed ✓ |
| Distribution shift — Normal score distributions | **Supported** |
| Threshold instability from high τ* | **Supported** (contributing) |
| Model overfitting | Partially supported, secondary |

The primary cause is a **genuine distributional difference between val and
test Normal examples**: val Normals score very low on the Pneumonia scale
(mean 0.07, median 0.001), while test Normals score substantially higher
(mean 0.48, median 0.44). This pattern is present for the 28×28 model too
(val/test specificity gap of 32.5 pp at τ*_28), confirming it is a
**dataset-level property** of the MedMNIST split, not an artifact of the
64×64 architecture or training procedure.

The AUROC improvement (0.9489 → 0.9532) has a 95% paired bootstrap CI of
[−0.0062, +0.0156], which includes zero. The improvement is not
statistically distinguishable from noise at the 95% level.

### Investigation outputs

| File | Description |
|------|-------------|
| `split_stats.json` | Split sizes and class distributions (both models) |
| `preprocessing_check.json` | Norm-stat consistency check |
| `prob_distributions.png` | Class-conditional score histograms, val vs test |
| `prob_stats.json` | Per-group score distribution statistics |
| `threshold_sweep.png` | Specificity/sensitivity/Youden J across all thresholds |
| `threshold_sweep.json` | Numerical threshold sweep data |
| `baseline_threshold.json` | 28×28 Youden J threshold and val/test metrics |
| `unified_comparison.json` | All four operating points side by side |
| `unified_comparison.png` | Grouped bar chart of all four operating points |
| `bootstrap_auroc.json` | Paired bootstrap 95% CI for AUROC difference |
| `bootstrap_auroc.png` | Bootstrap difference histogram |
| `diagnostic_report.txt` | Full written diagnostic narrative |

---

## Reproducibility

All random seeds are fixed via `SEED = 42` in both `config.py` and
`config_64.py`, applied to Python's `random`, NumPy, and PyTorch (including
CuDNN deterministic mode). Re-running either pipeline from scratch will
produce identical results.

---

## License and Attribution

- **Dataset:** PneumoniaMNIST from [MedMNIST v2](https://medmnist.com/) —
  Jiancheng Yang et al., 2023.
- **Original chest X-ray data:** Guangzhou Women and Children's Medical
  Center, licensed for research use.
- **This code:** Educational demonstration for the AWS Training Workshop.
  Not for clinical use.
