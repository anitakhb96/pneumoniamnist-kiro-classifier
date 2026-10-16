# PneumoniaMNIST Diagnostic Classifier

A lightweight CNN trained on the [PneumoniaMNIST](https://medmnist.com/) dataset to classify chest X-ray images as **Normal** or **Pneumonia**. Built for an AWS training workshop as an educational demonstration of end-to-end medical image classification with PyTorch.

> ⚠️ **Disclaimer:** This project is for **educational and research demonstration only**. It must **not** be used for clinical diagnosis or any medical decision-making. Always consult a qualified healthcare professional for medical advice.

---

## Project Overview

This project covers the full machine learning pipeline:

1. Automatic dataset download and preprocessing (MedMNIST)
2. Training-set–only normalization (no data leakage)
3. CNN training with validation-based model selection
4. Evaluation on the held-out test set with clinical metrics
5. Diagnostic plots: loss curves, ROC curve, confusion matrix
6. Interactive Streamlit inference UI

---

## Dataset

**PneumoniaMNIST** is derived from the Guangzhou Women and Children's Medical Center chest X-ray dataset. Images are resized to 28×28 grayscale and split into:

| Split | Size |
|-------|-----:|
| Train | 4708 |
| Val   |  524 |
| Test  |  624 |

- **Class 0 — Normal**
- **Class 1 — Pneumonia**

The dataset is downloaded automatically on first run via the `medmnist` Python package.

---

## Project Structure

```
medical_image_classifier/
├── README.md
├── requirements.txt
├── config.py          # Hyperparameters, paths, seeds
├── dataset.py         # Data loading, training-set normalization, dataloaders
├── model.py           # CNN architecture (PneumoniaCNN)
├── train.py           # Training loop, val-based model selection, loss curve
├── evaluate.py        # Test set evaluation, metrics, ROC curve, confusion matrix
├── app.py             # Streamlit inference UI
└── outputs/           # Auto-created — all saved artifacts
    ├── best_model.pth
    ├── norm_stats.json
    ├── metrics.json
    ├── loss_curve.png
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── sample_grid.png
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

## Usage

Run the scripts in order:

### 1. Train the model

```bash
python train.py
```

- Downloads PneumoniaMNIST on first run
- Computes training-set normalization statistics → `outputs/norm_stats.json`
- Trains for 15 epochs, saves best checkpoint → `outputs/best_model.pth`
- Saves training/validation loss curve → `outputs/loss_curve.png`

### 2. Evaluate on the test set

```bash
python evaluate.py
```

- Loads `best_model.pth`
- Runs inference on the held-out test set only
- Saves metrics → `outputs/metrics.json`
- Saves ROC curve → `outputs/roc_curve.png`
- Saves confusion matrix → `outputs/confusion_matrix.png`

### 3. Launch the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Two input modes are available:

- **Select test example** — sample a random image from the test set (true label shown)
- **Upload image** — upload any PNG/JPG chest X-ray (resized to 28×28 grayscale)

The app displays the predicted class and raw Pneumonia probability.

---

## Model Architecture

**PneumoniaCNN** — two convolutional blocks followed by a fully-connected head.

```
Input: (B, 1, 28, 28) grayscale

Block 1: Conv2d(1→16, 3×3) → BatchNorm2d → ReLU → MaxPool(2×2)   [→ 14×14]
Block 2: Conv2d(16→32, 3×3) → BatchNorm2d → ReLU → MaxPool(2×2)  [→ 7×7]

FC head: Flatten → Linear(1568→64) → ReLU → Dropout(0.3) → Linear(64→1)

Output: single logit per image
        → torch.sigmoid() → Pneumonia probability ∈ [0, 1]
```

| Property | Value |
|----------|-------|
| Trainable parameters | 105,329 |
| Input size | 28×28 grayscale |
| Loss function | BCEWithLogitsLoss |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| Training epochs | 15 |
| Model selection | Best val loss |

---

## Results

Evaluated on the held-out test set (624 images). The model is **not** tuned on the test set.

### Metrics

| Metric | Value |
|--------|------:|
| Accuracy | 0.8253 |
| AUROC | 0.9489 |
| Precision | 0.7827 |
| Recall (Sensitivity) | 0.9974 |
| Specificity | 0.5385 |
| F1-score | 0.8771 |

AUROC is computed from predicted probabilities (continuous scores), not binary predictions.

### Confusion Matrix (test set)

|  | Predicted Normal | Predicted Pneumonia |
|--|:-:|:-:|
| **True Normal** | 126 (TN) | 108 (FP) |
| **True Pneumonia** | 1 (FN) | 389 (TP) |

The model achieves very high recall (99.7%) — it misses almost no pneumonia cases — at the cost of moderate specificity (53.8%), meaning it over-predicts pneumonia on normal cases. This asymmetry is expected with a simple CNN on a small dataset.

---

## Outputs

After running `train.py` and `evaluate.py`, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `best_model.pth` | Trained model weights (best validation loss) |
| `norm_stats.json` | Training-set mean and std for normalization |
| `metrics.json` | All test-set metrics in JSON format |
| `loss_curve.png` | Training and validation loss per epoch |
| `roc_curve.png` | ROC curve with AUROC annotation |
| `confusion_matrix.png` | Confusion matrix with class labels |
| `sample_grid.png` | Grid of sample training images |

---

## Reproducibility

All random seeds are fixed via `SEED = 42` in `config.py`, applied to Python's `random`, NumPy, and PyTorch (including CuDNN deterministic mode). Re-running `train.py` from scratch will produce identical results.

---

## License and Attribution

- **Dataset:** PneumoniaMNIST from [MedMNIST v2](https://medmnist.com/) — Jiancheng Yang et al., 2023.
- **Original chest X-ray data:** Guangzhou Women and Children's Medical Center, licensed for research use.
- **This code:** Educational demonstration for the AWS Training Workshop. Not for clinical use.
