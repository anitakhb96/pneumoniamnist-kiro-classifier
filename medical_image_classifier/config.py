"""
config.py — Central configuration for the PneumoniaMNIST Classifier.

All hyperparameters, paths, and reproducibility settings live here.
Import this module at the top of every other script.

Educational/research use only. Not for clinical diagnosis.
"""

import pathlib
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
OUTPUT_DIR: pathlib.Path = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH: pathlib.Path = OUTPUT_DIR / "best_model.pth"
NORM_STATS_PATH: pathlib.Path = OUTPUT_DIR / "norm_stats.json"
METRICS_PATH: pathlib.Path = OUTPUT_DIR / "metrics.json"
LOSS_CURVE_PATH: pathlib.Path = OUTPUT_DIR / "loss_curve.png"
ROC_CURVE_PATH: pathlib.Path = OUTPUT_DIR / "roc_curve.png"
CONFUSION_MATRIX_PATH: pathlib.Path = OUTPUT_DIR / "confusion_matrix.png"
SAMPLE_GRID_PATH: pathlib.Path = OUTPUT_DIR / "sample_grid.png"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_FLAG: str = "pneumoniamnist"       # MedMNIST dataset identifier
IMAGE_SIZE: int = 28                    # 28×28 grayscale input
NUM_CLASSES: int = 2                    # Normal (0) vs Pneumonia (1)
CLASS_NAMES: list = ["Normal", "Pneumonia"]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 15
WEIGHT_DECAY: float = 1e-4

# ---------------------------------------------------------------------------
# Device — use GPU if available, otherwise CPU
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device        : {DEVICE}")
    print(f"Output dir    : {OUTPUT_DIR}  (exists={OUTPUT_DIR.exists()})")
    print(f"Seed          : {SEED}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Epochs        : {NUM_EPOCHS}")
    print(f"Weight decay  : {WEIGHT_DECAY}")
    print(f"Classes       : {CLASS_NAMES}")
    print("config.py OK")
