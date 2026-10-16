"""
config_64.py — Configuration for the 64×64 enhanced PneumoniaMNIST pipeline.

Mirrors config.py but targets native 64×64 PneumoniaMNIST+ images, a deeper CNN,
early stopping, and a richer evaluation suite.

All artifacts are saved under outputs_64/ so the 28×28 baseline is never touched.

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

# Separate output directory — the 28×28 outputs/ is never written to
OUTPUT_DIR: pathlib.Path = BASE_DIR / "outputs_64"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Core model artifacts
MODEL_PATH:       pathlib.Path = OUTPUT_DIR / "best_model.pth"
NORM_STATS_PATH:  pathlib.Path = OUTPUT_DIR / "norm_stats.json"
METRICS_PATH:     pathlib.Path = OUTPUT_DIR / "metrics.json"
THRESHOLD_PATH:   pathlib.Path = OUTPUT_DIR / "threshold.json"

# Plots
LOSS_CURVE_PATH:           pathlib.Path = OUTPUT_DIR / "loss_curve.png"
ROC_CURVE_PATH:            pathlib.Path = OUTPUT_DIR / "roc_curve.png"
PR_CURVE_PATH:             pathlib.Path = OUTPUT_DIR / "pr_curve.png"
CONFUSION_MATRIX_05_PATH:  pathlib.Path = OUTPUT_DIR / "confusion_matrix_05.png"
CONFUSION_MATRIX_OPT_PATH: pathlib.Path = OUTPUT_DIR / "confusion_matrix_opt.png"
SAMPLE_GRID_PATH:          pathlib.Path = OUTPUT_DIR / "sample_grid.png"

# Comparison artifacts (written by compare.py)
COMPARISON_PATH:      pathlib.Path = OUTPUT_DIR / "comparison.json"
COMPARISON_PLOT_PATH: pathlib.Path = OUTPUT_DIR / "comparison.png"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_FLAG:   str  = "pneumoniamnist"   # MedMNIST dataset identifier
IMAGE_SIZE:  int  = 64                 # native 64×64 (MedMNIST+ variant)
NUM_CLASSES: int  = 2                  # Normal (0) vs Pneumonia (1)
CLASS_NAMES: list = ["Normal", "Pneumonia"]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE:    int   = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY:  float = 1e-4
NUM_EPOCHS:    int   = 30      # more headroom so early stopping can act

# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------
DROPOUT_RATE: float = 0.4

# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
EARLY_STOP_PATIENCE: int = 5   # stop if val loss does not improve for 5 epochs

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device              : {DEVICE}")
    print(f"Output dir          : {OUTPUT_DIR}  (exists={OUTPUT_DIR.exists()})")
    print(f"Seed                : {SEED}")
    print(f"Image size          : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Learning rate       : {LEARNING_RATE}")
    print(f"Weight decay        : {WEIGHT_DECAY}")
    print(f"Epochs (max)        : {NUM_EPOCHS}")
    print(f"Dropout rate        : {DROPOUT_RATE}")
    print(f"Early stop patience : {EARLY_STOP_PATIENCE}")
    print(f"Classes             : {CLASS_NAMES}")
    print()
    print("Artifact paths:")
    for name, path in [
        ("MODEL_PATH",              MODEL_PATH),
        ("NORM_STATS_PATH",         NORM_STATS_PATH),
        ("METRICS_PATH",            METRICS_PATH),
        ("THRESHOLD_PATH",          THRESHOLD_PATH),
        ("LOSS_CURVE_PATH",         LOSS_CURVE_PATH),
        ("ROC_CURVE_PATH",          ROC_CURVE_PATH),
        ("PR_CURVE_PATH",           PR_CURVE_PATH),
        ("CONFUSION_MATRIX_05",     CONFUSION_MATRIX_05_PATH),
        ("CONFUSION_MATRIX_OPT",    CONFUSION_MATRIX_OPT_PATH),
        ("SAMPLE_GRID_PATH",        SAMPLE_GRID_PATH),
        ("COMPARISON_PATH",         COMPARISON_PATH),
        ("COMPARISON_PLOT_PATH",    COMPARISON_PLOT_PATH),
    ]:
        print(f"  {name:<26} {path}")
    print()
    print("config_64.py OK")
