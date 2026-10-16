"""
train.py — Training loop for the PneumoniaMNIST CNN classifier.

Key design:
- Seeds all RNG sources (Python, NumPy, PyTorch) at startup for reproducibility.
- Uses BCEWithLogitsLoss + Adam optimiser.
- Saves best_model.pth whenever validation loss improves (model selection on val loss).
- Logs per-epoch train loss, val loss, and val accuracy to stdout.
- Saves outputs/loss_curve.png after training completes.

Run:
    python train.py

Educational/research use only. Not for clinical diagnosis.
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import get_dataloaders
from model import get_model, count_parameters


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN ops (no effect on CPU, but good practice)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training / validation step helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one full pass over the training set. Returns mean loss."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        # labels from MedMNIST are shape (B, 1) with dtype int64 — cast to float
        labels = labels.to(device).float()

        optimiser.zero_grad()
        logits = model(images)          # (B, 1)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run inference over a data split.

    Returns
    -------
    mean_loss : float
    accuracy  : float  (0–1)
    """
    model.eval()
    running_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()

    n = len(loader.dataset)
    return running_loss / n, correct / n


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_loss_curve(
    train_losses: list,
    val_losses: list,
    path: str = str(config.LOSS_CURVE_PATH),
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", marker="o", markersize=4)
    ax.plot(epochs, val_losses,   label="Val loss",   marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training and Validation Loss — PneumoniaMNIST CNN")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved → {path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    set_seeds()

    print("=" * 60)
    print("PneumoniaMNIST CNN — Training")
    print("=" * 60)
    print(f"Device  : {config.DEVICE}")
    print(f"Seed    : {config.SEED}")
    print(f"Epochs  : {config.NUM_EPOCHS}")
    print(f"Batch   : {config.BATCH_SIZE}")
    print(f"LR      : {config.LEARNING_RATE}")
    print()

    # --- Data ---
    train_loader, val_loader, _ = get_dataloaders(batch_size=config.BATCH_SIZE)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}\n")

    # --- Model ---
    model = get_model(seed=config.SEED, device=config.DEVICE)
    print(f"Model params  : {count_parameters(model):,}\n")

    # --- Loss and optimiser ---
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # --- Training loop ---
    train_losses: list = []
    val_losses: list = []
    best_val_loss = math.inf
    best_epoch = 0

    header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val Acc':>8}  {'':>6}"
    print(header)
    print("-" * len(header))

    t_start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, config.DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Sanity check: losses must be finite
        assert math.isfinite(train_loss), f"Epoch {epoch}: train loss is not finite ({train_loss})"
        assert math.isfinite(val_loss),   f"Epoch {epoch}: val loss is not finite ({val_loss})"

        # Model selection: save checkpoint on val loss improvement
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_PATH)
            improved = "✓ saved"

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}  {improved}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best checkpoint : epoch {best_epoch}  val_loss={best_val_loss:.4f}")
    print(f"Model saved     → {config.MODEL_PATH}")

    # --- Save loss curve ---
    save_loss_curve(train_losses, val_losses)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
