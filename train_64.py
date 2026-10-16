"""
train_64.py — Training loop for the 64×64 enhanced PneumoniaMNIST pipeline.

Key design decisions:
- Uses PneumoniaCNN64 on native 64×64 MedMNIST+ images.
- BCEWithLogitsLoss + Adam optimiser (same as baseline).
- Early stopping on validation loss (patience = config_64.EARLY_STOP_PATIENCE).
  Saves best_model.pth whenever val loss improves; stops as soon as patience
  is exhausted to reduce overfitting.
- Loss curve annotates the best (saved) epoch with a vertical dashed line.
- All artifacts written to outputs_64/; outputs/ (28×28 baseline) is never touched.

Run:
    python train_64.py

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

import config_64
from dataset_64 import get_dataloaders_64
from model_64 import get_model_64, count_parameters


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = config_64.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        labels = labels.to(device).float()   # MedMNIST labels: (B,1) int64 → float

        optimiser.zero_grad()
        logits = model(images)               # (B, 1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_acc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Compute mean BCE loss and accuracy over a data split.

    Returns
    -------
    mean_loss : float
    accuracy  : float (0–1)
    """
    model.eval()
    running_loss = 0.0
    correct      = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()

        logits = model(images)
        loss   = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        preds   = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()

    n = len(loader.dataset)
    return running_loss / n, correct / n


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def save_loss_curve(
    train_losses: list,
    val_losses:   list,
    best_epoch:   int,
    stopped_early: bool,
    path: str = str(config_64.LOSS_CURVE_PATH),
) -> None:
    """
    Save training and validation loss curves.

    A vertical dashed line marks the best (saved) epoch.
    The title notes whether training completed or stopped early.
    """
    epochs = range(1, len(train_losses) + 1)
    max_ep = config_64.NUM_EPOCHS
    title  = (
        f"Training and Validation Loss — 64×64 Enhanced Pipeline\n"
        f"({'Early stopped' if stopped_early else f'Completed {len(train_losses)}/{max_ep} epochs'})"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train loss", marker="o", markersize=4)
    ax.plot(epochs, val_losses,   label="Val loss",   marker="s", markersize=4)
    ax.axvline(
        x=best_epoch, color="green", linestyle="--", linewidth=1.5,
        label=f"Best epoch ({best_epoch})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title(title)
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

    print("=" * 65)
    print("PneumoniaMNIST+ (64×64) — Enhanced Pipeline Training")
    print("=" * 65)
    print(f"Device              : {config_64.DEVICE}")
    print(f"Seed                : {config_64.SEED}")
    print(f"Max epochs          : {config_64.NUM_EPOCHS}")
    print(f"Early stop patience : {config_64.EARLY_STOP_PATIENCE}")
    print(f"Batch size          : {config_64.BATCH_SIZE}")
    print(f"Learning rate       : {config_64.LEARNING_RATE}")
    print(f"Weight decay        : {config_64.WEIGHT_DECAY}")
    print(f"Dropout rate        : {config_64.DROPOUT_RATE}")
    print()

    # --- Data ---
    train_loader, val_loader, _ = get_dataloaders_64(batch_size=config_64.BATCH_SIZE)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}\n")

    # --- Model ---
    model = get_model_64(seed=config_64.SEED, device=config_64.DEVICE)
    print(f"Model         : PneumoniaCNN64")
    print(f"Parameters    : {count_parameters(model):,}\n")

    # --- Loss and optimiser ---
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config_64.LEARNING_RATE,
        weight_decay=config_64.WEIGHT_DECAY,
    )

    # --- Training loop with early stopping ---
    train_losses:    list = []
    val_losses:      list = []
    best_val_loss  = math.inf
    best_epoch     = 0
    patience_counter = 0
    stopped_early  = False

    header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val Acc':>8}  {'':>12}"
    print(header)
    print("-" * len(header))

    t_start = time.time()

    for epoch in range(1, config_64.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, config_64.DEVICE)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, config_64.DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        assert math.isfinite(train_loss), f"Epoch {epoch}: train loss not finite ({train_loss})"
        assert math.isfinite(val_loss),   f"Epoch {epoch}: val loss not finite ({val_loss})"

        # Model selection
        note = ""
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), config_64.MODEL_PATH)
            note = "✓ saved"
        else:
            patience_counter += 1
            note = f"patience {patience_counter}/{config_64.EARLY_STOP_PATIENCE}"

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}  {note}")

        # Early stopping check
        if patience_counter >= config_64.EARLY_STOP_PATIENCE:
            stopped_early = True
            print(f"\nEarly stopping triggered at epoch {epoch}. "
                  f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s")
    if not stopped_early:
        print(f"Completed all {config_64.NUM_EPOCHS} epochs.")
    print(f"Best checkpoint : epoch {best_epoch}  val_loss={best_val_loss:.4f}")
    print(f"Model saved     → {config_64.MODEL_PATH}")

    # --- Save loss curve ---
    save_loss_curve(train_losses, val_losses, best_epoch, stopped_early)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
