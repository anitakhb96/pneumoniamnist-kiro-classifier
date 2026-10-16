"""
dataset_64.py — Data loading and preprocessing for 64×64 PneumoniaMNIST+.

Key design decisions:
- Uses PneumoniaMNIST with size=64 (MedMNIST+ native resolution).
  Images are NOT upscaled from 28×28; the higher-resolution variant is
  downloaded directly from MedMNIST.
- Normalization statistics are computed from the TRAINING set only to
  prevent data leakage into validation and test sets.
- Stats are saved to outputs_64/norm_stats.json for inference scripts.
- An assertion on the first training batch confirms size=64 was honoured
  (guards against MedMNIST falling back to 28×28).

Educational/research use only. Not for clinical diagnosis.
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_64


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_train_stats() -> tuple:
    """
    Compute per-channel mean and std from the raw (un-normalised) 64×64
    training set.

    Returns
    -------
    mean : float
    std  : float
    """
    raw_train = PneumoniaMNIST(
        split="train",
        transform=transforms.ToTensor(),
        download=True,
        size=config_64.IMAGE_SIZE,
    )
    loader = DataLoader(raw_train, batch_size=512, shuffle=False, num_workers=0)

    # Confirm the spatial dimensions before computing stats
    sample_images, _ = next(iter(loader))
    assert sample_images.shape[1:] == (1, config_64.IMAGE_SIZE, config_64.IMAGE_SIZE), (
        f"Expected (1, {config_64.IMAGE_SIZE}, {config_64.IMAGE_SIZE}), "
        f"got {sample_images.shape[1:]}. "
        "Make sure the medmnist package supports size=64 (MedMNIST+)."
    )

    running_sum    = 0.0
    running_sq_sum = 0.0
    n_pixels       = 0

    for images, _ in loader:
        running_sum    += images.sum().item()
        running_sq_sum += (images ** 2).sum().item()
        n_pixels       += images.numel()

    mean = running_sum / n_pixels
    std  = ((running_sq_sum / n_pixels) - mean ** 2) ** 0.5
    return float(mean), float(std)


def _save_norm_stats(mean: float, std: float) -> None:
    stats = {"mean": mean, "std": std}
    with open(config_64.NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_norm_stats_64() -> tuple:
    """Load mean/std from disk. Raises FileNotFoundError if not yet computed."""
    with open(config_64.NORM_STATS_PATH, "r") as f:
        stats = json.load(f)
    return float(stats["mean"]), float(stats["std"])


def get_dataloaders_64(batch_size: int = config_64.BATCH_SIZE) -> tuple:
    """
    Download PneumoniaMNIST+ at size=64 (if necessary), compute training-set
    normalisation statistics, and return (train_loader, val_loader, test_loader).

    Normalisation stats are also saved to outputs_64/norm_stats.json.
    """
    print("Computing normalisation statistics from 64×64 training set ...")
    mean, std = _compute_train_stats()
    _save_norm_stats(mean, std)
    print(f"  mean={mean:.6f}  std={std:.6f}  → saved to {config_64.NORM_STATS_PATH}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    train_dataset = PneumoniaMNIST(
        split="train", transform=transform, download=True,
        size=config_64.IMAGE_SIZE,
    )
    val_dataset = PneumoniaMNIST(
        split="val", transform=transform, download=True,
        size=config_64.IMAGE_SIZE,
    )
    test_dataset = PneumoniaMNIST(
        split="test", transform=transform, download=True,
        size=config_64.IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, test_loader


def show_sample_grid(n_samples: int = 16) -> None:
    """
    Save a grid of raw (un-normalised) 64×64 training images to
    outputs_64/sample_grid.png.
    """
    raw_dataset = PneumoniaMNIST(
        split="train",
        transform=transforms.ToTensor(),
        download=True,
        size=config_64.IMAGE_SIZE,
    )
    loader = DataLoader(raw_dataset, batch_size=n_samples, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(
        f"PneumoniaMNIST+ ({config_64.IMAGE_SIZE}×{config_64.IMAGE_SIZE}) — Sample Training Images",
        fontsize=13,
    )

    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img = images[idx].squeeze().numpy()
            label_idx  = int(labels[idx].item())
            label_name = config_64.CLASS_NAMES[label_idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(label_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(config_64.SAMPLE_GRID_PATH, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Sample grid saved → {config_64.SAMPLE_GRID_PATH}")


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders_64()

    print(f"\nSplit sizes:")
    print(f"  Train : {len(train_loader.dataset):>5} images")
    print(f"  Val   : {len(val_loader.dataset):>5} images")
    print(f"  Test  : {len(test_loader.dataset):>5} images")

    # Verify tensor shape
    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (1, 64, 64), f"Unexpected shape: {images.shape}"
    unique_labels = labels.unique().tolist()
    assert all(int(l) in [0, 1] for l in unique_labels), f"Non-binary labels: {unique_labels}"
    print(f"\nBatch shape : {images.shape}")
    print(f"Label values: {[int(l) for l in unique_labels]}  (0=Normal, 1=Pneumonia)")

    # Verify norm_stats.json
    mean, std = load_norm_stats_64()
    assert 0.0 < mean < 1.0 and std > 0.0, "Suspicious norm stats"
    print(f"Norm stats  : mean={mean:.6f}  std={std:.6f}  [OK]")

    # Save sample grid
    show_sample_grid()
    print("\ndataset_64.py OK")
