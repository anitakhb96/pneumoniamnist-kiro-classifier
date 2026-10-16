"""
model_64.py — Lightweight CNN for 64×64 PneumoniaMNIST+ binary classification.

Architecture:
  Block 1: Conv2d(1→16,  3×3) → BatchNorm → ReLU → MaxPool(2×2)  [64→32]
  Block 2: Conv2d(16→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  [32→16]
  Block 3: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  [16→8]
           AdaptiveAvgPool2d(4×4)                                  [8→4]
  FC head: Flatten → Linear(64*4*4=1024 → 128) → ReLU → Dropout(0.4) → Linear(128→1)

AdaptiveAvgPool2d decouples the FC input size from the spatial dimensions, so
the model is robust to any future input-resolution changes.

Output: single raw logit (apply torch.sigmoid externally for probabilities).
Trainable parameters: ~140K — suitable for CPU training.

Educational/research use only. Not for clinical diagnosis.
"""

import torch
import torch.nn as nn

import config_64


class PneumoniaCNN64(nn.Module):
    """
    Three-block CNN for binary classification of 64×64 grayscale chest X-rays.

    Output is a single logit per image (shape [B, 1]).
    Use BCEWithLogitsLoss for training, torch.sigmoid for inference probabilities.
    """

    def __init__(self, dropout_rate: float = config_64.DROPOUT_RATE) -> None:
        super().__init__()

        # --- Convolutional blocks ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64×64 → 32×32
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32×32 → 16×16
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16×16 → 8×8
            nn.AdaptiveAvgPool2d((4, 4)),             #  8×8  → 4×4
        )

        # --- Fully-connected head ---
        # After AdaptiveAvgPool2d((4,4)): feature map is always 4×4 with 64 channels
        # FC input = 64 * 4 * 4 = 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),                       # single logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, 64, 64)

        Returns
        -------
        logits : torch.Tensor  shape (B, 1)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def get_model_64(
    seed: int = config_64.SEED,
    device: torch.device = config_64.DEVICE,
    dropout_rate: float = config_64.DROPOUT_RATE,
) -> PneumoniaCNN64:
    """
    Instantiate PneumoniaCNN64, seed weight initialisation, and move to device.

    Parameters
    ----------
    seed         : random seed for reproducible weight initialisation
    device       : torch.device
    dropout_rate : dropout probability for the FC head

    Returns
    -------
    model : PneumoniaCNN64 on the specified device
    """
    torch.manual_seed(seed)
    model = PneumoniaCNN64(dropout_rate=dropout_rate)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = get_model_64()
    n_params = count_parameters(model)

    print(model)
    print(f"\nDevice          : {config_64.DEVICE}")
    print(f"Dropout rate    : {config_64.DROPOUT_RATE}")
    print(f"Trainable params: {n_params:,}")

    # Forward-pass shape check with 64×64 input
    dummy_input = torch.zeros(8, 1, 64, 64, device=config_64.DEVICE)
    output = model(dummy_input)
    assert output.shape == (8, 1), f"Unexpected output shape: {output.shape}"
    print(f"Output shape    : {tuple(output.shape)}  (expected (8, 1))  [OK]")

    assert n_params < 300_000, f"Model too large: {n_params:,} params (limit 300K)"
    print(f"Parameter count : {n_params:,} < 300,000  [OK]")

    print("\nmodel_64.py OK")
